"""Boundary-safe residual diagnostics for complex forward PA models.

The residual convention is always ``e = measured_y - predicted_y``.  This
module is for architecture discovery on out-of-fold training residuals and
one-time validation confirmation.  It deliberately rejects a test split:
final test reporting belongs to the frozen evaluator, not to a feature
discovery loop.

No lagged feature may cross a caller-supplied segment boundary.  ``nperseg``
is used for spectral settings, but it is not silently assumed to identify
independent physical captures; the caller must provide ``segment_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .metrics import (
    as_complex,
    bin_am_am_am_pm,
    nmse_pooled_db,
    welch_numpy,
)

ResidualSplitRole = Literal[
    "train_oof",
    "validation_confirmation",
    "validation_reused_descriptive",
]


def _unique_integer_tuple(
    values: tuple[int, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique integers")
    return result


@dataclass(frozen=True)
class ResidualAnalysisSpec:
    """Preregistered feature families and spectral settings."""

    sample_rate_hz: float
    psd_nperseg: int
    main_bandwidth_hz: float
    adjacent_bandwidth_hz: float
    lags: tuple[int, ...] = tuple(range(-8, 65))
    envelope_lags: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    envelope_powers: tuple[int, ...] = (1, 2, 3)
    slow_time_constants_samples: tuple[float, ...] = (4.0, 16.0, 64.0, 256.0)
    amplitude_quantiles: tuple[float, ...] = (0.90, 0.95, 0.99)
    characteristic_bins: int = 32
    position_bins: int = 10
    amplitude_floor_fraction: float = 1e-6
    minimum_time_constants_per_segment: float = 20.0
    independent_capture_count: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")
        if not isinstance(self.psd_nperseg, int) or self.psd_nperseg <= 1:
            raise ValueError("psd_nperseg must be an integer greater than one")
        if (
            not np.isfinite(self.main_bandwidth_hz)
            or self.main_bandwidth_hz <= 0.0
        ):
            raise ValueError("main_bandwidth_hz must be positive and finite")
        if (
            not np.isfinite(self.adjacent_bandwidth_hz)
            or self.adjacent_bandwidth_hz <= 0.0
        ):
            raise ValueError(
                "adjacent_bandwidth_hz must be positive and finite"
            )
        lags = _unique_integer_tuple(self.lags, name="lags")
        envelope_lags = _unique_integer_tuple(
            self.envelope_lags,
            name="envelope_lags",
        )
        if any(lag < 0 for lag in envelope_lags):
            raise ValueError("envelope_lags must be causal and non-negative")
        powers = _unique_integer_tuple(
            self.envelope_powers,
            name="envelope_powers",
        )
        if any(power < 1 for power in powers):
            raise ValueError("envelope_powers must be positive")
        taus = tuple(float(value) for value in self.slow_time_constants_samples)
        if (
            not taus
            or len(set(taus)) != len(taus)
            or any(not np.isfinite(value) or value <= 0.0 for value in taus)
        ):
            raise ValueError(
                "slow_time_constants_samples must be unique positive values"
            )
        quantiles = tuple(float(value) for value in self.amplitude_quantiles)
        if (
            not quantiles
            or len(set(quantiles)) != len(quantiles)
            or any(not 0.0 < value < 1.0 for value in quantiles)
        ):
            raise ValueError("amplitude_quantiles must lie strictly in (0, 1)")
        if not isinstance(self.position_bins, int) or self.position_bins < 2:
            raise ValueError("position_bins must be an integer of at least two")
        if (
            not isinstance(self.characteristic_bins, int)
            or self.characteristic_bins < 2
        ):
            raise ValueError(
                "characteristic_bins must be an integer of at least two"
            )
        if (
            not np.isfinite(self.amplitude_floor_fraction)
            or self.amplitude_floor_fraction <= 0.0
        ):
            raise ValueError(
                "amplitude_floor_fraction must be positive and finite"
            )
        if (
            not np.isfinite(self.minimum_time_constants_per_segment)
            or self.minimum_time_constants_per_segment <= 0.0
        ):
            raise ValueError(
                "minimum_time_constants_per_segment must be positive"
            )
        if (
            not isinstance(self.independent_capture_count, int)
            or self.independent_capture_count < 0
        ):
            raise ValueError(
                "independent_capture_count must be a non-negative integer"
            )
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "envelope_lags", envelope_lags)
        object.__setattr__(self, "envelope_powers", powers)
        object.__setattr__(self, "slow_time_constants_samples", taus)
        object.__setattr__(self, "amplitude_quantiles", quantiles)


def _one_dimensional(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _paired_complex(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    predicted_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = as_complex(pa_input, name="pa_input")
    y = as_complex(measured_output, name="measured_output")
    prediction = as_complex(predicted_output, name="predicted_output")
    if x.ndim != 1 or y.ndim != 1 or prediction.ndim != 1:
        raise ValueError("PA input, measured output, and prediction must be 1-D")
    if x.shape != y.shape or x.shape != prediction.shape:
        raise ValueError("PA input, measured output, and prediction must align")
    return x, y, prediction


def boundary_safe_lagged(
    values: np.ndarray,
    segment_id: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``values[n-lag]`` and a mask that never crosses a segment.

    Positive lags are causal past samples.  Negative lags are future-sample
    diagnostics only and must never be turned into a deployable causal branch.
    """

    array = _one_dimensional(values, name="values")
    segments = _one_dimensional(segment_id, name="segment_id")
    if array.shape != segments.shape:
        raise ValueError("values and segment_id must have equal length")
    if not isinstance(lag, (int, np.integer)):
        raise TypeError("lag must be an integer")
    lag = int(lag)
    indices = np.arange(array.size)
    source = indices - lag
    in_range = (source >= 0) & (source < array.size)
    safe_source = np.clip(source, 0, array.size - 1)
    valid = in_range & (segments == segments[safe_source])
    lagged = np.zeros(array.shape, dtype=array.dtype)
    lagged[valid] = array[source[valid]]
    return lagged, valid


def _complex_correlations(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
) -> tuple[complex, complex]:
    selected_left = np.asarray(left)[valid].astype(np.complex128, copy=False)
    selected_right = np.asarray(right)[valid].astype(np.complex128, copy=False)
    if selected_left.size < 2:
        return complex(np.nan, np.nan), complex(np.nan, np.nan)
    selected_left = selected_left - np.mean(selected_left)
    selected_right = selected_right - np.mean(selected_right)
    denominator = float(
        np.sqrt(
            np.sum(np.abs(selected_left) ** 2)
            * np.sum(np.abs(selected_right) ** 2)
        )
    )
    if denominator <= 0.0:
        return complex(np.nan, np.nan), complex(np.nan, np.nan)
    proper = complex(
        np.sum(selected_left * np.conj(selected_right)) / denominator
    )
    pseudo = complex(
        np.sum(selected_left * selected_right) / denominator
    )
    return proper, pseudo


def _real_correlation(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
) -> float:
    selected_left = np.asarray(left, dtype=float)[valid]
    selected_right = np.asarray(right, dtype=float)[valid]
    if selected_left.size < 2:
        return float("nan")
    selected_left = selected_left - np.mean(selected_left)
    selected_right = selected_right - np.mean(selected_right)
    denominator = float(
        np.sqrt(
            np.sum(selected_left**2)
            * np.sum(selected_right**2)
        )
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(selected_left * selected_right) / denominator)


def _complex_value(value: complex) -> dict[str, float]:
    number = complex(value)
    return {
        "real": float(number.real),
        "imag": float(number.imag),
        "magnitude": float(abs(number)),
        "phase_rad": float(np.angle(number)),
    }


def radial_tangential_residual(
    pa_input: np.ndarray,
    residual: np.ndarray,
    *,
    amplitude_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate residual into input phase without dividing by input power."""

    x = as_complex(pa_input, name="pa_input")
    error = as_complex(residual, name="residual")
    if x.ndim != 1 or error.ndim != 1 or x.shape != error.shape:
        raise ValueError("pa_input and residual must be aligned 1-D arrays")
    if not np.isfinite(amplitude_floor) or amplitude_floor < 0.0:
        raise ValueError("amplitude_floor must be finite and non-negative")
    amplitude = np.abs(x)
    valid = amplitude > amplitude_floor
    rotated = np.zeros(x.shape, dtype=np.complex128)
    rotated[valid] = (
        error[valid] * np.conj(x[valid]) / amplitude[valid]
    )
    return rotated.real, rotated.imag, valid


def freeze_residual_reference(
    training_input: np.ndarray,
    spec: ResidualAnalysisSpec,
) -> dict[str, object]:
    """Freeze amplitude thresholds and initialization from training input."""

    x = as_complex(training_input, name="training_input")
    if x.ndim != 1:
        raise ValueError("training_input must be one-dimensional")
    amplitude = np.abs(x)
    maximum = float(np.max(amplitude))
    if maximum <= 0.0:
        raise ValueError("training_input must have non-zero amplitude")
    thresholds = {
        f"q{int(round(100.0 * quantile)):02d}": float(
            np.quantile(amplitude, quantile)
        )
        for quantile in spec.amplitude_quantiles
    }
    return {
        "amplitude_thresholds": thresholds,
        "mean_input_power": float(np.mean(amplitude**2)),
        "maximum_input_amplitude": maximum,
        "amplitude_floor": spec.amplitude_floor_fraction * maximum,
        "characteristic_bin_edges": np.linspace(
            0.0,
            maximum,
            spec.characteristic_bins + 1,
        ),
        "source": "training input only",
    }


def slow_envelope_state(
    input_power: np.ndarray,
    segment_id: np.ndarray,
    *,
    time_constant_samples: float,
    initial_power: float,
) -> np.ndarray:
    """Causal one-pole envelope state reset at explicit segment boundaries."""

    power = _one_dimensional(input_power, name="input_power").astype(float)
    segments = _one_dimensional(segment_id, name="segment_id")
    if power.shape != segments.shape:
        raise ValueError("input_power and segment_id must have equal length")
    if np.any(power < 0.0):
        raise ValueError("input_power cannot be negative")
    if not np.isfinite(time_constant_samples) or time_constant_samples <= 0.0:
        raise ValueError("time_constant_samples must be positive and finite")
    if not np.isfinite(initial_power) or initial_power < 0.0:
        raise ValueError("initial_power must be finite and non-negative")
    alpha = float(np.exp(-1.0 / float(time_constant_samples)))
    result = np.empty(power.shape, dtype=float)
    state = float(initial_power)
    previous_segment: object | None = None
    for index in range(power.size):
        current_segment = segments[index]
        if index == 0 or current_segment != previous_segment:
            state = float(initial_power)
        state = alpha * state + (1.0 - alpha) * power[index]
        result[index] = state
        previous_segment = current_segment
    return result


def _lag_diagnostics(
    x: np.ndarray,
    error: np.ndarray,
    segments: np.ndarray,
    base_valid: np.ndarray,
    spec: ResidualAnalysisSpec,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for lag in spec.lags:
        lagged_x, lag_valid = boundary_safe_lagged(x, segments, lag)
        lagged_error, error_lag_valid = boundary_safe_lagged(
            error,
            segments,
            lag,
        )
        valid = base_valid & lag_valid
        proper, pseudo = _complex_correlations(error, lagged_x, valid)
        acf_valid = base_valid & error_lag_valid
        residual_acf, _ = _complex_correlations(
            error,
            lagged_error,
            acf_valid,
        )
        result.append(
            {
                "lag_samples": lag,
                "lag_seconds": lag / spec.sample_rate_hz,
                "causal_feature_eligible": lag >= 0,
                "valid_sample_count": int(np.count_nonzero(valid)),
                "proper_complex_correlation": _complex_value(proper),
                "pseudo_complex_correlation": _complex_value(pseudo),
                "corr_error_i_input_i": _real_correlation(
                    error.real,
                    lagged_x.real,
                    valid,
                ),
                "corr_error_i_input_q": _real_correlation(
                    error.real,
                    lagged_x.imag,
                    valid,
                ),
                "corr_error_q_input_i": _real_correlation(
                    error.imag,
                    lagged_x.real,
                    valid,
                ),
                "corr_error_q_input_q": _real_correlation(
                    error.imag,
                    lagged_x.imag,
                    valid,
                ),
                "residual_acf": _complex_value(residual_acf),
            }
        )
    return result


def _envelope_diagnostics(
    x: np.ndarray,
    error: np.ndarray,
    radial: np.ndarray,
    tangential: np.ndarray,
    rotated_valid: np.ndarray,
    segments: np.ndarray,
    base_valid: np.ndarray,
    spec: ResidualAnalysisSpec,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for lag in spec.envelope_lags:
        lagged_x, lag_valid = boundary_safe_lagged(x, segments, lag)
        amplitude = np.abs(lagged_x)
        valid = base_valid & lag_valid & rotated_valid
        for power in spec.envelope_powers:
            envelope_feature = amplitude**power
            complex_feature = lagged_x * amplitude ** (power - 1)
            proper, pseudo = _complex_correlations(
                error,
                complex_feature,
                valid,
            )
            result.append(
                {
                    "lag_samples": lag,
                    "lag_seconds": lag / spec.sample_rate_hz,
                    "envelope_power": power,
                    "valid_sample_count": int(np.count_nonzero(valid)),
                    "corr_radial_envelope": _real_correlation(
                        radial,
                        envelope_feature,
                        valid,
                    ),
                    "corr_tangential_envelope": _real_correlation(
                        tangential,
                        envelope_feature,
                        valid,
                    ),
                    "proper_candidate_feature_correlation": _complex_value(
                        proper
                    ),
                    "pseudo_candidate_feature_correlation": _complex_value(
                        pseudo
                    ),
                }
            )
    return result


def _residualize_slow_state(
    state: np.ndarray,
    power: np.ndarray,
    segments: np.ndarray,
    base_valid: np.ndarray,
    short_lags: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, int]:
    columns = [np.ones(power.size, dtype=float)]
    valid = base_valid.copy()
    for lag in short_lags:
        lagged, lag_valid = boundary_safe_lagged(power, segments, lag)
        columns.append(lagged.astype(float))
        valid &= lag_valid
    design = np.column_stack(columns)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design[valid],
        state[valid],
        rcond=None,
    )
    residual = state - design @ coefficients
    return residual, valid, int(rank)


def _slow_state_diagnostics(
    x: np.ndarray,
    radial: np.ndarray,
    tangential: np.ndarray,
    rotated_valid: np.ndarray,
    segments: np.ndarray,
    base_valid: np.ndarray,
    spec: ResidualAnalysisSpec,
    frozen_reference: dict[str, object],
) -> list[dict[str, object]]:
    power = np.abs(x) ** 2
    unique_segments, counts = np.unique(segments, return_counts=True)
    minimum_segment_length = int(np.min(counts))
    short_lags = tuple(
        lag for lag in spec.envelope_lags if lag <= max(8, min(spec.envelope_lags))
    )
    result: list[dict[str, object]] = []
    for tau in spec.slow_time_constants_samples:
        state = slow_envelope_state(
            power,
            segments,
            time_constant_samples=tau,
            initial_power=float(frozen_reference["mean_input_power"]),
        )
        conditioned, valid, rank = _residualize_slow_state(
            state,
            power,
            segments,
            base_valid & rotated_valid,
            short_lags,
        )
        observable_ratio = minimum_segment_length / tau
        observable = (
            observable_ratio >= spec.minimum_time_constants_per_segment
        )
        result.append(
            {
                "time_constant_samples": tau,
                "time_constant_seconds": tau / spec.sample_rate_hz,
                "segment_count": int(unique_segments.size),
                "minimum_segment_length_samples": minimum_segment_length,
                "minimum_segment_time_constants": observable_ratio,
                "observable_by_project_rule": bool(observable),
                "eligible_for_state_branch_selection": bool(
                    observable and spec.independent_capture_count >= 2
                ),
                "declared_independent_capture_count": (
                    spec.independent_capture_count
                ),
                "short_power_lags_projected_out": list(short_lags),
                "residualization_rank": rank,
                "valid_sample_count": int(np.count_nonzero(valid)),
                "corr_radial_conditioned_slow_state": _real_correlation(
                    radial,
                    conditioned,
                    valid,
                ),
                "corr_tangential_conditioned_slow_state": _real_correlation(
                    tangential,
                    conditioned,
                    valid,
                ),
            }
        )
    return result


def _conditional_metrics(
    error: np.ndarray,
    reference: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int | None]:
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {
            "sample_count": 0,
            "mse": None,
            "relative_error_power": None,
            "nmse_db": None,
        }
    error_power = float(np.mean(np.abs(error[selected]) ** 2))
    reference_power = float(np.mean(np.abs(reference[selected]) ** 2))
    if reference_power <= 0.0:
        return {
            "sample_count": count,
            "mse": error_power,
            "relative_error_power": None,
            "nmse_db": None,
        }
    relative = error_power / reference_power
    return {
        "sample_count": count,
        "mse": error_power,
        "relative_error_power": relative,
        "nmse_db": (
            float(10.0 * np.log10(relative))
            if relative > 0.0
            else float("-inf")
        ),
    }


def _amplitude_region_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    error: np.ndarray,
    base_valid: np.ndarray,
    frozen_reference: dict[str, object],
) -> list[dict[str, object]]:
    amplitude = np.abs(x)
    result: list[dict[str, object]] = []
    thresholds = frozen_reference["amplitude_thresholds"]
    if not isinstance(thresholds, dict):
        raise ValueError("frozen amplitude_thresholds must be a mapping")
    for name, raw_threshold in thresholds.items():
        threshold = float(raw_threshold)
        high = base_valid & (amplitude >= threshold)
        low = base_valid & (amplitude < threshold)
        result.append(
            {
                "threshold_name": str(name),
                "threshold_amplitude": threshold,
                "high_region": _conditional_metrics(error, y, high),
                "complement_region": _conditional_metrics(error, y, low),
            }
        )
    return result


def _characteristic_residuals(
    x: np.ndarray,
    y: np.ndarray,
    prediction: np.ndarray,
    frozen_reference: dict[str, object],
) -> dict[str, object]:
    edges = np.asarray(
        frozen_reference["characteristic_bin_edges"],
        dtype=float,
    )
    measured = bin_am_am_am_pm(x, y, bins=edges)
    predicted = bin_am_am_am_pm(x, prediction, bins=edges)
    phase_residual = np.angle(
        np.exp(1j * (predicted["am_pm_rad"] - measured["am_pm_rad"]))
    )
    return {
        "bin_edges": edges,
        "bin_centers": measured["bin_centers"],
        "count": measured["count"],
        "measured_output_amplitude_mean": measured["output_amplitude_mean"],
        "predicted_output_amplitude_mean": predicted[
            "output_amplitude_mean"
        ],
        "output_amplitude_mean_residual": (
            predicted["output_amplitude_mean"]
            - measured["output_amplitude_mean"]
        ),
        "measured_am_am_gain": measured["am_am_gain"],
        "predicted_am_am_gain": predicted["am_am_gain"],
        "am_am_gain_residual": (
            predicted["am_am_gain"] - measured["am_am_gain"]
        ),
        "measured_am_pm_deg": measured["am_pm_deg"],
        "predicted_am_pm_deg": predicted["am_pm_deg"],
        "am_pm_residual_rad": phase_residual,
        "am_pm_residual_deg": np.rad2deg(phase_residual),
        "bin_definition": (
            "uniform amplitude edges frozen from training maximum"
        ),
    }


def _segment_position_diagnostics(
    y: np.ndarray,
    error: np.ndarray,
    segments: np.ndarray,
    base_valid: np.ndarray,
    *,
    bins: int,
) -> list[dict[str, object]]:
    position = np.zeros(error.size, dtype=float)
    for segment in np.unique(segments):
        indices = np.flatnonzero(segments == segment)
        if indices.size == 1:
            position[indices] = 0.0
        else:
            position[indices] = np.linspace(0.0, 1.0, indices.size)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.searchsorted(edges, position, side="right") - 1
    bin_index[position == 1.0] = bins - 1
    result: list[dict[str, object]] = []
    for index in range(bins):
        selected = base_valid & (bin_index == index)
        result.append(
            {
                "position_bin": index,
                "position_low": float(edges[index]),
                "position_high": float(edges[index + 1]),
                **_conditional_metrics(error, y, selected),
            }
        )
    return result


def _integrated_psd_ratio_db(
    frequencies: np.ndarray,
    error_density: np.ndarray,
    reference_density: np.ndarray,
    low: float,
    high: float,
) -> tuple[float | None, int]:
    selected = (frequencies >= low) & (frequencies < high)
    count = int(np.count_nonzero(selected))
    if count == 0:
        return None, 0
    numerator = float(np.sum(error_density[selected]))
    denominator = float(np.sum(reference_density[selected]))
    if denominator <= 0.0:
        return None, count
    if numerator == 0.0:
        return float("-inf"), count
    return float(10.0 * np.log10(numerator / denominator)), count


def _residual_psd(
    error: np.ndarray,
    reference: np.ndarray,
    segments: np.ndarray,
    spec: ResidualAnalysisSpec,
) -> dict[str, object]:
    error_spectra: list[np.ndarray] = []
    reference_spectra: list[np.ndarray] = []
    included_segments: list[object] = []
    excluded_segments: list[object] = []
    frequencies: np.ndarray | None = None
    for segment in np.unique(segments):
        selected = segments == segment
        if np.count_nonzero(selected) < spec.psd_nperseg:
            excluded_segments.append(segment.item() if hasattr(segment, "item") else segment)
            continue
        frequencies, error_density = welch_numpy(
            error[selected],
            fs=spec.sample_rate_hz,
            nperseg=spec.psd_nperseg,
            noverlap=spec.psd_nperseg // 2,
            scaling="density",
            detrend="constant",
        )
        _, reference_density = welch_numpy(
            reference[selected],
            fs=spec.sample_rate_hz,
            nperseg=spec.psd_nperseg,
            noverlap=spec.psd_nperseg // 2,
            scaling="density",
            detrend="constant",
        )
        error_spectra.append(error_density)
        reference_spectra.append(reference_density)
        included_segments.append(
            segment.item() if hasattr(segment, "item") else segment
        )
    if frequencies is None:
        return {
            "available": False,
            "reason": "no explicit segment is at least psd_nperseg samples",
            "included_segment_ids": included_segments,
            "excluded_segment_ids": excluded_segments,
        }

    frequencies = np.fft.fftshift(frequencies)
    error_density = np.fft.fftshift(np.mean(error_spectra, axis=0))
    reference_density = np.fft.fftshift(
        np.mean(reference_spectra, axis=0)
    )
    half_main = spec.main_bandwidth_hz / 2.0
    adjacent = spec.adjacent_bandwidth_hz
    bands = {
        "left_adjacent": (-half_main - adjacent, -half_main),
        "main": (-half_main, half_main),
        "right_adjacent": (half_main, half_main + adjacent),
    }
    integrated: dict[str, object] = {}
    for name, (low, high) in bands.items():
        ratio, bin_count = _integrated_psd_ratio_db(
            frequencies,
            error_density,
            reference_density,
            low,
            high,
        )
        integrated[name] = {
            "low_hz": low,
            "high_hz": high,
            "fft_bin_count": bin_count,
            "error_to_measured_output_power_db": ratio,
        }
    return {
        "available": True,
        "frequency_hz": frequencies,
        "error_density": error_density,
        "measured_output_density": reference_density,
        "integrated_bands": integrated,
        "included_segment_ids": included_segments,
        "excluded_segment_ids": excluded_segments,
        "window": "periodic_hann",
        "noverlap": spec.psd_nperseg // 2,
        "nperseg": spec.psd_nperseg,
        "detrend": "constant",
        "scaling": "density",
        "valid_mask_policy": (
            "full explicit segments including cold start; diagnostic valid_mask "
            "is not applied to spectral windows"
        ),
        "uncertainty_scope": (
            "within-capture descriptive; independent-capture CI unavailable"
        ),
    }


def analyze_pa_residuals(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    predicted_output: np.ndarray,
    *,
    segment_id: np.ndarray,
    valid_mask: np.ndarray,
    split_role: ResidualSplitRole,
    spec: ResidualAnalysisSpec,
    frozen_reference: dict[str, object] | None = None,
) -> dict[str, object]:
    """Analyze omitted structure without permitting test-driven discovery."""

    if split_role not in {
        "train_oof",
        "validation_confirmation",
        "validation_reused_descriptive",
    }:
        raise ValueError(
            "split_role must be train_oof, validation_confirmation, or "
            "validation_reused_descriptive; "
            "test residuals are report-only"
        )
    x, y, prediction = _paired_complex(
        pa_input,
        measured_output,
        predicted_output,
    )
    segments = _one_dimensional(segment_id, name="segment_id")
    valid = np.asarray(valid_mask)
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError("valid_mask must be a one-dimensional boolean array")
    if segments.shape != x.shape or valid.shape != x.shape:
        raise ValueError("segment_id and valid_mask must align with signals")
    if not np.any(valid):
        raise ValueError("valid_mask excludes every sample")
    if split_role == "train_oof":
        if frozen_reference is None:
            frozen_reference = freeze_residual_reference(x, spec)
    elif frozen_reference is None:
        raise ValueError(
            "validation roles require train-frozen thresholds/state"
        )
    assert frozen_reference is not None

    error = y - prediction
    amplitude_floor = float(frozen_reference["amplitude_floor"])
    radial, tangential, rotated_valid = radial_tangential_residual(
        x,
        error,
        amplitude_floor=amplitude_floor,
    )
    segment_count = int(np.unique(segments).size)
    return {
        "schema_version": 1,
        "scope": "forward_pa_model_residual_discovery",
        "split_role": split_role,
        "test_access_permitted": False,
        "residual_definition": "measured_y - predicted_y",
        "sample_count": int(x.size),
        "valid_sample_count": int(np.count_nonzero(valid)),
        "segment_count": segment_count,
        "segment_independence_status": (
            "caller-supplied boundaries; physical capture independence not assumed"
        ),
        "spec": asdict(spec),
        "frozen_reference": frozen_reference,
        "global_metrics": {
            "pooled_complex_nmse_db": nmse_pooled_db(
                prediction[valid],
                y[valid],
            ),
            "mean_residual_real": float(np.mean(error[valid].real)),
            "mean_residual_imag": float(np.mean(error[valid].imag)),
            "residual_power": float(np.mean(np.abs(error[valid]) ** 2)),
            "radial_bias": float(
                np.mean(radial[valid & rotated_valid])
            ),
            "tangential_bias": float(
                np.mean(tangential[valid & rotated_valid])
            ),
            "radial_rms": float(
                np.sqrt(np.mean(radial[valid & rotated_valid] ** 2))
            ),
            "tangential_rms": float(
                np.sqrt(np.mean(tangential[valid & rotated_valid] ** 2))
            ),
        },
        "lag_correlations": _lag_diagnostics(
            x,
            error,
            segments,
            valid,
            spec,
        ),
        "envelope_correlations": _envelope_diagnostics(
            x,
            error,
            radial,
            tangential,
            rotated_valid,
            segments,
            valid,
            spec,
        ),
        "slow_state_correlations": _slow_state_diagnostics(
            x,
            radial,
            tangential,
            rotated_valid,
            segments,
            valid,
            spec,
            frozen_reference,
        ),
        "amplitude_regions": _amplitude_region_diagnostics(
            x,
            y,
            error,
            valid,
            frozen_reference,
        ),
        "am_am_am_pm_residuals": _characteristic_residuals(
            x,
            y,
            prediction,
            frozen_reference,
        ),
        "segment_position": _segment_position_diagnostics(
            y,
            error,
            segments,
            valid,
            bins=spec.position_bins,
        ),
        "error_psd": _residual_psd(
            error,
            y,
            segments,
            spec,
        ),
        "statistical_limit": (
            "effect sizes are descriptive; no iid-sample p-values or "
            "independent-capture confidence intervals are claimed"
        ),
    }
