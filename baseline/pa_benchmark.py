"""Frozen, model-agnostic evaluator for forward PA identification.

The only scored direction in this module is

``measured PA input x -> PA model -> predicted measured PA output y_hat``.

Alignment, gain diagnostics, framing, spectral settings, and AM/AM-AM/PM bin
edges are frozen from the training split.  Validation and test outputs are
never used to refit delay, gain, or bins.  In particular, the evaluator does
not align a model prediction to the measured output after inference.

The evaluator intentionally accepts a callable rather than a target array as
the model input.  This makes it difficult to accidentally implement the
circular inverse diagnostic ``y -> inverse -> forward -> y`` in the PA-model
benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable, Literal

import numpy as np

from .alignment import (
    complex_ls_gain,
    estimate_integer_delay,
    fractional_delay_diagnostic,
    overlap_for_delay,
)
from .complexity import OperationCount
from .metrics import (
    as_complex,
    bin_am_am_am_pm,
    nmse_opendpd_db,
    nmse_pooled_db,
    time_domain_rms_evm_db,
    welch_numpy,
)
from .pa_models import segmented_steady_state_mask

EvaluationPurpose = Literal["model_selection", "diagnostic", "final_report"]


def _complex_vector(signal: np.ndarray, *, name: str) -> np.ndarray:
    array = as_complex(signal, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional complex sequence")
    return array


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class PAEvaluationProtocol:
    """Quantities frozen using the training split only."""

    schema_version: int
    dataset_label: str
    training_sample_count_raw: int
    training_sample_count_aligned: int
    alignment_delay_samples: int
    alignment_definition: str
    fractional_delay_estimate_samples: float
    fractional_delay_offset_samples: float
    fractional_delay_peak_score: float
    fractional_delay_reliable: bool
    fractional_delay_applied: bool
    training_complex_ls_gain_real: float
    training_complex_ls_gain_imag: float
    training_opendpd_peak_gain: float
    training_maximum_input_amplitude: float
    characteristic_bin_edges: tuple[float, ...]
    sample_rate_hz: float
    nperseg: int
    main_bandwidth_hz: float
    subchannel_count: int
    frame_state_policy: str = "reset_at_each_nperseg_frame"
    partial_final_frame_policy: str = "evaluate_as_short_independent_frame"
    score_gain_policy: str = "no_post_prediction_gain_fit"
    nonzero_delay_frame_policy: str = (
        "unsupported_until_frame_safe_alignment_is_explicitly_configured"
    )

    @property
    def training_complex_ls_gain(self) -> complex:
        return complex(
            self.training_complex_ls_gain_real,
            self.training_complex_ls_gain_imag,
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["characteristic_bin_edges"] = list(self.characteristic_bin_edges)
        return result


@dataclass(frozen=True)
class PAEvaluationResult:
    """Stable result schema for one model on one already-frozen split."""

    schema_version: int
    model_label: str
    split: str
    purpose: EvaluationPurpose
    scope: str
    direction: str
    precision_label: str
    protocol: dict[str, object]
    sample_counts: dict[str, int]
    full_record_metrics: dict[str, float]
    steady_state_metrics: dict[str, float | int]
    opendpd_compatible_metrics: dict[str, float | int | None]
    error_psd: dict[str, object]
    characteristic_residuals: dict[str, object]
    input_support: dict[str, float]
    timing: dict[str, float | int | None | str]
    complexity: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def freeze_pa_evaluation_protocol(
    training_input: np.ndarray,
    measured_training_output: np.ndarray,
    *,
    dataset_label: str,
    sample_rate_hz: float,
    nperseg: int,
    main_bandwidth_hz: float,
    subchannel_count: int,
    alignment_max_abs_delay: int = 32,
    alignment_delay: int | None = None,
    characteristic_bins: int = 32,
) -> PAEvaluationProtocol:
    """Freeze evaluator nuisance quantities from training data only.

    Fractional delay is recorded as a diagnostic but is not silently applied.
    Applying it requires a separately specified band-limited resampler and
    must create a new protocol version.
    """

    x_raw = _complex_vector(training_input, name="training_input")
    y_raw = _complex_vector(
        measured_training_output,
        name="measured_training_output",
    )
    if x_raw.shape != y_raw.shape:
        raise ValueError("training input and measured output must have equal length")
    if not isinstance(dataset_label, str) or not dataset_label.strip():
        raise ValueError("dataset_label must be a non-empty string")
    if (
        not isinstance(alignment_max_abs_delay, (int, np.integer))
        or int(alignment_max_abs_delay) < 0
    ):
        raise ValueError("alignment_max_abs_delay must be a non-negative integer")
    alignment_max_abs_delay = int(alignment_max_abs_delay)
    if alignment_delay is None:
        frozen_delay = estimate_integer_delay(
            x_raw,
            y_raw,
            alignment_max_abs_delay,
        )
        alignment_definition = (
            "maximum normalized complex-correlation power on training only"
        )
    else:
        if not isinstance(alignment_delay, (int, np.integer)):
            raise TypeError("alignment_delay must be an integer or None")
        frozen_delay = int(alignment_delay)
        alignment_definition = "explicit user-supplied delay; frozen for all splits"

    diagnostic = fractional_delay_diagnostic(
        x_raw,
        y_raw,
        max(alignment_max_abs_delay, abs(frozen_delay)),
    )
    x_train, y_train = overlap_for_delay(x_raw, y_raw, frozen_delay)
    gain = complex_ls_gain(x_train, y_train)
    input_peak = float(np.max(np.abs(x_train)))
    output_peak = float(np.max(np.abs(y_train)))
    if input_peak <= 0.0:
        raise ValueError("training input must have non-zero peak amplitude")

    if (
        not isinstance(characteristic_bins, (int, np.integer))
        or int(characteristic_bins) < 2
    ):
        raise ValueError("characteristic_bins must be an integer of at least two")
    characteristic_edges = np.linspace(
        0.0,
        input_peak,
        int(characteristic_bins) + 1,
    )
    if not isinstance(nperseg, (int, np.integer)) or int(nperseg) <= 1:
        raise ValueError("nperseg must be an integer greater than one")
    if (
        not isinstance(subchannel_count, (int, np.integer))
        or int(subchannel_count) <= 0
    ):
        raise ValueError("subchannel_count must be a positive integer")

    return PAEvaluationProtocol(
        schema_version=1,
        dataset_label=dataset_label.strip(),
        training_sample_count_raw=int(x_raw.size),
        training_sample_count_aligned=int(x_train.size),
        alignment_delay_samples=frozen_delay,
        alignment_definition=alignment_definition,
        fractional_delay_estimate_samples=float(diagnostic.estimated_delay),
        fractional_delay_offset_samples=float(diagnostic.fractional_offset),
        fractional_delay_peak_score=float(diagnostic.peak_score),
        fractional_delay_reliable=bool(diagnostic.reliable),
        fractional_delay_applied=False,
        training_complex_ls_gain_real=float(gain.real),
        training_complex_ls_gain_imag=float(gain.imag),
        training_opendpd_peak_gain=output_peak / input_peak,
        training_maximum_input_amplitude=input_peak,
        characteristic_bin_edges=tuple(float(v) for v in characteristic_edges),
        sample_rate_hz=_positive_float(sample_rate_hz, name="sample_rate_hz"),
        nperseg=int(nperseg),
        main_bandwidth_hz=_positive_float(
            main_bandwidth_hz,
            name="main_bandwidth_hz",
        ),
        subchannel_count=int(subchannel_count),
    )


def prepare_pa_split(
    pa_input: np.ndarray,
    measured_pa_output: np.ndarray,
    protocol: PAEvaluationProtocol,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the training-frozen integer delay to a later split."""

    x_raw = _complex_vector(pa_input, name="pa_input")
    y_raw = _complex_vector(measured_pa_output, name="measured_pa_output")
    if x_raw.shape != y_raw.shape:
        raise ValueError("PA input and measured output must have equal length")
    return overlap_for_delay(
        x_raw,
        y_raw,
        protocol.alignment_delay_samples,
    )


def _predict_framed(
    predictor: Callable[[np.ndarray], np.ndarray],
    pa_input: np.ndarray,
    *,
    nperseg: int,
) -> tuple[np.ndarray, float, int]:
    """Invoke a reset-per-frame predictor without a per-sample Python loop."""

    predictions: list[np.ndarray] = []
    start_time = time.perf_counter()
    call_count = 0
    for start in range(0, pa_input.size, nperseg):
        frame = pa_input[start : min(start + nperseg, pa_input.size)]
        predicted = _complex_vector(
            predictor(frame),
            name="predictor output",
        )
        if predicted.shape != frame.shape:
            raise ValueError("predictor output must match every input frame shape")
        predictions.append(predicted)
        call_count += 1
    elapsed = time.perf_counter() - start_time
    return np.concatenate(predictions), elapsed, call_count


def _time_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    error = estimate - reference
    mse = float(np.mean(np.abs(error) ** 2))
    reference_power = float(np.mean(np.abs(reference) ** 2))
    if reference_power <= 0.0:
        raise ValueError("measured PA output must have positive energy")
    return {
        "mse": mse,
        "reference_power": reference_power,
        "relative_error_power": mse / reference_power,
        "complex_nmse_pooled_db": nmse_pooled_db(estimate, reference),
        "time_domain_rms_sample_evm_db": time_domain_rms_evm_db(
            estimate,
            reference,
        ),
    }


def _steady_state_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    nperseg: int,
    warmup_samples: int,
) -> dict[str, float | int]:
    mask = segmented_steady_state_mask(
        estimate.size,
        segment_length=nperseg,
        warmup_samples=warmup_samples,
    )
    result: dict[str, float | int] = _time_metrics(
        estimate[mask],
        reference[mask],
    )
    result["warmup_samples_per_frame"] = int(warmup_samples)
    result["scored_sample_count"] = int(np.count_nonzero(mask))
    result["discarded_sample_count"] = int(mask.size - np.count_nonzero(mask))
    return result


def _opendpd_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    nperseg: int,
) -> dict[str, float | int | None]:
    complete_samples = (estimate.size // nperseg) * nperseg
    if complete_samples == 0:
        return {
            "nmse_mean_segment_db": None,
            "complete_frame_count": 0,
            "scored_sample_count": 0,
            "discarded_partial_tail_samples": int(estimate.size),
        }
    estimate_frames = estimate[:complete_samples].reshape(-1, nperseg)
    reference_frames = reference[:complete_samples].reshape(-1, nperseg)
    return {
        "nmse_mean_segment_db": nmse_opendpd_db(
            estimate_frames,
            reference_frames,
        ),
        "complete_frame_count": int(estimate_frames.shape[0]),
        "scored_sample_count": int(complete_samples),
        "discarded_partial_tail_samples": int(estimate.size - complete_samples),
    }


def _error_psd(
    error: np.ndarray,
    reference: np.ndarray,
    protocol: PAEvaluationProtocol,
) -> dict[str, object]:
    if error.size < protocol.nperseg:
        return {
            "available": False,
            "reason": "aligned split is shorter than nperseg",
            "frequency_hz": np.asarray([], dtype=float),
            "density": np.asarray([], dtype=float),
            "density_relative_to_reference_power_db_per_hz": np.asarray(
                [],
                dtype=float,
            ),
        }

    frequencies, density = welch_numpy(
        error,
        fs=protocol.sample_rate_hz,
        nperseg=protocol.nperseg,
        noverlap=protocol.nperseg // 2,
        scaling="density",
        detrend="constant",
    )
    _, reference_density = welch_numpy(
        reference,
        fs=protocol.sample_rate_hz,
        nperseg=protocol.nperseg,
        noverlap=protocol.nperseg // 2,
        scaling="density",
        detrend="constant",
    )
    frequencies = np.fft.fftshift(frequencies)
    density = np.fft.fftshift(density)
    reference_density = np.fft.fftshift(reference_density)
    bin_width = protocol.sample_rate_hz / protocol.nperseg
    reference_power = float(np.sum(reference_density) * bin_width)
    if reference_power <= 0.0:
        raise ValueError("reference Welch spectrum has zero integrated power")
    relative_density = density / reference_power
    relative_db = np.full(relative_density.shape, -np.inf, dtype=float)
    positive = relative_density > 0.0
    relative_db[positive] = 10.0 * np.log10(relative_density[positive])
    return {
        "available": True,
        "frequency_hz": frequencies,
        "density": density,
        "density_relative_to_reference_power_db_per_hz": relative_db,
        "welch_nperseg": protocol.nperseg,
        "welch_noverlap": protocol.nperseg // 2,
        "window": "periodic_hann",
        "detrend": "constant",
        "scaling": "density",
        "reference_power_from_integrated_density": reference_power,
    }


def _characteristic_residuals(
    pa_input: np.ndarray,
    prediction: np.ndarray,
    measured_output: np.ndarray,
    protocol: PAEvaluationProtocol,
) -> dict[str, object]:
    edges = np.asarray(protocol.characteristic_bin_edges, dtype=float)
    measured = bin_am_am_am_pm(
        pa_input,
        measured_output,
        bins=edges,
    )
    predicted = bin_am_am_am_pm(
        pa_input,
        prediction,
        bins=edges,
    )
    phase_residual = np.angle(
        np.exp(1j * (predicted["am_pm_rad"] - measured["am_pm_rad"]))
    )
    return {
        "bin_edges": edges,
        "bin_centers": measured["bin_centers"],
        "count": measured["count"],
        "measured_output_amplitude_mean": measured["output_amplitude_mean"],
        "predicted_output_amplitude_mean": predicted["output_amplitude_mean"],
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
        "bin_definition": "uniform amplitude edges frozen from training maximum",
    }


def _complexity_report(
    operation_count: OperationCount | None,
    *,
    trainable_real_parameter_count: int | None,
) -> dict[str, object] | None:
    if operation_count is None and trainable_real_parameter_count is None:
        return None
    if trainable_real_parameter_count is not None:
        if (
            not isinstance(trainable_real_parameter_count, (int, np.integer))
            or int(trainable_real_parameter_count) < 0
        ):
            raise ValueError(
                "trainable_real_parameter_count must be a non-negative integer"
            )
        trainable_real_parameter_count = int(trainable_real_parameter_count)
    operations = (
        operation_count.to_dict()
        if operation_count is not None
        else None
    )
    memory: dict[str, int] | None = None
    if operation_count is not None:
        coefficient_and_constant_reals = (
            operation_count.stored_real_coefficients
            + operation_count.stored_real_constants
        )
        memory = {
            "coefficient_bytes_fp32": 4
            * operation_count.stored_real_coefficients,
            "constant_bytes_fp32": 4 * operation_count.stored_real_constants,
            "state_bytes_fp32": 4 * operation_count.state_real_values,
            "total_model_and_state_bytes_fp32": 4
            * (
                coefficient_and_constant_reals
                + operation_count.state_real_values
            ),
        }
    return {
        "operation_count_per_complex_sample": operations,
        "trainable_real_parameter_count": trainable_real_parameter_count,
        "memory": memory,
        "complex_multiply_convention": "4 real MUL + 2 real ADD",
        "fma_convention": "1 real MUL + 1 real ADD",
    }


def evaluate_pa_predictor(
    predictor: Callable[[np.ndarray], np.ndarray],
    pa_input: np.ndarray,
    measured_pa_output: np.ndarray,
    *,
    protocol: PAEvaluationProtocol,
    model_label: str,
    split: Literal["train", "validation", "test"],
    purpose: EvaluationPurpose,
    common_warmup_samples: int = 0,
    operation_count: OperationCount | None = None,
    trainable_real_parameter_count: int | None = None,
    fit_seconds: float | None = None,
    precision_label: str = "numpy_complex128",
) -> tuple[PAEvaluationResult, np.ndarray]:
    """Evaluate one frozen forward PA model and return its aligned prediction.

    A test split cannot be marked for model selection.  This is a guardrail,
    not a complete experiment tracker: callers must still freeze the selected
    architecture and coefficients before invoking the final test report.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if purpose not in {"model_selection", "diagnostic", "final_report"}:
        raise ValueError(
            "purpose must be model_selection, diagnostic, or final_report"
        )
    if split == "test" and purpose == "model_selection":
        raise ValueError("test data cannot be used for model selection")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be a non-empty string")
    if protocol.alignment_delay_samples != 0:
        raise NotImplementedError(
            "non-zero delay would shift flattened nperseg boundaries; "
            "regenerate frame-aligned pairs or add an explicit frame-safe "
            "alignment policy before scoring"
        )
    if (
        not isinstance(common_warmup_samples, (int, np.integer))
        or int(common_warmup_samples) < 0
        or int(common_warmup_samples) >= protocol.nperseg
    ):
        raise ValueError(
            "common_warmup_samples must satisfy 0 <= warmup < nperseg"
        )
    common_warmup_samples = int(common_warmup_samples)
    if fit_seconds is not None:
        fit_seconds = float(fit_seconds)
        if not np.isfinite(fit_seconds) or fit_seconds < 0.0:
            raise ValueError("fit_seconds must be finite and non-negative")

    raw_sample_count = int(_complex_vector(pa_input, name="pa_input").size)
    x_aligned, y_aligned = prepare_pa_split(
        pa_input,
        measured_pa_output,
        protocol,
    )
    prediction, prediction_seconds, predictor_calls = _predict_framed(
        predictor,
        x_aligned,
        nperseg=protocol.nperseg,
    )
    full_metrics = _time_metrics(prediction, y_aligned)
    steady_metrics = _steady_state_metrics(
        prediction,
        y_aligned,
        nperseg=protocol.nperseg,
        warmup_samples=common_warmup_samples,
    )
    support_extrapolation = float(
        np.mean(
            np.abs(x_aligned)
            > protocol.training_maximum_input_amplitude
        )
    )
    throughput = (
        float(x_aligned.size / prediction_seconds)
        if prediction_seconds > 0.0
        else float("inf")
    )
    result = PAEvaluationResult(
        schema_version=1,
        model_label=model_label.strip(),
        split=split,
        purpose=purpose,
        scope="measured_forward_pa_identification",
        direction="x_split -> frozen PA model -> y_hat_split; compare with measured y",
        precision_label=precision_label,
        protocol=protocol.to_dict(),
        sample_counts={
            "raw": raw_sample_count,
            "aligned": int(x_aligned.size),
            "discarded_by_frozen_integer_alignment": int(
                raw_sample_count - x_aligned.size
            ),
        },
        full_record_metrics=full_metrics,
        steady_state_metrics=steady_metrics,
        opendpd_compatible_metrics=_opendpd_metrics(
            prediction,
            y_aligned,
            nperseg=protocol.nperseg,
        ),
        error_psd=_error_psd(
            prediction - y_aligned,
            y_aligned,
            protocol,
        ),
        characteristic_residuals=_characteristic_residuals(
            x_aligned,
            prediction,
            y_aligned,
            protocol,
        ),
        input_support={
            "training_maximum_input_amplitude": (
                protocol.training_maximum_input_amplitude
            ),
            "split_maximum_input_amplitude": float(
                np.max(np.abs(x_aligned))
            ),
            "fraction_above_training_maximum": support_extrapolation,
        },
        timing={
            "fit_seconds": fit_seconds,
            "inference_batch_seconds_single_call": prediction_seconds,
            "inference_batch_sample_count": int(x_aligned.size),
            "inference_batch_throughput_complex_samples_per_second": throughput,
            "predictor_call_count": predictor_calls,
            "timing_scope": (
                "host batch wall-clock diagnostic; not hardware sample latency"
            ),
        },
        complexity=_complexity_report(
            operation_count,
            trainable_real_parameter_count=trainable_real_parameter_count,
        ),
    )
    return result, prediction
