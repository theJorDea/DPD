r"""Phase-equivariant causal spline-Hammerstein PA inference.

The standalone forward PA model is

.. math::

   v[n] = x[n] C(s[n]),
   \qquad
   \hat y[n] = v[n] + \sum_{l=1}^{L-1} h_l v[n-l].

The direct FIR tap is fixed to ``h[0]=1`` and is neither stored nor
multiplied.  This removes the otherwise arbitrary scale split between the
spline and FIR.  ``s`` is either amplitude or power; these are distinct bases,
not two arithmetic implementations of one fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Literal

import numpy as np

from .complex_spline_dpd import (
    _strict_knots,
    local_spline_coordinates,
    spline_basis,
)
from .complexity import (
    ComplexMultiplyConvention,
    OperationCount,
    spline_hammerstein_pa_cost,
)

SplineCoordinate = Literal["amplitude", "power"]
SplineKnotVariant = Literal[
    "amplitude_uniform",
    "amplitude_uniform_power_placement",
    "amplitude_quantile",
    "amplitude_compression_aware_p2",
    "power_uniform",
]


def _complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    result = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _complex_coefficients(
    values: np.ndarray,
    *,
    name: str,
    allow_empty: bool,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (not allow_empty and array.size == 0):
        qualifier = "one-dimensional" if allow_empty else "non-empty 1-D"
        raise ValueError(f"{name} must be a {qualifier} complex array")
    dtype = (
        np.complex64
        if array.dtype == np.dtype(np.complex64)
        else np.complex128
    )
    result = np.asarray(array, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_segment_length(segment_length: int) -> int:
    if isinstance(segment_length, bool) or not isinstance(
        segment_length,
        Integral,
    ):
        raise TypeError("segment_length must be an integer")
    result = int(segment_length)
    if result < 1:
        raise ValueError("segment_length must be positive")
    return result


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def sph_coordinate_values(
    signal: np.ndarray,
    coordinate: SplineCoordinate,
) -> np.ndarray:
    """Return amplitude or power without hiding a sqrt in power mode."""

    samples = _complex_vector(signal, name="signal")
    if coordinate not in {"amplitude", "power"}:
        raise ValueError(f"unknown spline coordinate: {coordinate}")
    power = samples.real * samples.real + samples.imag * samples.imag
    return np.sqrt(power) if coordinate == "amplitude" else power


def make_sph_knots(
    pa_input: np.ndarray,
    count: int,
    variant: SplineKnotVariant,
) -> tuple[np.ndarray, SplineCoordinate]:
    """Construct one preregistered SPH coordinate/knot variant.

    Quantile duplicates are rejected instead of silently reducing ``K``.  The
    requested model size therefore remains an auditable part of a recipe.
    """

    knot_count = _validate_positive_integer(count, name="count")
    if knot_count < 2:
        raise ValueError("count must be at least two")
    samples = _complex_vector(pa_input, name="pa_input")
    power = sph_coordinate_values(samples, "power")
    maximum_power = float(np.max(power, initial=0.0))
    if maximum_power <= 0.0:
        raise ValueError("pa_input must contain a non-zero sample")
    maximum_amplitude = float(np.sqrt(maximum_power))
    unit = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)

    if variant == "amplitude_uniform":
        coordinate: SplineCoordinate = "amplitude"
        knots = maximum_amplitude * unit
    elif variant == "amplitude_uniform_power_placement":
        coordinate = "amplitude"
        knots = maximum_amplitude * np.sqrt(unit)
    elif variant == "amplitude_quantile":
        coordinate = "amplitude"
        amplitude = np.sqrt(power)
        knots = np.quantile(amplitude, unit)
        knots[0] = 0.0
        knots[-1] = maximum_amplitude
        if np.unique(knots).size != knot_count:
            raise ValueError(
                "amplitude quantiles contain duplicate knots; requested K "
                "must not be changed silently"
            )
    elif variant == "amplitude_compression_aware_p2":
        coordinate = "amplitude"
        knots = maximum_amplitude * (1.0 - np.square(1.0 - unit))
    elif variant == "power_uniform":
        coordinate = "power"
        knots = maximum_power * unit
    else:
        raise ValueError(f"unknown SPH knot variant: {variant}")
    return _strict_knots(knots), coordinate


def sph_spline_design_matrix(
    signal: np.ndarray,
    knots: np.ndarray,
    coordinate: SplineCoordinate,
) -> np.ndarray:
    """Return ``Phi[n,k] = x[n] B_k(s[n])`` for forward PA fitting."""

    samples = _complex_vector(signal, name="signal")
    knot_array = _strict_knots(knots)
    values = sph_coordinate_values(samples, coordinate)
    return samples[:, None] * spline_basis(values, knot_array)


def _segmented_delay_rows(
    values: np.ndarray,
    delay: int,
    *,
    segment_length: int,
) -> np.ndarray:
    """Delay vector/matrix rows without crossing explicit frame boundaries."""

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim not in {1, 2} or array.shape[0] == 0:
        raise ValueError("values must be a non-empty vector or matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain non-finite entries")
    if isinstance(delay, bool) or not isinstance(delay, Integral):
        raise TypeError("delay must be an integer")
    normalized_delay = int(delay)
    if normalized_delay < 0:
        raise ValueError("delay must be causal and non-negative")
    length = _validate_segment_length(segment_length)
    if normalized_delay >= length:
        raise ValueError("delay must be shorter than each explicit frame")
    output = np.zeros_like(array, dtype=np.complex128)
    if normalized_delay == 0:
        output[...] = array
        return output
    for start in range(0, array.shape[0], length):
        stop = min(start + length, array.shape[0])
        if stop - start > normalized_delay:
            output[start + normalized_delay : stop] = array[
                start : stop - normalized_delay
            ]
    return output


def sph_fir_tail_design_matrix(
    nonlinear_output: np.ndarray,
    fir_length: int,
    *,
    segment_length: int,
) -> np.ndarray:
    """Return columns ``v[n-l]`` for ``l=1..L-1`` with frame resets."""

    nonlinear = _complex_vector(nonlinear_output, name="nonlinear_output")
    length = _validate_positive_integer(fir_length, name="fir_length")
    frame_length = _validate_segment_length(segment_length)
    if length > frame_length:
        raise ValueError("fir_length must not exceed the explicit frame length")
    if length == 1:
        return np.zeros((nonlinear.size, 0), dtype=np.complex128)
    return np.column_stack(
        [
            _segmented_delay_rows(
                nonlinear,
                delay,
                segment_length=frame_length,
            )
            for delay in range(1, length)
        ]
    )


def sph_filtered_control_design_matrix(
    spline_design: np.ndarray,
    fir_tail: np.ndarray,
    *,
    segment_length: int,
) -> np.ndarray:
    """Filter every spline feature by the fixed ``[1, fir_tail]``."""

    design = np.asarray(spline_design, dtype=np.complex128)
    if design.ndim != 2 or min(design.shape) < 1:
        raise ValueError("spline_design must be a non-empty matrix")
    if not np.all(np.isfinite(design)):
        raise ValueError("spline_design contains non-finite entries")
    tail = _complex_coefficients(
        fir_tail,
        name="fir_tail",
        allow_empty=True,
    ).astype(np.complex128, copy=False)
    length = _validate_segment_length(segment_length)
    if tail.size >= length:
        raise ValueError("FIR tail must be shorter than each explicit frame")
    filtered = design.copy()
    for delay, coefficient in enumerate(tail, start=1):
        filtered += coefficient * _segmented_delay_rows(
            design,
            delay,
            segment_length=length,
        )
    return filtered


@dataclass(frozen=True)
class SplineHammersteinState:
    """Nonlinear-output history ordered from oldest to newest."""

    history: np.ndarray

    def __post_init__(self) -> None:
        history = np.asarray(self.history)
        if history.ndim != 1:
            raise ValueError("SPH streaming history must be one-dimensional")
        history = np.asarray(history, dtype=np.complex128)
        if not np.all(np.isfinite(history)):
            raise ValueError("SPH streaming history must be finite")
        object.__setattr__(self, "history", history.copy())


@dataclass(frozen=True)
class SplineHammersteinPA:
    """Immutable spline control points and causal complex FIR tail."""

    knots: np.ndarray
    control_points: np.ndarray
    fir_tail: np.ndarray
    coordinate: SplineCoordinate = "amplitude"
    knot_strategy: str = "explicit"

    def __post_init__(self) -> None:
        knots = _strict_knots(self.knots)
        controls = _complex_coefficients(
            self.control_points,
            name="control_points",
            allow_empty=False,
        )
        if controls.size != knots.size:
            raise ValueError("one complex control point is required per knot")
        tail = _complex_coefficients(
            self.fir_tail,
            name="fir_tail",
            allow_empty=True,
        )
        if self.coordinate not in {"amplitude", "power"}:
            raise ValueError(f"unknown spline coordinate: {self.coordinate}")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "control_points", controls.copy())
        object.__setattr__(self, "fir_tail", tail.copy())

    @property
    def knot_count(self) -> int:
        return int(self.knots.size)

    @property
    def fir_length(self) -> int:
        return int(self.fir_tail.size + 1)

    @property
    def stored_real_coefficients(self) -> int:
        return 2 * self.knot_count + 2 * int(self.fir_tail.size)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "phase_equivariant_spline_hammerstein_pa",
            "direction": "measured_pa_input_to_predicted_pa_output",
            "coordinate": self.coordinate,
            "knot_strategy": self.knot_strategy,
            "knot_count": self.knot_count,
            "fir_length": self.fir_length,
            "h0": "1+0j fixed and not stored",
            "phase_behavior": "phase-equivariant",
            "causality": "zero lookahead; zero history after reset",
            "continuous_streaming": "carry SplineHammersteinState",
        }

    def coordinate_values(self, signal: np.ndarray) -> np.ndarray:
        """Return the fitted spline coordinate without a hidden power sqrt."""

        return sph_coordinate_values(signal, self.coordinate)

    def correction(self, coordinate_values: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinate_values, dtype=np.float64)
        left, weight = local_spline_coordinates(values, self.knots)
        return self.control_points[left] + weight * (
            self.control_points[left + 1] - self.control_points[left]
        )

    def nonlinear_output(self, signal: np.ndarray) -> np.ndarray:
        samples = _complex_vector(signal, name="signal")
        return samples * self.correction(self.coordinate_values(samples))

    def initial_state(self) -> SplineHammersteinState:
        return SplineHammersteinState(
            np.zeros(self.fir_length - 1, dtype=np.complex128)
        )

    def _validate_state(self, state: SplineHammersteinState) -> np.ndarray:
        if not isinstance(state, SplineHammersteinState):
            raise TypeError("state must be a SplineHammersteinState")
        if state.history.size != self.fir_length - 1:
            raise ValueError("state history length must equal fir_length minus one")
        return state.history

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: SplineHammersteinState,
    ) -> tuple[np.ndarray, SplineHammersteinState]:
        """Predict one continuous chunk and return the next nonlinear state."""

        original = np.asarray(signal)
        samples = _complex_vector(original, name="signal")
        history = self._validate_state(state)
        nonlinear = self.nonlinear_output(samples)
        history_length = self.fir_length - 1
        delay_line = np.concatenate((history, nonlinear))
        output = nonlinear.copy()
        for delay, coefficient in enumerate(self.fir_tail, start=1):
            start = history_length - delay
            output += coefficient * delay_line[start : start + samples.size]

        next_history = (
            delay_line[-history_length:].copy()
            if history_length
            else np.asarray([], dtype=np.complex128)
        )
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        return (
            output.astype(target_dtype, copy=False),
            SplineHammersteinState(next_history),
        )

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Predict one independent record with zero initial state."""

        output, _ = self.predict_chunk(signal, self.initial_state())
        return output

    __call__ = predict

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        """Predict frames independently, including a final partial frame."""

        original = np.asarray(signal)
        samples = _complex_vector(original, name="signal")
        length = _validate_segment_length(segment_length)
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        output = np.empty(samples.size, dtype=target_dtype)
        for start in range(0, samples.size, length):
            stop = min(start + length, samples.size)
            output[start:stop] = self.predict(samples[start:stop])
        return output

    def operation_count(
        self,
        *,
        convention: ComplexMultiplyConvention = "4m2a",
        indexing: Literal["binary", "uniform"] = "binary",
    ) -> OperationCount:
        return spline_hammerstein_pa_cost(
            self.knot_count,
            self.fir_length,
            convention=convention,
            indexing=indexing,
            coordinate=self.coordinate,
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray("phase_equivariant_spline_hammerstein_pa"),
            knots=self.knots,
            control_points=self.control_points,
            fir_tail=self.fir_tail,
            coordinate=np.asarray(self.coordinate),
            knot_strategy=np.asarray(self.knot_strategy),
            h0_contract=np.asarray("1+0j fixed and not stored"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SplineHammersteinPA":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported spline-Hammerstein model schema")
            if str(data["model_type"]) != (
                "phase_equivariant_spline_hammerstein_pa"
            ):
                raise ValueError("unexpected spline-Hammerstein model type")
            if str(data["h0_contract"]) != "1+0j fixed and not stored":
                raise ValueError("unexpected spline-Hammerstein h0 contract")
            return cls(
                knots=data["knots"],
                control_points=data["control_points"],
                fir_tail=data["fir_tail"],
                coordinate=str(data["coordinate"]),
                knot_strategy=str(data["knot_strategy"]),
            )
