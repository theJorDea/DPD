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

from .complex_spline_dpd import _strict_knots, local_spline_coordinates
from .complexity import (
    ComplexMultiplyConvention,
    OperationCount,
    spline_hammerstein_pa_cost,
)

SplineCoordinate = Literal["amplitude", "power"]


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

        samples = _complex_vector(signal, name="signal")
        power = samples.real * samples.real + samples.imag * samples.imag
        if self.coordinate == "power":
            return power
        return np.sqrt(power)

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
