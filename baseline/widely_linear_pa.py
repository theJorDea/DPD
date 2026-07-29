r"""Causal widely-linear residual correction for a frozen PA model.

The component implemented here is deliberately narrow:

.. math::

   \Delta\hat y[n] = \sum_{d\in D} b_d x^*[n-d],
   \qquad
   \hat y_{WL}[n] = \hat y_{base}[n] + \Delta\hat y[n].

It tests a conjugate-linear residual direction that can arise from PA
asymmetry, feedback-path IQ imbalance, or other measurement effects.  A fit
improvement must not be attributed to PA physics without an independent
measurement-path audit.

All delays are non-negative.  Independent records use zero history, while
continuous streaming carries an explicit raw complex input state.  Calibration
uses a column-scaled complex augmented least-squares system and never fits an
intercept.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable

import numpy as np

from .complexity import (
    ComplexMultiplyConvention,
    OperationCount,
    widely_linear_residual_correction_cost,
)
from .metrics import nmse_pooled_db


def _complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    result = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _normalize_delays(delays: Iterable[int]) -> tuple[int, ...]:
    values = tuple(delays)
    if not values:
        raise ValueError("widely-linear delays must not be empty")
    if any(
        isinstance(delay, bool) or not isinstance(delay, Integral)
        for delay in values
    ):
        raise TypeError("widely-linear delays must be integers")
    result = tuple(int(delay) for delay in values)
    if any(delay < 0 for delay in result):
        raise ValueError("widely-linear delays must be causal and non-negative")
    if len(set(result)) != len(result):
        raise ValueError("widely-linear delays must be unique")
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


def _causal_delay(values: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.complex128)
    if delay == 0:
        result[:] = values
    elif delay < values.size:
        result[delay:] = values[:-delay]
    return result


def widely_linear_design_matrix(
    signal: np.ndarray,
    delays: Iterable[int],
) -> np.ndarray:
    """Return columns ``conj(x[n-d])`` with zero pre-record context."""

    samples = _complex_vector(signal, name="signal")
    delay_tuple = _normalize_delays(delays)
    return np.column_stack(
        [np.conjugate(_causal_delay(samples, delay)) for delay in delay_tuple]
    )


def widely_linear_segmented_design_matrix(
    signal: np.ndarray,
    delays: Iterable[int],
    *,
    segment_length: int,
) -> np.ndarray:
    """Build a design that resets delayed input state at each frame."""

    samples = _complex_vector(signal, name="signal")
    delay_tuple = _normalize_delays(delays)
    length = _validate_segment_length(segment_length)
    return np.vstack(
        [
            widely_linear_design_matrix(
                samples[start : min(start + length, samples.size)],
                delay_tuple,
            )
            for start in range(0, samples.size, length)
        ]
    )


@dataclass(frozen=True)
class WidelyLinearStreamingState:
    """Raw input history ordered from oldest to newest."""

    history: np.ndarray

    def __post_init__(self) -> None:
        history = np.asarray(self.history)
        if history.ndim != 1:
            raise ValueError("widely-linear streaming history must be 1-D")
        history = np.asarray(history, dtype=np.complex128)
        if not np.all(np.isfinite(history)):
            raise ValueError("widely-linear streaming history must be finite")
        object.__setattr__(self, "history", history.copy())


@dataclass(frozen=True)
class WidelyLinearFitDiagnostics:
    """Auditable diagnostics for one deterministic residual fit."""

    sample_count: int
    segment_length: int
    segment_count: int
    tap_count: int
    feature_count: int
    ridge: float
    column_rms_minimum: float
    column_rms_maximum: float
    solver_rank: int
    scaled_augmented_condition_number: float
    residual_target_power: float
    training_correction_mse: float
    training_correction_nmse_db: float
    coefficient_l2_norm: float
    causal_warmup_samples: int
    coefficient_dtype: str
    solver: str = "numpy_column_scaled_complex_ridge_lstsq"


@dataclass(frozen=True)
class WidelyLinearResidualCorrection:
    """Immutable conjugate-FIR residual coefficient set."""

    delays: tuple[int, ...]
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        delays = _normalize_delays(self.delays)
        coefficients = np.asarray(self.coefficients)
        if coefficients.shape != (len(delays),):
            raise ValueError(
                "coefficients must contain one complex value per delay"
            )
        coefficients = coefficients.astype(
            (
                np.complex64
                if coefficients.dtype == np.dtype(np.complex64)
                else np.complex128
            ),
            copy=False,
        )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("widely-linear coefficients must be finite")
        object.__setattr__(self, "delays", delays)
        object.__setattr__(self, "coefficients", coefficients.copy())

    @property
    def tap_count(self) -> int:
        return len(self.delays)

    @property
    def maximum_delay(self) -> int:
        return max(self.delays)

    @property
    def stored_complex_coefficients(self) -> int:
        return int(self.coefficients.size)

    @property
    def stored_real_coefficients(self) -> int:
        return 2 * self.stored_complex_coefficients

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "causal_widely_linear_residual_correction",
            "equation": "sum_d b_d * conj(x[n-d])",
            "delays": list(self.delays),
            "tap_count": self.tap_count,
            "maximum_delay": self.maximum_delay,
            "phase_behavior": "conjugate-linear, intentionally not phase-equivariant",
            "causal_padding": "zeros_before_record_or_reset_segment_start",
            "continuous_streaming": "carry WidelyLinearStreamingState",
            "physical_attribution": "not identified by this model alone",
        }

    def initial_state(self) -> WidelyLinearStreamingState:
        return WidelyLinearStreamingState(
            np.zeros(self.maximum_delay, dtype=np.complex128)
        )

    def _validate_state(self, state: WidelyLinearStreamingState) -> np.ndarray:
        if not isinstance(state, WidelyLinearStreamingState):
            raise TypeError("state must be a WidelyLinearStreamingState")
        if state.history.size != self.maximum_delay:
            raise ValueError(
                "state history length must equal model.maximum_delay"
            )
        return state.history

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: WidelyLinearStreamingState,
    ) -> tuple[np.ndarray, WidelyLinearStreamingState]:
        """Predict one continuous chunk and return its next input state."""

        original = np.asarray(signal)
        samples = _complex_vector(original, name="signal")
        history = self._validate_state(state)
        history_length = self.maximum_delay
        delay_line = np.concatenate((history, samples))
        output = np.zeros(samples.size, dtype=np.complex128)
        for coefficient, delay in zip(
            self.coefficients,
            self.delays,
            strict=True,
        ):
            start = history_length - delay
            delayed = delay_line[start : start + samples.size]
            output += coefficient * np.conjugate(delayed)

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
            WidelyLinearStreamingState(next_history),
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
        """Predict independent frames and reset state at every boundary."""

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

    def correct(
        self,
        signal: np.ndarray,
        base_prediction: np.ndarray,
    ) -> np.ndarray:
        """Add this residual correction to one independent base prediction."""

        samples = _complex_vector(signal, name="signal")
        base = _complex_vector(base_prediction, name="base_prediction")
        if samples.shape != base.shape:
            raise ValueError("signal and base_prediction must have equal length")
        return base + self.predict(samples)

    def correct_segments(
        self,
        signal: np.ndarray,
        base_prediction: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        """Add a reset-at-frame correction to a segmented base prediction."""

        samples = _complex_vector(signal, name="signal")
        base = _complex_vector(base_prediction, name="base_prediction")
        if samples.shape != base.shape:
            raise ValueError("signal and base_prediction must have equal length")
        return base + self.predict_segments(samples, segment_length)

    def operation_count(
        self,
        *,
        convention: ComplexMultiplyConvention = "4m2a",
        reuse_input_delay_state: bool = False,
    ) -> OperationCount:
        """Return the incremental cost of adding this branch to a base output."""

        return widely_linear_residual_correction_cost(
            self.delays,
            convention=convention,
            reuse_input_delay_state=reuse_input_delay_state,
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray("causal_widely_linear_residual_correction"),
            delays=np.asarray(self.delays, dtype=np.int64),
            coefficients=self.coefficients,
            coefficient_order=np.asarray("one coefficient per listed delay"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WidelyLinearResidualCorrection":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported widely-linear model schema")
            if str(data["model_type"]) != (
                "causal_widely_linear_residual_correction"
            ):
                raise ValueError("unexpected widely-linear model type")
            return cls(
                tuple(int(delay) for delay in data["delays"]),
                data["coefficients"],
            )


def fit_widely_linear_residual_correction(
    pa_input: np.ndarray,
    residual_target: np.ndarray,
    *,
    delays: Iterable[int],
    ridge: float = 1e-8,
    segment_length: int,
    coefficient_dtype: np.dtype = np.complex128,
) -> tuple[WidelyLinearResidualCorrection, WidelyLinearFitDiagnostics]:
    """Fit conjugate coefficients without normal equations or an intercept."""

    samples = _complex_vector(pa_input, name="pa_input")
    target = _complex_vector(residual_target, name="residual_target")
    if samples.shape != target.shape:
        raise ValueError("PA input and residual target must have equal length")
    delay_tuple = _normalize_delays(delays)
    length = _validate_segment_length(segment_length)
    if max(delay_tuple) >= length:
        raise ValueError("widely-linear memory must be shorter than each frame")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    dtype = np.dtype(coefficient_dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError("coefficient_dtype must be a complex dtype")

    design = widely_linear_segmented_design_matrix(
        samples,
        delay_tuple,
        segment_length=length,
    )
    if design.shape[1] >= design.shape[0]:
        raise ValueError("widely-linear least-squares system must be overdetermined")
    column_rms = np.sqrt(np.mean(np.abs(design) ** 2, axis=0))
    if np.any(~np.isfinite(column_rms)) or np.any(column_rms <= 0.0):
        raise ValueError("widely-linear design has invalid or all-zero columns")

    normalization = np.sqrt(float(samples.size))
    solve_design = (design / column_rms) / normalization
    solve_target = target / normalization
    if ridge > 0.0:
        solve_design = np.vstack(
            (
                solve_design,
                np.sqrt(ridge)
                * np.eye(design.shape[1], dtype=np.complex128),
            )
        )
        solve_target = np.concatenate(
            (
                solve_target,
                np.zeros(design.shape[1], dtype=np.complex128),
            )
        )
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        solve_design,
        solve_target,
        rcond=None,
    )
    coefficients = (scaled_coefficients / column_rms).astype(
        dtype,
        copy=False,
    )
    model = WidelyLinearResidualCorrection(delay_tuple, coefficients)
    prediction = model.predict_segments(samples, length)
    error = target - prediction
    target_power = float(np.mean(np.abs(target) ** 2))
    if target_power <= 0.0:
        raise ValueError("residual_target must have non-zero power")
    if singular_values.size and singular_values[-1] > 0.0:
        condition = float(singular_values[0] / singular_values[-1])
    else:
        condition = float("inf")
    diagnostics = WidelyLinearFitDiagnostics(
        sample_count=int(samples.size),
        segment_length=length,
        segment_count=int(np.ceil(samples.size / length)),
        tap_count=len(delay_tuple),
        feature_count=len(delay_tuple),
        ridge=float(ridge),
        column_rms_minimum=float(np.min(column_rms)),
        column_rms_maximum=float(np.max(column_rms)),
        solver_rank=int(rank),
        scaled_augmented_condition_number=condition,
        residual_target_power=target_power,
        training_correction_mse=float(np.mean(np.abs(error) ** 2)),
        training_correction_nmse_db=nmse_pooled_db(prediction, target),
        coefficient_l2_norm=float(np.linalg.norm(coefficients)),
        causal_warmup_samples=max(delay_tuple),
        coefficient_dtype=str(coefficients.dtype),
    )
    return model, diagnostics
