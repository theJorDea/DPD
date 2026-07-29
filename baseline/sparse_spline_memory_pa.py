"""Phase-equivariant non-factorized sparse spline-memory PA model.

The forward model is

    y_hat[n] = sum_b x[n-m_b] * C_b(abs(x[n-d_b])),

where each ``C_b`` is a complex local linear spline.  Signal and envelope
delays are explicit, non-negative and causal.  This module is deliberately a
forward PA-identification API; it does not contain ILA/DPD target transforms.

The implementation reuses the already audited spline-memory inference kernel,
but gives it a distinct PA model type, forward-direction metadata and a
segment-safe complex ridge fitter.  Fit segments are never concatenated before
delay features are formed, so a frame boundary cannot leak state into another
fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .complex_spline_dpd import KnotStrategy, _strict_knots, make_knots
from .spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    SplineMemoryState,
    _normalize_branches,
    _error_statistics,
    spline_memory_design_matrix,
)


SparseSplineMemoryPAState = SplineMemoryState
SparseSplineMemoryPABranch = SplineMemoryBranch


def _complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _normalize_segments(
    values: Iterable[np.ndarray],
    *,
    name: str,
) -> tuple[np.ndarray, ...]:
    segments = tuple(
        _complex_vector(segment, name=f"{name}[{index}]")
        for index, segment in enumerate(values)
    )
    if not segments:
        raise ValueError(f"{name} must contain at least one segment")
    return segments


def sparse_spline_memory_pa_design_matrix(
    signal_segments: Iterable[np.ndarray],
    knots: np.ndarray,
    branches: Iterable[SplineMemoryBranch | tuple[int, int]],
) -> np.ndarray:
    """Build a vertically stacked, frame-safe complex design matrix.

    Each segment is passed independently to the audited single-record design
    builder.  Delayed samples before a segment boundary are therefore zero,
    exactly matching reset-per-frame inference.
    """

    segments = _normalize_segments(signal_segments, name="signal_segments")
    knot_array = _strict_knots(knots)
    branch_tuple = _normalize_branches(branches)
    return np.vstack(
        [
            spline_memory_design_matrix(segment, knot_array, branch_tuple)
            for segment in segments
        ]
    )


@dataclass(frozen=True)
class SparseSplineMemoryPAFitDiagnostics:
    """Numerical diagnostics for one deterministic complex PA fit."""

    sample_count: int
    segment_count: int
    branch_count: int
    knot_count: int
    feature_count: int
    ridge: float
    data_design_rank: int
    data_design_condition_number: float
    augmented_solver_rank: int
    augmented_design_condition_number: float
    solver: str
    causal_warmup_samples: int
    training_mse_full: float
    training_relative_error_power_full: float
    training_nmse_db_full: float
    training_mse_after_warmup: float
    training_relative_error_power_after_warmup: float
    training_nmse_db_after_warmup: float
    maximum_calibration_radius: float
    maximum_absolute_coefficient: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "segment_count": self.segment_count,
            "branch_count": self.branch_count,
            "knot_count": self.knot_count,
            "feature_count": self.feature_count,
            "ridge": self.ridge,
            "data_design_rank": self.data_design_rank,
            "data_design_condition_number": self.data_design_condition_number,
            "augmented_solver_rank": self.augmented_solver_rank,
            "augmented_design_condition_number": (
                self.augmented_design_condition_number
            ),
            "solver": self.solver,
            "causal_warmup_samples": self.causal_warmup_samples,
            "training_mse_full": self.training_mse_full,
            "training_relative_error_power_full": (
                self.training_relative_error_power_full
            ),
            "training_nmse_db_full": self.training_nmse_db_full,
            "training_mse_after_warmup": self.training_mse_after_warmup,
            "training_relative_error_power_after_warmup": (
                self.training_relative_error_power_after_warmup
            ),
            "training_nmse_db_after_warmup": self.training_nmse_db_after_warmup,
            "maximum_calibration_radius": self.maximum_calibration_radius,
            "maximum_absolute_coefficient": self.maximum_absolute_coefficient,
        }


@dataclass(frozen=True)
class SparseSplineMemoryPA:
    """Immutable causal phase-equivariant sparse spline-memory PA."""

    knots: np.ndarray
    branches: tuple[SplineMemoryBranch, ...]
    coefficients: np.ndarray
    knot_strategy: str = "explicit"

    def __post_init__(self) -> None:
        knots = _strict_knots(self.knots)
        branches = _normalize_branches(self.branches)
        coefficients = np.asarray(self.coefficients)
        expected = (len(branches), knots.size)
        if coefficients.shape != expected:
            raise ValueError(
                f"coefficients must have shape (branches, knots) == {expected}"
            )
        if not np.issubdtype(coefficients.dtype, np.complexfloating):
            coefficients = coefficients.astype(np.complex128)
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients contain NaN or infinite values")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "coefficients", coefficients.copy())

    @property
    def branch_count(self) -> int:
        return len(self.branches)

    @property
    def knot_count(self) -> int:
        return int(self.knots.size)

    @property
    def maximum_delay(self) -> int:
        return max(
            max(branch.signal_delay, branch.envelope_delay)
            for branch in self.branches
        )

    @property
    def stored_real_coefficients(self) -> int:
        return int(2 * self.coefficients.size)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "phase_equivariant_non_factorized_sparse_spline_memory_pa",
            "direction": "measured_pa_input_to_predicted_pa_output",
            "equation": "sum_b x[n-m_b] * C_b(abs(x[n-d_b]))",
            "phase_behavior": "phase-equivariant",
            "branches": [
                {
                    "signal_delay": branch.signal_delay,
                    "envelope_delay": branch.envelope_delay,
                }
                for branch in self.branches
            ],
            "knot_count": self.knot_count,
            "knot_strategy": self.knot_strategy,
            "local_active_control_points_per_branch": 2,
            "causality": "zero lookahead; zero history after reset",
            "horizon_samples": self.maximum_delay,
        }

    def _core(self) -> SparseSplineMemoryDPD:
        return SparseSplineMemoryDPD(
            knots=self.knots,
            branches=self.branches,
            coefficients=self.coefficients,
            knot_strategy=self.knot_strategy,
        )

    def initial_state(self) -> SparseSplineMemoryPAState:
        return self._core().initial_state()

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: SparseSplineMemoryPAState,
    ) -> tuple[np.ndarray, SparseSplineMemoryPAState]:
        return self._core().predict_chunk(signal, state)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        return self._core().predict(signal)

    __call__ = predict

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        return self._core().predict_segments(signal, segment_length)

    def operation_count(self, **kwargs: object):
        return self._core().operation_count(**kwargs)

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray(
                "phase_equivariant_non_factorized_sparse_spline_memory_pa"
            ),
            knots=self.knots,
            signal_delays=np.asarray(
                [branch.signal_delay for branch in self.branches],
                dtype=np.int64,
            ),
            envelope_delays=np.asarray(
                [branch.envelope_delay for branch in self.branches],
                dtype=np.int64,
            ),
            coefficients=self.coefficients,
            knot_strategy=np.asarray(self.knot_strategy),
            direction=np.asarray("measured_pa_input_to_predicted_pa_output"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SparseSplineMemoryPA":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported sparse spline-memory PA schema")
            expected = (
                "phase_equivariant_non_factorized_sparse_spline_memory_pa"
            )
            if str(data["model_type"]) != expected:
                raise ValueError("unexpected sparse spline-memory PA model type")
            if str(data["direction"]) != (
                "measured_pa_input_to_predicted_pa_output"
            ):
                raise ValueError("unexpected sparse spline-memory PA direction")
            signal_delays = np.asarray(data["signal_delays"])
            envelope_delays = np.asarray(data["envelope_delays"])
            if signal_delays.shape != envelope_delays.shape:
                raise ValueError("saved branch-delay arrays have different shapes")
            branches = tuple(
                SplineMemoryBranch(int(signal), int(envelope))
                for signal, envelope in zip(
                    signal_delays,
                    envelope_delays,
                    strict=True,
                )
            )
            return cls(
                knots=data["knots"],
                branches=branches,
                coefficients=data["coefficients"],
                knot_strategy=str(data["knot_strategy"]),
            )


def _fit_design(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, int, float, int, float]:
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    design = np.asarray(design, dtype=np.complex128)
    target = _complex_vector(target, name="target")
    if design.ndim != 2 or design.shape[0] != target.size:
        raise ValueError("design and target dimensions do not match")
    if not np.all(np.isfinite(design)):
        raise ValueError("design contains NaN or infinite values")
    data_rank = int(np.linalg.matrix_rank(design))
    data_condition = float(np.linalg.cond(design))
    scale = np.sqrt(float(target.size))
    normalized_design = design / scale
    normalized_target = target / scale
    feature_count = design.shape[1]
    if ridge > 0.0:
        augmented_design = np.vstack(
            (
                normalized_design,
                np.sqrt(float(ridge))
                * np.eye(feature_count, dtype=np.complex128),
            )
        )
        augmented_target = np.concatenate(
            (
                normalized_target,
                np.zeros(feature_count, dtype=np.complex128),
            )
        )
    else:
        augmented_design = normalized_design
        augmented_target = normalized_target
    augmented_condition = float(np.linalg.cond(augmented_design))
    coefficients, _, augmented_rank, _ = np.linalg.lstsq(
        augmented_design,
        augmented_target,
        rcond=None,
    )
    return (
        coefficients,
        data_rank,
        data_condition,
        int(augmented_rank),
        augmented_condition,
    )


def fit_sparse_spline_memory_pa_segments(
    pa_input_segments: Iterable[np.ndarray],
    measured_output_segments: Iterable[np.ndarray],
    *,
    branches: Iterable[SplineMemoryBranch | tuple[int, int]],
    knot_count: int | None = None,
    knot_strategy: KnotStrategy = "uniform_amplitude",
    knots: np.ndarray | None = None,
    ridge: float = 1e-8,
    compression_power: float = 2.0,
) -> tuple[SparseSplineMemoryPA, SparseSplineMemoryPAFitDiagnostics]:
    """Fit one forward PA model without crossing segment boundaries."""

    input_segments = _normalize_segments(pa_input_segments, name="pa_input_segments")
    output_segments = _normalize_segments(
        measured_output_segments,
        name="measured_output_segments",
    )
    if len(input_segments) != len(output_segments):
        raise ValueError("input and output segment counts must match")
    if any(
        input_segment.size != output_segment.size
        for input_segment, output_segment in zip(
            input_segments,
            output_segments,
            strict=True,
        )
    ):
        raise ValueError("input/output segment lengths must match")
    branch_tuple = _normalize_branches(branches)
    calibration_input = np.concatenate(input_segments)
    target = np.concatenate(output_segments)
    if knots is None:
        if knot_count is None:
            raise ValueError("knot_count is required when knots are not supplied")
        knot_array = make_knots(
            calibration_input,
            knot_count,
            knot_strategy,
            compression_power=compression_power,
        )
        strategy_label = knot_strategy
    else:
        knot_array = _strict_knots(knots)
        if knot_count is not None and knot_count != knot_array.size:
            raise ValueError("knot_count does not match explicit knots")
        strategy_label = "explicit"

    design = sparse_spline_memory_pa_design_matrix(
        input_segments,
        knot_array,
        branch_tuple,
    )
    (
        flat_coefficients,
        data_rank,
        data_condition,
        augmented_rank,
        augmented_condition,
    ) = _fit_design(design, target, ridge=ridge)
    coefficients = flat_coefficients.reshape(
        len(branch_tuple),
        knot_array.size,
    )
    model = SparseSplineMemoryPA(
        knots=knot_array,
        branches=branch_tuple,
        coefficients=coefficients,
        knot_strategy=strategy_label,
    )
    predictions = np.concatenate(
        [model.predict(segment) for segment in input_segments]
    )
    mse_full, relative_full, nmse_full = _error_statistics(predictions, target)
    warmup = model.maximum_delay
    steady_predictions: list[np.ndarray] = []
    steady_targets: list[np.ndarray] = []
    for input_segment, output_segment in zip(
        input_segments,
        output_segments,
        strict=True,
    ):
        if input_segment.size <= warmup:
            continue
        steady_predictions.append(model.predict(input_segment)[warmup:])
        steady_targets.append(output_segment[warmup:])
    if not steady_predictions:
        raise ValueError("segments are too short for the model delay")
    steady_prediction = np.concatenate(steady_predictions)
    steady_target = np.concatenate(steady_targets)
    mse_steady, relative_steady, nmse_steady = _error_statistics(
        steady_prediction,
        steady_target,
    )
    diagnostics = SparseSplineMemoryPAFitDiagnostics(
        sample_count=int(target.size),
        segment_count=len(input_segments),
        branch_count=model.branch_count,
        knot_count=model.knot_count,
        feature_count=int(design.shape[1]),
        ridge=float(ridge),
        data_design_rank=data_rank,
        data_design_condition_number=data_condition,
        augmented_solver_rank=augmented_rank,
        augmented_design_condition_number=augmented_condition,
        solver="augmented_complex_lstsq",
        causal_warmup_samples=warmup,
        training_mse_full=mse_full,
        training_relative_error_power_full=relative_full,
        training_nmse_db_full=nmse_full,
        training_mse_after_warmup=mse_steady,
        training_relative_error_power_after_warmup=relative_steady,
        training_nmse_db_after_warmup=nmse_steady,
        maximum_calibration_radius=float(np.max(np.abs(calibration_input))),
        maximum_absolute_coefficient=float(np.max(np.abs(coefficients))),
    )
    return model, diagnostics


def fit_sparse_spline_memory_pa(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    **kwargs: object,
) -> tuple[SparseSplineMemoryPA, SparseSplineMemoryPAFitDiagnostics]:
    """Fit a forward PA model as one independent zero-state record."""

    return fit_sparse_spline_memory_pa_segments(
        (pa_input,),
        (measured_output,),
        **kwargs,
    )

