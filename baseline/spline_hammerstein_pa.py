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
import time
from typing import Literal

import numpy as np

from .complex_spline_dpd import (
    _second_derivative_operator,
    _strict_knots,
    local_spline_coordinates,
    spline_basis,
)
from .metrics import nmse_pooled_db
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
class SplineHammersteinBlockDiagnostics:
    """Rank, conditioning, and objective change for one exact block solve."""

    iteration: int
    block: str
    feature_count: int
    data_design_rank: int
    data_design_condition_number: float
    data_minimum_singular_value: float
    data_maximum_singular_value: float
    augmented_solver_rank: int
    augmented_condition_number: float
    augmented_minimum_singular_value: float
    augmented_maximum_singular_value: float
    objective_before: float
    objective_after: float
    relative_objective_decrease: float
    coefficient_l2_norm: float


@dataclass(frozen=True)
class SplineHammersteinFitDiagnostics:
    """Auditable result of deterministic alternating complex LS."""

    sample_count: int
    segment_length: int
    segment_count: int
    knot_count: int
    knot_strategy: str
    coordinate: str
    fir_length: int
    control_ridge: float
    smoothness: float
    fir_ridge: float
    maximum_alternations: int
    minimum_alternations: int
    completed_alternations: int
    convergence_tolerance: float
    objective_increase_tolerance: float
    converged: bool
    convergence_reason: str
    zero_model_objective: float
    memoryless_initial_objective: float
    optimization_final_objective: float
    serialized_model_objective: float
    all_updates_monotonic: bool
    all_data_designs_full_column_rank: bool
    minimum_nonzero_control_feature_samples: int
    maximum_calibration_coordinate: float
    control_point_l2_norm: float
    fir_tail_l2_norm: float
    target_power: float
    training_mse: float
    training_relative_error_power: float
    training_nmse_db: float
    coefficient_dtype: str
    fit_wall_time_seconds: float
    solver: str
    h0_contract: str
    updates: tuple[SplineHammersteinBlockDiagnostics, ...]


@dataclass(frozen=True)
class _ComplexSolveDiagnostics:
    feature_count: int
    data_design_rank: int
    data_design_condition_number: float
    data_minimum_singular_value: float
    data_maximum_singular_value: float
    augmented_solver_rank: int
    augmented_condition_number: float
    augmented_minimum_singular_value: float
    augmented_maximum_singular_value: float


def _rank_and_condition(
    singular_values: np.ndarray,
    shape: tuple[int, int],
) -> tuple[int, float, float, float]:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return 0, float("inf"), 0.0, 0.0
    maximum = float(values[0])
    minimum = float(values[-1])
    tolerance = maximum * max(shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(values > tolerance))
    condition = (
        float(maximum / minimum)
        if rank == shape[1] and minimum > np.finfo(float).tiny
        else float("inf")
    )
    return rank, condition, minimum, maximum


def _regularized_complex_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    penalties: tuple[tuple[float, np.ndarray], ...],
) -> tuple[np.ndarray, _ComplexSolveDiagnostics]:
    """Solve a mean-square complex LS objective via augmented rows."""

    matrix = np.asarray(design, dtype=np.complex128)
    desired = _complex_vector(target, name="target")
    if matrix.ndim != 2 or matrix.shape[0] != desired.size:
        raise ValueError("design rows must match the one-dimensional target")
    if matrix.shape[1] < 1 or matrix.shape[1] >= matrix.shape[0]:
        raise ValueError("complex LS design must be overdetermined and non-empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("complex LS design contains non-finite values")

    normalization = np.sqrt(float(matrix.shape[0]))
    normalized = matrix / normalization
    data_singular_values = np.linalg.svd(normalized, compute_uv=False)
    data_rank, data_condition, data_minimum, data_maximum = (
        _rank_and_condition(data_singular_values, matrix.shape)
    )
    blocks = [normalized]
    targets = [desired / normalization]
    for strength, operator in penalties:
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("penalty strengths must be finite and non-negative")
        penalty = np.asarray(operator, dtype=np.complex128)
        if penalty.ndim != 2 or penalty.shape[1] != matrix.shape[1]:
            raise ValueError("penalty operator has the wrong feature count")
        if not np.all(np.isfinite(penalty)):
            raise ValueError("penalty operator contains non-finite values")
        if strength > 0.0 and penalty.shape[0] > 0:
            blocks.append(np.sqrt(strength) * penalty)
            targets.append(np.zeros(penalty.shape[0], dtype=np.complex128))

    augmented = np.vstack(blocks)
    augmented_target = np.concatenate(targets)
    coefficients, _, solver_rank, augmented_singular_values = np.linalg.lstsq(
        augmented,
        augmented_target,
        rcond=None,
    )
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("complex LS returned non-finite coefficients")
    _, augmented_condition, augmented_minimum, augmented_maximum = (
        _rank_and_condition(augmented_singular_values, augmented.shape)
    )
    return coefficients, _ComplexSolveDiagnostics(
        feature_count=int(matrix.shape[1]),
        data_design_rank=data_rank,
        data_design_condition_number=data_condition,
        data_minimum_singular_value=data_minimum,
        data_maximum_singular_value=data_maximum,
        augmented_solver_rank=int(solver_rank),
        augmented_condition_number=augmented_condition,
        augmented_minimum_singular_value=augmented_minimum,
        augmented_maximum_singular_value=augmented_maximum,
    )


def _sph_objective(
    prediction: np.ndarray,
    target: np.ndarray,
    control_points: np.ndarray,
    fir_tail: np.ndarray,
    second_derivative: np.ndarray,
    *,
    control_ridge: float,
    smoothness: float,
    fir_ridge: float,
) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    value = float(np.mean(np.abs(error) ** 2))
    value += float(control_ridge * np.sum(np.abs(control_points) ** 2))
    if second_derivative.shape[0]:
        curvature = second_derivative @ control_points
        value += float(smoothness * np.sum(np.abs(curvature) ** 2))
    value += float(fir_ridge * np.sum(np.abs(fir_tail) ** 2))
    if not np.isfinite(value):
        raise FloatingPointError("spline-Hammerstein objective is non-finite")
    return value


def _block_diagnostics(
    *,
    iteration: int,
    block: str,
    solve: _ComplexSolveDiagnostics,
    objective_before: float,
    objective_after: float,
    coefficient_norm: float,
    increase_tolerance: float,
    numerical_scale_floor: float,
) -> SplineHammersteinBlockDiagnostics:
    scale = max(abs(objective_before), numerical_scale_floor)
    increase = objective_after - objective_before
    if increase > increase_tolerance * scale:
        raise FloatingPointError(
            f"{block} block increased the frozen objective by {increase / scale:.3e}"
        )
    return SplineHammersteinBlockDiagnostics(
        iteration=iteration,
        block=block,
        feature_count=solve.feature_count,
        data_design_rank=solve.data_design_rank,
        data_design_condition_number=solve.data_design_condition_number,
        data_minimum_singular_value=solve.data_minimum_singular_value,
        data_maximum_singular_value=solve.data_maximum_singular_value,
        augmented_solver_rank=solve.augmented_solver_rank,
        augmented_condition_number=solve.augmented_condition_number,
        augmented_minimum_singular_value=solve.augmented_minimum_singular_value,
        augmented_maximum_singular_value=solve.augmented_maximum_singular_value,
        objective_before=float(objective_before),
        objective_after=float(objective_after),
        relative_objective_decrease=float((objective_before - objective_after) / scale),
        coefficient_l2_norm=float(coefficient_norm),
    )


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


def fit_spline_hammerstein_pa(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    *,
    knot_count: int | None = None,
    knot_variant: SplineKnotVariant = "amplitude_uniform",
    knots: np.ndarray | None = None,
    coordinate: SplineCoordinate | None = None,
    fir_length: int = 1,
    segment_length: int,
    control_ridge: float = 1e-8,
    smoothness: float = 1e-6,
    fir_ridge: float = 1e-8,
    maximum_alternations: int = 20,
    minimum_alternations: int = 2,
    convergence_tolerance: float = 1e-7,
    objective_increase_tolerance: float = 1e-10,
    coefficient_dtype: np.dtype = np.complex128,
) -> tuple[SplineHammersteinPA, SplineHammersteinFitDiagnostics]:
    """Fit a forward SPH PA using deterministic alternating complex LS.

    The function consumes only measured PA input and output.  It performs no
    gain/delay fit, has no random initialization, and does not accept a
    validation or test sequence.  Every block solve minimizes the same frozen
    mean-square plus ridge/smoothness objective.
    """

    start_time = time.perf_counter()
    samples = _complex_vector(pa_input, name="pa_input")
    target = _complex_vector(measured_output, name="measured_output")
    if samples.shape != target.shape:
        raise ValueError("pa_input and measured_output must have equal length")
    frame_length = _validate_segment_length(segment_length)
    length = _validate_positive_integer(fir_length, name="fir_length")
    if length > frame_length:
        raise ValueError("fir_length must not exceed the explicit frame length")

    maximum_iterations = _validate_positive_integer(
        maximum_alternations,
        name="maximum_alternations",
    )
    minimum_iterations = _validate_positive_integer(
        minimum_alternations,
        name="minimum_alternations",
    )
    if minimum_iterations > maximum_iterations:
        raise ValueError(
            "minimum_alternations must not exceed maximum_alternations"
        )
    regularization_values = (control_ridge, smoothness, fir_ridge)
    if any(
        not np.isfinite(value) or value < 0.0
        for value in regularization_values
    ):
        raise ValueError("regularization strengths must be finite and non-negative")
    if not np.isfinite(convergence_tolerance) or convergence_tolerance < 0.0:
        raise ValueError("convergence_tolerance must be finite and non-negative")
    if (
        not np.isfinite(objective_increase_tolerance)
        or objective_increase_tolerance < 0.0
    ):
        raise ValueError(
            "objective_increase_tolerance must be finite and non-negative"
        )
    dtype = np.dtype(coefficient_dtype)
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError("coefficient_dtype must be a complex dtype")

    if knots is None:
        if knot_count is None:
            raise ValueError("knot_count is required without explicit knots")
        knot_array, inferred_coordinate = make_sph_knots(
            samples,
            knot_count,
            knot_variant,
        )
        if coordinate is not None and coordinate != inferred_coordinate:
            raise ValueError("coordinate conflicts with the selected knot variant")
        fitted_coordinate = inferred_coordinate
        strategy_label = knot_variant
    else:
        knot_array = _strict_knots(knots)
        if knot_count is not None:
            if (
                isinstance(knot_count, bool)
                or not isinstance(knot_count, Integral)
            ):
                raise TypeError("knot_count must be an integer")
            if int(knot_count) != knot_array.size:
                raise ValueError("knot_count does not match explicit knots")
        if coordinate is None:
            raise ValueError("coordinate is required with explicit knots")
        if coordinate not in {"amplitude", "power"}:
            raise ValueError(f"unknown spline coordinate: {coordinate}")
        fitted_coordinate = coordinate
        strategy_label = "explicit"

    if knot_array.size >= samples.size:
        raise ValueError("SPH control solve must be overdetermined")
    target_power = float(np.mean(np.abs(target) ** 2))
    if target_power <= 0.0:
        raise ValueError("measured_output must contain non-zero power")

    spline_design = sph_spline_design_matrix(
        samples,
        knot_array,
        fitted_coordinate,
    ).astype(np.complex128, copy=False)
    derivative = _second_derivative_operator(knot_array).astype(
        np.complex128,
        copy=False,
    )
    control_identity = np.eye(knot_array.size, dtype=np.complex128)
    control_penalties = (
        (float(control_ridge), control_identity),
        (float(smoothness), derivative),
    )
    minimum_feature_samples = int(
        np.min(np.count_nonzero(np.abs(spline_design) > 0.0, axis=0))
    )
    maximum_coordinate = float(
        np.max(sph_coordinate_values(samples, fitted_coordinate))
    )
    numerical_scale_floor = np.finfo(np.float64).eps * target_power

    updates: list[SplineHammersteinBlockDiagnostics] = []
    zero_controls = np.zeros(knot_array.size, dtype=np.complex128)
    fir_tail = np.zeros(length - 1, dtype=np.complex128)
    zero_objective = _sph_objective(
        np.zeros_like(target),
        target,
        zero_controls,
        fir_tail,
        derivative,
        control_ridge=float(control_ridge),
        smoothness=float(smoothness),
        fir_ridge=float(fir_ridge),
    )
    control_points, initial_solve = _regularized_complex_lstsq(
        spline_design,
        target,
        control_penalties,
    )
    prediction = spline_design @ control_points
    objective = _sph_objective(
        prediction,
        target,
        control_points,
        fir_tail,
        derivative,
        control_ridge=float(control_ridge),
        smoothness=float(smoothness),
        fir_ridge=float(fir_ridge),
    )
    updates.append(
        _block_diagnostics(
            iteration=0,
            block="memoryless_initial_control",
            solve=initial_solve,
            objective_before=zero_objective,
            objective_after=objective,
            coefficient_norm=float(np.linalg.norm(control_points)),
            increase_tolerance=float(objective_increase_tolerance),
            numerical_scale_floor=numerical_scale_floor,
        )
    )
    memoryless_objective = objective

    converged = False
    convergence_reason = "maximum_alternations_reached"
    completed_alternations = 0
    for iteration in range(1, maximum_iterations + 1):
        iteration_start_objective = objective
        nonlinear = spline_design @ control_points

        if length > 1:
            tail_design = sph_fir_tail_design_matrix(
                nonlinear,
                length,
                segment_length=frame_length,
            )
            fir_identity = np.eye(length - 1, dtype=np.complex128)
            next_tail, tail_solve = _regularized_complex_lstsq(
                tail_design,
                target - nonlinear,
                ((float(fir_ridge), fir_identity),),
            )
            next_prediction = nonlinear + tail_design @ next_tail
            next_objective = _sph_objective(
                next_prediction,
                target,
                control_points,
                next_tail,
                derivative,
                control_ridge=float(control_ridge),
                smoothness=float(smoothness),
                fir_ridge=float(fir_ridge),
            )
            updates.append(
                _block_diagnostics(
                    iteration=iteration,
                    block="fir_tail",
                    solve=tail_solve,
                    objective_before=objective,
                    objective_after=next_objective,
                    coefficient_norm=float(np.linalg.norm(next_tail)),
                    increase_tolerance=float(objective_increase_tolerance),
                    numerical_scale_floor=numerical_scale_floor,
                )
            )
            fir_tail = next_tail
            prediction = next_prediction
            objective = next_objective

        filtered_control_design = sph_filtered_control_design_matrix(
            spline_design,
            fir_tail,
            segment_length=frame_length,
        )
        next_controls, control_solve = _regularized_complex_lstsq(
            filtered_control_design,
            target,
            control_penalties,
        )
        next_prediction = filtered_control_design @ next_controls
        next_objective = _sph_objective(
            next_prediction,
            target,
            next_controls,
            fir_tail,
            derivative,
            control_ridge=float(control_ridge),
            smoothness=float(smoothness),
            fir_ridge=float(fir_ridge),
        )
        updates.append(
            _block_diagnostics(
                iteration=iteration,
                block="control_points",
                solve=control_solve,
                objective_before=objective,
                objective_after=next_objective,
                coefficient_norm=float(np.linalg.norm(next_controls)),
                increase_tolerance=float(objective_increase_tolerance),
                numerical_scale_floor=numerical_scale_floor,
            )
        )
        control_points = next_controls
        prediction = next_prediction
        objective = next_objective
        completed_alternations = iteration

        scale = max(abs(iteration_start_objective), numerical_scale_floor)
        relative_full_decrease = (
            iteration_start_objective - objective
        ) / scale
        if (
            iteration >= minimum_iterations
            and relative_full_decrease <= convergence_tolerance
        ):
            converged = True
            convergence_reason = "relative_objective_converged"
            break

    model = SplineHammersteinPA(
        knots=knot_array,
        control_points=control_points.astype(dtype, copy=False),
        fir_tail=fir_tail.astype(dtype, copy=False),
        coordinate=fitted_coordinate,
        knot_strategy=strategy_label,
    )
    serialized_prediction = model.predict_segments(samples, frame_length).astype(
        np.complex128,
        copy=False,
    )
    serialized_objective = _sph_objective(
        serialized_prediction,
        target,
        model.control_points.astype(np.complex128, copy=False),
        model.fir_tail.astype(np.complex128, copy=False),
        derivative,
        control_ridge=float(control_ridge),
        smoothness=float(smoothness),
        fir_ridge=float(fir_ridge),
    )
    training_mse = float(np.mean(np.abs(serialized_prediction - target) ** 2))
    relative_error = training_mse / target_power
    all_full_rank = all(
        update.data_design_rank == update.feature_count for update in updates
    )
    diagnostics = SplineHammersteinFitDiagnostics(
        sample_count=int(samples.size),
        segment_length=frame_length,
        segment_count=int(np.ceil(samples.size / frame_length)),
        knot_count=int(knot_array.size),
        knot_strategy=strategy_label,
        coordinate=fitted_coordinate,
        fir_length=length,
        control_ridge=float(control_ridge),
        smoothness=float(smoothness),
        fir_ridge=float(fir_ridge),
        maximum_alternations=maximum_iterations,
        minimum_alternations=minimum_iterations,
        completed_alternations=completed_alternations,
        convergence_tolerance=float(convergence_tolerance),
        objective_increase_tolerance=float(objective_increase_tolerance),
        converged=converged,
        convergence_reason=convergence_reason,
        zero_model_objective=float(zero_objective),
        memoryless_initial_objective=float(memoryless_objective),
        optimization_final_objective=float(objective),
        serialized_model_objective=float(serialized_objective),
        all_updates_monotonic=all(
            update.objective_after
            <= update.objective_before
            + objective_increase_tolerance
            * max(abs(update.objective_before), numerical_scale_floor)
            for update in updates
        ),
        all_data_designs_full_column_rank=all_full_rank,
        minimum_nonzero_control_feature_samples=minimum_feature_samples,
        maximum_calibration_coordinate=maximum_coordinate,
        control_point_l2_norm=float(np.linalg.norm(model.control_points)),
        fir_tail_l2_norm=float(np.linalg.norm(model.fir_tail)),
        target_power=target_power,
        training_mse=training_mse,
        training_relative_error_power=relative_error,
        training_nmse_db=nmse_pooled_db(serialized_prediction, target),
        coefficient_dtype=str(model.control_points.dtype),
        fit_wall_time_seconds=float(time.perf_counter() - start_time),
        solver="deterministic_alternating_augmented_complex_lstsq",
        h0_contract="1+0j fixed and not stored",
        updates=tuple(updates),
    )
    return model, diagnostics
