"""Phase-equivariant complex linear-spline digital predistorter.

The deployed model is

    z[n] = x[n] C(|x[n]|),

where ``C`` is a complex, continuous, piecewise-linear function.  Only the two
control points bracketing ``|x[n]|`` are active for a sample.  Calibration is a
small complex ridge regression; inference is vectorized and has no Python loop
over samples.

This module intentionally distinguishes postdistorter calibration inputs from
predistorter deployment inputs:

* ILA calibration: ``u = measured_pa_output / gain`` and ``target = pa_input``;
* deployment: ``desired_signal -> model.predict -> physical/surrogate PA``.

An inverse reconstruction score on ``u`` is diagnostic only.  It is not the
deployment score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

KnotStrategy = Literal[
    "uniform_amplitude",
    "uniform_power",
    "quantile",
    "compression_aware",
]


def _as_complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional complex sequence")
    if not np.issubdtype(array.dtype, np.complexfloating):
        array = array.astype(np.complex128)
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _strict_knots(knots: np.ndarray) -> np.ndarray:
    """Validate and return strictly increasing float64 knots.

    Earlier revisions silently replaced duplicate/descending knots with
    ``nextafter`` values.  That changes the interpolation domain by an
    implementation-dependent epsilon and is especially harmful for quantile
    knots when a capture contains repeated amplitudes.  Knot construction now
    owns any duplicate removal; explicit knots are rejected so a malformed
    deployment table cannot pass unnoticed.
    """

    result = np.asarray(knots, dtype=np.float64).copy()
    if result.ndim != 1 or result.size < 2:
        raise ValueError("at least two one-dimensional knots are required")
    if not np.all(np.isfinite(result)):
        raise ValueError("knots contain NaN or infinite values")
    if result[0] < 0.0:
        raise ValueError("amplitude-domain knots must be non-negative")
    if not np.all(np.diff(result) > 0):
        raise ValueError(
            "knots must be strictly increasing; remove duplicate values "
            "during knot construction rather than perturbing them"
        )
    return result


def make_knots(
    calibration_input: np.ndarray,
    count: int,
    strategy: KnotStrategy = "uniform_amplitude",
    *,
    compression_power: float = 2.0,
) -> np.ndarray:
    """Construct amplitude-domain knots from calibration samples.

    ``uniform_power`` changes knot *placement*: its amplitude knots are the
    square roots of an equally spaced power grid.  Interpolation remains linear
    in amplitude, as required by ``C(r)``.  A separate power-coordinate spline
    would have a different basis and must be benchmarked as a distinct model.
    """

    if count < 2:
        raise ValueError("count must be at least two")
    samples = _as_complex_vector(calibration_input, name="calibration_input")
    radius = np.abs(samples).astype(np.float64, copy=False)
    maximum = float(np.max(radius, initial=0.0))
    if maximum <= 0.0:
        raise ValueError("calibration_input must contain a non-zero sample")

    unit = np.linspace(0.0, 1.0, count, dtype=np.float64)
    if strategy == "uniform_amplitude":
        knots = maximum * unit
    elif strategy == "uniform_power":
        knots = maximum * np.sqrt(unit)
    elif strategy == "quantile":
        knots = np.quantile(radius, unit)
        # Cover zero-valued desired inputs even if the finite calibration record
        # happens not to contain an exact zero.
        knots[0] = 0.0
        knots[-1] = maximum
        # A quantile of a heavily quantized/low-PAPR capture can contain
        # repeated radii.  Keep the unique support and expose the effective
        # knot count through the fitted model diagnostics instead of fabricating
        # near-zero intervals.
        knots = np.unique(knots)
    elif strategy == "compression_aware":
        if not np.isfinite(compression_power) or compression_power <= 1.0:
            raise ValueError(
                "compression_power must be finite and greater than one"
            )
        # Shrinking intervals towards the high-amplitude compression region.
        knots = maximum * (1.0 - np.power(1.0 - unit, compression_power))
    else:
        raise ValueError(f"unknown knot strategy: {strategy}")

    return _strict_knots(knots)


def local_spline_coordinates(
    radius: np.ndarray,
    knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left-knot indices and interpolation weights.

    Values outside the calibrated range use endpoint control points.  This is
    constant extrapolation of ``C`` (not of the complex output ``x*C``), which
    avoids an unconstrained slope beyond the last observed amplitude.
    """

    knot_array = _strict_knots(knots)
    radii = np.asarray(radius, dtype=np.float64)
    if np.any(~np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("radius must be finite and non-negative")
    clipped = np.clip(radii, knot_array[0], knot_array[-1])
    left = np.searchsorted(knot_array, clipped, side="right") - 1
    left = np.clip(left, 0, knot_array.size - 2)
    width = knot_array[left + 1] - knot_array[left]
    weight = (clipped - knot_array[left]) / width
    return left.astype(np.int64, copy=False), weight


def spline_basis(radius: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Build the dense training basis; each row has at most two non-zeros."""

    radii = np.asarray(radius, dtype=np.float64)
    flat = radii.reshape(-1)
    knot_array = _strict_knots(knots)
    left, weight = local_spline_coordinates(flat, knot_array)
    basis = np.zeros((flat.size, knot_array.size), dtype=np.float64)
    rows = np.arange(flat.size)
    basis[rows, left] = 1.0 - weight
    basis[rows, left + 1] += weight
    return basis.reshape(radii.shape + (knot_array.size,))


def complex_design_matrix(samples: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return ``Phi[n,k] = samples[n] * B_k(|samples[n]|)``."""

    values = _as_complex_vector(samples, name="samples")
    basis = spline_basis(np.abs(values), knots)
    return values[:, None] * basis


@dataclass(frozen=True)
class SplineFitDiagnostics:
    """Numerical diagnostics for one closed-form fit."""

    sample_count: int
    knot_count: int
    ridge: float
    smoothness: float
    # Legacy-compatible field: condition(augmented_design)^2, not used for
    # solving.  Use the explicitly named fields below for new reports.
    gram_condition_number: float
    augmented_design_condition_number: float
    solver_rank: int
    solver: str
    data_design_rank: int
    data_design_condition_number: float
    minimum_nonzero_feature_samples: int
    training_mse: float
    training_relative_error_power: float
    training_nmse_db: float
    maximum_calibration_radius: float


@dataclass(frozen=True)
class ComplexLinearSplineDPD:
    """Immutable complex spline coefficient set."""

    knots: np.ndarray
    coefficients: np.ndarray
    knot_strategy: str = "explicit"

    def __post_init__(self) -> None:
        knots = _strict_knots(self.knots)
        coefficients = np.asarray(self.coefficients)
        if coefficients.ndim != 1 or coefficients.size != knots.size:
            raise ValueError("one complex coefficient is required per knot")
        if not np.issubdtype(coefficients.dtype, np.complexfloating):
            coefficients = coefficients.astype(np.complex128)
        if not np.all(np.isfinite(coefficients.real)) or not np.all(
            np.isfinite(coefficients.imag)
        ):
            raise ValueError("coefficients contain NaN or infinite values")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "coefficients", coefficients.copy())

    @property
    def knot_count(self) -> int:
        return int(self.knots.size)

    @property
    def stored_real_coefficients(self) -> int:
        return 2 * self.knot_count

    def correction(self, radius: np.ndarray) -> np.ndarray:
        radii = np.asarray(radius, dtype=np.float64)
        left, weight = local_spline_coordinates(radii, self.knots)
        return self.coefficients[left] + weight * (
            self.coefficients[left + 1] - self.coefficients[left]
        )

    def predict(self, desired_signal: np.ndarray) -> np.ndarray:
        """Predistort a desired complex baseband sequence."""

        original = np.asarray(desired_signal)
        shape = original.shape
        flat = _as_complex_vector(original.reshape(-1), name="desired_signal")
        correction = self.correction(np.abs(flat))
        output = flat * correction
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        return output.astype(target_dtype, copy=False).reshape(shape)

    __call__ = predict

    def save(self, path: str | Path) -> None:
        """Save the small coefficient set without serializing Python objects."""

        destination = Path(path)
        np.savez(
            destination,
            schema_version=np.asarray(1, dtype=np.int64),
            knots=self.knots,
            coefficients=self.coefficients,
            knot_strategy=np.asarray(self.knot_strategy),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ComplexLinearSplineDPD":
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["schema_version"])
            if version != 1:
                raise ValueError(f"unsupported spline schema version: {version}")
            return cls(
                knots=data["knots"],
                coefficients=data["coefficients"],
                knot_strategy=str(data["knot_strategy"]),
            )


def _second_derivative_operator(knots: np.ndarray) -> np.ndarray:
    """Return a spacing-aware second-derivative finite-difference operator.

    For non-uniform knots, the ordinary index-space ``[1,-2,1]`` penalty
    over-penalizes wide intervals and under-penalizes narrow ones.  Each row
    below is the second derivative of the piecewise-linear control-point
    sequence at the middle knot, using the two adjacent interval widths.
    """

    knot_array = _strict_knots(knots)
    count = knot_array.size
    if count < 3:
        return np.zeros((0, count), dtype=np.float64)
    widths = np.diff(knot_array)
    operator = np.zeros((count - 2, count), dtype=np.float64)
    for row in range(count - 2):
        left_width = widths[row]
        right_width = widths[row + 1]
        total = left_width + right_width
        # Quadrature weight makes ||D c||^2 approximate the integral of
        # squared curvature rather than a raw sum whose scale changes with
        # local knot density.
        quadrature_weight = np.sqrt(total / 2.0)
        operator[row, row] = (
            quadrature_weight * 2.0 / (left_width * total)
        )
        operator[row, row + 1] = (
            -quadrature_weight * 2.0 / (left_width * right_width)
        )
        operator[row, row + 2] = (
            quadrature_weight * 2.0 / (right_width * total)
        )
    return operator


def _second_difference_penalty(knots: np.ndarray) -> np.ndarray:
    """Return the Hermitian penalty matrix for backwards-compatible callers."""

    operator = _second_derivative_operator(knots)
    return operator.T @ operator


def fit_complex_linear_spline(
    calibration_input: np.ndarray,
    target: np.ndarray,
    *,
    knot_count: int | None = None,
    knot_strategy: KnotStrategy = "uniform_amplitude",
    knots: np.ndarray | None = None,
    ridge: float = 0.0,
    smoothness: float = 0.0,
    compression_power: float = 2.0,
    coefficient_dtype: np.dtype = np.complex128,
) -> tuple[ComplexLinearSplineDPD, SplineFitDiagnostics]:
    """Fit a complex spline using normalized ridge/smoothness penalties.

    The solved objective is

    ``mean(|Phi c - target|^2) + ridge*||c||^2
       + smoothness*||D2 c||^2``.

    The solve uses an augmented complex least-squares system (QR/SVD in
    ``numpy.linalg.lstsq``), rather than explicitly solving normal equations.
    This avoids squaring the condition number for narrow/quantile intervals
    while retaining one joint complex solution for I and Q.
    """

    values = _as_complex_vector(calibration_input, name="calibration_input")
    desired = _as_complex_vector(target, name="target")
    if values.size != desired.size:
        raise ValueError("calibration_input and target must have equal length")
    if values.size == 0:
        raise ValueError("cannot fit an empty sequence")
    if (
        not np.isfinite(ridge)
        or not np.isfinite(smoothness)
        or ridge < 0.0
        or smoothness < 0.0
    ):
        raise ValueError(
            "regularization strengths must be finite and non-negative"
        )
    coefficient_dtype = np.dtype(coefficient_dtype)
    if not np.issubdtype(coefficient_dtype, np.complexfloating):
        raise ValueError("coefficient_dtype must be a complex floating dtype")

    if knots is None:
        if knot_count is None:
            raise ValueError("knot_count is required when knots are not supplied")
        knot_array = make_knots(
            values,
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

    phi = complex_design_matrix(values, knot_array).astype(
        np.complex128, copy=False
    )
    desired128 = desired.astype(np.complex128, copy=False)
    sample_count = values.size
    if float(np.mean(np.abs(desired128) ** 2)) <= 0.0:
        raise ValueError("target must contain non-zero energy")

    # Identifiability is a property of the data design, not of the augmented
    # ridge system (ridge makes the latter full rank by construction).  Use
    # singular values directly so the reported rank/condition is not itself
    # affected by squaring a narrow direction in Phi^H Phi.
    data_singular_values = np.linalg.svd(
        phi / np.sqrt(float(sample_count)),
        compute_uv=False,
    )
    maximum_singular_value = float(
        np.max(data_singular_values, initial=0.0)
    )
    rank_tolerance = (
        maximum_singular_value
        * max(phi.shape)
        * np.finfo(np.float64).eps
    )
    data_design_rank = int(
        np.count_nonzero(data_singular_values > rank_tolerance)
    )
    if data_design_rank < knot_array.size or data_design_rank == 0:
        data_design_condition_number = float("inf")
    else:
        positive = data_singular_values[
            data_singular_values > rank_tolerance
        ]
        data_design_condition_number = float(
            np.max(positive) / np.min(positive)
        )
    minimum_nonzero_feature_samples = int(
        np.min(np.count_nonzero(np.abs(phi) > 0.0, axis=0))
    )

    # Scaling the data rows by sqrt(1/N) makes the augmented least-squares
    # objective exactly match the documented mean-square objective.
    blocks = [phi / np.sqrt(float(sample_count))]
    right_hand_side = [desired128 / np.sqrt(float(sample_count))]
    if ridge:
        blocks.append(
            np.sqrt(float(ridge))
            * np.eye(knot_array.size, dtype=np.complex128)
        )
        right_hand_side.append(
            np.zeros(knot_array.size, dtype=np.complex128)
        )
    if smoothness:
        derivative = _second_derivative_operator(knot_array)
        blocks.append(np.sqrt(float(smoothness)) * derivative)
        right_hand_side.append(
            np.zeros(derivative.shape[0], dtype=np.complex128)
        )
    augmented = np.vstack(blocks)
    augmented_target = np.concatenate(right_hand_side)
    coefficients, _, solver_rank, singular_values = np.linalg.lstsq(
        augmented,
        augmented_target,
        rcond=None,
    )
    if singular_values.size == 0 or singular_values[-1] <= np.finfo(float).tiny:
        augmented_condition_number = float("inf")
    else:
        augmented_condition_number = float(
            singular_values[0] / singular_values[-1]
        )
    if np.isfinite(augmented_condition_number):
        max_float_sqrt = np.sqrt(np.finfo(float).max)
        gram_condition_number = float(
            augmented_condition_number**2
            if augmented_condition_number <= max_float_sqrt
            else np.inf
        )
    else:
        gram_condition_number = float("inf")

    coefficients = coefficients.astype(coefficient_dtype, copy=False)
    model = ComplexLinearSplineDPD(
        knots=knot_array,
        coefficients=coefficients,
        knot_strategy=strategy_label,
    )
    prediction = model.predict(values).astype(np.complex128, copy=False)
    error_power = float(np.mean(np.abs(prediction - desired128) ** 2))
    reference_power = float(np.mean(np.abs(desired128) ** 2))
    relative = error_power / reference_power if reference_power > 0 else np.inf
    nmse = 10.0 * np.log10(max(relative, np.finfo(float).tiny))
    diagnostics = SplineFitDiagnostics(
        sample_count=sample_count,
        knot_count=knot_array.size,
        ridge=float(ridge),
        smoothness=float(smoothness),
        gram_condition_number=gram_condition_number,
        augmented_design_condition_number=augmented_condition_number,
        solver_rank=int(solver_rank),
        solver="augmented_complex_lstsq",
        data_design_rank=data_design_rank,
        data_design_condition_number=data_design_condition_number,
        minimum_nonzero_feature_samples=minimum_nonzero_feature_samples,
        training_mse=error_power,
        training_relative_error_power=relative,
        training_nmse_db=float(nmse),
        maximum_calibration_radius=float(np.max(np.abs(values))),
    )
    return model, diagnostics


def fit_ila_postdistorter(
    known_pa_input: np.ndarray,
    measured_pa_output: np.ndarray,
    gain: complex,
    **fit_kwargs: object,
) -> tuple[ComplexLinearSplineDPD, SplineFitDiagnostics]:
    """Fit ``x ~= D(y/g)`` for ILA, without performing a circular test."""

    if not np.isfinite(gain.real) or not np.isfinite(gain.imag) or abs(gain) == 0:
        raise ValueError("gain must be finite and non-zero")
    pa_input = _as_complex_vector(known_pa_input, name="known_pa_input")
    pa_output = _as_complex_vector(measured_pa_output, name="measured_pa_output")
    if pa_input.size != pa_output.size:
        raise ValueError("known_pa_input and measured_pa_output must match")
    normalized_observation = pa_output / gain
    return fit_complex_linear_spline(
        normalized_observation,
        pa_input,
        **fit_kwargs,
    )
