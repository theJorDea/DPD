"""Branch-specific residual hysteresis extension for spline-memory DPD.

The existing baseline is preserved as the shared coefficient bank ``C``.  The
extension adds a regularized residual bank ``H`` and a ternary branch-local gate
``g_m[n] in {-1, 0, +1}``:

    z[n] = sum_m x[n-m] * (C_m(r_m[n]) + g_m[n] H_m(r_m[n]))
    r_m[n] = |x[n-m]|

The gate uses the direction of the *same delayed envelope* as its branch:

    delta_r_m[n] = |x[n-m]| - |x[n-m-1]|.

This keeps the topology phase-equivariant and avoids silently replacing the
validated aligned-delay BlackBox model by a common-envelope model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .complexity import OperationCount, complex_multiply_cost
from .spline_memory_dpd import (
    BranchLike,
    SplineMemoryBranch,
    _as_complex_vector,
    _causal_delay,
    _normalize_branches,
    spline_memory_design_matrix,
)
from .complex_spline_dpd import _strict_knots, make_knots


def branch_hysteresis_gate(
    signal: np.ndarray,
    branches: Iterable[BranchLike],
    deadband: float,
) -> np.ndarray:
    """Return one ternary ``up/steady/down`` gate per branch and sample.

    ``deadband`` is in the same normalized amplitude units as ``signal``.
    Samples with ``delta_r > deadband`` receive ``+1``; samples with
    ``delta_r < -deadband`` receive ``-1``; all others receive ``0``.
    """

    samples = _as_complex_vector(signal, name="signal")
    if not np.isfinite(deadband) or deadband < 0.0:
        raise ValueError("deadband must be finite and non-negative")
    branch_tuple = _normalize_branches(branches)
    gates = np.zeros((samples.size, len(branch_tuple)), dtype=np.int8)
    for branch_index, branch in enumerate(branch_tuple):
        current = np.abs(_causal_delay(samples, branch.envelope_delay))
        previous = np.abs(_causal_delay(samples, branch.envelope_delay + 1))
        delta = current - previous
        gates[delta > deadband, branch_index] = 1
        gates[delta < -deadband, branch_index] = -1
    return gates


def hysteresis_design_matrices(
    signal: np.ndarray,
    knots: np.ndarray,
    branches: Iterable[BranchLike],
    deadband: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build baseline and branch-local residual spline dictionaries."""

    branch_tuple = _normalize_branches(branches)
    knot_array = _strict_knots(knots)
    base = spline_memory_design_matrix(signal, knot_array, branch_tuple)
    gates = branch_hysteresis_gate(signal, branch_tuple, deadband)
    residual = np.zeros_like(base)
    for branch_index in range(len(branch_tuple)):
        start = branch_index * knot_array.size
        stop = start + knot_array.size
        residual[:, start:stop] = base[:, start:stop] * gates[:, branch_index, None]
    return base, residual, gates


def _second_difference_matrix(knot_count: int) -> np.ndarray:
    if knot_count < 3:
        return np.zeros((0, knot_count), dtype=float)
    matrix = np.zeros((knot_count - 2, knot_count), dtype=float)
    rows = np.arange(knot_count - 2)
    matrix[rows, rows] = 1.0
    matrix[rows, rows + 1] = -2.0
    matrix[rows, rows + 2] = 1.0
    return matrix


@dataclass(frozen=True)
class HysteresisFitDiagnostics:
    """Diagnostics retained for model-selection and residual audits."""

    sample_count: int
    warmup: int
    deadband: float
    up_fraction: tuple[float, ...]
    steady_fraction: tuple[float, ...]
    down_fraction: tuple[float, ...]
    augmented_condition_number: float
    solver_rank: int
    baseline_norm: float
    residual_norm: float
    residual_to_baseline_norm: float
    training_nmse_db: float


@dataclass(frozen=True)
class ResidualHysteresisSplineMemoryDPD:
    """Immutable baseline-plus-residual hysteresis DPD."""

    knots: np.ndarray
    branches: tuple[SplineMemoryBranch, ...]
    baseline_coefficients: np.ndarray
    residual_coefficients: np.ndarray
    deadband: float = 0.0

    def __post_init__(self) -> None:
        knots = _strict_knots(self.knots)
        branches = _normalize_branches(self.branches)
        expected = (len(branches), knots.size)
        baseline = np.asarray(self.baseline_coefficients, dtype=np.complex128)
        residual = np.asarray(self.residual_coefficients, dtype=np.complex128)
        if baseline.shape != expected or residual.shape != expected:
            raise ValueError(f"coefficient arrays must have shape {expected}")
        if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(residual)):
            raise ValueError("coefficients must be finite")
        if not np.isfinite(self.deadband) or self.deadband < 0.0:
            raise ValueError("deadband must be finite and non-negative")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "baseline_coefficients", baseline.copy())
        object.__setattr__(self, "residual_coefficients", residual.copy())
        object.__setattr__(self, "deadband", float(self.deadband))

    @property
    def knot_count(self) -> int:
        return int(self.knots.size)

    @property
    def branch_count(self) -> int:
        return len(self.branches)

    @property
    def maximum_delay(self) -> int:
        return max(
            max(branch.signal_delay, branch.envelope_delay + 1)
            for branch in self.branches
        )

    @property
    def stored_complex_coefficients(self) -> int:
        return int(self.baseline_coefficients.size + self.residual_coefficients.size)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Evaluate one independent causal record with zero history."""

        samples = _as_complex_vector(signal, name="signal")
        base, residual, _ = hysteresis_design_matrices(
            samples, self.knots, self.branches, self.deadband
        )
        feature_count = self.branch_count * self.knot_count
        output = (
            base @ self.baseline_coefficients.reshape(-1)
            + residual @ self.residual_coefficients.reshape(-1)
        )
        assert base.shape[1] == feature_count
        return output

    __call__ = predict

    def operation_count(self, *, precomputed_banks: bool = True) -> OperationCount:
        """Count fast-path work under the project's 4M+2A complex convention.

        With ``precomputed_banks=True`` the offline combinations ``C+H`` and
        ``C-H`` are stored, so runtime interpolation is baseline-like and only
        gate subtraction/comparisons and an extra previous-envelope history are
        added.  ``False`` reports the online ``C + g*H`` arithmetic instead.
        """

        cmul, cadd = complex_multiply_cost("4m2a")
        envelope_groups = len({b.envelope_delay for b in self.branches})
        baseline_mul = 3 * envelope_groups + self.branch_count * (2 + cmul)
        baseline_add = 2 * envelope_groups + self.branch_count * (4 + cadd)
        baseline_add += 2 * (self.branch_count - 1)
        if precomputed_banks:
            extra_mul = 0
            extra_add = self.branch_count  # delta r subtraction
            extra_nonlinear = 0
        else:
            extra_mul = 2 * self.branch_count
            extra_add = 7 * self.branch_count  # delta + C/H combination
            extra_nonlinear = 0
        return OperationCount(
            real_multiplications=baseline_mul + extra_mul,
            real_additions=baseline_add + extra_add,
            nonlinear_operations=envelope_groups + extra_nonlinear,
            comparisons=envelope_groups * int(np.ceil(np.log2(self.knot_count)))
            + self.branch_count,
            lookups=(2 * self.branch_count) * (1 if precomputed_banks else 2),
            state_real_values=2 * self.maximum_delay,
            stored_real_coefficients=2 * self.stored_complex_coefficients,
            notes=(
                "branch-local ternary gate; g in {-1,0,+1}",
                "precomputed C+H/C/C-H banks" if precomputed_banks else "online residual interpolation",
                "physical memory traffic and control latency remain platform-dependent",
            ),
        )


def fit_residual_hysteresis_spline_memory_dpd(
    calibration_input: np.ndarray,
    target: np.ndarray,
    *,
    branches: Iterable[BranchLike],
    knot_count: int,
    deadband: float,
    ridge_baseline: float = 1e-4,
    ridge_residual: float = 1e-3,
    smoothness_baseline: float = 0.0,
    smoothness_residual: float = 0.0,
    knots: np.ndarray | None = None,
) -> tuple[ResidualHysteresisSplineMemoryDPD, HysteresisFitDiagnostics]:
    """Fit a shared baseline and regularized branch-local hysteresis residual."""

    values = _as_complex_vector(calibration_input, name="calibration_input")
    desired = _as_complex_vector(target, name="target")
    if values.shape != desired.shape:
        raise ValueError("calibration_input and target must have equal length")
    branch_tuple = _normalize_branches(branches)
    knot_array = make_knots(values, knot_count, "quantile") if knots is None else _strict_knots(knots)
    warmup = max(max(b.signal_delay, b.envelope_delay + 1) for b in branch_tuple)
    base, residual, gates = hysteresis_design_matrices(values, knot_array, branch_tuple, deadband)
    base = base[warmup:]
    residual = residual[warmup:]
    target_steady = desired[warmup:]
    sample_count, feature_count = base.shape
    joint = np.column_stack((base, residual)) / np.sqrt(float(sample_count))
    normalized_target = target_steady / np.sqrt(float(sample_count))

    regularizer_rows: list[np.ndarray] = []
    if ridge_baseline < 0 or ridge_residual < 0 or smoothness_baseline < 0 or smoothness_residual < 0:
        raise ValueError("regularization values must be non-negative")
    if ridge_baseline:
        row = np.zeros((feature_count, 2 * feature_count), dtype=np.complex128)
        row[:, :feature_count] = np.sqrt(ridge_baseline) * np.eye(feature_count)
        regularizer_rows.append(row)
    if ridge_residual:
        row = np.zeros((feature_count, 2 * feature_count), dtype=np.complex128)
        row[:, feature_count:] = np.sqrt(ridge_residual) * np.eye(feature_count)
        regularizer_rows.append(row)
    d2 = _second_difference_matrix(knot_array.size)
    if d2.size and (smoothness_baseline or smoothness_residual):
        for strength, offset in ((smoothness_baseline, 0), (smoothness_residual, feature_count)):
            if strength:
                row = np.zeros((d2.shape[0] * len(branch_tuple), 2 * feature_count), dtype=np.complex128)
                for branch_index in range(len(branch_tuple)):
                    start = branch_index * knot_array.size
                    stop = start + knot_array.size
                    row[branch_index * d2.shape[0]:(branch_index + 1) * d2.shape[0], offset + start:offset + stop] = np.sqrt(strength) * d2
                regularizer_rows.append(row)
    augmented = np.vstack([joint, *regularizer_rows]) if regularizer_rows else joint
    augmented_target = np.concatenate([normalized_target, np.zeros(augmented.shape[0] - sample_count, dtype=np.complex128)])
    condition = float(np.linalg.cond(augmented))
    coefficients, _, rank, _ = np.linalg.lstsq(augmented, augmented_target, rcond=None)
    model = ResidualHysteresisSplineMemoryDPD(
        knots=knot_array,
        branches=branch_tuple,
        baseline_coefficients=coefficients[:feature_count].reshape(len(branch_tuple), knot_array.size),
        residual_coefficients=coefficients[feature_count:].reshape(len(branch_tuple), knot_array.size),
        deadband=deadband,
    )
    prediction = model.predict(values)
    error = prediction[warmup:] - desired[warmup:]
    reference = desired[warmup:]
    nmse_db = float(10.0 * np.log10(np.mean(np.abs(error) ** 2) / np.mean(np.abs(reference) ** 2)))
    up_fraction = tuple(float(np.mean(gates[warmup:, i] == 1)) for i in range(len(branch_tuple)))
    steady_fraction = tuple(float(np.mean(gates[warmup:, i] == 0)) for i in range(len(branch_tuple)))
    down_fraction = tuple(float(np.mean(gates[warmup:, i] == -1)) for i in range(len(branch_tuple)))
    baseline_norm = float(np.linalg.norm(model.baseline_coefficients))
    residual_norm = float(np.linalg.norm(model.residual_coefficients))
    diagnostics = HysteresisFitDiagnostics(
        sample_count=sample_count,
        warmup=warmup,
        deadband=float(deadband),
        up_fraction=up_fraction,
        steady_fraction=steady_fraction,
        down_fraction=down_fraction,
        augmented_condition_number=condition,
        solver_rank=int(rank),
        baseline_norm=baseline_norm,
        residual_norm=residual_norm,
        residual_to_baseline_norm=float(residual_norm / max(baseline_norm, 1e-30)),
        training_nmse_db=nmse_db,
    )
    return model, diagnostics
