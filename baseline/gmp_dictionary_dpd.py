"""Composite spline-memory + sparse GMP-class DPD with greedy selection.

The deployed model is

    u[n] = D_spline(x)[n] + sum_i c_i * x[n-m_i] * q[n-d_i]^{k_i},
    q[n] = |x[n]|^2,

where every dictionary member ``x[n-m] * q[n-d]^k`` is phase equivariant
(``q`` is a non-negative real envelope power), so the composite preserves

    D(x * e^{j phi}) = e^{j phi} * D(x)

exactly, including the bit-exact 90-degree rotation behavior required by
the fixed-point gate.

Selection is greedy orthogonal matching pursuit over the normalized
candidate grid (doubly-orthogonalized flavor: selection and coefficient
estimation operate on an orthonormalized Gram system).  The per-sample
operation count mirrors the repository conventions: one shared envelope
generator per distinct envelope delay, one real multiplication per extra
power of ``q`` per distinct envelope delay, and one complex multiply plus
accumulate per selected member.  A candidate configuration is deployable
only if its composite count stays within the 1000 real-multiplications
budget.

Everything in this module is offline calibration support.  The deployed
object is a frozen linear-in-parameters causal filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .complexity import (
    ComplexMultiplyConvention,
    OperationCount,
    complex_multiply_cost,
)
from .spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryState,
)

__all__ = [
    "GmpMember",
    "GmpDictionaryGrid",
    "CompositeSplineGmpDPD",
    "CompositeSplineGmpState",
    "gmp_member_grid",
    "gmp_dictionary_columns",
    "gmp_member_operation_count",
    "composite_design_matrix",
    "orthogonal_matching_pursuit",
    "fit_gmp_residual_members",
]


def _complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _delay(values: np.ndarray, delay: int) -> np.ndarray:
    """Return ``values[n-delay]`` with zeros before the record start."""

    result = np.zeros(values.shape, dtype=values.dtype)
    if delay == 0:
        result[:] = values
    elif 0 < delay < values.size:
        result[delay:] = values[:-delay]
    return result


@dataclass(frozen=True)
class GmpMember:
    """One phase-equivariant dictionary member."""

    signal_delay: int
    envelope_delay: int
    exponent: int

    def __post_init__(self) -> None:
        for name in ("signal_delay", "envelope_delay", "exponent"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if int(self.exponent) < 1:
            raise ValueError("exponent must be at least 1 (q^1)")


@dataclass(frozen=True)
class GmpDictionaryGrid:
    """Candidate grid: m in [0, M), d in [0, D), k in [1, K]."""

    maximum_signal_delay: int
    maximum_envelope_delay: int
    maximum_exponent: int

    def __post_init__(self) -> None:
        for name in (
            "maximum_signal_delay",
            "maximum_envelope_delay",
            "maximum_exponent",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def members(self) -> tuple[GmpMember, ...]:
        return tuple(
            GmpMember(m, d, k)
            for k in range(1, self.maximum_exponent + 1)
            for d in range(self.maximum_envelope_delay)
            for m in range(self.maximum_signal_delay)
        )


def gmp_dictionary_columns(
    signal: np.ndarray,
    members: tuple[GmpMember, ...],
) -> np.ndarray:
    """Design matrix (N, len(members)) of phase-equivariant members."""

    samples = _complex_vector(signal, name="signal")
    envelope_square = samples.real * samples.real + samples.imag * samples.imag
    delayed_signal_cache: dict[int, np.ndarray] = {}
    delayed_power_cache: dict[tuple[int, int], np.ndarray] = {}

    def delayed_signal(delay: int) -> np.ndarray:
        if delay not in delayed_signal_cache:
            delayed_signal_cache[delay] = _delay(samples, delay)
        return delayed_signal_cache[delay]

    def delayed_power(delay: int, exponent: int) -> np.ndarray:
        key = (delay, exponent)
        if key not in delayed_power_cache:
            power = np.ones(samples.shape, dtype=float)
            for _ in range(exponent):
                power = power * _delay(envelope_square, delay)
            delayed_power_cache[key] = power
        return delayed_power_cache[key]

    columns = np.empty(
        (samples.size, len(members)),
        dtype=np.complex128,
    )
    for index, member in enumerate(members):
        columns[:, index] = delayed_signal(member.signal_delay) * (
            delayed_power(member.envelope_delay, member.exponent)
        )
    return columns


def gmp_member_operation_count(
    members: tuple[GmpMember, ...],
    *,
    shared_envelope_delays: frozenset[int] | set[int] = frozenset(),
    convention: ComplexMultiplyConvention = "4m2a",
) -> OperationCount:
    """Per-sample cost of the selected member set (adds to a host model).

    Accounting: one shared ``|x|^2`` generator (2M+1A) and one ``sqrt`` per
    distinct envelope delay not already provided by the host model; one
    real multiplication per extra power of ``q`` per distinct envelope
    delay; one complex multiply (and accumulate) per selected member.
    """

    if not members:
        return OperationCount(
            real_multiplications=0,
            real_additions=0,
            nonlinear_operations=0,
            comparisons=0,
            lookups=0,
            real_memory_reads=0,
            real_memory_writes=0,
            stored_real_coefficients=0,
            stored_real_constants=0,
            state_real_values=0,
            notes=("empty GMP member set",),
        )
    cmul, cadd = complex_multiply_cost(convention)
    envelope_delays = sorted({member.envelope_delay for member in members})
    shared = frozenset(shared_envelope_delays)
    new_delays = len(set(envelope_delays) - shared)
    generator_multiplications = 2 * new_delays
    generator_additions = 1 * new_delays
    power_multiplications = sum(
        max(
            (
                member.exponent
                for member in members
                if member.envelope_delay == delay
            ),
            default=1,
        )
        - 1
        for delay in envelope_delays
    )
    member_multiplications = len(members) * cmul
    member_additions = len(members) * (cadd + 2)
    maximum_delay = max(
        max(member.signal_delay for member in members),
        max(member.envelope_delay for member in members),
    )
    return OperationCount(
        real_multiplications=(
            generator_multiplications
            + power_multiplications
            + member_multiplications
        ),
        real_additions=generator_additions + member_additions,
        nonlinear_operations=new_delays,
        comparisons=0,
        lookups=0,
        real_memory_reads=(
            len(members) * 4 + 2 * len(envelope_delays)
        ),
        real_memory_writes=2 * max(maximum_delay, 1),
        stored_real_coefficients=2 * len(members),
        stored_real_constants=0,
        state_real_values=2 * maximum_delay,
        notes=(
            f"complex multiply convention {convention}",
            f"{len(envelope_delays)} distinct envelope delays "
            f"({new_delays} new versus host model)",
            "shared q=|x|^2 generator and delayed power streams",
            "one complex multiply plus accumulate per selected member",
        ),
    )


@dataclass(frozen=True)
class CompositeSplineGmpState:
    """Causal state: raw complex history shared by both model parts."""

    history: np.ndarray

    def __post_init__(self) -> None:
        history = np.asarray(self.history, dtype=np.complex128)
        if history.ndim != 1 or not np.all(np.isfinite(history)):
            raise ValueError("composite state history must be finite and 1-D")
        object.__setattr__(self, "history", history.copy())


@dataclass(frozen=True)
class CompositeSplineGmpDPD:
    """Frozen composite: spline-memory DPD plus sparse GMP-class residual."""

    spline: SparseSplineMemoryDPD
    members: tuple[GmpMember, ...]
    member_coefficients: np.ndarray

    def __post_init__(self) -> None:
        members = tuple(self.members)
        coefficients = np.asarray(self.member_coefficients)
        if coefficients.ndim != 1 or coefficients.size != len(members):
            raise ValueError(
                "member_coefficients must be one value per member"
            )
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("member coefficients contain non-finite values")
        delays = [0]
        for member in members:
            if member.signal_delay > self.spline.maximum_delay + 64:
                raise ValueError("member signal delay is unreasonably large")
            delays.append(max(member.signal_delay, member.envelope_delay))
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "member_coefficients", coefficients.copy())
        object.__setattr__(self, "_maximum_delay", int(max(delays)))

    @property
    def maximum_delay(self) -> int:
        return max(self.spline.maximum_delay, self._maximum_delay)

    def initial_state(self) -> CompositeSplineGmpState:
        return CompositeSplineGmpState(
            np.zeros(self.maximum_delay, dtype=np.complex128)
        )

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: CompositeSplineGmpState,
    ) -> tuple[np.ndarray, CompositeSplineGmpState]:
        samples = _complex_vector(signal, name="signal")
        history_length = self.maximum_delay
        stored_history = np.asarray(state.history, dtype=np.complex128)
        if stored_history.size != history_length:
            raise ValueError(
                "state history length must equal the model maximum delay"
            )
        combined = np.concatenate((stored_history, samples))
        if self.spline.maximum_delay:
            spline_state = SplineMemoryState(
                stored_history[-self.spline.maximum_delay:]
            )
        else:
            spline_state = SplineMemoryState(
                np.empty(0, dtype=np.complex128)
            )
        spline_output, next_spline_state = self.spline.predict_chunk(
            samples,
            spline_state,
        )
        tail = combined[combined.size - history_length :] if history_length else np.empty(0, dtype=np.complex128)
        if not self.members:
            return spline_output, CompositeSplineGmpState(tail)
        envelope_square = (
            combined.real * combined.real + combined.imag * combined.imag
        )
        output = spline_output.astype(np.complex128, copy=True)
        for member, coefficient in zip(
            self.members,
            self.member_coefficients,
            strict=True,
        ):
            offset = history_length
            signal_stop = offset - member.signal_delay + samples.size
            envelope_stop = (
                offset - member.envelope_delay + samples.size
            )
            signal_part = combined[
                offset - member.signal_delay : signal_stop
            ]
            power_part = np.ones(samples.size, dtype=float)
            envelope = envelope_square[
                offset - member.envelope_delay : envelope_stop
            ]
            for _ in range(member.exponent):
                power_part = power_part * envelope
            output += coefficient * signal_part * power_part
        return output, CompositeSplineGmpState(tail)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        output, _ = self.predict_chunk(signal, self.initial_state())
        return output

    __call__ = predict

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        samples = _complex_vector(signal, name="signal")
        output = np.empty(samples.size, dtype=np.complex128)
        for start in range(0, samples.size, segment_length):
            stop = min(start + segment_length, samples.size)
            output[start:stop] = self.predict(samples[start:stop])
        return output

    def operation_count(
        self,
        *,
        convention: ComplexMultiplyConvention = "4m2a",
        indexing: Literal["binary", "uniform"] = "binary",
    ) -> OperationCount:
        spline_count = self.spline.operation_count(
            convention=convention,
            indexing=indexing,
        )
        member_count = gmp_member_operation_count(
            self.members,
            shared_envelope_delays={
                branch.envelope_delay for branch in self.spline.branches
            },
            convention=convention,
        )
        return spline_count + member_count

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray(
                "phase_equivariant_composite_spline_gmp_dpd"
            ),
            spline_knots=self.spline.knots,
            spline_signal_delays=np.asarray(
                [branch.signal_delay for branch in self.spline.branches],
                dtype=np.int64,
            ),
            spline_envelope_delays=np.asarray(
                [branch.envelope_delay for branch in self.spline.branches],
                dtype=np.int64,
            ),
            spline_coefficients=self.spline.coefficients,
            knot_strategy=np.asarray(self.spline.knot_strategy),
            member_signal_delays=np.asarray(
                [member.signal_delay for member in self.members],
                dtype=np.int64,
            ),
            member_envelope_delays=np.asarray(
                [member.envelope_delay for member in self.members],
                dtype=np.int64,
            ),
            member_exponents=np.asarray(
                [member.exponent for member in self.members],
                dtype=np.int64,
            ),
            member_coefficients=self.member_coefficients,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompositeSplineGmpDPD":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported composite schema version")
            if str(data["model_type"]) != (
                "phase_equivariant_composite_spline_gmp_dpd"
            ):
                raise ValueError("unexpected composite model type")
            spline = SparseSplineMemoryDPD(
                knots=data["spline_knots"],
                branches=tuple(
                    (int(m), int(d))
                    for m, d in zip(
                        data["spline_signal_delays"],
                        data["spline_envelope_delays"],
                        strict=True,
                    )
                ),
                coefficients=data["spline_coefficients"],
                knot_strategy=str(data["knot_strategy"]),
            )
            members = tuple(
                GmpMember(int(m), int(d), int(k))
                for m, d, k in zip(
                    data["member_signal_delays"],
                    data["member_envelope_delays"],
                    data["member_exponents"],
                    strict=True,
                )
            )
            return cls(
                spline=spline,
                members=members,
                member_coefficients=data["member_coefficients"],
            )


def composite_design_matrix(
    composite_like: CompositeSplineGmpDPD,
    signal: np.ndarray,
) -> np.ndarray:
    """Full linear-in-parameters dictionary of the composite model.

    Columns ``0..B*K-1`` follow the spline branch-major/knot-major order;
    the remaining columns are the GMP member columns.  For a composite
    with an empty member set this reduces to the spline dictionary.
    """

    from .spline_memory_dpd import spline_memory_design_matrix

    spline_design = spline_memory_design_matrix(
        signal,
        composite_like.spline.knots,
        composite_like.spline.branches,
    )
    if not composite_like.members:
        return spline_design
    member_design = gmp_dictionary_columns(
        signal,
        composite_like.members,
    )
    return np.concatenate((spline_design, member_design), axis=1)


def orthogonal_matching_pursuit(
    design: np.ndarray,
    target: np.ndarray,
    *,
    maximum_members: int,
    ridge: float,
    minimum_improvement: float = 1e-10,
) -> tuple[list[int], np.ndarray, list[float]]:
    """Greedy OMP on column-normalized design with ridge Cholesky solves.

    Returns (selected column indices in selection order, unnormalized
    complex coefficients aligned with ``design`` columns, residual-power
    history).  The residual power is the exact ridge-regularized value
    ``min_c ||Phi_S c - t||^2 + ridge*||c||^2``.
    """

    design = np.asarray(design, dtype=np.complex128)
    target = _complex_vector(target, name="target")
    if design.shape[0] != target.size:
        raise ValueError("design rows must match target length")
    if maximum_members < 1:
        raise ValueError("maximum_members must be at least one")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    column_norms = np.linalg.norm(design, axis=0)
    usable = column_norms > 0.0
    if not np.any(usable):
        raise ValueError("design has no non-zero columns")
    normalized = design[:, usable] / column_norms[usable]
    gram = normalized.conj().T @ normalized
    correlation = normalized.conj().T @ target
    target_power = float(np.vdot(target, target).real)
    if target_power <= 0.0:
        raise ValueError("target must have positive power")

    candidate_count = normalized.shape[1]
    selected: list[int] = []
    residual_history: list[float] = [target_power]
    basis = np.empty((target.size, 0), dtype=np.complex128)
    residual = target.copy()
    coefficients_normalized = np.empty(0, dtype=np.complex128)
    for _ in range(min(maximum_members, candidate_count)):
        correlations = np.abs(normalized.conj().T @ residual)
        correlations[selected] = -1.0
        candidate = int(np.argmax(correlations))
        selected.append(candidate)
        basis = np.column_stack(
            (basis, normalized[:, candidate])
        )
        gram_selected = gram[np.ix_(selected, selected)]
        gram_selected += ridge * np.eye(len(selected))
        correlation_vector = correlation[selected]
        try:
            cholesky = np.linalg.cholesky(gram_selected)
        except np.linalg.LinAlgError:
            selected.pop()
            break
        weights_forward = np.linalg.solve(
            cholesky,
            correlation_vector,
        )
        coefficients_normalized = np.linalg.solve(
            cholesky.conj().T,
            weights_forward,
        )
        residual = target - basis @ coefficients_normalized
        residual_power = float(np.vdot(residual, residual).real) + ridge * (
            float(np.vdot(coefficients_normalized, coefficients_normalized).real)
        )
        if residual_power > residual_history[-1] + minimum_improvement:
            # Ridge makes tiny positive steps possible; only reject real
            # regressions, otherwise stop on no further gain.
            if residual_power - residual_history[-1] > 1e-9 * target_power:
                selected.pop()
                break
        residual_history.append(residual_power)
    original_indices = np.flatnonzero(usable)
    selected_original = [int(original_indices[index]) for index in selected]
    coefficients = np.zeros(design.shape[1], dtype=np.complex128)
    for position, column_index in enumerate(selected):
        original_index = selected_original[position]
        coefficients[original_index] = (
            coefficients_normalized[position]
            / column_norms[original_index]
        )
    return selected_original, coefficients, residual_history


def fit_gmp_residual_members(
    desired_x: np.ndarray,
    *,
    spline_model: SparseSplineMemoryDPD,
    pa,
    gain: complex,
    grid: GmpDictionaryGrid,
    member_budgets: tuple[int, ...],
    ridge_values: tuple[float, ...],
    warmup: int,
    consensus_secondary: tuple[object, complex] | None = None,
) -> dict:
    """Stage 1 of composite fitting: select and fit residual members.

    The residual drive estimate is the linearized inverse of the frozen
    cascade residual: ``delta_u = (g*x - P(u_spline)) / gain``.  When
    ``consensus_secondary`` supplies a second frozen evaluator ``(pa_b,
    gain_b)``, the target becomes the *consensus residual*
    ``(delta_u_a + delta_u_b) / 2``: for shared coefficients across two
    stacked least-squares problems with the same dictionary, minimizing
    ``||Phi c - d_a||^2 + ||Phi c - d_b||^2`` is exactly equivalent to
    minimizing ``2 ||Phi c - (d_a + d_b)/2||^2``, so a single consensus
    target suppresses improvements that exploit one surrogate's
    idiosyncratic approximation error.

    Member selection (OMP) and coefficient fitting use the same train
    block; scoring across (budget, ridge) candidates is the caller's job
    on a disjoint train block.
    """

    from .direct_learning import nmse_db

    gain = complex(gain)
    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("gain must be finite and non-zero")
    desired_x = _complex_vector(desired_x, name="desired_x")
    if not member_budgets or not ridge_values:
        raise ValueError("budget and ridge grids must not be empty")
    warmup = int(warmup)
    if warmup < 0 or desired_x.size <= warmup:
        raise ValueError("warmup must leave scored samples")

    spline_drive = spline_model.predict(desired_x)
    ideal = gain * desired_x

    def _residual_target(evaluator, evaluator_gain: complex) -> np.ndarray:
        cascade = np.asarray(
            evaluator.predict(spline_drive), dtype=np.complex128
        )
        return (evaluator_gain * desired_x - cascade) / evaluator_gain

    delta_target = _residual_target(pa, gain)
    if consensus_secondary is not None:
        secondary_pa, secondary_gain = consensus_secondary
        secondary_gain = complex(secondary_gain)
        if not np.isfinite(secondary_gain) or abs(secondary_gain) == 0.0:
            raise ValueError("secondary gain must be finite and non-zero")
        delta_target = 0.5 * (
            delta_target + _residual_target(secondary_pa, secondary_gain)
        )

    cascade = np.asarray(pa.predict(spline_drive), dtype=np.complex128)

    grid_members = grid.members
    design = gmp_dictionary_columns(desired_x, grid_members)
    no_dpd_nmse = nmse_db(cascade, ideal, warmup)

    candidates: list[dict] = []
    best: tuple[int, float] | None = None
    for budget in member_budgets:
        if budget < 1:
            raise ValueError("member budgets must be positive")
        selected, _, residual_history = orthogonal_matching_pursuit(
            design,
            delta_target,
            maximum_members=budget,
            ridge=0.0,
        )
        if not selected:
            candidates.append(
                {
                    "member_budget": budget,
                    "selected_count": 0,
                    "note": "omp produced no members",
                }
            )
            continue
        member_design = design[:, selected]
        for ridge in ridge_values:
            normalization = np.sqrt(float(desired_x.size))
            augmented = np.vstack(
                (
                    member_design / normalization,
                    np.sqrt(ridge) * np.eye(len(selected)),
                )
            )
            stacked_target = np.concatenate(
                (
                    delta_target / normalization,
                    np.zeros(len(selected), dtype=np.complex128),
                )
            )
            coefficients, *_ = np.linalg.lstsq(
                augmented,
                stacked_target,
                rcond=None,
            )
            candidates.append(
                {
                    "member_budget": budget,
                    "ridge": float(ridge),
                    "selected_count": len(selected),
                    "selected_members": [
                        {
                            "signal_delay": grid_members[index].signal_delay,
                            "envelope_delay": grid_members[index].envelope_delay,
                            "exponent": grid_members[index].exponent,
                        }
                        for index in selected
                    ],
                    "member_coefficients": coefficients,
                    "omp_residual_power_history": residual_history,
                }
            )
    return {
        "cascade_nmse_before_members_db": no_dpd_nmse,
        "delta_target_summary": {
            "rms": float(np.sqrt(np.mean(np.abs(delta_target) ** 2))),
            "peak": float(np.max(np.abs(delta_target))),
        },
        "grid_member_count": len(grid_members),
        "candidates": candidates,
    }
