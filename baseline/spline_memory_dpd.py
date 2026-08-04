r"""Causal phase-equivariant sparse spline-memory DPD.

The deployed model is

.. math::

   z[n] = \sum_b x[n-m_b] C_b(|x[n-d_b]|),

where every ``C_b`` is a complex linear spline over one shared amplitude-knot
grid.  Only two adjacent complex control points per branch are active for one
sample.  Signal and envelope delays are non-negative, so the implementation
has no future look-ahead.

Three inference semantics are deliberately separate:

``predict``
    One independent record, with zero history before sample zero.
``predict_chunk``
    Continuous streaming, with an explicit state returned to the caller.
``predict_segments``
    Independent fixed-length records, resetting history at every boundary.

Calibration is one joint complex augmented least-squares solve.  I and Q are
not fitted as unrelated models, and the architecture obeys
``D(x*exp(1j*phi)) == D(x)*exp(1j*phi)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from .complex_spline_dpd import (
    KnotStrategy,
    _strict_knots,
    make_knots,
    spline_basis,
)
from .complexity import (
    ComplexMultiplyConvention,
    OperationCount,
    complex_multiply_cost,
)


def _as_complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@dataclass(frozen=True, order=True)
class SplineMemoryBranch:
    """One ``x[n-m] * C(|x[n-d]|)`` branch."""

    signal_delay: int
    envelope_delay: int

    def __post_init__(self) -> None:
        for name, value in (
            ("signal_delay", self.signal_delay),
            ("envelope_delay", self.envelope_delay),
        ):
            if not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))


BranchLike = SplineMemoryBranch | tuple[int, int]


def _normalize_branches(
    branches: Iterable[BranchLike],
) -> tuple[SplineMemoryBranch, ...]:
    result: list[SplineMemoryBranch] = []
    for branch in branches:
        if isinstance(branch, SplineMemoryBranch):
            normalized = branch
        else:
            try:
                signal_delay, envelope_delay = branch
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "each branch must be SplineMemoryBranch or an (m, d) pair"
                ) from error
            normalized = SplineMemoryBranch(signal_delay, envelope_delay)
        result.append(normalized)
    if not result:
        raise ValueError("at least one spline-memory branch is required")
    if len(set(result)) != len(result):
        raise ValueError("duplicate (signal_delay, envelope_delay) branches")
    return tuple(result)


def _causal_delay(signal: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros(signal.shape, dtype=np.complex128)
    if delay == 0:
        result[:] = signal
    elif delay < signal.size:
        result[delay:] = signal[:-delay]
    return result


def _local_spline_coordinates_from_envelope(
    envelope: np.ndarray,
    knots: np.ndarray,
    reciprocal_knot_widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local coordinates for one already selected envelope delay.

    This is the optimized inference-path counterpart of
    :func:`baseline.complex_spline_dpd.local_spline_coordinates`.  The caller
    supplies one reciprocal per knot interval, so interpolation uses a
    multiply rather than a division for every sample.  Model construction
    validates the knots, and ``SparseSplineMemoryDPD`` owns the read-only
    reciprocal table; this private helper therefore avoids repeating those
    deployment-table checks in every streaming chunk.
    """

    radii = np.abs(envelope)
    clipped = np.clip(radii, knots[0], knots[-1])
    left = np.searchsorted(knots, clipped, side="right") - 1
    left = np.clip(left, 0, knots.size - 2)
    weight = (clipped - knots[left]) * reciprocal_knot_widths[left]
    return left.astype(np.int64, copy=False), weight


def spline_memory_design_matrix(
    signal: np.ndarray,
    knots: np.ndarray,
    branches: Iterable[BranchLike],
) -> np.ndarray:
    """Return the joint complex calibration dictionary.

    Column ordering is branch-major and then knot-major:

    ``Phi[n, b*K+k] = x[n-m_b] * B_k(|x[n-d_b]|)``.

    Every branch contributes at most two nonzero basis terms for a sample.
    Delayed samples before the start of the record are exactly zero.
    """

    samples = _as_complex_vector(signal, name="signal")
    knot_array = _strict_knots(knots)
    branch_tuple = _normalize_branches(branches)
    feature_count = len(branch_tuple) * knot_array.size
    design = np.empty((samples.size, feature_count), dtype=np.complex128)
    for branch_index, branch in enumerate(branch_tuple):
        signal_lag = _causal_delay(samples, branch.signal_delay)
        envelope_lag = _causal_delay(samples, branch.envelope_delay)
        basis = spline_basis(np.abs(envelope_lag), knot_array)
        start = branch_index * knot_array.size
        design[:, start:start + knot_array.size] = signal_lag[:, None] * basis
    return design


@dataclass(frozen=True)
class SplineMemoryState:
    """Input history carried between continuous streaming chunks.

    ``history`` is ordered oldest to newest and contains exactly
    ``model.maximum_delay`` samples.
    """

    history: np.ndarray

    def __post_init__(self) -> None:
        history = np.asarray(self.history)
        if history.ndim != 1:
            raise ValueError("streaming history must be one-dimensional")
        history = np.asarray(history, dtype=np.complex128)
        if not np.all(np.isfinite(history)):
            raise ValueError("streaming history contains NaN or infinite values")
        object.__setattr__(self, "history", history.copy())


@dataclass(frozen=True)
class SplineMemoryFitDiagnostics:
    """Numerical and boundary diagnostics for one joint complex fit."""

    sample_count: int
    branch_count: int
    knot_count: int
    feature_count: int
    ridge: float
    gram_condition_number: float
    augmented_design_condition_number: float
    solver_rank: int
    solver: str
    causal_warmup_samples: int
    training_mse_full: float
    training_relative_error_power_full: float
    training_nmse_db_full: float
    training_mse_after_warmup: float
    training_relative_error_power_after_warmup: float
    training_nmse_db_after_warmup: float
    maximum_calibration_radius: float


@dataclass(frozen=True)
class SparseSplineMemoryDPD:
    """Immutable shared-knot sparse spline-memory coefficient set."""

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
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients contain NaN or infinite values")
        # ``frozen=True`` prevents attribute rebinding but not in-place ndarray
        # mutation.  The reciprocal-width cache is derived from this table, so
        # make the copied authoritative knots genuinely immutable as well.
        knots.setflags(write=False)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "coefficients", coefficients.copy())

    @cached_property
    def _reciprocal_knot_widths(self) -> np.ndarray:
        """Read-only reciprocal interval widths, outside sample-rate work.

        The derived table is deliberately absent from the constructor and NPZ
        schema.  It is created once per loaded/constructed model, cached in the
        instance, and recomputed from the authoritative knot table after a
        save/load round trip.
        """

        reciprocal = np.reciprocal(np.diff(self.knots))
        reciprocal.setflags(write=False)
        return reciprocal

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
    def stored_complex_coefficients(self) -> int:
        return int(self.coefficients.size)

    @property
    def stored_real_coefficients(self) -> int:
        return 2 * self.stored_complex_coefficients

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "phase_equivariant_sparse_spline_memory_dpd",
            "equation": "sum_b x[n-m_b] * C_b(abs(x[n-d_b]))",
            "branches": [
                {
                    "signal_delay": branch.signal_delay,
                    "envelope_delay": branch.envelope_delay,
                }
                for branch in self.branches
            ],
            "shared_amplitude_knots": True,
            "knot_count": self.knot_count,
            "local_active_control_points_per_branch": 2,
            "causal_padding": "zeros_before_record_or_reset_segment_start",
            "continuous_streaming": "use predict_chunk and carry returned state",
            "independent_segments": "predict_segments resets every boundary",
            "partial_final_segment": "evaluated_without_end_padding",
            "maximum_delay": self.maximum_delay,
        }

    def initial_state(self) -> SplineMemoryState:
        """Return explicit zero history for a new continuous stream."""

        return SplineMemoryState(
            np.zeros(self.maximum_delay, dtype=np.complex128)
        )

    def _validate_state(self, state: SplineMemoryState) -> np.ndarray:
        if not isinstance(state, SplineMemoryState):
            raise TypeError("state must be a SplineMemoryState")
        if state.history.size != self.maximum_delay:
            raise ValueError(
                "state history length must equal model.maximum_delay"
            )
        return state.history

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: SplineMemoryState,
    ) -> tuple[np.ndarray, SplineMemoryState]:
        """Evaluate a chunk and return the state needed by the next chunk."""

        original = np.asarray(signal)
        samples = _as_complex_vector(original, name="signal")
        history = self._validate_state(state)
        delay_line = np.concatenate((history, samples))
        history_length = self.maximum_delay
        output = np.zeros(samples.size, dtype=np.complex128)

        # Several branches may use the same delayed envelope.  Addressing the
        # spline separately inside the branch loop would repeat magnitude,
        # interval search, and interpolation-weight work that operation_count
        # explicitly treats as shared.  Cache one vectorized coordinate pair
        # per unique envelope delay for this chunk instead.
        coordinates_by_envelope_delay: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}
        reciprocal_knot_widths = self._reciprocal_knot_widths
        for envelope_delay in dict.fromkeys(
            branch.envelope_delay for branch in self.branches
        ):
            envelope_start = history_length - envelope_delay
            envelope_lag = delay_line[
                envelope_start:envelope_start + samples.size
            ]
            coordinates_by_envelope_delay[envelope_delay] = (
                _local_spline_coordinates_from_envelope(
                    envelope_lag,
                    self.knots,
                    reciprocal_knot_widths,
                )
            )

        for branch_index, branch in enumerate(self.branches):
            signal_start = history_length - branch.signal_delay
            signal_lag = delay_line[
                signal_start:signal_start + samples.size
            ]
            left, weight = coordinates_by_envelope_delay[
                branch.envelope_delay
            ]
            branch_coefficients = self.coefficients[branch_index]
            correction = branch_coefficients[left] + weight * (
                branch_coefficients[left + 1] - branch_coefficients[left]
            )
            output += signal_lag * correction

        if history_length:
            next_history = delay_line[-history_length:]
        else:
            next_history = np.empty(0, dtype=np.complex128)
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        return (
            output.astype(target_dtype, copy=False),
            SplineMemoryState(next_history),
        )

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Evaluate one independent record with zero initial history."""

        output, _ = self.predict_chunk(signal, self.initial_state())
        return output

    __call__ = predict

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        """Evaluate independent segments, resetting history at every boundary."""

        original = np.asarray(signal)
        if original.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if not isinstance(segment_length, (int, np.integer)):
            raise TypeError("segment_length must be an integer")
        segment_length = int(segment_length)
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        samples = _as_complex_vector(original, name="signal")
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        output = np.empty(samples.size, dtype=target_dtype)
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
        """Return an auditable per-sample arithmetic/storage count.

        This is the arithmetic count of the optimized vectorized/reference
        datapath implemented by :meth:`predict_chunk`; it is not a Python
        wall-clock or target-hardware timing measurement.  Host timing is
        benchmarked separately because array dispatch, allocation, memory
        hierarchy, and chunk size are not represented by this count.

        Envelope magnitude, knot address, and interpolation weight are shared
        by branches having the same ``envelope_delay``.  Delay-line control and
        physical cache/bus behavior remain implementation-dependent and are
        called out in ``notes``.
        """

        if indexing not in {"binary", "uniform"}:
            raise ValueError("indexing must be binary or uniform")
        complex_mult, complex_add = complex_multiply_cost(convention)
        envelope_groups = len(
            {branch.envelope_delay for branch in self.branches}
        )
        unique_input_delays = len(
            {
                delay
                for branch in self.branches
                for delay in (branch.signal_delay, branch.envelope_delay)
            }
        )
        comparisons_per_group = (
            int(math.ceil(math.log2(self.knot_count)))
            if indexing == "binary"
            else 2
        )

        # Per unique envelope: |x|, sqrt, and t=(r-r0)*inv_width.
        real_multiplications = 3 * envelope_groups
        real_additions = 2 * envelope_groups
        nonlinear_operations = envelope_groups

        # Per branch: complex coefficient interpolation and x_lag*C.
        real_multiplications += self.branch_count * (2 + complex_mult)
        real_additions += self.branch_count * (4 + complex_add)
        real_additions += 2 * (self.branch_count - 1)

        return OperationCount(
            real_multiplications=real_multiplications,
            real_additions=real_additions,
            nonlinear_operations=nonlinear_operations,
            comparisons=envelope_groups * comparisons_per_group,
            lookups=2 * self.branch_count,
            real_memory_reads=(
                4 * self.branch_count + 2 * unique_input_delays
            ),
            real_memory_writes=2 if self.maximum_delay else 0,
            stored_real_coefficients=self.stored_real_coefficients,
            stored_real_constants=2 * self.knot_count - 1,
            state_real_values=2 * self.maximum_delay,
            notes=(
                f"complex multiply convention {convention}",
                f"{indexing} interval selection",
                (
                    f"{envelope_groups} unique envelope delays; address and "
                    "weight shared within each group"
                ),
                (
                    "reciprocal knot widths are precomputed once per model; "
                    "sample-path interpolation has no division"
                ),
                (
                    f"{unique_input_delays} unique input delays; delayed sample "
                    "reads shared when signal/envelope taps coincide"
                ),
                (
                    "delay-line state size included; physical memory traffic "
                    "and control latency are implementation-dependent"
                ),
                (
                    "optimized vectorized/reference datapath arithmetic only; "
                    "Python wall-clock and target timing are measured separately"
                ),
            ),
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray(
                "phase_equivariant_sparse_spline_memory_dpd"
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
            causal_padding=np.asarray(
                "zeros_before_record_or_reset_segment_start"
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SparseSplineMemoryDPD":
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["schema_version"])
            if version != 1:
                raise ValueError(
                    f"unsupported spline-memory schema version: {version}"
                )
            model_type = str(data["model_type"])
            if model_type != "phase_equivariant_sparse_spline_memory_dpd":
                raise ValueError(f"unexpected model type: {model_type}")
            signal_delays = data["signal_delays"]
            envelope_delays = data["envelope_delays"]
            if signal_delays.shape != envelope_delays.shape:
                raise ValueError("saved branch-delay arrays have different shapes")
            branches = tuple(
                SplineMemoryBranch(int(signal_delay), int(envelope_delay))
                for signal_delay, envelope_delay in zip(
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


def _error_statistics(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float]:
    error_power = float(np.mean(np.abs(estimate - reference) ** 2))
    reference_power = float(np.mean(np.abs(reference) ** 2))
    if reference_power <= 0.0:
        raise ValueError("target must have positive energy")
    relative = error_power / reference_power
    with np.errstate(divide="ignore"):
        nmse_db = float(10.0 * np.log10(relative))
    return error_power, relative, nmse_db


def fit_sparse_spline_memory_dpd(
    calibration_input: np.ndarray,
    target: np.ndarray,
    *,
    branches: Iterable[BranchLike],
    knot_count: int | None = None,
    knot_strategy: KnotStrategy = "uniform_amplitude",
    knots: np.ndarray | None = None,
    ridge: float = 1e-8,
    compression_power: float = 2.0,
    coefficient_dtype: np.dtype = np.complex128,
) -> tuple[SparseSplineMemoryDPD, SplineMemoryFitDiagnostics]:
    """Jointly fit every complex spline-memory branch by augmented ridge LS.

    The objective is

    ``mean(abs(Phi @ c - target)**2) + ridge*sum(abs(c)**2)``.
    """

    values = _as_complex_vector(
        calibration_input,
        name="calibration_input",
    )
    desired = _as_complex_vector(target, name="target")
    if values.shape != desired.shape:
        raise ValueError("calibration_input and target must have equal length")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    coefficient_dtype = np.dtype(coefficient_dtype)
    if not np.issubdtype(coefficient_dtype, np.complexfloating):
        raise ValueError("coefficient_dtype must be a complex floating dtype")
    branch_tuple = _normalize_branches(branches)
    maximum_delay = max(
        max(branch.signal_delay, branch.envelope_delay)
        for branch in branch_tuple
    )
    if maximum_delay >= values.size:
        raise ValueError(
            "maximum branch delay must be shorter than the calibration record"
        )

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

    design = spline_memory_design_matrix(
        values,
        knot_array,
        branch_tuple,
    )
    sample_count = values.size
    normalization = np.sqrt(float(sample_count))
    normalized_design = design / normalization
    normalized_target = desired / normalization
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
    if np.isfinite(augmented_condition):
        max_float_sqrt = np.sqrt(np.finfo(float).max)
        gram_condition = float(
            augmented_condition**2
            if augmented_condition <= max_float_sqrt
            else np.inf
        )
    else:
        gram_condition = float(np.inf)
    flat_coefficients, _, rank, _ = np.linalg.lstsq(
        augmented_design,
        augmented_target,
        rcond=None,
    )
    coefficients = flat_coefficients.reshape(
        len(branch_tuple),
        knot_array.size,
    ).astype(coefficient_dtype, copy=False)
    model = SparseSplineMemoryDPD(
        knots=knot_array,
        branches=branch_tuple,
        coefficients=coefficients,
        knot_strategy=strategy_label,
    )
    prediction = model.predict(values).astype(np.complex128, copy=False)
    mse_full, relative_full, nmse_full = _error_statistics(
        prediction,
        desired,
    )
    warmup = model.maximum_delay
    mse_steady, relative_steady, nmse_steady = _error_statistics(
        prediction[warmup:],
        desired[warmup:],
    )
    diagnostics = SplineMemoryFitDiagnostics(
        sample_count=sample_count,
        branch_count=model.branch_count,
        knot_count=model.knot_count,
        feature_count=feature_count,
        ridge=float(ridge),
        gram_condition_number=gram_condition,
        augmented_design_condition_number=augmented_condition,
        solver_rank=int(rank),
        solver="augmented_complex_lstsq",
        causal_warmup_samples=warmup,
        training_mse_full=mse_full,
        training_relative_error_power_full=relative_full,
        training_nmse_db_full=nmse_full,
        training_mse_after_warmup=mse_steady,
        training_relative_error_power_after_warmup=relative_steady,
        training_nmse_db_after_warmup=nmse_steady,
        maximum_calibration_radius=float(np.max(np.abs(values))),
    )
    return model, diagnostics


def fit_ila_sparse_spline_memory_dpd(
    known_pa_input: np.ndarray,
    measured_pa_output: np.ndarray,
    gain: complex,
    **fit_kwargs: object,
) -> tuple[SparseSplineMemoryDPD, SplineMemoryFitDiagnostics]:
    """Fit the ILA mapping ``u=y/g -> known x``.

    This calibrates a postdistorter.  A deployment evaluation must still feed
    a desired ``x`` to the returned model and pass its output through an
    independent physical PA or explicitly labelled PA surrogate.
    """

    gain = complex(gain)
    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("gain must be finite and non-zero")
    pa_input = _as_complex_vector(
        known_pa_input,
        name="known_pa_input",
    )
    pa_output = _as_complex_vector(
        measured_pa_output,
        name="measured_pa_output",
    )
    if pa_input.shape != pa_output.shape:
        raise ValueError("known_pa_input and measured_pa_output must match")
    normalized_observation = pa_output / gain
    return fit_sparse_spline_memory_dpd(
        normalized_observation,
        pa_input,
        **fit_kwargs,
    )
