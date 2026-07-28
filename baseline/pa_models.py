r"""Deterministic complex memory-polynomial PA surrogate.

The model implemented here is

.. math::

   \hat y[n] = \sum_{d \in D}\sum_{p \in P}
       a_{d,p}\,x[n-d]\,|x[n-d]|^{p-1}.

All delays are causal and non-negative.  Samples before the beginning of an
input array are explicitly zero padded; no sample is wrapped from the end of a
record.  Splits are therefore independent, and callers must decide whether to
score or discard the first ``max(delays)`` transient samples.

This is a PA *surrogate*, not a physical-PA result.  A DPD score obtained by
cascading through this object must always be labelled surrogate-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


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


def _integer_tuple(
    values: Iterable[int],
    *,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"every {name} entry must be an integer")
        integer = int(value)
        if integer < minimum:
            raise ValueError(f"every {name} entry must be at least {minimum}")
        result.append(integer)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return tuple(result)


def memory_polynomial_design_matrix(
    signal: np.ndarray,
    *,
    orders: Iterable[int] = (1, 3, 5, 7),
    delays: Iterable[int] = (0, 1, 2),
) -> np.ndarray:
    """Build the causal complex memory-polynomial feature matrix.

    Columns use delay-major ordering: ``(delay[0], order[0])``,
    ``(delay[0], order[1])``, ..., ``(delay[-1], order[-1])``.
    A delayed sequence is formed as

    ``lagged[d:] = signal[:-d]`` and ``lagged[:d] = 0``.

    Thus every returned row corresponds to the same output time index as the
    input row, with no circular wraparound and no future samples.
    """

    samples = _as_complex_vector(signal, name="signal")
    order_tuple = _integer_tuple(orders, name="orders", minimum=1)
    delay_tuple = _integer_tuple(delays, name="delays", minimum=0)
    features = np.empty(
        (samples.size, len(order_tuple) * len(delay_tuple)),
        dtype=np.complex128,
    )

    column = 0
    for delay in delay_tuple:
        lagged = np.zeros(samples.shape, dtype=np.complex128)
        if delay == 0:
            lagged[:] = samples
        elif delay < samples.size:
            lagged[delay:] = samples[:-delay]
        amplitude = np.abs(lagged)
        for order in order_tuple:
            if order == 1:
                features[:, column] = lagged
            else:
                features[:, column] = lagged * np.power(amplitude, order - 1)
            column += 1
    return features


def memory_polynomial_segmented_design_matrix(
    signal: np.ndarray,
    *,
    segment_length: int,
    orders: Iterable[int] = (1, 3, 5, 7),
    delays: Iterable[int] = (0, 1, 2),
) -> np.ndarray:
    """Build features with zero history at every independent frame boundary."""

    samples = _as_complex_vector(signal, name="signal")
    if not isinstance(segment_length, (int, np.integer)):
        raise TypeError("segment_length must be an integer")
    segment_length = int(segment_length)
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    matrices = [
        memory_polynomial_design_matrix(
            samples[start : min(start + segment_length, samples.size)],
            orders=orders,
            delays=delays,
        )
        for start in range(0, samples.size, segment_length)
    ]
    return np.vstack(matrices)


def segmented_steady_state_mask(
    sample_count: int,
    *,
    segment_length: int,
    warmup_samples: int,
) -> np.ndarray:
    """Return samples remaining after discarding warm-up in every segment."""

    if not isinstance(sample_count, (int, np.integer)) or int(sample_count) < 1:
        raise ValueError("sample_count must be a positive integer")
    if not isinstance(segment_length, (int, np.integer)) or int(segment_length) < 1:
        raise ValueError("segment_length must be a positive integer")
    if not isinstance(warmup_samples, (int, np.integer)) or int(warmup_samples) < 0:
        raise ValueError("warmup_samples must be a non-negative integer")
    sample_count = int(sample_count)
    segment_length = int(segment_length)
    warmup_samples = int(warmup_samples)
    mask = np.ones(sample_count, dtype=bool)
    for start in range(0, sample_count, segment_length):
        stop = min(start + warmup_samples, min(start + segment_length, sample_count))
        mask[start:stop] = False
    if not np.any(mask):
        raise ValueError("per-segment warmup consumes the full record")
    return mask


@dataclass(frozen=True)
class MemoryPolynomialFitDiagnostics:
    """Auditable diagnostics for one deterministic complex ridge fit."""

    sample_count: int
    feature_count: int
    ridge: float
    gram_condition_number: float
    training_mse_full: float
    training_relative_error_power_full: float
    training_nmse_db_full: float
    causal_warmup_samples: int
    training_mse_after_warmup: float
    training_relative_error_power_after_warmup: float
    training_nmse_db_after_warmup: float
    maximum_training_input_amplitude: float
    segment_length: int | None = None
    segment_count: int = 1
    state_reset_policy: str = "once_at_record_start"
    training_scored_samples_after_warmup: int | None = None
    # Kept in addition to ``gram_condition_number`` for backwards-compatible
    # reports.  The former describes the legacy normal-equation matrix; the
    # latter describes the matrix actually passed to augmented lstsq.
    augmented_design_condition_number: float | None = None
    solver_rank: int | None = None
    solver: str = "augmented_complex_lstsq"


@dataclass(frozen=True)
class MemoryPolynomialPA:
    """Immutable coefficient set for a complex memory-polynomial PA."""

    orders: tuple[int, ...]
    delays: tuple[int, ...]
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        orders = _integer_tuple(self.orders, name="orders", minimum=1)
        delays = _integer_tuple(self.delays, name="delays", minimum=0)
        coefficients = np.asarray(self.coefficients)
        expected_shape = (len(delays), len(orders))
        if coefficients.shape != expected_shape:
            raise ValueError(
                "coefficients must have shape "
                f"(len(delays), len(orders)) == {expected_shape}"
            )
        if not np.issubdtype(coefficients.dtype, np.complexfloating):
            coefficients = coefficients.astype(np.complex128)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients contain NaN or infinite values")
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "delays", delays)
        object.__setattr__(self, "coefficients", coefficients.copy())

    @property
    def feature_count(self) -> int:
        return len(self.orders) * len(self.delays)

    @property
    def causal_warmup_samples(self) -> int:
        return max(self.delays)

    @property
    def stored_complex_coefficients(self) -> int:
        return int(self.coefficients.size)

    @property
    def stored_real_coefficients(self) -> int:
        return 2 * self.stored_complex_coefficients

    @property
    def metadata(self) -> dict[str, object]:
        """Return deployment-relevant semantics without serializing the model.

        ``predict`` treats one supplied array as one independent record.
        ``predict_segments`` applies the same zero-before-record-start rule at
        every segment boundary.  A short final segment is evaluated as-is;
        there is no circular wrap and no end padding.
        """

        return {
            "model_type": "complex_memory_polynomial_pa",
            "orders": list(self.orders),
            "delays": list(self.delays),
            "feature_order": "delay_major_then_order",
            "causal_padding": "zeros_before_record_or_segment_start",
            "segment_state_reset": "zero_state_at_each_predict_segments_boundary",
            "partial_final_segment": "evaluated_without_end_padding",
            "surrogate_scope": "behavioral_model_only",
        }

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Apply the causal surrogate with explicit zero-padding at index zero."""

        original = np.asarray(signal)
        original_shape = original.shape
        samples = _as_complex_vector(original.reshape(-1), name="signal")
        design = memory_polynomial_design_matrix(
            samples,
            orders=self.orders,
            delays=self.delays,
        )
        result = design @ self.coefficients.reshape(-1)
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        return result.astype(target_dtype, copy=False).reshape(original_shape)

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        """Evaluate independent streaming segments with a reset at each start.

        The operation is intentionally segment-aware rather than a reshape
        followed by one continuous call.  Every segment gets zero-padded
        history for its positive delays, matching how OpenDPD evaluates
        independently framed records.  If the record length is not divisible
        by ``segment_length``, the final shorter segment is processed without
        adding samples and is returned at its original length.
        """

        original = np.asarray(signal)
        if original.ndim != 1:
            raise ValueError("signal must be one-dimensional for segmented prediction")
        if not isinstance(segment_length, (int, np.integer)):
            raise TypeError("segment_length must be an integer")
        segment_length = int(segment_length)
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        samples = _as_complex_vector(original, name="signal")
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        result = np.empty(samples.size, dtype=target_dtype)
        for start in range(0, samples.size, segment_length):
            stop = min(start + segment_length, samples.size)
            result[start:stop] = self.predict(samples[start:stop])
        return result

    __call__ = predict

    def save(self, path: str | Path) -> None:
        """Save coefficients and their exact feature ordering as an NPZ file."""

        np.savez(
            Path(path),
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray("complex_memory_polynomial_pa"),
            orders=np.asarray(self.orders, dtype=np.int64),
            delays=np.asarray(self.delays, dtype=np.int64),
            coefficients=self.coefficients,
            causal_padding=np.asarray("zeros_before_record_start"),
            feature_order=np.asarray("delay_major_then_order"),
            segment_state_reset=np.asarray(
                "zero_state_at_each_predict_segments_boundary"
            ),
            partial_final_segment=np.asarray("evaluated_without_end_padding"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryPolynomialPA":
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["schema_version"])
            if version != 1:
                raise ValueError(
                    f"unsupported memory-polynomial schema version: {version}"
                )
            model_type = str(data["model_type"])
            if model_type != "complex_memory_polynomial_pa":
                raise ValueError(f"unexpected model type: {model_type}")
            return cls(
                orders=tuple(int(value) for value in data["orders"]),
                delays=tuple(int(value) for value in data["delays"]),
                coefficients=data["coefficients"],
            )


def _error_statistics(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float]:
    error_power = float(np.mean(np.abs(estimate - reference) ** 2))
    reference_power = float(np.mean(np.abs(reference) ** 2))
    if reference_power <= 0.0:
        raise ValueError("PA target must have positive energy")
    relative = error_power / reference_power
    with np.errstate(divide="ignore"):
        nmse_db = float(10.0 * np.log10(relative))
    return error_power, relative, nmse_db


def fit_memory_polynomial_pa(
    pa_input: np.ndarray,
    measured_pa_output: np.ndarray,
    *,
    orders: Iterable[int] = (1, 3, 5, 7),
    delays: Iterable[int] = (0, 1, 2),
    ridge: float = 1e-8,
    segment_length: int | None = None,
    coefficient_dtype: np.dtype = np.complex128,
) -> tuple[MemoryPolynomialPA, MemoryPolynomialFitDiagnostics]:
    """Fit a deterministic complex PA surrogate by normalized ridge regression.

    The objective is

    ``mean(abs(Phi @ a - measured_pa_output)**2) + ridge*sum(abs(a)**2)``.

    Only the supplied calibration record is used.  No random initialization,
    validation data, or test data enters the closed-form fit.
    """

    samples = _as_complex_vector(pa_input, name="pa_input")
    target = _as_complex_vector(
        measured_pa_output,
        name="measured_pa_output",
    )
    if samples.shape != target.shape:
        raise ValueError("pa_input and measured_pa_output must have equal length")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")

    order_tuple = _integer_tuple(orders, name="orders", minimum=1)
    delay_tuple = _integer_tuple(delays, name="delays", minimum=0)
    if max(delay_tuple) >= samples.size:
        raise ValueError("every fitted delay must be shorter than the record")

    if segment_length is None:
        design = memory_polynomial_design_matrix(
            samples,
            orders=order_tuple,
            delays=delay_tuple,
        )
        reset_policy = "once_at_record_start"
        segment_count = 1
    else:
        if not isinstance(segment_length, (int, np.integer)):
            raise TypeError("segment_length must be an integer or None")
        segment_length = int(segment_length)
        if segment_length <= 0:
            raise ValueError("segment_length must be positive")
        design = memory_polynomial_segmented_design_matrix(
            samples,
            segment_length=segment_length,
            orders=order_tuple,
            delays=delay_tuple,
        )
        reset_policy = "zero_history_at_every_segment_start"
        segment_count = int(np.ceil(samples.size / segment_length))
    sample_count = samples.size
    # Solve the normalized ridge objective through an augmented least-squares
    # system.  We intentionally do not form Phi^H Phi: normal equations square
    # the condition number and are unnecessarily fragile for high-order PA
    # dictionaries.
    normalization = np.sqrt(float(sample_count))
    normalized_design = design / normalization
    normalized_target = target / normalization
    if ridge > 0.0:
        augmented_design = np.vstack(
            (
                normalized_design,
                np.sqrt(float(ridge))
                * np.eye(design.shape[1], dtype=np.complex128),
            )
        )
        augmented_target = np.concatenate(
            (
                normalized_target,
                np.zeros(design.shape[1], dtype=np.complex128),
            )
        )
    else:
        augmented_design = normalized_design
        augmented_target = normalized_target
    augmented_condition_number = float(np.linalg.cond(augmented_design))
    # Preserve the historical ``gram_condition_number`` field as the
    # mathematically equivalent condition estimate (cond(A^H A)=cond(A)^2),
    # without constructing the normal-equation matrix.
    if np.isfinite(augmented_condition_number):
        max_float_sqrt = np.sqrt(np.finfo(float).max)
        condition_number = float(
            augmented_condition_number**2
            if augmented_condition_number <= max_float_sqrt
            else np.inf
        )
    else:
        condition_number = float(np.inf)
    flat_coefficients, _, solver_rank, _ = np.linalg.lstsq(
        augmented_design,
        augmented_target,
        rcond=None,
    )

    coefficients = flat_coefficients.reshape(
        len(delay_tuple),
        len(order_tuple),
    ).astype(coefficient_dtype, copy=False)
    model = MemoryPolynomialPA(order_tuple, delay_tuple, coefficients)
    prediction = (
        model.predict(samples)
        if segment_length is None
        else model.predict_segments(samples, segment_length)
    ).astype(np.complex128, copy=False)
    mse_full, relative_full, nmse_full = _error_statistics(prediction, target)

    warmup = model.causal_warmup_samples
    if warmup >= sample_count:
        raise ValueError("causal warmup consumes the full calibration record")
    if segment_length is None:
        steady_mask = np.arange(sample_count) >= warmup
    else:
        steady_mask = segmented_steady_state_mask(
            sample_count,
            segment_length=segment_length,
            warmup_samples=warmup,
        )
    mse_steady, relative_steady, nmse_steady = _error_statistics(
        prediction[steady_mask],
        target[steady_mask],
    )
    diagnostics = MemoryPolynomialFitDiagnostics(
        sample_count=sample_count,
        feature_count=model.feature_count,
        ridge=float(ridge),
        gram_condition_number=condition_number,
        training_mse_full=mse_full,
        training_relative_error_power_full=relative_full,
        training_nmse_db_full=nmse_full,
        causal_warmup_samples=warmup,
        training_mse_after_warmup=mse_steady,
        training_relative_error_power_after_warmup=relative_steady,
        training_nmse_db_after_warmup=nmse_steady,
        maximum_training_input_amplitude=float(np.max(np.abs(samples))),
        segment_length=segment_length,
        segment_count=segment_count,
        state_reset_policy=reset_policy,
        training_scored_samples_after_warmup=int(np.count_nonzero(steady_mask)),
        augmented_design_condition_number=augmented_condition_number,
        solver_rank=int(solver_rank),
        solver="augmented_complex_lstsq",
    )
    return model, diagnostics
