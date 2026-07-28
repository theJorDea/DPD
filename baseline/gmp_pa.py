r"""Complex Generalized Memory Polynomial PA model.

The dictionary matches OpenDPD's polynomial benchmark:

.. math::

   \sum_{k=0}^{K_a-1}\sum_q a_{kq}x[n-q]|x[n-q]|^k

   + \sum_{k=1}^{K_b}\sum_{q,l}
       b_{kql}x[n-q]|x[n-q-l]|^k

   + \sum_{k=1}^{K_c}\sum_{q,l}
       c_{kql}x[n-q]|x[n-q+l]|^k.

``opendpd_exact`` retains every leading-envelope term and therefore needs up
to ``Mc`` future samples.  ``causal_leading`` stores only terms with
``lead <= q`` so every envelope delay is non-negative.  The factorized
inference path groups all envelope terms sharing ``x[n-q]`` and never
materializes the dense design matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .complexity import GMPLeadingPolicy, gmp_inference_cost
from .metrics import nmse_pooled_db
from .pa_models import segmented_steady_state_mask


def _complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class GMPConfig:
    """GMP dimensions and explicit leading-envelope causality policy."""

    ka: int
    la: int
    kb: int = 0
    lb: int = 0
    mb: int = 0
    kc: int = 0
    lc: int = 0
    mc: int = 0
    leading_policy: GMPLeadingPolicy = "causal_leading"

    def __post_init__(self) -> None:
        # Reuse the strict deployment counter validation as one source of truth.
        gmp_inference_cost(
            ka=self.ka,
            la=self.la,
            kb=self.kb,
            lb=self.lb,
            mb=self.mb,
            kc=self.kc,
            lc=self.lc,
            mc=self.mc,
            leading_policy=self.leading_policy,
        )

    @property
    def base_delay_count(self) -> int:
        return max(
            self.la,
            self.lb if self.kb else 0,
            self.lc if self.kc else 0,
        )

    @property
    def lookahead_samples(self) -> int:
        return (
            self.mc
            if self.kc and self.leading_policy == "opendpd_exact"
            else 0
        )

    @property
    def causal_warmup_samples(self) -> int:
        candidates = [self.base_delay_count - 1]
        if self.kb:
            candidates.append(self.lb - 1 + self.mb)
        if self.kc:
            candidates.append(max(self.lc - 2, 0))
        return max(candidates)

    @property
    def coefficient_count(self) -> int:
        return len(gmp_terms(self))


@dataclass(frozen=True)
class GMPTerm:
    """One complex coefficient in exact OpenDPD column order."""

    branch: Literal["aligned", "lagging", "leading"]
    exponent: int
    signal_delay: int
    envelope_delay: int
    cross_offset: int


def gmp_terms(config: GMPConfig) -> tuple[GMPTerm, ...]:
    terms: list[GMPTerm] = []
    for exponent in range(config.ka):
        for signal_delay in range(config.la):
            terms.append(
                GMPTerm(
                    "aligned",
                    exponent,
                    signal_delay,
                    signal_delay,
                    0,
                )
            )
    for exponent in range(1, config.kb + 1):
        for signal_delay in range(config.lb):
            for lag in range(1, config.mb + 1):
                terms.append(
                    GMPTerm(
                        "lagging",
                        exponent,
                        signal_delay,
                        signal_delay + lag,
                        lag,
                    )
                )
    for exponent in range(1, config.kc + 1):
        for signal_delay in range(config.lc):
            for lead in range(1, config.mc + 1):
                if (
                    config.leading_policy == "causal_leading"
                    and lead > signal_delay
                ):
                    continue
                terms.append(
                    GMPTerm(
                        "leading",
                        exponent,
                        signal_delay,
                        signal_delay - lead,
                        lead,
                    )
                )
    return tuple(terms)


def _delay(values: np.ndarray, delay: int) -> np.ndarray:
    """Return ``values[n-delay]`` with zeros outside one record."""

    array = np.asarray(values)
    result = np.zeros(array.shape, dtype=array.dtype)
    if delay == 0:
        result[:] = array
    elif 0 < delay < array.size:
        result[delay:] = array[:-delay]
    elif -array.size < delay < 0:
        result[:delay] = array[-delay:]
    return result


def gmp_design_matrix(
    signal: np.ndarray,
    config: GMPConfig,
) -> np.ndarray:
    """Build the dense calibration/reference matrix in OpenDPD column order."""

    samples = _complex_vector(signal, name="signal")
    columns: list[np.ndarray] = []
    for term in gmp_terms(config):
        delayed_signal = _delay(samples, term.signal_delay)
        if term.exponent == 0:
            columns.append(delayed_signal)
        else:
            envelope = np.abs(_delay(samples, term.envelope_delay))
            columns.append(delayed_signal * envelope**term.exponent)
    return np.column_stack(columns)


def gmp_segmented_design_matrix(
    signal: np.ndarray,
    config: GMPConfig,
    *,
    segment_length: int,
) -> np.ndarray:
    """Build a design with zero past/future context at every frame boundary."""

    samples = _complex_vector(signal, name="signal")
    if not isinstance(segment_length, (int, np.integer)) or int(segment_length) < 1:
        raise ValueError("segment_length must be a positive integer")
    segment_length = int(segment_length)
    return np.vstack(
        [
            gmp_design_matrix(
                samples[start : min(start + segment_length, samples.size)],
                config,
            )
            for start in range(0, samples.size, segment_length)
        ]
    )


@dataclass(frozen=True)
class GMPStreamingState:
    """Raw complex history for the causal reference streaming path."""

    history: np.ndarray

    def __post_init__(self) -> None:
        history = np.asarray(self.history, dtype=np.complex128)
        if history.ndim != 1 or not np.all(np.isfinite(history)):
            raise ValueError("GMP streaming history must be finite and 1-D")
        object.__setattr__(self, "history", history.copy())


@dataclass(frozen=True)
class GMPFitDiagnostics:
    sample_count: int
    feature_count: int
    ridge: float
    column_rms_minimum: float
    column_rms_maximum: float
    solver_rank: int
    scaled_augmented_condition_number: float
    training_nmse_db_full: float
    training_nmse_db_interior: float
    causal_warmup_samples: int
    future_cooldown_samples: int
    training_scored_samples_interior: int
    segment_length: int
    segment_count: int
    solver: str = "column_rms_scaled_augmented_complex_lstsq"


@dataclass(frozen=True)
class GeneralizedMemoryPolynomialPA:
    config: GMPConfig
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients)
        if coefficients.shape != (self.config.coefficient_count,):
            raise ValueError(
                "coefficients must have shape "
                f"({self.config.coefficient_count},)"
            )
        coefficients = coefficients.astype(np.complex128, copy=False)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("GMP coefficients contain non-finite values")
        object.__setattr__(self, "coefficients", coefficients.copy())

    @property
    def stored_complex_coefficients(self) -> int:
        return int(self.coefficients.size)

    @property
    def stored_real_coefficients(self) -> int:
        return 2 * self.stored_complex_coefficients

    @property
    def operation_count(self):
        return gmp_inference_cost(
            ka=self.config.ka,
            la=self.config.la,
            kb=self.config.kb,
            lb=self.config.lb,
            mb=self.config.mb,
            kc=self.config.kc,
            lc=self.config.lc,
            mc=self.config.mc,
            leading_policy=self.config.leading_policy,
        )

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Factorized vector inference without constructing ``Phi``."""

        original = np.asarray(signal)
        original_shape = original.shape
        samples = _complex_vector(original.reshape(-1), name="signal")
        maximum_exponent = max(
            self.config.ka - 1,
            self.config.kb,
            self.config.kc,
        )
        current_powers: dict[int, np.ndarray] = {}
        if maximum_exponent:
            # Form q=I²+Q² once so the vector implementation follows the
            # arithmetic schedule reported by gmp_inference_cost.  In
            # particular, the linear-only model does not compute a magnitude.
            power = samples.real * samples.real + samples.imag * samples.imag
            current_powers[1] = np.sqrt(power)
            if maximum_exponent >= 2:
                current_powers[2] = power
            for exponent in range(3, maximum_exponent + 1):
                current_powers[exponent] = (
                    current_powers[exponent - 2] * current_powers[2]
                )
        delayed_signal_cache: dict[int, np.ndarray] = {}
        delayed_power_cache: dict[tuple[int, int], np.ndarray] = {}

        def delayed_signal(delay: int) -> np.ndarray:
            if delay not in delayed_signal_cache:
                delayed_signal_cache[delay] = _delay(samples, delay)
            return delayed_signal_cache[delay]

        def delayed_power(delay: int, exponent: int) -> np.ndarray:
            key = (delay, exponent)
            if key not in delayed_power_cache:
                delayed_power_cache[key] = _delay(
                    current_powers[exponent],
                    delay,
                )
            return delayed_power_cache[key]

        weights = [
            np.zeros(samples.shape, dtype=np.complex128)
            for _ in range(self.config.base_delay_count)
        ]
        for coefficient, term in zip(
            self.coefficients,
            gmp_terms(self.config),
            strict=True,
        ):
            if term.exponent == 0:
                weights[term.signal_delay] += coefficient
            else:
                weights[term.signal_delay] += (
                    coefficient
                    * delayed_power(term.envelope_delay, term.exponent)
                )
        prediction = np.zeros(samples.shape, dtype=np.complex128)
        for signal_delay, weight in enumerate(weights):
            prediction += delayed_signal(signal_delay) * weight
        target_dtype = (
            np.complex64 if original.dtype == np.complex64 else np.complex128
        )
        return prediction.astype(target_dtype, copy=False).reshape(original_shape)

    __call__ = predict

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        original = np.asarray(signal)
        if original.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if not isinstance(segment_length, (int, np.integer)) or int(segment_length) < 1:
            raise ValueError("segment_length must be a positive integer")
        segment_length = int(segment_length)
        result = np.empty(
            original.shape,
            dtype=(
                np.complex64
                if original.dtype == np.complex64
                else np.complex128
            ),
        )
        for start in range(0, original.size, segment_length):
            stop = min(start + segment_length, original.size)
            result[start:stop] = self.predict(original[start:stop])
        return result

    def predict_streaming_chunk(
        self,
        signal: np.ndarray,
        state: GMPStreamingState | None = None,
    ) -> tuple[np.ndarray, GMPStreamingState]:
        """Reference causal chunking with carried raw history.

        This proves chunk equivalence.  It recomputes envelope powers over the
        stored raw history and is not the optimized state schedule counted by
        :func:`gmp_inference_cost`.
        """

        if self.config.leading_policy != "causal_leading":
            raise ValueError("exact leading mode requires future lookahead")
        samples = _complex_vector(signal, name="signal")
        history = (
            np.asarray([], dtype=np.complex128)
            if state is None
            else state.history
        )
        required = self.config.causal_warmup_samples
        if history.size > required:
            raise ValueError("streaming state contains excess history")
        combined = np.concatenate((history, samples))
        prediction = self.predict(combined)[history.size:]
        next_history = (
            combined[-required:].copy()
            if required > 0
            else np.asarray([], dtype=np.complex128)
        )
        return prediction, GMPStreamingState(next_history)

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray("complex_generalized_memory_polynomial_pa"),
            ka=np.asarray(self.config.ka, dtype=np.int64),
            la=np.asarray(self.config.la, dtype=np.int64),
            kb=np.asarray(self.config.kb, dtype=np.int64),
            lb=np.asarray(self.config.lb, dtype=np.int64),
            mb=np.asarray(self.config.mb, dtype=np.int64),
            kc=np.asarray(self.config.kc, dtype=np.int64),
            lc=np.asarray(self.config.lc, dtype=np.int64),
            mc=np.asarray(self.config.mc, dtype=np.int64),
            leading_policy=np.asarray(self.config.leading_policy),
            coefficients=self.coefficients,
            coefficient_order=np.asarray(
                "aligned(k,q),lagging(k,q,lag),leading(k,q,lead)"
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GeneralizedMemoryPolynomialPA":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported GMP model schema")
            if str(data["model_type"]) != "complex_generalized_memory_polynomial_pa":
                raise ValueError("unexpected GMP model type")
            config = GMPConfig(
                ka=int(data["ka"]),
                la=int(data["la"]),
                kb=int(data["kb"]),
                lb=int(data["lb"]),
                mb=int(data["mb"]),
                kc=int(data["kc"]),
                lc=int(data["lc"]),
                mc=int(data["mc"]),
                leading_policy=str(data["leading_policy"]),
            )
            return cls(config, data["coefficients"])


def _interior_mask(
    sample_count: int,
    *,
    segment_length: int,
    warmup: int,
    cooldown: int,
) -> np.ndarray:
    mask = segmented_steady_state_mask(
        sample_count,
        segment_length=segment_length,
        warmup_samples=warmup,
    )
    if cooldown:
        for start in range(0, sample_count, segment_length):
            stop = min(start + segment_length, sample_count)
            mask[max(start, stop - cooldown):stop] = False
    if not np.any(mask):
        raise ValueError("GMP warmup/cooldown consumes every training sample")
    return mask


def fit_gmp_pa(
    pa_input: np.ndarray,
    measured_pa_output: np.ndarray,
    *,
    config: GMPConfig,
    ridge: float = 1e-8,
    segment_length: int,
    coefficient_dtype: np.dtype = np.complex128,
) -> tuple[GeneralizedMemoryPolynomialPA, GMPFitDiagnostics]:
    """Fit complex GMP coefficients with column-RMS-scaled ridge least squares."""

    samples = _complex_vector(pa_input, name="pa_input")
    target = _complex_vector(
        measured_pa_output,
        name="measured_pa_output",
    )
    if samples.shape != target.shape:
        raise ValueError("PA input and measured output must have equal length")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if not isinstance(segment_length, (int, np.integer)) or int(segment_length) < 1:
        raise ValueError("segment_length must be a positive integer")
    segment_length = int(segment_length)
    if (
        config.causal_warmup_samples >= segment_length
        or config.lookahead_samples >= segment_length
    ):
        raise ValueError("GMP memory/lookahead must be shorter than each frame")

    design = gmp_segmented_design_matrix(
        samples,
        config,
        segment_length=segment_length,
    )
    if design.shape[1] >= design.shape[0]:
        raise ValueError("GMP least-squares system must be overdetermined")
    column_rms = np.sqrt(np.mean(np.abs(design) ** 2, axis=0))
    if np.any(~np.isfinite(column_rms)) or np.any(column_rms <= 0.0):
        raise ValueError("GMP design contains invalid or all-zero columns")
    scaled_design = design / column_rms
    normalization = np.sqrt(float(samples.size))
    solve_design = scaled_design / normalization
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
    coefficients = (
        scaled_coefficients / column_rms
    ).astype(coefficient_dtype, copy=False)
    model = GeneralizedMemoryPolynomialPA(config, coefficients)
    prediction = model.predict_segments(samples, segment_length)
    interior = _interior_mask(
        samples.size,
        segment_length=segment_length,
        warmup=config.causal_warmup_samples,
        cooldown=config.lookahead_samples,
    )
    if singular_values.size and singular_values[-1] > 0.0:
        condition = float(singular_values[0] / singular_values[-1])
    else:
        condition = float("inf")
    diagnostics = GMPFitDiagnostics(
        sample_count=int(samples.size),
        feature_count=config.coefficient_count,
        ridge=float(ridge),
        column_rms_minimum=float(np.min(column_rms)),
        column_rms_maximum=float(np.max(column_rms)),
        solver_rank=int(rank),
        scaled_augmented_condition_number=condition,
        training_nmse_db_full=nmse_pooled_db(prediction, target),
        training_nmse_db_interior=nmse_pooled_db(
            prediction[interior],
            target[interior],
        ),
        causal_warmup_samples=config.causal_warmup_samples,
        future_cooldown_samples=config.lookahead_samples,
        training_scored_samples_interior=int(np.count_nonzero(interior)),
        segment_length=segment_length,
        segment_count=int(np.ceil(samples.size / segment_length)),
    )
    return model, diagnostics
