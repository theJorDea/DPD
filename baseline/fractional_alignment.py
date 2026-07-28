"""Frame-safe fractional-alignment transform for sensitivity experiments.

This module does not estimate a delay and does not claim that a correlation
peak is a calibrated measurement-path delay.  It only applies an explicitly
supplied, training-frozen delay as a versioned *sensitivity transform*.

The sign convention is identical to :mod:`baseline.alignment`: for delay
``d``,

``observed[n] ~= gain * reference[n - d]``.

A positive ``d`` therefore means that ``observed`` lags ``reference``.  The
transform first applies the exact integer overlap and then advances the
overlapped observed signal by the residual fractional delay.  Linear
convolution is used, never circular convolution.  Exactly half the FIR length
is discarded from both ends of both aligned signals, independently for every
frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Iterable

import numpy as np

from .alignment import overlap_for_delay


ALGORITHM_VERSION = "frame_safe_windowed_sinc_v1"
SIGN_CONVENTION = "observed[n] ~= gain * reference[n - d]"
SENSITIVITY_PURPOSE = (
    "sensitivity_analysis_only_not_automatic_measurement_path_truth"
)


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_odd_integer(value: Integral, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 3 or result % 2 == 0:
        raise ValueError(f"{name} must be an odd integer of at least three")
    return result


def _decompose_delay(delay_samples: float) -> tuple[int, float]:
    """Split a delay into an integer and a residual in ``[-0.5, 0.5)``.

    Half-sample ties are assigned to the larger integer.  This rule is
    explicit so coefficient generation is independent of Python's
    round-to-even behavior.
    """

    integer_delay = int(math.floor(delay_samples + 0.5))
    fractional_delay = float(delay_samples - integer_delay)
    # Suppress negative zero in metadata and the exact-integer fast path.
    if fractional_delay == 0.0:
        fractional_delay = 0.0
    if not -0.5 <= fractional_delay < 0.5:
        raise RuntimeError("internal delay decomposition invariant failed")
    return integer_delay, fractional_delay


@dataclass(frozen=True, slots=True)
class FractionalAlignmentConfig:
    """Immutable configuration for one frozen sensitivity transform.

    ``observed_delay_samples`` is the complete real-valued delay under
    :data:`SIGN_CONVENTION`.  The implementation separates it into an exact
    integer overlap and a residual fractional FIR shift.

    Only a Kaiser-windowed sinc is supported in version 1.  Restricting the
    choice makes the transform and its operation order easy to reproduce.
    """

    observed_delay_samples: float
    tap_count: int = 65
    kaiser_beta: float = 8.6
    algorithm_version: str = field(
        default=ALGORITHM_VERSION,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        delay = _finite_real(
            self.observed_delay_samples,
            "observed_delay_samples",
        )
        tap_count = _positive_odd_integer(self.tap_count, "tap_count")
        beta = _finite_real(self.kaiser_beta, "kaiser_beta")
        if beta < 0.0:
            raise ValueError("kaiser_beta must be non-negative")
        if delay == 0.0:
            delay = 0.0
        if beta == 0.0:
            beta = 0.0
        object.__setattr__(self, "observed_delay_samples", delay)
        object.__setattr__(self, "tap_count", tap_count)
        object.__setattr__(self, "kaiser_beta", beta)


def _coefficient_digest(coefficients: tuple[float, ...]) -> str:
    # A fixed byte order makes the digest independent of host endianness.
    canonical = np.asarray(coefficients, dtype=">f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _protocol_payload(
    *,
    config: FractionalAlignmentConfig,
    integer_delay_samples: int,
    fractional_delay_samples: float,
    guard_samples: int,
    coefficients: tuple[float, ...],
    coefficient_sha256: str,
) -> dict[str, object]:
    """Return the exact, canonical payload covered by ``protocol_sha256``."""

    return {
        "schema_version": 1,
        "algorithm_version": config.algorithm_version,
        "purpose": SENSITIVITY_PURPOSE,
        "sign_convention": SIGN_CONVENTION,
        "observed_delay_samples_float64_hex": (
            config.observed_delay_samples.hex()
        ),
        "integer_delay_samples": integer_delay_samples,
        "fractional_delay_samples_float64_hex": fractional_delay_samples.hex(),
        "tap_count": config.tap_count,
        "window": "kaiser",
        "kaiser_beta_float64_hex": config.kaiser_beta.hex(),
        "normalization": "unit_dc_gain",
        "convolution": "linear_centered_same_no_circular_wrap",
        "frame_policy": "independent_per_frame",
        "guard_policy": "exact_symmetric_half_fir_after_integer_overlap",
        "guard_samples_each_side": guard_samples,
        "coefficient_dtype": "float64",
        "coefficient_float64_hex": tuple(
            coefficient.hex() for coefficient in coefficients
        ),
        "coefficient_sha256": coefficient_sha256,
    }


def _payload_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenFractionalAlignment:
    """Immutable FIR coefficients and hashes for a frozen transform."""

    config: FractionalAlignmentConfig
    integer_delay_samples: int
    fractional_delay_samples: float
    guard_samples: int
    coefficients: tuple[float, ...]
    coefficient_sha256: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, FractionalAlignmentConfig):
            raise TypeError("config must be a FractionalAlignmentConfig")
        expected_integer, expected_fractional = _decompose_delay(
            self.config.observed_delay_samples
        )
        if self.integer_delay_samples != expected_integer:
            raise ValueError("integer delay is inconsistent with config")
        if self.fractional_delay_samples != expected_fractional:
            raise ValueError("fractional delay is inconsistent with config")
        expected_guard = self.config.tap_count // 2
        if self.guard_samples != expected_guard:
            raise ValueError("guard must equal half the odd FIR tap count")
        if not isinstance(self.coefficients, tuple):
            raise TypeError("coefficients must use an immutable tuple")
        if len(self.coefficients) != self.config.tap_count:
            raise ValueError("coefficient count does not match tap_count")
        if not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError("coefficients must be finite")
        expected_coefficient_sha256 = _coefficient_digest(self.coefficients)
        if self.coefficient_sha256 != expected_coefficient_sha256:
            raise ValueError("coefficient_sha256 does not match coefficients")
        payload = _protocol_payload(
            config=self.config,
            integer_delay_samples=self.integer_delay_samples,
            fractional_delay_samples=self.fractional_delay_samples,
            guard_samples=self.guard_samples,
            coefficients=self.coefficients,
            coefficient_sha256=self.coefficient_sha256,
        )
        if self.protocol_sha256 != _payload_digest(payload):
            raise ValueError("protocol_sha256 does not match frozen transform")

    def coefficient_array(self) -> np.ndarray:
        """Return a read-only copy of the real float64 FIR coefficients."""

        array = np.asarray(self.coefficients, dtype=np.float64)
        array.setflags(write=False)
        return array

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-serializable exact coefficients and versioned hashes."""

        payload = _protocol_payload(
            config=self.config,
            integer_delay_samples=self.integer_delay_samples,
            fractional_delay_samples=self.fractional_delay_samples,
            guard_samples=self.guard_samples,
            coefficients=self.coefficients,
            coefficient_sha256=self.coefficient_sha256,
        )
        # Decimal values are included for readability.  The exact hexadecimal
        # float encodings above are the values covered by the protocol hash.
        payload.update(
            {
                "observed_delay_samples": self.config.observed_delay_samples,
                "fractional_delay_samples": self.fractional_delay_samples,
                "kaiser_beta": self.config.kaiser_beta,
                "protocol_sha256": self.protocol_sha256,
            }
        )
        return payload


def freeze_fractional_alignment(
    config: FractionalAlignmentConfig,
) -> FrozenFractionalAlignment:
    """Create deterministic coefficients for an explicit, train-frozen delay."""

    if not isinstance(config, FractionalAlignmentConfig):
        raise TypeError("config must be a FractionalAlignmentConfig")

    integer_delay, fractional_delay = _decompose_delay(
        config.observed_delay_samples
    )
    half_length = config.tap_count // 2
    if fractional_delay == 0.0:
        coefficient_array = np.zeros(config.tap_count, dtype=np.float64)
        coefficient_array[half_length] = 1.0
    else:
        offsets = np.arange(-half_length, half_length + 1, dtype=np.float64)
        coefficient_array = (
            np.sinc(offsets + fractional_delay)
            * np.kaiser(config.tap_count, config.kaiser_beta)
        )
        dc_gain = float(np.sum(coefficient_array, dtype=np.float64))
        if not math.isfinite(dc_gain) or abs(dc_gain) <= np.finfo(float).tiny:
            raise ValueError("fractional-delay FIR has invalid DC gain")
        coefficient_array /= dc_gain

    coefficients = tuple(float(value) for value in coefficient_array)
    coefficient_sha256 = _coefficient_digest(coefficients)
    payload = _protocol_payload(
        config=config,
        integer_delay_samples=integer_delay,
        fractional_delay_samples=fractional_delay,
        guard_samples=half_length,
        coefficients=coefficients,
        coefficient_sha256=coefficient_sha256,
    )
    return FrozenFractionalAlignment(
        config=config,
        integer_delay_samples=integer_delay,
        fractional_delay_samples=fractional_delay,
        guard_samples=half_length,
        coefficients=coefficients,
        coefficient_sha256=coefficient_sha256,
        protocol_sha256=_payload_digest(payload),
    )


def _as_complex_frame(values: np.ndarray, name: str) -> np.ndarray:
    frame = np.asarray(values)
    if frame.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional frame")
    if frame.size == 0:
        raise ValueError(f"{name} must not be empty")
    frame = np.asarray(frame, dtype=np.complex128)
    if not np.all(np.isfinite(frame)):
        raise ValueError(f"{name} contains non-finite values")
    return frame


def symmetric_guard_crop(
    reference: np.ndarray,
    observed: np.ndarray,
    guard_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop exactly ``guard_samples`` from both ends of a same-length pair."""

    reference_frame = _as_complex_frame(reference, "reference")
    observed_frame = _as_complex_frame(observed, "observed")
    if reference_frame.shape != observed_frame.shape:
        raise ValueError("reference and observed frames must have the same shape")
    if (
        isinstance(guard_samples, (bool, np.bool_))
        or not isinstance(guard_samples, Integral)
    ):
        raise TypeError("guard_samples must be an integer")
    guard = int(guard_samples)
    if guard < 0:
        raise ValueError("guard_samples must be non-negative")
    if 2 * guard >= reference_frame.size:
        raise ValueError("symmetric guard leaves no scored frame samples")
    if guard == 0:
        return reference_frame, observed_frame
    return reference_frame[guard:-guard], observed_frame[guard:-guard]


def _centered_linear_fir(
    frame: np.ndarray,
    coefficients: tuple[float, ...],
) -> np.ndarray:
    """Apply an odd centered FIR with zero extension and no circular wrap."""

    half_length = len(coefficients) // 2
    full = np.convolve(
        frame,
        np.asarray(coefficients, dtype=np.float64),
        mode="full",
    )
    return np.asarray(
        full[half_length:half_length + frame.size],
        dtype=np.complex128,
    )


def apply_fractional_alignment_frame(
    reference: np.ndarray,
    observed: np.ndarray,
    frozen: FrozenFractionalAlignment,
) -> tuple[np.ndarray, np.ndarray]:
    """Align and valid-crop one frame without consulting any other frame.

    The exact integer overlap is applied first.  For a residual fractional
    delay ``r``, the centered FIR approximates
    ``observed_aligned[n] = observed_overlap[n + r]``.  Both the reference and
    filtered observed arrays are then cropped by exactly
    ``frozen.guard_samples`` at each end.
    """

    if not isinstance(frozen, FrozenFractionalAlignment):
        raise TypeError("frozen must be a FrozenFractionalAlignment")
    reference_frame = _as_complex_frame(reference, "reference")
    observed_frame = _as_complex_frame(observed, "observed")
    if reference_frame.shape != observed_frame.shape:
        raise ValueError("reference and observed frames must have the same shape")

    reference_overlap, observed_overlap = overlap_for_delay(
        reference_frame,
        observed_frame,
        frozen.integer_delay_samples,
    )
    observed_shifted = _centered_linear_fir(
        observed_overlap,
        frozen.coefficients,
    )
    return symmetric_guard_crop(
        reference_overlap,
        observed_shifted,
        frozen.guard_samples,
    )


def _materialize_frames(
    frames: Iterable[np.ndarray] | np.ndarray,
    name: str,
) -> tuple[np.ndarray, ...]:
    if isinstance(frames, np.ndarray):
        if frames.ndim == 1:
            return (frames,)
        if frames.ndim == 2:
            return tuple(frames[index] for index in range(frames.shape[0]))
        raise ValueError(f"{name} must be a 1-D frame or a 2-D frame array")
    try:
        result = tuple(frames)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of frames") from error
    if not result:
        raise ValueError(f"{name} must contain at least one frame")
    return result


def apply_fractional_alignment_frames(
    reference_frames: Iterable[np.ndarray] | np.ndarray,
    observed_frames: Iterable[np.ndarray] | np.ndarray,
    frozen: FrozenFractionalAlignment,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Apply one frozen transform independently to each paired frame.

    The function intentionally returns frame tuples rather than silently
    flattening them.  This prevents FIR state, integer overlap, or edge
    transients from crossing capture/frame boundaries.
    """

    references = _materialize_frames(reference_frames, "reference_frames")
    observations = _materialize_frames(observed_frames, "observed_frames")
    if len(references) != len(observations):
        raise ValueError("reference and observed frame counts must match")

    aligned_pairs = tuple(
        apply_fractional_alignment_frame(reference, observed, frozen)
        for reference, observed in zip(references, observations)
    )
    return (
        tuple(pair[0] for pair in aligned_pairs),
        tuple(pair[1] for pair in aligned_pairs),
    )


__all__ = [
    "ALGORITHM_VERSION",
    "SIGN_CONVENTION",
    "SENSITIVITY_PURPOSE",
    "FractionalAlignmentConfig",
    "FrozenFractionalAlignment",
    "freeze_fractional_alignment",
    "symmetric_guard_crop",
    "apply_fractional_alignment_frame",
    "apply_fractional_alignment_frames",
]
