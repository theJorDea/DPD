"""Bit-accurate integer primitives for forward PA model emulation.

The helpers in this module are intentionally model-agnostic.  They define the
numeric contract used by the later GMP and sparse-spline PA simulators:

* signed two's-complement codes with an explicit fractional-bit count;
* round-to-nearest-even for float quantization and integer right shifts;
* explicit saturation counters (never silent wrap);
* an integer square-root primitive with deterministic tie handling.

This is a reference model, not an RTL timing/resource model.  All arithmetic
is kept in ``int64`` after callers have checked that intermediate products fit.
The model-specific code records wider intermediate maxima and fails loudly if
an ``int64`` product would overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)


def _validate_integer(value: int, *, name: str, minimum: int | None = None) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class QuantizedReal:
    """Integer codes and the number of values clipped to the signed range."""

    codes: np.ndarray
    saturation_count: int

    def __post_init__(self) -> None:
        codes = np.asarray(self.codes, dtype=np.int64)
        if not np.all(np.isfinite(codes)):
            raise ValueError("quantized codes must be finite")
        if int(self.saturation_count) < 0:
            raise ValueError("saturation_count must be non-negative")
        object.__setattr__(self, "codes", codes.copy())
        object.__setattr__(self, "saturation_count", int(self.saturation_count))


@dataclass(frozen=True)
class QuantizedComplex:
    """Independent signed integer I/Q codes and aggregate saturation count."""

    real: np.ndarray
    imag: np.ndarray
    saturation_count: int

    def __post_init__(self) -> None:
        real = np.asarray(self.real, dtype=np.int64)
        imag = np.asarray(self.imag, dtype=np.int64)
        if real.shape != imag.shape:
            raise ValueError("real and imag code shapes must match")
        if int(self.saturation_count) < 0:
            raise ValueError("saturation_count must be non-negative")
        object.__setattr__(self, "real", real.copy())
        object.__setattr__(self, "imag", imag.copy())
        object.__setattr__(self, "saturation_count", int(self.saturation_count))


@dataclass(frozen=True)
class SaturatedCodes:
    """Codes after signed clipping and the number of clipped elements."""

    codes: np.ndarray
    saturation_count: int

    def __post_init__(self) -> None:
        codes = np.asarray(self.codes, dtype=np.int64)
        object.__setattr__(self, "codes", codes.copy())
        object.__setattr__(self, "saturation_count", int(self.saturation_count))


@dataclass(frozen=True)
class FixedPointFormat:
    """Signed two's-complement fixed-point representation.

    A code ``c`` represents ``c * 2**(-fractional_bits)``.  The asymmetric
    signed range is retained: ``[-2**(bits-1), 2**(bits-1)-1]``.
    """

    bits: int
    fractional_bits: int
    label: str = ""

    def __post_init__(self) -> None:
        bits = _validate_integer(self.bits, name="bits", minimum=2)
        fractional_bits = _validate_integer(
            self.fractional_bits,
            name="fractional_bits",
            minimum=0,
        )
        if bits > 62:
            raise ValueError("bits above 62 are not supported by int64 reference")
        if fractional_bits >= bits:
            raise ValueError("fractional_bits must be smaller than bits")
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "fractional_bits", fractional_bits)

    @property
    def scale(self) -> float:
        return float(2.0 ** (-self.fractional_bits))

    @property
    def minimum_code(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def maximum_code(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def representable_minimum(self) -> float:
        return self.minimum_code * self.scale

    @property
    def representable_maximum(self) -> float:
        return self.maximum_code * self.scale

    def quantize(self, values: Any) -> QuantizedReal:
        """Quantize real values using NumPy's ties-to-even ``rint``."""

        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{self.label or 'values'} must be finite")
        raw = np.rint(array / self.scale)
        # The format is deliberately bounded before conversion to int64.
        if np.any(raw < _INT64_MIN) or np.any(raw > _INT64_MAX):
            raise OverflowError("quantization code exceeds int64 range")
        clipped = np.clip(raw, self.minimum_code, self.maximum_code)
        count = int(np.count_nonzero(raw != clipped))
        return QuantizedReal(clipped.astype(np.int64), count)

    def quantize_complex(self, values: Any) -> QuantizedComplex:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.complexfloating):
            array = np.asarray(array, dtype=np.complex128)
        real = self.quantize(np.asarray(array.real))
        imag = self.quantize(np.asarray(array.imag))
        return QuantizedComplex(
            real=real.codes,
            imag=imag.codes,
            saturation_count=real.saturation_count + imag.saturation_count,
        )

    def dequantize(self, codes: Any) -> np.ndarray:
        array = np.asarray(codes, dtype=np.int64)
        return array.astype(np.float64) * self.scale

    def dequantize_complex(
        self,
        real_codes: Any,
        imag_codes: Any,
    ) -> np.ndarray:
        real = np.asarray(real_codes, dtype=np.int64)
        imag = np.asarray(imag_codes, dtype=np.int64)
        if real.shape != imag.shape:
            raise ValueError("real and imag code shapes must match")
        return (real.astype(np.float64) + 1j * imag.astype(np.float64)) * self.scale


def round_shift_even(values: Any, shift: int) -> np.ndarray:
    """Arithmetic right shift with round-to-nearest-even.

    ``values`` must already be an integer array whose products fit in int64.
    Negative values are rounded symmetrically rather than toward zero.
    """

    shift = _validate_integer(shift, name="shift", minimum=0)
    array = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return array.copy()
    if shift >= 63:
        raise ValueError("shift must be smaller than 63")
    divisor = 1 << shift
    absolute = np.abs(array)
    quotient = absolute // divisor
    remainder = absolute % divisor
    half = divisor // 2
    increment = (remainder > half) | (
        (remainder == half) & ((quotient & 1) == 1)
    )
    rounded = quotient + increment.astype(np.int64)
    return np.where(array < 0, -rounded, rounded).astype(np.int64)


def round_divide_even(numerator: int, denominator: int) -> int:
    """Round an integer quotient to nearest-even, including negative values."""

    numerator = _validate_integer(numerator, name="numerator")
    denominator = _validate_integer(denominator, name="denominator", minimum=1)
    sign = -1 if numerator < 0 else 1
    absolute = abs(numerator)
    quotient, remainder = divmod(absolute, denominator)
    twice = 2 * remainder
    if twice > denominator or (
        twice == denominator and quotient % 2 == 1
    ):
        quotient += 1
    return sign * quotient


def saturate_codes(values: Any, fmt: FixedPointFormat) -> SaturatedCodes:
    """Clip integer codes to ``fmt`` and report clipped elements."""

    array = np.asarray(values, dtype=np.int64)
    clipped = np.clip(array, fmt.minimum_code, fmt.maximum_code)
    return SaturatedCodes(
        codes=clipped.astype(np.int64, copy=False),
        saturation_count=int(np.count_nonzero(array != clipped)),
    )


def integer_sqrt_round_even(value: int) -> int:
    """Return round-to-nearest-even ``sqrt(value)`` for a non-negative integer."""

    value = _validate_integer(value, name="value", minimum=0)
    root = math.isqrt(value)
    lower_error = value - root * root
    upper = root + 1
    upper_error = upper * upper - value
    if upper_error < lower_error:
        return upper
    if upper_error > lower_error:
        return root
    return upper if root % 2 == 1 else root


def integer_sqrt_array(values: Any) -> np.ndarray:
    """Vector convenience wrapper around :func:`integer_sqrt_round_even`."""

    array = np.asarray(values)
    if np.any(array < 0):
        raise ValueError("integer square-root input must be non-negative")
    flat = array.astype(object, copy=False).reshape(-1)
    result = np.asarray(
        [integer_sqrt_round_even(int(item)) for item in flat],
        dtype=np.int64,
    )
    return result.reshape(array.shape)


def checked_int64_product(left: int, right: int, *, label: str = "product") -> int:
    """Multiply Python integers and fail before an int64 wrap could occur."""

    result = int(left) * int(right)
    if result < _INT64_MIN or result > _INT64_MAX:
        raise OverflowError(f"{label} exceeds signed int64 reference range")
    return result

