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

from .gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA, gmp_terms


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

    @classmethod
    def for_full_scale(
        cls,
        bits: int,
        full_scale: float,
        *,
        label: str = "",
        guard_ratio: float = 1.001,
    ) -> "FixedPointFormat":
        """Choose the finest fractional precision covering a frozen peak.

        ``full_scale`` must come from an already-frozen calibration split.  A
        small guard ratio avoids making the positive signed endpoint itself a
        normal operating value.
        """

        bits = _validate_integer(bits, name="bits", minimum=2)
        if not np.isfinite(full_scale) or full_scale <= 0.0:
            raise ValueError("full_scale must be positive and finite")
        if not np.isfinite(guard_ratio) or guard_ratio < 1.0:
            raise ValueError("guard_ratio must be finite and >= 1")
        available = float((1 << (bits - 1)) - 1)
        required = float(full_scale) * float(guard_ratio)
        fractional_bits = int(np.floor(np.log2(available / required)))
        fractional_bits = max(0, min(bits - 1, fractional_bits))
        result = cls(bits, fractional_bits, label=label)
        if result.representable_maximum + np.finfo(float).eps < required:
            raise ValueError("no representable signed format covers full_scale")
        return result

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


@dataclass(frozen=True)
class FixedPointPAConfig:
    """Shared numeric contract for fixed-point forward PA models."""

    input_format: FixedPointFormat
    coefficient_format: FixedPointFormat
    power_format: FixedPointFormat
    accumulator_bits: int = 56
    scalar_accumulator_bits: int = 56
    output_format: FixedPointFormat | None = None
    interpolation_fraction_bits: int = 16

    def __post_init__(self) -> None:
        accumulator_bits = _validate_integer(
            self.accumulator_bits,
            name="accumulator_bits",
            minimum=2,
        )
        scalar_bits = _validate_integer(
            self.scalar_accumulator_bits,
            name="scalar_accumulator_bits",
            minimum=2,
        )
        if accumulator_bits > 62 or scalar_bits > 62:
            raise ValueError("accumulator widths above 62 are unsupported")
        interpolation_bits = _validate_integer(
            self.interpolation_fraction_bits,
            name="interpolation_fraction_bits",
            minimum=1,
        )
        if interpolation_bits > 30:
            raise ValueError("interpolation_fraction_bits must be <= 30")
        output_format = self.input_format if self.output_format is None else self.output_format
        if not isinstance(output_format, FixedPointFormat):
            raise TypeError("output_format must be a FixedPointFormat")
        object.__setattr__(self, "accumulator_bits", accumulator_bits)
        object.__setattr__(self, "scalar_accumulator_bits", scalar_bits)
        object.__setattr__(self, "interpolation_fraction_bits", interpolation_bits)
        object.__setattr__(self, "output_format", output_format)

    @classmethod
    def for_activation_bits(
        cls,
        bits: int,
        *,
        accumulator_bits: int = 56,
        scalar_accumulator_bits: int = 56,
        coefficient_fraction_bits: int | None = None,
        power_bits: int = 48,
        interpolation_fraction_bits: int = 16,
    ) -> "FixedPointPAConfig":
        """Build the preregistered Q-format family for 16/14/12-bit sweeps.

        Inputs/outputs use Q1.(bits-1).  Coefficients use a deliberately wider
        physical range (approximately ``[-8,8)``), hence ``bits-4`` fractional
        bits by default.  The choice is a format contract, not a value tuned
        on the evaluation split.
        """

        bits = _validate_integer(bits, name="bits", minimum=4)
        coefficient_fraction_bits = (
            bits - 4
            if coefficient_fraction_bits is None
            else int(coefficient_fraction_bits)
        )
        input_format = FixedPointFormat(bits, bits - 1, label="input_output")
        coefficient_format = FixedPointFormat(
            bits,
            coefficient_fraction_bits,
            label="coefficient",
        )
        power_format = FixedPointFormat(
            power_bits,
            bits - 1,
            label="power",
        )
        return cls(
            input_format=input_format,
            coefficient_format=coefficient_format,
            power_format=power_format,
            accumulator_bits=accumulator_bits,
            scalar_accumulator_bits=scalar_accumulator_bits,
            interpolation_fraction_bits=interpolation_fraction_bits,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "input": {
                "bits": self.input_format.bits,
                "fractional_bits": self.input_format.fractional_bits,
            },
            "coefficient": {
                "bits": self.coefficient_format.bits,
                "fractional_bits": self.coefficient_format.fractional_bits,
            },
            "power": {
                "bits": self.power_format.bits,
                "fractional_bits": self.power_format.fractional_bits,
            },
            "output": {
                "bits": self.output_format.bits,
                "fractional_bits": self.output_format.fractional_bits,
            },
            "accumulator_bits": self.accumulator_bits,
            "scalar_accumulator_bits": self.scalar_accumulator_bits,
            "interpolation_fraction_bits": self.interpolation_fraction_bits,
            "rounding": "nearest_even",
            "overflow": "saturate_and_count",
        }


@dataclass(frozen=True)
class FixedPointPAState:
    """Quantized causal I/Q history carried between streaming chunks."""

    real_codes: np.ndarray
    imag_codes: np.ndarray

    def __post_init__(self) -> None:
        real = np.asarray(self.real_codes, dtype=np.int64)
        imag = np.asarray(self.imag_codes, dtype=np.int64)
        if real.ndim != 1 or imag.ndim != 1 or real.shape != imag.shape:
            raise ValueError("fixed-point state must contain matching 1-D I/Q arrays")
        object.__setattr__(self, "real_codes", real.copy())
        object.__setattr__(self, "imag_codes", imag.copy())

    @property
    def size(self) -> int:
        return int(self.real_codes.size)


@dataclass(frozen=True)
class FixedPointPAStats:
    """All numerical clipping/occupancy diagnostics for one prediction call."""

    sample_count: int
    input_saturations: int
    coefficient_saturations: int
    power_saturations: int
    scalar_accumulator_saturations: int
    accumulator_saturations: int
    output_saturations: int
    maximum_power_magnitude: int
    maximum_scalar_accumulator_magnitude: int
    maximum_accumulator_magnitude: int
    knot_code_collision_count: int = 0
    maximum_knot_code_shift: int = 0

    def __post_init__(self) -> None:
        for name in (
            "sample_count",
            "input_saturations",
            "coefficient_saturations",
            "power_saturations",
            "scalar_accumulator_saturations",
            "accumulator_saturations",
            "output_saturations",
            "maximum_power_magnitude",
            "maximum_scalar_accumulator_magnitude",
            "maximum_accumulator_magnitude",
            "knot_code_collision_count",
            "maximum_knot_code_shift",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in (
                "sample_count",
                "input_saturations",
                "coefficient_saturations",
                "power_saturations",
                "scalar_accumulator_saturations",
                "accumulator_saturations",
                "output_saturations",
                "maximum_power_magnitude",
                "maximum_scalar_accumulator_magnitude",
                "maximum_accumulator_magnitude",
                "knot_code_collision_count",
                "maximum_knot_code_shift",
            )
        }


@dataclass(frozen=True)
class FixedPointPAResult:
    """Output, next quantized state and diagnostics for a fixed-point chunk."""

    output: np.ndarray
    next_state: FixedPointPAState
    stats: FixedPointPAStats

    def __post_init__(self) -> None:
        output = np.asarray(self.output, dtype=np.complex128)
        if output.ndim != 1 or not np.all(np.isfinite(output)):
            raise ValueError("fixed-point output must be finite and one-dimensional")
        object.__setattr__(self, "output", output.copy())


@dataclass
class _CounterAccumulator:
    input_saturations: int = 0
    coefficient_saturations: int = 0
    power_saturations: int = 0
    scalar_accumulator_saturations: int = 0
    accumulator_saturations: int = 0
    output_saturations: int = 0
    maximum_power_magnitude: int = 0
    maximum_scalar_accumulator_magnitude: int = 0
    maximum_accumulator_magnitude: int = 0
    knot_code_collision_count: int = 0
    maximum_knot_code_shift: int = 0

    def freeze(self, sample_count: int) -> FixedPointPAStats:
        return FixedPointPAStats(
            sample_count=sample_count,
            input_saturations=self.input_saturations,
            coefficient_saturations=self.coefficient_saturations,
            power_saturations=self.power_saturations,
            scalar_accumulator_saturations=self.scalar_accumulator_saturations,
            accumulator_saturations=self.accumulator_saturations,
            output_saturations=self.output_saturations,
            maximum_power_magnitude=self.maximum_power_magnitude,
            maximum_scalar_accumulator_magnitude=self.maximum_scalar_accumulator_magnitude,
            maximum_accumulator_magnitude=self.maximum_accumulator_magnitude,
            knot_code_collision_count=self.knot_code_collision_count,
            maximum_knot_code_shift=self.maximum_knot_code_shift,
        )


def _signed_scalar_saturate(value: int, fmt: FixedPointFormat) -> tuple[int, int]:
    if value < fmt.minimum_code:
        return fmt.minimum_code, 1
    if value > fmt.maximum_code:
        return fmt.maximum_code, 1
    return int(value), 0


def _signed_bits_saturate(value: int, bits: int) -> tuple[int, int]:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if value < minimum:
        return minimum, 1
    if value > maximum:
        return maximum, 1
    return int(value), 0


def _saturate_array_bits(values: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    array = np.asarray(values, dtype=np.int64)
    clipped = np.clip(array, minimum, maximum).astype(np.int64, copy=False)
    return clipped, int(np.count_nonzero(array != clipped))


def _saturate_array_format(
    values: np.ndarray,
    fmt: FixedPointFormat,
) -> tuple[np.ndarray, int]:
    """Clip an integer-code array to a fixed-point format's code range."""

    array = np.asarray(values, dtype=np.int64)
    clipped = np.clip(
        array,
        fmt.minimum_code,
        fmt.maximum_code,
    ).astype(np.int64, copy=False)
    return clipped, int(np.count_nonzero(array != clipped))


def _checked_array_product(
    left: np.ndarray,
    right: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("integer product operands must have matching shapes")
    maximum_left = int(np.max(np.abs(left), initial=0))
    maximum_right = int(np.max(np.abs(right), initial=0))
    checked_int64_product(maximum_left, maximum_right, label=label)
    return (left * right).astype(np.int64, copy=False)


def _checked_array_add(
    left: np.ndarray,
    right: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("integer add operands must have matching shapes")
    maximum_left = int(np.max(np.abs(left), initial=0))
    maximum_right = int(np.max(np.abs(right), initial=0))
    checked_int64_product(maximum_left + maximum_right, 1, label=label)
    return (left + right).astype(np.int64, copy=False)


def _delay_codes(values: np.ndarray, delay: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if delay < 0:
        raise ValueError("fixed-point delay must be non-negative")
    result = np.zeros(values.shape, dtype=np.int64)
    if delay == 0:
        result[:] = values
    elif delay < values.size:
        result[delay:] = values[:-delay]
    return result


def _round_shift_scalar(value: int, shift: int, *, label: str) -> int:
    checked_int64_product(value, 1, label=label)
    return int(round_shift_even(np.asarray([value], dtype=np.int64), shift)[0])


def _complex_product_int(
    left_real: int,
    left_imag: int,
    right_real: int,
    right_imag: int,
    *,
    label: str,
) -> tuple[int, int]:
    rr = checked_int64_product(left_real, right_real, label=f"{label}.rr")
    ii = checked_int64_product(left_imag, right_imag, label=f"{label}.ii")
    ri = checked_int64_product(left_real, right_imag, label=f"{label}.ri")
    ir = checked_int64_product(left_imag, right_real, label=f"{label}.ir")
    real = rr - ii
    imag = ri + ir
    if not _INT64_MIN <= real <= _INT64_MAX:
        raise OverflowError(f"{label}.real exceeds signed int64 range")
    if not _INT64_MIN <= imag <= _INT64_MAX:
        raise OverflowError(f"{label}.imag exceeds signed int64 range")
    return int(real), int(imag)


def _gmp_history_length(config: GMPConfig) -> int:
    terms = gmp_terms(config)
    return max(
        max((term.signal_delay for term in terms), default=0),
        max((term.envelope_delay for term in terms), default=0),
    )


def _integer_power_codes(
    real_codes: np.ndarray,
    imag_codes: np.ndarray,
    *,
    maximum_exponent: int,
    config: FixedPointPAConfig,
    counters: _CounterAccumulator,
) -> dict[int, np.ndarray]:
    """Compute amplitude powers in the explicitly declared power format."""

    powers: dict[int, np.ndarray] = {}
    if maximum_exponent <= 0:
        return powers
    real_codes = np.asarray(real_codes, dtype=np.int64)
    imag_codes = np.asarray(imag_codes, dtype=np.int64)
    real_square = _checked_array_product(
        real_codes,
        real_codes,
        label="input.real.square",
    )
    imag_square = _checked_array_product(
        imag_codes,
        imag_codes,
        label="input.imag.square",
    )
    radicand = _checked_array_add(
        real_square,
        imag_square,
        label="input.power",
    )
    amplitude_raw = integer_sqrt_array(radicand)
    amplitude, count = _saturate_array_format(
        amplitude_raw,
        config.power_format,
    )
    counters.power_saturations += count
    counters.maximum_power_magnitude = int(
        max(
            counters.maximum_power_magnitude,
            np.max(np.abs(amplitude), initial=0),
        )
    )
    powers[1] = amplitude
    for exponent in range(2, maximum_exponent + 1):
        previous = powers[exponent - 1]
        product = _checked_array_product(
            previous,
            amplitude,
            label=f"power^{exponent}.product",
        )
        rounded = round_shift_even(
            product,
            config.input_format.fractional_bits,
        )
        current, count = _saturate_array_format(
            rounded,
            config.power_format,
        )
        counters.power_saturations += count
        counters.maximum_power_magnitude = int(
            max(
                counters.maximum_power_magnitude,
                np.max(np.abs(current), initial=0),
            )
        )
        powers[exponent] = current
    return powers


class FixedPointGMPPA:
    """Bit-accurate causal GMP forward model using integer I/Q state."""

    def __init__(
        self,
        model: GeneralizedMemoryPolynomialPA,
        config: FixedPointPAConfig,
    ) -> None:
        if not isinstance(model, GeneralizedMemoryPolynomialPA):
            raise TypeError("model must be GeneralizedMemoryPolynomialPA")
        if model.config.leading_policy != "causal_leading":
            raise ValueError("fixed-point GMP requires causal_leading policy")
        self.model = model
        self.config = config
        coefficients = config.coefficient_format.quantize_complex(model.coefficients)
        self._coefficient_real = coefficients.real
        self._coefficient_imag = coefficients.imag
        self.coefficient_saturation_count = coefficients.saturation_count
        self._terms = gmp_terms(model.config)
        self.history_length = _gmp_history_length(model.config)
        self._terms_by_signal_delay: dict[int, list[tuple[int, object]]] = {}
        for coefficient_index, term in enumerate(self._terms):
            self._terms_by_signal_delay.setdefault(term.signal_delay, []).append(
                (coefficient_index, term)
            )

    def initial_state(self) -> FixedPointPAState:
        return FixedPointPAState(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    def _predict_codes(
        self,
        real_codes: np.ndarray,
        imag_codes: np.ndarray,
        counters: _CounterAccumulator,
    ) -> np.ndarray:
        maximum_exponent = max(
            (term.exponent for term in self._terms),
            default=0,
        )
        powers = _integer_power_codes(
            real_codes,
            imag_codes,
            maximum_exponent=maximum_exponent,
            config=self.config,
            counters=counters,
        )
        size = int(real_codes.size)
        output_accumulator_real = np.zeros(size, dtype=np.int64)
        output_accumulator_imag = np.zeros(size, dtype=np.int64)
        coefficient_fraction_bits = self.config.coefficient_format.fractional_bits
        power_fraction_bits = self.config.power_format.fractional_bits
        scalar_shift = coefficient_fraction_bits
        scalar_scale = 1 << power_fraction_bits
        output_shift = (
            self.config.input_format.fractional_bits
            + power_fraction_bits
            - self.config.output_format.fractional_bits
        )

        for signal_delay, delayed_terms in self._terms_by_signal_delay.items():
            scalar_real = np.zeros(size, dtype=np.int64)
            scalar_imag = np.zeros(size, dtype=np.int64)
            for coefficient_index, term in delayed_terms:
                coefficient_real = int(self._coefficient_real[coefficient_index])
                coefficient_imag = int(self._coefficient_imag[coefficient_index])
                if term.exponent == 0:
                    power = np.full(size, scalar_scale, dtype=np.int64)
                else:
                    power = _delay_codes(
                        powers[term.exponent],
                        term.envelope_delay,
                    )
                coefficient_real_codes = np.full(
                    size,
                    coefficient_real,
                    dtype=np.int64,
                )
                coefficient_imag_codes = np.full(
                    size,
                    coefficient_imag,
                    dtype=np.int64,
                )
                scalar_real = _checked_array_add(
                    scalar_real,
                    _checked_array_product(
                        coefficient_real_codes,
                        power,
                        label="gmp.scalar.real",
                    ),
                    label="gmp.scalar.real.sum",
                )
                scalar_imag = _checked_array_add(
                    scalar_imag,
                    _checked_array_product(
                        coefficient_imag_codes,
                        power,
                        label="gmp.scalar.imag",
                    ),
                    label="gmp.scalar.imag.sum",
                )

            scalar_real = round_shift_even(scalar_real, scalar_shift)
            scalar_imag = round_shift_even(scalar_imag, scalar_shift)
            scalar_real, sat_r = _saturate_array_bits(
                scalar_real,
                self.config.scalar_accumulator_bits,
            )
            scalar_imag, sat_i = _saturate_array_bits(
                scalar_imag,
                self.config.scalar_accumulator_bits,
            )
            counters.scalar_accumulator_saturations += sat_r + sat_i
            counters.maximum_scalar_accumulator_magnitude = int(
                max(
                    counters.maximum_scalar_accumulator_magnitude,
                    np.max(np.abs(scalar_real), initial=0),
                    np.max(np.abs(scalar_imag), initial=0),
                )
            )
            delayed_real = _delay_codes(real_codes, signal_delay)
            delayed_imag = _delay_codes(imag_codes, signal_delay)
            product_real = _checked_array_add(
                _checked_array_product(
                    delayed_real,
                    scalar_real,
                    label="gmp.output.rr",
                ),
                -_checked_array_product(
                    delayed_imag,
                    scalar_imag,
                    label="gmp.output.ii",
                ),
                label="gmp.output.real",
            )
            product_imag = _checked_array_add(
                _checked_array_product(
                    delayed_real,
                    scalar_imag,
                    label="gmp.output.ri",
                ),
                _checked_array_product(
                    delayed_imag,
                    scalar_real,
                    label="gmp.output.ir",
                ),
                label="gmp.output.imag",
            )
            output_accumulator_real = _checked_array_add(
                output_accumulator_real,
                product_real,
                label="gmp.output.accumulator.real",
            )
            output_accumulator_imag = _checked_array_add(
                output_accumulator_imag,
                product_imag,
                label="gmp.output.accumulator.imag",
            )

        output_accumulator_real, sat_r = _saturate_array_bits(
            output_accumulator_real,
            self.config.accumulator_bits,
        )
        output_accumulator_imag, sat_i = _saturate_array_bits(
            output_accumulator_imag,
            self.config.accumulator_bits,
        )
        counters.accumulator_saturations += sat_r + sat_i
        counters.maximum_accumulator_magnitude = int(
            max(
                counters.maximum_accumulator_magnitude,
                np.max(np.abs(output_accumulator_real), initial=0),
                np.max(np.abs(output_accumulator_imag), initial=0),
            )
        )
        if output_shift >= 0:
            output_real = round_shift_even(output_accumulator_real, output_shift)
            output_imag = round_shift_even(output_accumulator_imag, output_shift)
        else:
            output_real = _checked_array_product(
                output_accumulator_real,
                np.full(size, 1 << (-output_shift), dtype=np.int64),
                label="gmp.output.real.left_shift",
            )
            output_imag = _checked_array_product(
                output_accumulator_imag,
                np.full(size, 1 << (-output_shift), dtype=np.int64),
                label="gmp.output.imag.left_shift",
            )
        output_real, sat_r = _saturate_array_format(
            output_real,
            self.config.output_format,
        )
        output_imag, sat_i = _saturate_array_format(
            output_imag,
            self.config.output_format,
        )
        counters.output_saturations += sat_r + sat_i
        return (
            self.config.output_format.dequantize(output_real)
            + 1j * self.config.output_format.dequantize(output_imag)
        )

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: FixedPointPAState | None = None,
    ) -> FixedPointPAResult:
        array = np.asarray(signal)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("signal must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError("signal must be finite")
        quantized = self.config.input_format.quantize_complex(array)
        counters = _CounterAccumulator(
            input_saturations=quantized.saturation_count,
            coefficient_saturations=self.coefficient_saturation_count,
        )
        if state is None:
            state = self.initial_state()
        if state.size > self.history_length:
            raise ValueError("state contains more history than this GMP requires")
        combined_real = np.concatenate((state.real_codes, quantized.real))
        combined_imag = np.concatenate((state.imag_codes, quantized.imag))
        prediction = self._predict_codes(combined_real, combined_imag, counters)
        output = prediction[state.size :]
        if self.history_length:
            next_real = combined_real[-self.history_length :].copy()
            next_imag = combined_imag[-self.history_length :].copy()
        else:
            next_real = np.zeros(0, dtype=np.int64)
            next_imag = np.zeros(0, dtype=np.int64)
        return FixedPointPAResult(
            output=output,
            next_state=FixedPointPAState(next_real, next_imag),
            stats=counters.freeze(int(array.size)),
        )

    def predict(self, signal: np.ndarray) -> np.ndarray:
        return self.predict_chunk(signal).output

    __call__ = predict

    def predict_segments(self, signal: np.ndarray, segment_length: int) -> np.ndarray:
        array = np.asarray(signal)
        if array.ndim != 1 or int(segment_length) < 1:
            raise ValueError("signal must be 1-D and segment_length positive")
        outputs = [
            self.predict_chunk(array[start : start + int(segment_length)]).output
            for start in range(0, array.size, int(segment_length))
        ]
        return np.concatenate(outputs)
