"""Fixed-point reference emulation for the complex linear-spline DPD.

This is a deterministic arithmetic reference, not an RTL performance model.
It quantizes I/Q samples, knots, complex control points and interpolation
weights, uses integer interpolation and integer complex products, applies a
finite signed accumulator, and reports saturation.  It is suitable for
measuring numerical degradation before an HLS/RTL implementation exists.

Rounding ties use the documented round-to-nearest-even convention.  The square
root used to form the integer magnitude code is evaluated with NumPy and
rounded to the nearest integer.  Hardware must implement an equivalent
integer-sqrt/LUT approximation; its latency and resources are not claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .complex_spline_dpd import ComplexLinearSplineDPD


def _signed_limit(bits: int) -> int:
    if not isinstance(bits, (int, np.integer)) or int(bits) < 2:
        raise ValueError("signed bit width must be an integer of at least two")
    return (1 << (int(bits) - 1)) - 1


def _symmetric_scale(full_scale: float, bits: int) -> float:
    if not np.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError("full_scale must be positive and finite")
    return float(full_scale) / _signed_limit(bits)


def _quantize_signed(
    values: np.ndarray,
    *,
    scale: float,
    bits: int,
) -> tuple[np.ndarray, int]:
    limit = _signed_limit(bits)
    raw = np.rint(np.asarray(values, dtype=np.float64) / scale)
    clipped = np.clip(raw, -limit - 1, limit)
    saturation_count = int(np.count_nonzero(raw != clipped))
    return clipped.astype(np.int64), saturation_count


def _round_shift_nearest(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    if fractional_bits == 0:
        return values
    divisor = 1 << fractional_bits
    values = np.asarray(values, dtype=np.int64)
    absolute = np.abs(values)
    quotient = absolute // divisor
    remainder = absolute % divisor
    half = divisor // 2
    rounded = quotient + (
        (remainder > half)
        | ((remainder == half) & ((quotient & 1) == 1))
    )
    return np.where(values < 0, -rounded, rounded).astype(np.int64)


def _saturate_signed(values: np.ndarray, bits: int) -> tuple[np.ndarray, int]:
    limit = _signed_limit(bits)
    array = np.asarray(values, dtype=np.int64)
    clipped = np.clip(array, -limit - 1, limit)
    count = int(np.count_nonzero(array != clipped))
    return clipped.astype(np.int64, copy=False), count


@dataclass(frozen=True)
class FixedPointConfig:
    input_bits: int = 16
    coefficient_bits: int = 16
    interpolation_fraction_bits: int = 16
    accumulator_bits: int = 40
    input_full_scale: float = 1.0
    coefficient_full_scale: float | None = None
    output_bits: int | None = None
    output_full_scale: float | None = None

    def validate(self) -> None:
        _signed_limit(self.input_bits)
        _signed_limit(self.coefficient_bits)
        _signed_limit(self.accumulator_bits)
        if self.input_bits > 24 or self.coefficient_bits > 24:
            raise ValueError("reference int64 path supports input/coeff widths <= 24")
        if not isinstance(self.interpolation_fraction_bits, (int, np.integer)):
            raise TypeError("interpolation_fraction_bits must be an integer")
        if not 1 <= int(self.interpolation_fraction_bits) <= 24:
            raise ValueError("interpolation_fraction_bits must be in [1, 24]")
        if not np.isfinite(self.input_full_scale) or self.input_full_scale <= 0:
            raise ValueError("input_full_scale must be positive and finite")
        if self.coefficient_full_scale is not None and (
            not np.isfinite(self.coefficient_full_scale)
            or self.coefficient_full_scale <= 0
        ):
            raise ValueError("coefficient_full_scale must be positive and finite")
        if self.output_bits is not None:
            _signed_limit(self.output_bits)
            if self.output_full_scale is None:
                raise ValueError(
                    "output_full_scale is required when output_bits is set"
                )
        elif self.output_full_scale is not None:
            raise ValueError(
                "output_bits is required when output_full_scale is set"
            )
        if self.output_full_scale is not None and (
            not np.isfinite(self.output_full_scale)
            or self.output_full_scale <= 0.0
        ):
            raise ValueError("output_full_scale must be positive and finite")


@dataclass(frozen=True)
class FixedPointResult:
    output: np.ndarray
    input_scale: float
    coefficient_scale: float
    input_saturations: int
    coefficient_saturations: int
    accumulator_saturations: int
    output_saturations: int
    maximum_accumulator_magnitude: int
    interpolation_fraction_bits: int
    knot_code_collision_count: int
    maximum_knot_code_shift: int
    output_scale: float


def predict_fixed_point(
    model: ComplexLinearSplineDPD,
    desired_signal: np.ndarray,
    config: FixedPointConfig,
) -> FixedPointResult:
    """Evaluate the spline with an explicit integer arithmetic schedule."""

    config.validate()
    signal = np.asarray(desired_signal)
    shape = signal.shape
    flat = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("desired_signal must be finite and non-empty")

    input_scale = _symmetric_scale(config.input_full_scale, config.input_bits)
    coefficient_peak = float(
        max(
            np.max(np.abs(model.coefficients.real), initial=0.0),
            np.max(np.abs(model.coefficients.imag), initial=0.0),
        )
    )
    coefficient_full_scale = (
        config.coefficient_full_scale
        if config.coefficient_full_scale is not None
        else max(coefficient_peak, np.finfo(float).tiny)
    )
    coefficient_scale = _symmetric_scale(
        coefficient_full_scale,
        config.coefficient_bits,
    )

    input_i, sat_i = _quantize_signed(
        flat.real,
        scale=input_scale,
        bits=config.input_bits,
    )
    input_q, sat_q = _quantize_signed(
        flat.imag,
        scale=input_scale,
        bits=config.input_bits,
    )
    coefficient_i, sat_ci = _quantize_signed(
        model.coefficients.real,
        scale=coefficient_scale,
        bits=config.coefficient_bits,
    )
    coefficient_q, sat_cq = _quantize_signed(
        model.coefficients.imag,
        scale=coefficient_scale,
        bits=config.coefficient_bits,
    )

    # Radius and knots use the same integer unit as I/Q.
    radius_code = np.rint(
        np.sqrt(
            input_i.astype(np.float64) ** 2
            + input_q.astype(np.float64) ** 2
        )
    ).astype(np.int64)
    fractional_scale = 1 << int(config.interpolation_fraction_bits)
    knot_float_codes = model.knots / input_scale
    int64_limit = float(np.iinfo(np.int64).max - 1)
    if (
        not np.all(np.isfinite(knot_float_codes))
        or np.max(np.abs(knot_float_codes), initial=0.0) > int64_limit
    ):
        raise ValueError(
            "knot address codes exceed the int64 reference range; choose a "
            "compatible input_full_scale/address width"
        )
    original_knot_codes = np.rint(knot_float_codes).astype(np.int64)
    knot_codes = original_knot_codes.copy()
    # Quantization can merge adjacent knots.  Enforce a monotone address table;
    # this is recorded indirectly by the resulting numeric error and prevents
    # division by zero in the reference.
    for index in range(1, knot_codes.size):
        if knot_codes[index] <= knot_codes[index - 1]:
            knot_codes[index] = knot_codes[index - 1] + 1
    knot_code_collision_count = int(
        np.count_nonzero(knot_codes != original_knot_codes)
    )
    maximum_knot_code_shift = int(
        np.max(np.abs(knot_codes - original_knot_codes), initial=0)
    )
    maximum_address = int(np.max(np.abs(knot_codes), initial=0))
    if (
        maximum_address
        > np.iinfo(np.int64).max // fractional_scale
    ):
        raise ValueError(
            "knot address multiplied by interpolation scale would overflow "
            "the int64 reference arithmetic"
        )

    clipped_radius = np.clip(radius_code, knot_codes[0], knot_codes[-1])
    left = np.searchsorted(knot_codes, clipped_radius, side="right") - 1
    left = np.clip(left, 0, knot_codes.size - 2)
    numerator = clipped_radius - knot_codes[left]
    denominator = knot_codes[left + 1] - knot_codes[left]
    # All operands are non-negative, so add half the denominator before an
    # integer division to implement deterministic round-to-nearest.
    weight_floor, weight_remainder = np.divmod(
        numerator * fractional_scale,
        denominator,
    )
    weight_code = weight_floor + (
        (2 * weight_remainder > denominator)
        | (
            (2 * weight_remainder == denominator)
            & ((weight_floor & 1) == 1)
        )
    )
    weight_code = np.clip(weight_code, 0, fractional_scale)

    delta_i = coefficient_i[left + 1] - coefficient_i[left]
    delta_q = coefficient_q[left + 1] - coefficient_q[left]
    interpolated_i = coefficient_i[left] + _round_shift_nearest(
        weight_code * delta_i,
        int(config.interpolation_fraction_bits),
    )
    interpolated_q = coefficient_q[left] + _round_shift_nearest(
        weight_code * delta_q,
        int(config.interpolation_fraction_bits),
    )

    accumulator_i = input_i * interpolated_i - input_q * interpolated_q
    accumulator_q = input_i * interpolated_q + input_q * interpolated_i
    maximum_accumulator = int(
        max(
            np.max(np.abs(accumulator_i), initial=0),
            np.max(np.abs(accumulator_q), initial=0),
        )
    )
    accumulator_i, sat_ai = _saturate_signed(
        accumulator_i,
        config.accumulator_bits,
    )
    accumulator_q, sat_aq = _saturate_signed(
        accumulator_q,
        config.accumulator_bits,
    )

    accumulator_output_scale = input_scale * coefficient_scale
    output_unquantized = (
        accumulator_i.astype(np.float64)
        + 1j * accumulator_q.astype(np.float64)
    ) * accumulator_output_scale
    output_saturations = 0
    if config.output_bits is None:
        output = output_unquantized
        output_scale = accumulator_output_scale
    else:
        assert config.output_full_scale is not None
        output_scale = _symmetric_scale(
            config.output_full_scale,
            config.output_bits,
        )
        output_i, sat_oi = _quantize_signed(
            output_unquantized.real,
            scale=output_scale,
            bits=config.output_bits,
        )
        output_q, sat_oq = _quantize_signed(
            output_unquantized.imag,
            scale=output_scale,
            bits=config.output_bits,
        )
        output_saturations = sat_oi + sat_oq
        output = (
            output_i.astype(np.float64)
            + 1j * output_q.astype(np.float64)
        ) * output_scale
    return FixedPointResult(
        output=output.reshape(shape),
        input_scale=input_scale,
        coefficient_scale=coefficient_scale,
        input_saturations=sat_i + sat_q,
        coefficient_saturations=sat_ci + sat_cq,
        accumulator_saturations=sat_ai + sat_aq,
        output_saturations=output_saturations,
        maximum_accumulator_magnitude=maximum_accumulator,
        interpolation_fraction_bits=int(config.interpolation_fraction_bits),
        knot_code_collision_count=knot_code_collision_count,
        maximum_knot_code_shift=maximum_knot_code_shift,
        output_scale=output_scale,
    )


def predict_fp16_storage(
    model: ComplexLinearSplineDPD,
    desired_signal: np.ndarray,
) -> np.ndarray:
    """Emulate FP16 storage/rounding with FP32 arithmetic.

    NumPy has no portable complex-FP16 dtype.  Real and imaginary components
    are rounded independently to float16 and promoted to float32 for the
    interpolation/product.  This is therefore an FP16-like storage test, not a
    claim about a specific accelerator's FP16 accumulator.
    """

    signal = np.asarray(desired_signal)
    signal32 = (
        signal.real.astype(np.float16).astype(np.float32)
        + 1j * signal.imag.astype(np.float16).astype(np.float32)
    ).astype(np.complex64)
    knots32 = model.knots.astype(np.float16).astype(np.float32)
    if not np.all(np.diff(knots32) > 0.0):
        raise ValueError(
            "FP16 knot storage collapses adjacent knots; use fewer knots or "
            "a wider knot/address format"
        )
    coefficients32 = (
        model.coefficients.real.astype(np.float16).astype(np.float32)
        + 1j * model.coefficients.imag.astype(np.float16).astype(np.float32)
    )
    radius32 = np.sqrt(
        signal32.real * signal32.real + signal32.imag * signal32.imag
    ).astype(np.float32)
    clipped = np.clip(radius32, knots32[0], knots32[-1]).astype(np.float32)
    left = np.searchsorted(knots32, clipped, side="right") - 1
    left = np.clip(left, 0, knots32.size - 2)
    width = (knots32[left + 1] - knots32[left]).astype(np.float32)
    weight = ((clipped - knots32[left]) / width).astype(np.float32)
    correction = (
        coefficients32[left]
        + weight * (coefficients32[left + 1] - coefficients32[left])
    ).astype(np.complex64)
    output = (signal32 * correction).astype(np.complex64)
    return (
        output.real.astype(np.float16).astype(np.float32)
        + 1j * output.imag.astype(np.float16).astype(np.float32)
    )
