"""Bit-accurate fixed-point emulation of a sparse spline-memory PA.

The floating-point forward model is

    y[n] = sum_b x[n-m_b] C_b(|x[n-d_b]|),

where each ``C_b`` is a complex piecewise-linear function.  This module
implements the same causal schedule with integer I/Q codes:

* input and control points are quantized with explicit signed formats;
* the envelope is obtained from an integer square root;
* only the two control points surrounding one quantized knot interval are
  accessed;
* interpolation uses deterministic round-to-nearest-even integer division;
* complex products and accumulators are checked before int64 arithmetic;
* every stream boundary is explicit through :class:`FixedPointPAState`.

This is a numerical reference model.  It is intended to freeze the arithmetic
contract before an RTL/HLS implementation; Python/NumPy runtime is not a
hardware latency claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .complexity import (
    ComplexMultiplyConvention,
    OperationCount,
    complex_multiply_cost,
)
from .fixed_point_pa import (
    FixedPointPAConfig,
    FixedPointPAResult,
    FixedPointPAState,
    _CounterAccumulator,
    _checked_array_add,
    _checked_array_product,
    _delay_codes,
    _integer_power_codes,
    _saturate_array_format,
    _saturate_array_bits,
    round_shift_even,
)
from .sparse_spline_memory_pa import SparseSplineMemoryPA


@dataclass(frozen=True)
class _KnotAddressTable:
    """Quantized knot table and the deterministic collision diagnostics."""

    codes: np.ndarray
    collision_count: int
    maximum_shift: int

    def __post_init__(self) -> None:
        codes = np.asarray(self.codes, dtype=np.int64)
        if codes.ndim != 1 or codes.size < 2:
            raise ValueError("at least two quantized knots are required")
        if not np.all(np.diff(codes) > 0):
            raise ValueError("quantized knot table must be strictly increasing")
        object.__setattr__(self, "codes", codes.copy())
        object.__setattr__(self, "collision_count", int(self.collision_count))
        object.__setattr__(self, "maximum_shift", int(self.maximum_shift))


def _round_divide_array_even(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Round non-negative integer quotients to nearest-even.

    The comparison ``remainder`` versus ``denominator - remainder`` avoids a
    potentially overflowing ``2 * remainder`` operation.
    """

    numerator = np.asarray(numerator, dtype=np.int64)
    denominator = np.asarray(denominator, dtype=np.int64)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    if np.any(numerator < 0):
        raise ValueError(f"{label} numerator must be non-negative")
    if np.any(denominator <= 0):
        raise ValueError(f"{label} denominator must be positive")
    quotient, remainder = np.divmod(numerator, denominator)
    increment = (remainder > (denominator - remainder)) | (
        (remainder == (denominator - remainder))
        & ((quotient & 1) == 1)
    )
    result = quotient + increment.astype(np.int64)
    if np.any(result < 0) or np.any(result > np.iinfo(np.int64).max):
        raise OverflowError(f"{label} exceeds signed int64 range")
    return result.astype(np.int64, copy=False)


def _scaled_array(
    values: np.ndarray,
    shift: int,
    *,
    label: str,
) -> np.ndarray:
    """Multiply integer codes by ``2**shift`` after an overflow check."""

    values = np.asarray(values, dtype=np.int64)
    if shift < 0:
        raise ValueError("left scaling shift must be non-negative")
    if shift == 0:
        return values.copy()
    factor = np.full(values.shape, 1 << int(shift), dtype=np.int64)
    return _checked_array_product(values, factor, label=label)


def _quantized_knot_table(
    knots: np.ndarray,
    config: FixedPointPAConfig,
) -> _KnotAddressTable:
    """Quantize and monotonically repair the amplitude address table.

    The radius code is in ``power_format`` units.  A collision is not silently
    ignored: the following knot is moved to the next representable code and
    both the number and the largest movement are exposed in every result.
    """

    if (
        config.power_format.fractional_bits
        != config.input_format.fractional_bits
    ):
        raise ValueError(
            "power_format and input_format must use the same fractional scale "
            "for amplitude interpolation"
        )
    quantized = config.power_format.quantize(knots)
    if quantized.saturation_count:
        raise ValueError(
            "spline knots exceed the representable power/address range"
        )
    original = quantized.codes
    repaired = original.copy()
    for index in range(1, repaired.size):
        if repaired[index] <= repaired[index - 1]:
            repaired[index] = repaired[index - 1] + 1
    if np.any(repaired > config.power_format.maximum_code):
        raise OverflowError("monotone knot repair exceeds address code range")
    changed = repaired != original
    return _KnotAddressTable(
        repaired,
        collision_count=int(np.count_nonzero(changed)),
        maximum_shift=int(np.max(np.abs(repaired - original), initial=0)),
    )


class FixedPointSparseSplineMemoryPA:
    """Causal fixed-point evaluator for :class:`SparseSplineMemoryPA`."""

    def __init__(
        self,
        model: SparseSplineMemoryPA,
        config: FixedPointPAConfig,
    ) -> None:
        if not isinstance(model, SparseSplineMemoryPA):
            raise TypeError("model must be SparseSplineMemoryPA")
        if not isinstance(config, FixedPointPAConfig):
            raise TypeError("config must be FixedPointPAConfig")
        self.model = model
        self.config = config
        self.history_length = int(model.maximum_delay)
        self._terms = tuple(model.branches)
        coefficients = config.coefficient_format.quantize_complex(
            model.coefficients.reshape(-1)
        )
        self._coefficient_real = coefficients.real.reshape(
            model.coefficients.shape
        )
        self._coefficient_imag = coefficients.imag.reshape(
            model.coefficients.shape
        )
        self.coefficient_saturation_count = coefficients.saturation_count
        self._knots = _quantized_knot_table(model.knots, config)
        self.knot_code_collision_count = self._knots.collision_count
        self.maximum_knot_code_shift = self._knots.maximum_shift
        self._envelope_delays = tuple(
            sorted({branch.envelope_delay for branch in self._terms})
        )
        self._branches_by_envelope: dict[int, list[int]] = {}
        for branch_index, branch in enumerate(self._terms):
            self._branches_by_envelope.setdefault(
                branch.envelope_delay,
                [],
            ).append(branch_index)

    @property
    def knot_codes(self) -> np.ndarray:
        """Return a copy of the integer knot address table."""

        return self._knots.codes.copy()

    def initial_state(self) -> FixedPointPAState:
        return FixedPointPAState(
            np.zeros(self.history_length, dtype=np.int64),
            np.zeros(self.history_length, dtype=np.int64),
        )

    def _validate_state(self, state: FixedPointPAState) -> None:
        if not isinstance(state, FixedPointPAState):
            raise TypeError("state must be FixedPointPAState")
        if state.size > self.history_length:
            raise ValueError("state contains more history than this model needs")
        if state.size:
            if np.any(
                state.real_codes < self.config.input_format.minimum_code
            ) or np.any(
                state.real_codes > self.config.input_format.maximum_code
            ):
                raise ValueError("state real codes exceed input format")
            if np.any(
                state.imag_codes < self.config.input_format.minimum_code
            ) or np.any(
                state.imag_codes > self.config.input_format.maximum_code
            ):
                raise ValueError("state imag codes exceed input format")

    def _interpolation_coordinates(
        self,
        radius_codes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return local knot indices and Q-weight codes for each sample."""

        radius_codes = np.asarray(radius_codes, dtype=np.int64)
        knots = self._knots.codes
        clipped = np.clip(
            radius_codes,
            int(knots[0]),
            int(knots[-1]),
        )
        left = np.searchsorted(knots, clipped, side="right") - 1
        left = np.clip(left, 0, knots.size - 2).astype(np.int64)
        numerator = clipped - knots[left]
        denominator = knots[left + 1] - knots[left]
        scaled = _scaled_array(
            numerator,
            self.config.interpolation_fraction_bits,
            label="spline.interpolation.numerator",
        )
        weight = _round_divide_array_even(
            scaled,
            denominator,
            label="spline.interpolation.weight",
        )
        weight = np.clip(
            weight,
            0,
            1 << self.config.interpolation_fraction_bits,
        ).astype(np.int64, copy=False)
        return left, weight

    def _interpolated_coefficients(
        self,
        left: np.ndarray,
        weight: np.ndarray,
        branch_index: int,
        counters: _CounterAccumulator,
    ) -> tuple[np.ndarray, np.ndarray]:
        real = self._coefficient_real[branch_index]
        imag = self._coefficient_imag[branch_index]
        delta_real = real[left + 1] - real[left]
        delta_imag = imag[left + 1] - imag[left]
        real_product = _checked_array_product(
            delta_real,
            weight,
            label="spline.coefficient.real",
        )
        imag_product = _checked_array_product(
            delta_imag,
            weight,
            label="spline.coefficient.imag",
        )
        real = real[left] + round_shift_even(
            real_product,
            self.config.interpolation_fraction_bits,
        )
        imag = imag[left] + round_shift_even(
            imag_product,
            self.config.interpolation_fraction_bits,
        )
        real, sat_real = _saturate_array_format(
            real,
            self.config.coefficient_format,
        )
        imag, sat_imag = _saturate_array_format(
            imag,
            self.config.coefficient_format,
        )
        counters.interpolation_saturations += sat_real + sat_imag
        counters.maximum_scalar_accumulator_magnitude = int(
            max(
                counters.maximum_scalar_accumulator_magnitude,
                np.max(np.abs(real), initial=0),
                np.max(np.abs(imag), initial=0),
            )
        )
        return real, imag

    def _predict_codes(
        self,
        real_codes: np.ndarray,
        imag_codes: np.ndarray,
        counters: _CounterAccumulator,
    ) -> np.ndarray:
        real_codes = np.asarray(real_codes, dtype=np.int64)
        imag_codes = np.asarray(imag_codes, dtype=np.int64)
        if real_codes.shape != imag_codes.shape:
            raise ValueError("integer input code shapes must match")
        size = int(real_codes.size)
        powers = _integer_power_codes(
            real_codes,
            imag_codes,
            maximum_exponent=1,
            config=self.config,
            counters=counters,
        )
        radius_by_delay = {
            delay: _delay_codes(powers[1], delay)
            for delay in self._envelope_delays
        }
        coordinates = {
            delay: self._interpolation_coordinates(radius)
            for delay, radius in radius_by_delay.items()
        }
        output_real = np.zeros(size, dtype=np.int64)
        output_imag = np.zeros(size, dtype=np.int64)

        for branch_index, branch in enumerate(self._terms):
            left, weight = coordinates[branch.envelope_delay]
            coefficient_real, coefficient_imag = (
                self._interpolated_coefficients(
                    left,
                    weight,
                    branch_index,
                    counters,
                )
            )
            delayed_real = _delay_codes(real_codes, branch.signal_delay)
            delayed_imag = _delay_codes(imag_codes, branch.signal_delay)
            product_real = _checked_array_add(
                _checked_array_product(
                    delayed_real,
                    coefficient_real,
                    label="spline.output.rr",
                ),
                -_checked_array_product(
                    delayed_imag,
                    coefficient_imag,
                    label="spline.output.ii",
                ),
                label="spline.output.real",
            )
            product_imag = _checked_array_add(
                _checked_array_product(
                    delayed_real,
                    coefficient_imag,
                    label="spline.output.ri",
                ),
                _checked_array_product(
                    delayed_imag,
                    coefficient_real,
                    label="spline.output.ir",
                ),
                label="spline.output.imag",
            )
            output_real = _checked_array_add(
                output_real,
                product_real,
                label="spline.output.accumulator.real",
            )
            output_imag = _checked_array_add(
                output_imag,
                product_imag,
                label="spline.output.accumulator.imag",
            )

        output_real, sat_real = _saturate_array_bits(
            output_real,
            self.config.accumulator_bits,
        )
        output_imag, sat_imag = _saturate_array_bits(
            output_imag,
            self.config.accumulator_bits,
        )
        counters.accumulator_saturations += sat_real + sat_imag
        counters.maximum_accumulator_magnitude = int(
            max(
                counters.maximum_accumulator_magnitude,
                np.max(np.abs(output_real), initial=0),
                np.max(np.abs(output_imag), initial=0),
            )
        )
        output_shift = (
            self.config.input_format.fractional_bits
            + self.config.coefficient_format.fractional_bits
            - self.config.output_format.fractional_bits
        )
        if output_shift >= 0:
            output_real = round_shift_even(output_real, output_shift)
            output_imag = round_shift_even(output_imag, output_shift)
        else:
            output_real = _scaled_array(
                output_real,
                -output_shift,
                label="spline.output.real.left_shift",
            )
            output_imag = _scaled_array(
                output_imag,
                -output_shift,
                label="spline.output.imag.left_shift",
            )
        output_real, sat_real = _saturate_array_format(
            output_real,
            self.config.output_format,
        )
        output_imag, sat_imag = _saturate_array_format(
            output_imag,
            self.config.output_format,
        )
        counters.output_saturations += sat_real + sat_imag
        return self.config.output_format.dequantize_complex(
            output_real,
            output_imag,
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
        if state is None:
            state = self.initial_state()
        self._validate_state(state)
        counters = _CounterAccumulator(
            input_saturations=quantized.saturation_count,
            coefficient_saturations=self.coefficient_saturation_count,
            knot_code_collision_count=self.knot_code_collision_count,
            maximum_knot_code_shift=self.maximum_knot_code_shift,
        )
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
        """Evaluate one independent record with zero initial history."""

        return self.predict_chunk(signal).output

    __call__ = predict

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        array = np.asarray(signal)
        if array.ndim != 1 or int(segment_length) < 1:
            raise ValueError("signal must be 1-D and segment_length positive")
        outputs = [
            self.predict_chunk(array[start : start + int(segment_length)]).output
            for start in range(0, array.size, int(segment_length))
        ]
        return np.concatenate(outputs)

    def operation_count(
        self,
        *,
        convention: ComplexMultiplyConvention = "4m2a",
        indexing: str = "binary",
    ) -> OperationCount:
        """Return the declared integer arithmetic cost per complex sample."""

        if indexing not in {"binary", "uniform"}:
            raise ValueError("indexing must be binary or uniform")
        complex_mult, complex_add = complex_multiply_cost(convention)
        envelope_groups = len(self._envelope_delays)
        branches = len(self._terms)
        knot_count = self.model.knot_count
        comparisons = (
            int(math.ceil(math.log2(knot_count)))
            if indexing == "binary"
            else 2
        )
        # Two integer squares and one radius-square addition per envelope.
        real_multiplications = 2 * envelope_groups
        real_additions = 3 * envelope_groups
        real_divisions = envelope_groups
        nonlinear_operations = envelope_groups  # integer sqrt
        # Each branch interpolates real/imaginary control points and then
        # performs one proper complex multiply.
        real_multiplications += branches * (2 + complex_mult)
        real_additions += branches * (4 + complex_add)
        real_additions += 2 * max(branches - 1, 0)
        return OperationCount(
            real_multiplications=real_multiplications,
            real_additions=real_additions,
            real_divisions=real_divisions,
            nonlinear_operations=nonlinear_operations,
            comparisons=envelope_groups * comparisons,
            lookups=2 * envelope_groups + 2 * branches,
            real_memory_reads=(
                4 * envelope_groups
                + 4 * branches
                + 4 * branches
            ),
            real_memory_writes=2 if self.history_length else 0,
            stored_real_coefficients=2 * branches * knot_count,
            stored_real_constants=knot_count,
            state_real_values=2 * self.history_length,
            notes=(
                f"complex multiply convention {convention}",
                f"{indexing} integer knot interval selection",
                "integer sqrt is counted as one nonlinear operation",
                "weight division may map to reciprocal multiply plus shift in RTL",
                "interpolation uses two active control points per branch",
            ),
        )


__all__ = ["FixedPointSparseSplineMemoryPA"]
