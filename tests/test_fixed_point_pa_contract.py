import unittest

import numpy as np

from baseline.fixed_point_pa import (
    FixedPointFormat,
    integer_sqrt_array,
    integer_sqrt_round_even,
    round_divide_even,
    round_shift_even,
    saturate_codes,
)


class FixedPointContractTests(unittest.TestCase):
    def test_format_quantizes_and_reports_saturation(self) -> None:
        fmt = FixedPointFormat(bits=4, fractional_bits=2, label="q")
        result = fmt.quantize(np.asarray([-3.0, -1.25, 0.125, 1.9, 4.0]))
        np.testing.assert_array_equal(result.codes, [-8, -5, 0, 7, 7])
        self.assertEqual(result.saturation_count, 3)
        np.testing.assert_allclose(
            fmt.dequantize(result.codes),
            [-2.0, -1.25, 0.0, 1.75, 1.75],
        )

    def test_round_shift_is_symmetric_and_ties_even(self) -> None:
        values = np.asarray([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
        expected = np.asarray([-4, -3, -2, -2, -2, -1, 0, 0, 0, 1, 2, 2, 2, 3, 4])
        np.testing.assert_array_equal(round_shift_even(values, 1), expected)
        self.assertEqual(round_divide_even(-5, 2), -2)
        self.assertEqual(round_divide_even(-7, 2), -4)
        self.assertEqual(round_divide_even(5, 2), 2)
        self.assertEqual(round_divide_even(7, 2), 4)

    def test_integer_square_root_uses_nearest_even_ties(self) -> None:
        self.assertEqual(integer_sqrt_round_even(0), 0)
        self.assertEqual(integer_sqrt_round_even(3), 2)
        self.assertEqual(integer_sqrt_round_even(2), 1)
        np.testing.assert_array_equal(
            integer_sqrt_array(np.asarray([0, 1, 2, 3, 4, 8, 9])),
            [0, 1, 1, 2, 2, 3, 3],
        )

    def test_complex_quantization_and_explicit_saturation(self) -> None:
        fmt = FixedPointFormat(bits=8, fractional_bits=4)
        result = fmt.quantize_complex(np.asarray([0.5 + 0.25j, 10.0 - 10.0j]))
        np.testing.assert_array_equal(result.real, [8, 127])
        np.testing.assert_array_equal(result.imag, [4, -128])
        self.assertEqual(result.saturation_count, 2)
        clipped = saturate_codes(np.asarray([-200, -1, 1, 200]), fmt)
        np.testing.assert_array_equal(clipped.codes, [-128, -1, 1, 127])
        self.assertEqual(clipped.saturation_count, 2)

    def test_invalid_format_and_negative_sqrt_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FixedPointFormat(bits=8, fractional_bits=8)
        with self.assertRaises(ValueError):
            integer_sqrt_round_even(-1)
        with self.assertRaises(ValueError):
            integer_sqrt_array(np.asarray([1, -1]))

    def test_full_scale_helper_preserves_headroom(self) -> None:
        fmt = FixedPointFormat.for_full_scale(16, 2.52, label="output")
        self.assertGreaterEqual(fmt.representable_maximum, 2.52 * 1.001)
        self.assertEqual(fmt.fractional_bits, 13)


if __name__ == "__main__":
    unittest.main()
