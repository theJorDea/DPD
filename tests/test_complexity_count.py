import unittest

from baseline.complexity import (
    complex_spline_inference_cost,
    esn_fan_complex_pair_cost,
    esn_fan_scalar_cost,
    memory_polynomial_inference_cost,
)


class ComplexityCountTests(unittest.TestCase):
    def test_memoryless_spline_primary_convention(self) -> None:
        cost = complex_spline_inference_cost(
            16,
            convention="4m2a",
            indexing="binary",
            reciprocal_widths=True,
            amplitude_coordinate=True,
        )
        self.assertEqual(cost.real_multiplications, 9)
        self.assertEqual(cost.real_additions, 8)
        self.assertEqual(cost.real_divisions, 0)
        self.assertEqual(cost.nonlinear_operations, 1)
        self.assertEqual(cost.comparisons, 4)
        self.assertEqual(cost.stored_real_coefficients, 32)

    def test_gauss_complex_multiply_tradeoff(self) -> None:
        normal = complex_spline_inference_cost(8, convention="4m2a")
        gauss = complex_spline_inference_cost(8, convention="3m5a")
        self.assertEqual(gauss.real_multiplications, normal.real_multiplications - 1)
        self.assertEqual(gauss.real_additions, normal.real_additions + 3)

    def test_egor_dense_esn_exact_counts(self) -> None:
        dpd_scalar = esn_fan_scalar_cost(600)
        dpd_pair = esn_fan_complex_pair_cost(600)
        pa_pair = esn_fan_complex_pair_cost(800)
        self.assertEqual(dpd_scalar.real_multiplications, 364_311)
        self.assertEqual(dpd_scalar.real_additions, 363_076)
        self.assertEqual(dpd_pair.real_multiplications, 728_622)
        self.assertEqual(dpd_pair.real_additions, 726_152)
        self.assertEqual(pa_pair.real_multiplications, 1_291_422)
        self.assertEqual(pa_pair.real_additions, 1_288_152)
        self.assertEqual(dpd_pair.stored_real_coefficients, 727_432)

    def test_memory_polynomial_count_shares_envelope_powers(self) -> None:
        cost = memory_polynomial_inference_cost(
            orders=(1, 3, 5, 7, 9),
            delays=(0, 1, 2, 3, 4),
        )
        self.assertEqual(cost.real_multiplications, 170)
        self.assertEqual(cost.real_additions, 103)
        self.assertEqual(cost.stored_real_coefficients, 50)


if __name__ == "__main__":
    unittest.main()
