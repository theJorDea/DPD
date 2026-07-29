import unittest

from baseline.complexity import (
    OperationCount,
    complex_spline_inference_cost,
    esn_fan_complex_pair_cost,
    esn_fan_scalar_cost,
    memory_polynomial_inference_cost,
    widely_linear_residual_correction_cost,
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
        self.assertEqual(cost.real_multiplications, 165)
        self.assertEqual(cost.real_additions, 103)
        self.assertEqual(cost.nonlinear_operations, 0)
        self.assertEqual(cost.stored_real_coefficients, 50)
        self.assertEqual(cost.state_real_values, 8)

    def test_full_mp_count_shares_one_magnitude_per_delay(self) -> None:
        cost = memory_polynomial_inference_cost(
            orders=tuple(range(1, 10)),
            delays=(0,),
        )
        self.assertEqual(cost.real_multiplications, 60)
        self.assertEqual(cost.real_additions, 35)
        self.assertEqual(cost.nonlinear_operations, 1)
        self.assertEqual(cost.stored_real_coefficients, 18)
        self.assertEqual(cost.state_real_values, 0)

    def test_linear_mp_does_not_compute_unused_magnitude(self) -> None:
        cost = memory_polynomial_inference_cost(
            orders=(1,),
            delays=(0,),
        )
        self.assertEqual(cost.real_multiplications, 4)
        self.assertEqual(cost.real_additions, 2)
        self.assertEqual(cost.nonlinear_operations, 0)
        self.assertEqual(cost.state_real_values, 0)

    def test_mp_delay_line_state_is_counted_separately(self) -> None:
        dpa = memory_polynomial_inference_cost(
            orders=(1, 3, 5, 7, 9),
            delays=tuple(range(24)),
        )
        apa = memory_polynomial_inference_cost(
            orders=(1, 2, 3, 4, 5),
            delays=tuple(range(30)),
        )
        self.assertEqual(dpa.state_real_values, 46)
        self.assertEqual(apa.state_real_values, 58)
        self.assertEqual(dpa.real_memory_writes, 2)
        self.assertEqual(apa.real_memory_writes, 2)

    def test_widely_linear_residual_cost_matches_preregistered_apa_budget(
        self,
    ) -> None:
        base = OperationCount(
            real_multiplications=954,
            real_additions=947,
            nonlinear_operations=1,
            stored_real_coefficients=888,
            state_real_values=236,
        )
        correction = widely_linear_residual_correction_cost(
            delays=(0, 1, 2, 3, 4),
            convention="4m2a",
            reuse_input_delay_state=True,
        )
        self.assertEqual(correction.real_multiplications, 20)
        self.assertEqual(correction.real_additions, 20)
        self.assertEqual(correction.real_memory_reads, 20)
        self.assertEqual(correction.real_memory_writes, 0)
        self.assertEqual(correction.stored_real_coefficients, 10)
        self.assertEqual(correction.state_real_values, 0)

        combined = base + correction
        self.assertEqual(combined.real_multiplications, 974)
        self.assertEqual(combined.real_additions, 967)
        self.assertEqual(combined.nonlinear_operations, 1)
        self.assertEqual(combined.stored_real_coefficients, 898)
        self.assertEqual(combined.state_real_values, 236)

    def test_widely_linear_gauss_and_standalone_state_tradeoff(self) -> None:
        gauss = widely_linear_residual_correction_cost(
            delays=(0, 1, 2, 3, 4),
            convention="3m5a",
            reuse_input_delay_state=True,
        )
        self.assertEqual(gauss.real_multiplications, 15)
        self.assertEqual(gauss.real_additions, 35)

        standalone = widely_linear_residual_correction_cost(
            delays=(0, 3),
            reuse_input_delay_state=False,
        )
        self.assertEqual(standalone.state_real_values, 6)
        self.assertEqual(standalone.real_memory_writes, 2)
        self.assertEqual(standalone.stored_real_constants, 2)

    def test_widely_linear_cost_rejects_noncausal_or_ambiguous_delays(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            widely_linear_residual_correction_cost(())
        with self.assertRaisesRegex(ValueError, "causal"):
            widely_linear_residual_correction_cost((0, -1))
        with self.assertRaisesRegex(ValueError, "unique"):
            widely_linear_residual_correction_cost((0, 0))
        with self.assertRaisesRegex(TypeError, "integers"):
            widely_linear_residual_correction_cost((0, 1.0))


if __name__ == "__main__":
    unittest.main()
