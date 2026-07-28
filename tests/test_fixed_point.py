import unittest

import numpy as np

from baseline.complex_spline_dpd import ComplexLinearSplineDPD
from baseline.fixed_point import (
    FixedPointConfig,
    predict_fixed_point,
    predict_fp16_storage,
)
from baseline.metrics import nmse_pooled_db


class FixedPointSplineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ComplexLinearSplineDPD(
            knots=np.linspace(0.0, 1.0, 12),
            coefficients=np.linspace(0.8, 1.4, 12)
            + 1j * np.linspace(0.2, -0.15, 12),
            knot_strategy="uniform_amplitude",
        )
        rng = np.random.default_rng(128)
        radius = rng.uniform(0.0, 0.9, 1000)
        phase = rng.uniform(-np.pi, np.pi, 1000)
        self.signal = radius * np.exp(1j * phase)

    def test_16_bit_is_more_accurate_than_12_bit(self) -> None:
        reference = self.model.predict(self.signal)
        result16 = predict_fixed_point(
            self.model,
            self.signal,
            FixedPointConfig(
                input_bits=16,
                coefficient_bits=16,
                input_full_scale=1.0,
                accumulator_bits=40,
            ),
        )
        result12 = predict_fixed_point(
            self.model,
            self.signal,
            FixedPointConfig(
                input_bits=12,
                coefficient_bits=12,
                input_full_scale=1.0,
                accumulator_bits=32,
            ),
        )
        self.assertLess(
            nmse_pooled_db(result16.output, reference),
            nmse_pooled_db(result12.output, reference),
        )
        self.assertEqual(result16.accumulator_saturations, 0)
        self.assertEqual(result12.accumulator_saturations, 0)

    def test_accumulator_saturation_is_reported(self) -> None:
        result = predict_fixed_point(
            self.model,
            self.signal,
            FixedPointConfig(
                input_bits=12,
                coefficient_bits=12,
                input_full_scale=1.0,
                accumulator_bits=12,
            ),
        )
        self.assertGreater(result.accumulator_saturations, 0)

    def test_fp16_like_path_preserves_shape_and_is_finite(self) -> None:
        output = predict_fp16_storage(self.model, self.signal)
        self.assertEqual(output.shape, self.signal.shape)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertLess(
            nmse_pooled_db(output, self.model.predict(self.signal)),
            -50.0,
        )

    def test_phase_equivariance_degradation_is_bounded_without_saturation(self) -> None:
        phase = np.exp(1j * 0.731)
        config = FixedPointConfig(
            input_bits=12,
            coefficient_bits=12,
            input_full_scale=1.0,
            accumulator_bits=32,
        )
        rotated = predict_fixed_point(self.model, self.signal * phase, config)
        reference = predict_fixed_point(self.model, self.signal, config)
        self.assertEqual(rotated.input_saturations, 0)
        self.assertEqual(reference.input_saturations, 0)
        self.assertLess(
            nmse_pooled_db(rotated.output, reference.output * phase),
            -50.0,
        )

    def test_fp16_rejects_collapsed_knot_table(self) -> None:
        model = ComplexLinearSplineDPD(
            knots=np.asarray([0.0, 0.5, 1.0, 1.00001]),
            coefficients=np.ones(4, dtype=complex),
        )
        with self.assertRaisesRegex(ValueError, "collapses"):
            predict_fp16_storage(model, np.asarray([0.7 + 0.0j]))

    def test_integer_path_rejects_unrepresentable_knot_address(self) -> None:
        model = ComplexLinearSplineDPD(
            knots=np.asarray([0.0, 1e300, 2e300]),
            coefficients=np.ones(3, dtype=complex),
        )
        with self.assertRaisesRegex(ValueError, "int64"):
            predict_fixed_point(
                model,
                np.asarray([0.2 + 0.0j]),
                FixedPointConfig(input_bits=12, coefficient_bits=12),
            )

    def test_integer_path_rejects_interpolation_product_overflow(self) -> None:
        model = ComplexLinearSplineDPD(
            knots=np.asarray([0.0, 1e12, 2e12]),
            coefficients=np.ones(3, dtype=complex),
        )
        with self.assertRaisesRegex(ValueError, "interpolation scale"):
            predict_fixed_point(
                model,
                np.asarray([0.2 + 0.0j]),
                FixedPointConfig(
                    input_bits=12,
                    coefficient_bits=12,
                    interpolation_fraction_bits=24,
                ),
            )


if __name__ == "__main__":
    unittest.main()
