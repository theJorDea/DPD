import unittest

import numpy as np

from baseline.fixed_point_pa import FixedPointFormat, FixedPointPAConfig
from baseline.fixed_point_sparse_spline_pa import (
    FixedPointSparseSplineMemoryPA,
)
from baseline.metrics import nmse_pooled_db
from baseline.sparse_spline_memory_pa import (
    SparseSplineMemoryPA,
    SparseSplineMemoryPABranch,
)


class FixedPointSparseSplinePATests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SparseSplineMemoryPA(
            knots=np.asarray([0.0, 0.08, 0.20, 0.42, 0.70]),
            branches=(
                SparseSplineMemoryPABranch(0, 0),
                SparseSplineMemoryPABranch(1, 0),
                SparseSplineMemoryPABranch(2, 1),
            ),
            coefficients=np.asarray(
                [
                    [0.98 + 0.02j, 0.96 + 0.03j, 0.91 + 0.06j, 0.84 + 0.10j, 0.78 + 0.13j],
                    [0.06 - 0.01j, 0.04 - 0.02j, 0.02 - 0.02j, 0.01 - 0.01j, 0.0 + 0.0j],
                    [0.03 + 0.01j, 0.02 + 0.01j, 0.01 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
                dtype=np.complex128,
            ),
        )
        rng = np.random.default_rng(17)
        self.signal = (
            rng.normal(size=257) + 1j * rng.normal(size=257)
        ) * 0.18
        self.config = FixedPointPAConfig.for_activation_bits(
            16,
            accumulator_bits=56,
            scalar_accumulator_bits=56,
        )
        self.fixed = FixedPointSparseSplineMemoryPA(
            self.model,
            self.config,
        )

    def test_tracks_float_forward_model_without_saturation(self) -> None:
        result = self.fixed.predict_chunk(self.signal)
        reference = self.model.predict(self.signal)
        self.assertEqual(result.output.shape, self.signal.shape)
        self.assertEqual(result.stats.input_saturations, 0)
        self.assertEqual(result.stats.output_saturations, 0)
        self.assertEqual(result.stats.accumulator_saturations, 0)
        self.assertEqual(result.stats.knot_code_collision_count, 0)
        self.assertLess(nmse_pooled_db(result.output, reference), -50.0)

    def test_streaming_chunks_are_bit_identical(self) -> None:
        full = self.fixed.predict_chunk(self.signal)
        first = self.fixed.predict_chunk(self.signal[:73])
        second = self.fixed.predict_chunk(
            self.signal[73:149],
            first.next_state,
        )
        third = self.fixed.predict_chunk(
            self.signal[149:],
            second.next_state,
        )
        streamed = np.concatenate((first.output, second.output, third.output))
        np.testing.assert_array_equal(streamed, full.output)
        np.testing.assert_array_equal(
            third.next_state.real_codes,
            full.next_state.real_codes,
        )
        np.testing.assert_array_equal(
            third.next_state.imag_codes,
            full.next_state.imag_codes,
        )

    def test_segment_reset_matches_independent_records(self) -> None:
        segmented = self.fixed.predict_segments(self.signal, 31)
        independent = np.concatenate(
            [
                self.fixed.predict_chunk(
                    self.signal[start : start + 31]
                ).output
                for start in range(0, self.signal.size, 31)
            ]
        )
        np.testing.assert_array_equal(segmented, independent)
        continuous = self.fixed.predict(self.signal)
        self.assertGreater(float(np.max(np.abs(segmented - continuous))), 0.0)

    def test_knot_collision_is_explicitly_reported(self) -> None:
        model = SparseSplineMemoryPA(
            knots=np.asarray([0.0, 1e-5, 2e-5, 0.5]),
            branches=(SparseSplineMemoryPABranch(0, 0),),
            coefficients=np.asarray([[1.0 + 0j] * 4]),
        )
        fixed = FixedPointSparseSplineMemoryPA(
            model,
            FixedPointPAConfig.for_activation_bits(12),
        )
        result = fixed.predict_chunk(self.signal[:11])
        self.assertGreater(result.stats.knot_code_collision_count, 0)
        self.assertGreaterEqual(result.stats.maximum_knot_code_shift, 1)
        self.assertTrue(np.all(np.isfinite(result.output)))

    def test_saturation_is_reported(self) -> None:
        narrow = FixedPointPAConfig(
            input_format=FixedPointFormat(12, 8, label="input"),
            coefficient_format=FixedPointFormat(12, 8, label="coefficient"),
            power_format=FixedPointFormat(32, 8, label="power"),
            accumulator_bits=12,
            scalar_accumulator_bits=12,
        )
        large = SparseSplineMemoryPA(
            knots=self.model.knots,
            branches=(SparseSplineMemoryPABranch(0, 0),),
            coefficients=self.model.coefficients[:1] * 16.0,
        )
        result = FixedPointSparseSplineMemoryPA(
            large,
            narrow,
        ).predict_chunk(self.signal)
        self.assertGreater(result.stats.coefficient_saturations, 0)
        self.assertGreater(result.stats.accumulator_saturations, 0)

    def test_operation_count_is_explicit(self) -> None:
        cost = self.fixed.operation_count()
        self.assertEqual(cost.real_multiplications, 2 * 2 + 3 * 6)
        self.assertEqual(cost.real_divisions, 2)
        self.assertEqual(cost.nonlinear_operations, 2)
        self.assertEqual(cost.stored_real_coefficients, 2 * 3 * 5)
        self.assertEqual(cost.state_real_values, 4)

    def test_mismatched_amplitude_fractional_scales_are_rejected(self) -> None:
        config = FixedPointPAConfig(
            input_format=FixedPointFormat(12, 10),
            coefficient_format=FixedPointFormat(12, 8),
            power_format=FixedPointFormat(32, 9),
        )
        with self.assertRaisesRegex(ValueError, "same fractional scale"):
            FixedPointSparseSplineMemoryPA(self.model, config)


if __name__ == "__main__":
    unittest.main()
