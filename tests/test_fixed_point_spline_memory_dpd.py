import unittest

import numpy as np

from baseline.fixed_point_pa import FixedPointFormat
from baseline.fixed_point_spline_memory_dpd import (
    FixedPointDPDConfig,
    FixedPointDPDState,
    FixedPointSparseSplineMemoryDPD,
)
from baseline.metrics import nmse_pooled_db
from baseline.sparse_spline_memory_pa import SparseSplineMemoryPA
from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
)


class FixedPointSplineMemoryDPDTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SparseSplineMemoryDPD(
            knots=np.asarray([0.0, 0.08, 0.20, 0.42, 0.70]),
            branches=(
                SplineMemoryBranch(0, 0),
                SplineMemoryBranch(1, 0),
                SplineMemoryBranch(2, 1),
            ),
            coefficients=np.asarray(
                [
                    [
                        1.02 - 0.01j,
                        1.04 - 0.02j,
                        1.08 - 0.04j,
                        1.14 - 0.07j,
                        1.22 - 0.10j,
                    ],
                    [
                        0.04 + 0.01j,
                        0.03 + 0.01j,
                        0.02 + 0.02j,
                        0.01 + 0.01j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.02 - 0.01j,
                        0.01 - 0.01j,
                        0.01 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                ],
                dtype=np.complex128,
            ),
            knot_strategy="explicit_test",
        )
        rng = np.random.default_rng(23)
        self.signal = (
            rng.normal(size=257) + 1j * rng.normal(size=257)
        ) * 0.18
        self.config = FixedPointDPDConfig.for_activation_bits(16)
        self.fixed = FixedPointSparseSplineMemoryDPD(
            self.model,
            self.config,
        )

    def test_tracks_float_dpd_and_preserves_direction(self) -> None:
        result = self.fixed.predict_chunk(self.signal)
        reference = self.model.predict(self.signal)
        self.assertLess(nmse_pooled_db(result.output, reference), -50.0)
        self.assertEqual(result.stats.input_saturations, 0)
        self.assertEqual(result.stats.output_saturations, 0)
        self.assertEqual(result.stats.accumulator_saturations, 0)
        self.assertEqual(
            self.fixed.metadata["direction"],
            "desired_input_to_predistorted_drive",
        )
        self.assertIs(self.fixed.model, self.model)

    def test_arbitrary_chunks_are_bit_identical_and_state_is_exact(self) -> None:
        full = self.fixed.predict_chunk(self.signal)
        state = self.fixed.initial_state()
        self.assertEqual(state.size, self.model.maximum_delay)
        outputs: list[np.ndarray] = []
        for start, stop in ((0, 73), (73, 149), (149, 211), (211, 257)):
            chunk = self.fixed.predict_chunk(self.signal[start:stop], state)
            outputs.append(chunk.output)
            state = chunk.next_state
            self.assertEqual(state.size, self.model.maximum_delay)
        np.testing.assert_array_equal(np.concatenate(outputs), full.output)
        np.testing.assert_array_equal(
            state.real_codes,
            full.next_state.real_codes,
        )
        np.testing.assert_array_equal(
            state.imag_codes,
            full.next_state.imag_codes,
        )

        short_state = FixedPointDPDState(
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "exactly maximum_delay"):
            self.fixed.predict_chunk(self.signal[:8], short_state)

    def test_segment_reset_matches_independent_frames(self) -> None:
        segmented = self.fixed.predict_segments(self.signal, 31)
        independent = np.concatenate(
            [
                self.fixed.predict(self.signal[start : start + 31])
                for start in range(0, self.signal.size, 31)
            ]
        )
        np.testing.assert_array_equal(segmented, independent)
        self.assertGreater(
            float(np.max(np.abs(segmented - self.fixed.predict(self.signal)))),
            0.0,
        )

    def test_future_input_cannot_change_past_predistorted_drive(self) -> None:
        changed = self.signal.copy()
        changed[101:] = (0.7 - 0.4j) * np.exp(
            1j * np.linspace(0.0, 4.0, changed.size - 101)
        )
        original_output = self.fixed.predict(self.signal)
        changed_output = self.fixed.predict(changed)
        np.testing.assert_array_equal(
            original_output[:101],
            changed_output[:101],
        )

    def test_pa_model_and_invalid_config_are_rejected(self) -> None:
        pa_model = SparseSplineMemoryPA(
            knots=self.model.knots,
            branches=self.model.branches,
            coefficients=self.model.coefficients,
        )
        with self.assertRaisesRegex(TypeError, "SparseSplineMemoryDPD"):
            FixedPointSparseSplineMemoryDPD(pa_model, self.config)
        with self.assertRaisesRegex(TypeError, "FixedPointDPDConfig"):
            FixedPointSparseSplineMemoryDPD(self.model, object())

    def test_collision_and_saturation_diagnostics_are_exposed(self) -> None:
        collision_model = SparseSplineMemoryDPD(
            knots=np.asarray([0.0, 1e-5, 2e-5, 0.5]),
            branches=(SplineMemoryBranch(0, 0),),
            coefficients=np.asarray([[1.0 + 0j] * 4]),
        )
        collision = FixedPointSparseSplineMemoryDPD(
            collision_model,
            FixedPointDPDConfig.for_activation_bits(12),
        ).predict_chunk(self.signal[:11])
        self.assertGreater(collision.stats.knot_code_collision_count, 0)
        self.assertGreaterEqual(collision.stats.maximum_knot_code_shift, 1)

        narrow = FixedPointDPDConfig(
            input_format=FixedPointFormat(12, 8, label="input"),
            coefficient_format=FixedPointFormat(
                12,
                8,
                label="coefficient",
            ),
            power_format=FixedPointFormat(32, 8, label="power"),
            accumulator_bits=12,
            scalar_accumulator_bits=12,
        )
        large = SparseSplineMemoryDPD(
            knots=self.model.knots,
            branches=(SplineMemoryBranch(0, 0),),
            coefficients=self.model.coefficients[:1] * 16.0,
        )
        saturated = FixedPointSparseSplineMemoryDPD(
            large,
            narrow,
        ).predict_chunk(self.signal)
        self.assertGreater(saturated.stats.coefficient_saturations, 0)
        self.assertGreater(saturated.stats.accumulator_saturations, 0)

    def test_selected_topology_fixed_schedule_is_exact(self) -> None:
        for knot_count, comparisons, coefficients in (
            (24, 5, 144),
            (8, 3, 48),
        ):
            model = SparseSplineMemoryDPD(
                knots=np.linspace(0.0, 1.0, knot_count),
                branches=(
                    SplineMemoryBranch(0, 0),
                    SplineMemoryBranch(1, 0),
                    SplineMemoryBranch(2, 0),
                ),
                coefficients=np.ones(
                    (3, knot_count),
                    dtype=np.complex128,
                ),
            )
            cost = FixedPointSparseSplineMemoryDPD(
                model,
                self.config,
            ).operation_count()
            self.assertEqual(cost.real_multiplications, 20)
            self.assertEqual(cost.real_additions, 25)
            self.assertEqual(cost.real_divisions, 1)
            self.assertEqual(cost.nonlinear_operations, 1)
            self.assertEqual(cost.comparisons, comparisons)
            self.assertEqual(cost.lookups, 8)
            self.assertEqual(cost.real_memory_reads, 28)
            self.assertEqual(cost.real_memory_writes, 2)
            self.assertEqual(cost.stored_real_coefficients, coefficients)
            self.assertEqual(cost.state_real_values, 4)


if __name__ == "__main__":
    unittest.main()
