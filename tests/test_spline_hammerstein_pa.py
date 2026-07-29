import tempfile
from pathlib import Path
import unittest

import numpy as np

from baseline.spline_hammerstein_pa import (
    SplineHammersteinPA,
    SplineHammersteinState,
    fit_spline_hammerstein_pa,
    make_sph_knots,
    _segmented_delay_rows,
    sph_filtered_control_design_matrix,
    sph_fir_tail_design_matrix,
    sph_spline_design_matrix,
)


class SplineHammersteinInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(1709)
        self.signal = (
            0.35 * self.rng.normal(size=173)
            + 0.35j * self.rng.normal(size=173)
        )
        self.model = SplineHammersteinPA(
            knots=np.asarray([0.0, 0.2, 0.55, 1.2]),
            control_points=np.asarray(
                [
                    1.3 + 0.1j,
                    1.2 + 0.04j,
                    1.0 - 0.08j,
                    0.72 - 0.2j,
                ]
            ),
            fir_tail=np.asarray([0.08 + 0.03j, -0.025 + 0.01j, 0.01j]),
            coordinate="amplitude",
            knot_strategy="explicit_test",
        )

    def _explicit_prediction(self, signal: np.ndarray) -> np.ndarray:
        nonlinear = self.model.nonlinear_output(signal)
        output = nonlinear.copy()
        for delay, coefficient in enumerate(self.model.fir_tail, start=1):
            output[delay:] += coefficient * nonlinear[:-delay]
        return output

    def test_prediction_matches_equation_and_is_phase_equivariant(self) -> None:
        expected = self._explicit_prediction(self.signal)
        np.testing.assert_allclose(
            self.model.predict(self.signal),
            expected,
            rtol=2e-14,
            atol=2e-14,
        )
        rotation = np.exp(0.617j)
        np.testing.assert_allclose(
            self.model.predict(rotation * self.signal),
            rotation * expected,
            rtol=3e-14,
            atol=3e-14,
        )

    def test_power_coordinate_uses_distinct_basis_without_sqrt_contract(
        self,
    ) -> None:
        power_model = SplineHammersteinPA(
            knots=np.asarray([0.0, 0.09, 0.36, 1.44]),
            control_points=self.model.control_points,
            fir_tail=np.asarray([], dtype=np.complex128),
            coordinate="power",
        )
        expected_power = self.signal.real**2 + self.signal.imag**2
        np.testing.assert_array_equal(
            power_model.coordinate_values(self.signal),
            expected_power,
        )
        self.assertEqual(power_model.operation_count().nonlinear_operations, 0)
        self.assertEqual(self.model.operation_count().nonlinear_operations, 1)

    def test_causal_prefix_and_arbitrary_chunk_streaming_are_exact(self) -> None:
        modified = self.signal.copy()
        modified[91:] += 2.0 - 1.5j
        np.testing.assert_array_equal(
            self.model.predict(self.signal)[:91],
            self.model.predict(modified)[:91],
        )

        expected = self.model.predict(self.signal)
        state = self.model.initial_state()
        chunks: list[np.ndarray] = []
        start = 0
        for length in (1, 2, 19, 3, 71, 77):
            stop = start + length
            output, state = self.model.predict_chunk(
                self.signal[start:stop],
                state,
            )
            chunks.append(output)
            start = stop
        self.assertEqual(start, self.signal.size)
        np.testing.assert_array_equal(np.concatenate(chunks), expected)
        np.testing.assert_array_equal(
            state.history,
            self.model.nonlinear_output(self.signal)[-3:],
        )

    def test_segment_reset_matches_independent_partial_frames(self) -> None:
        segmented = self.model.predict_segments(self.signal, 48)
        expected = np.concatenate(
            [
                self.model.predict(self.signal[start : start + 48])
                for start in range(0, self.signal.size, 48)
            ]
        )
        np.testing.assert_array_equal(segmented, expected)
        self.assertGreater(
            abs(segmented[48] - self.model.predict(self.signal)[48]),
            1e-7,
        )

    def test_save_load_dtype_metadata_and_cost(self) -> None:
        model = SplineHammersteinPA(
            knots=self.model.knots,
            control_points=self.model.control_points.astype(np.complex64),
            fir_tail=self.model.fir_tail.astype(np.complex64),
            coordinate="amplitude",
            knot_strategy="amplitude_uniform",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sph_pa.npz"
            model.save(path)
            restored = SplineHammersteinPA.load(path)
        np.testing.assert_array_equal(restored.knots, model.knots)
        np.testing.assert_array_equal(restored.control_points, model.control_points)
        np.testing.assert_array_equal(restored.fir_tail, model.fir_tail)
        self.assertEqual(restored.control_points.dtype, np.dtype(np.complex64))
        self.assertEqual(restored.fir_tail.dtype, np.dtype(np.complex64))
        self.assertEqual(restored.coordinate, "amplitude")
        self.assertEqual(restored.fir_length, 4)
        self.assertEqual(restored.stored_real_coefficients, 14)
        self.assertEqual(restored.metadata["h0"], "1+0j fixed and not stored")
        self.assertEqual(
            restored.predict(self.signal.astype(np.complex64)).dtype,
            np.dtype(np.complex64),
        )
        cost = restored.operation_count()
        self.assertEqual(cost.real_multiplications, 21)
        self.assertEqual(cost.real_additions, 20)
        self.assertEqual(cost.stored_real_coefficients, 14)

    def test_invalid_coefficients_coordinate_and_state_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one complex control"):
            SplineHammersteinPA(
                self.model.knots,
                np.ones(3),
                np.asarray([], dtype=complex),
            )
        with self.assertRaisesRegex(ValueError, "coordinate"):
            SplineHammersteinPA(
                self.model.knots,
                self.model.control_points,
                self.model.fir_tail,
                coordinate="radius",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "history length"):
            self.model.predict_chunk(
                self.signal,
                SplineHammersteinState(np.zeros(1)),
            )
        with self.assertRaisesRegex(TypeError, "SplineHammersteinState"):
            self.model.predict_chunk(self.signal, object())  # type: ignore[arg-type]


class SplineHammersteinDesignTests(unittest.TestCase):
    def test_all_preregistered_knot_variants_are_exact(self) -> None:
        signal = np.linspace(0.01, 1.0, 1001).astype(np.complex128)
        unit = np.linspace(0.0, 1.0, 5)
        expected = {
            "amplitude_uniform": (unit, "amplitude"),
            "amplitude_uniform_power_placement": (
                np.sqrt(unit),
                "amplitude",
            ),
            "amplitude_compression_aware_p2": (
                1.0 - np.square(1.0 - unit),
                "amplitude",
            ),
            "power_uniform": (unit, "power"),
        }
        for variant, (expected_knots, expected_coordinate) in expected.items():
            with self.subTest(variant=variant):
                knots, coordinate = make_sph_knots(signal, 5, variant)
                np.testing.assert_allclose(knots, expected_knots, atol=1e-15)
                self.assertEqual(coordinate, expected_coordinate)

        quantile, coordinate = make_sph_knots(
            signal,
            5,
            "amplitude_quantile",
        )
        np.testing.assert_allclose(quantile, np.asarray([0.0, 0.2575, 0.505, 0.7525, 1.0]))
        self.assertEqual(coordinate, "amplitude")

    def test_quantile_duplicates_do_not_silently_change_k(self) -> None:
        signal = np.asarray([0.0] * 20 + [0.5] * 20 + [1.0] * 20, complex)
        with self.assertRaisesRegex(ValueError, "duplicate knots"):
            make_sph_knots(signal, 16, "amplitude_quantile")

    def test_spline_design_has_two_local_features_and_partition(self) -> None:
        signal = np.asarray(
            [0.05 + 0.02j, 0.2 - 0.1j, -0.4 + 0.2j, 0.9j]
        )
        knots = np.asarray([0.0, 0.2, 0.5, 1.0])
        design = sph_spline_design_matrix(signal, knots, "amplitude")
        basis = design / signal[:, None]
        np.testing.assert_allclose(np.sum(basis, axis=1), 1.0, atol=1e-14)
        self.assertTrue(np.all(np.count_nonzero(basis, axis=1) <= 2))

    def test_training_design_matches_segmented_inference(self) -> None:
        rng = np.random.default_rng(1733)
        signal = 0.3 * (
            rng.normal(size=131) + 1j * rng.normal(size=131)
        )
        model = SplineHammersteinPA(
            knots=np.asarray([0.0, 0.2, 0.5, 1.2]),
            control_points=np.asarray(
                [1.2 + 0.1j, 1.1, 0.9 - 0.1j, 0.7 - 0.2j]
            ),
            fir_tail=np.asarray([0.07 + 0.02j, -0.03 + 0.01j]),
        )
        spline_design = sph_spline_design_matrix(
            signal,
            model.knots,
            model.coordinate,
        )
        filtered = sph_filtered_control_design_matrix(
            spline_design,
            model.fir_tail,
            segment_length=48,
        )
        nonlinear = spline_design @ model.control_points
        tail_design = sph_fir_tail_design_matrix(
            nonlinear,
            model.fir_length,
            segment_length=48,
        )
        expected = model.predict_segments(signal, 48)
        np.testing.assert_allclose(
            filtered @ model.control_points,
            expected,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            nonlinear + tail_design @ model.fir_tail,
            expected,
            rtol=2e-14,
            atol=2e-14,
        )

    def test_design_rejects_memory_that_crosses_every_frame(self) -> None:
        nonlinear = np.ones(32, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            sph_fir_tail_design_matrix(
                nonlinear,
                17,
                segment_length=16,
            )
        with self.assertRaisesRegex(ValueError, "shorter"):
            sph_filtered_control_design_matrix(
                np.ones((32, 3), dtype=np.complex128),
                np.ones(16, dtype=np.complex128),
                segment_length=16,
            )
        with self.assertRaisesRegex(TypeError, "delay must be an integer"):
            _segmented_delay_rows(nonlinear, True, segment_length=16)
        with self.assertRaisesRegex(ValueError, "causal"):
            _segmented_delay_rows(nonlinear, -1, segment_length=16)


class SplineHammersteinFitTests(unittest.TestCase):
    def test_memoryless_complex_fit_recovers_known_control_points(self) -> None:
        rng = np.random.default_rng(1777)
        radius = rng.uniform(0.01, 1.0, size=2048)
        phase = rng.uniform(-np.pi, np.pi, size=2048)
        signal = radius * np.exp(1j * phase)
        truth = SplineHammersteinPA(
            knots=np.asarray([0.0, 0.2, 0.45, 0.7, 1.0]),
            control_points=np.asarray(
                [1.4 + 0.1j, 1.3 + 0.03j, 1.1 - 0.08j, 0.9 - 0.16j, 0.7 - 0.22j]
            ),
            fir_tail=np.asarray([], dtype=np.complex128),
        )
        target = truth.predict_segments(signal, 512)
        fitted, diagnostics = fit_spline_hammerstein_pa(
            signal,
            target,
            knots=truth.knots,
            coordinate="amplitude",
            fir_length=1,
            segment_length=512,
            control_ridge=0.0,
            smoothness=0.0,
            fir_ridge=0.0,
            maximum_alternations=4,
            minimum_alternations=2,
            convergence_tolerance=1e-12,
        )
        np.testing.assert_allclose(
            fitted.control_points,
            truth.control_points,
            rtol=3e-14,
            atol=3e-14,
        )
        self.assertEqual(fitted.fir_length, 1)
        self.assertTrue(diagnostics.converged)
        self.assertEqual(diagnostics.completed_alternations, 2)
        self.assertTrue(diagnostics.all_updates_monotonic)
        self.assertTrue(diagnostics.all_data_designs_full_column_rank)
        self.assertLess(diagnostics.training_nmse_db, -260.0)

    def test_alternating_fit_recovers_nonzero_short_fir_mapping(self) -> None:
        rng = np.random.default_rng(1783)
        radius = rng.uniform(0.02, 1.0, size=4096)
        phase = rng.uniform(-np.pi, np.pi, size=4096)
        signal = radius * np.exp(1j * phase)
        truth = SplineHammersteinPA(
            knots=np.asarray([0.0, 0.18, 0.42, 0.68, 1.0]),
            control_points=np.asarray(
                [1.45 + 0.08j, 1.32, 1.12 - 0.07j, 0.91 - 0.15j, 0.69 - 0.22j]
            ),
            fir_tail=np.asarray([0.07 + 0.025j, -0.025 + 0.012j]),
        )
        target = truth.predict_segments(signal, 512)
        fitted, diagnostics = fit_spline_hammerstein_pa(
            signal,
            target,
            knots=truth.knots,
            coordinate="amplitude",
            fir_length=truth.fir_length,
            segment_length=512,
            control_ridge=0.0,
            smoothness=0.0,
            fir_ridge=0.0,
            maximum_alternations=30,
            minimum_alternations=2,
            convergence_tolerance=1e-12,
        )
        self.assertTrue(diagnostics.all_updates_monotonic)
        self.assertTrue(diagnostics.all_data_designs_full_column_rank)
        self.assertLess(
            diagnostics.optimization_final_objective,
            diagnostics.memoryless_initial_objective * 1e-8,
        )
        self.assertLess(diagnostics.training_nmse_db, -100.0)
        np.testing.assert_allclose(
            fitted.predict_segments(signal, 512),
            target,
            rtol=2e-5,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            fitted.fir_tail,
            truth.fir_tail,
            rtol=2e-5,
            atol=2e-7,
        )

    def test_fit_is_deterministic_and_records_regularized_objective(self) -> None:
        rng = np.random.default_rng(1789)
        signal = 0.4 * (
            rng.normal(size=1024) + 1j * rng.normal(size=1024)
        )
        target = (1.2 - 0.1j) * signal + 0.01 * (
            rng.normal(size=1024) + 1j * rng.normal(size=1024)
        )
        kwargs = dict(
            knot_count=8,
            knot_variant="power_uniform",
            fir_length=2,
            segment_length=256,
            control_ridge=1e-8,
            smoothness=1e-6,
            fir_ridge=1e-8,
            maximum_alternations=5,
            minimum_alternations=2,
        )
        first, first_diagnostics = fit_spline_hammerstein_pa(
            signal,
            target,
            **kwargs,
        )
        second, second_diagnostics = fit_spline_hammerstein_pa(
            signal,
            target,
            **kwargs,
        )
        np.testing.assert_array_equal(first.knots, second.knots)
        np.testing.assert_array_equal(
            first.control_points,
            second.control_points,
        )
        np.testing.assert_array_equal(first.fir_tail, second.fir_tail)
        self.assertEqual(first.coordinate, "power")
        self.assertEqual(
            first_diagnostics.optimization_final_objective,
            second_diagnostics.optimization_final_objective,
        )
        self.assertTrue(first_diagnostics.all_updates_monotonic)
        self.assertEqual(first_diagnostics.h0_contract, "1+0j fixed and not stored")

    def test_fit_rejects_ambiguous_or_invalid_protocol(self) -> None:
        signal = np.ones(64, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "coordinate is required"):
            fit_spline_hammerstein_pa(
                signal,
                signal,
                knots=np.asarray([0.0, 0.5, 1.0]),
                segment_length=32,
            )
        with self.assertRaisesRegex(ValueError, "non-zero power"):
            fit_spline_hammerstein_pa(
                signal,
                np.zeros_like(signal),
                knot_count=4,
                segment_length=32,
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            fit_spline_hammerstein_pa(
                signal,
                signal,
                knot_count=4,
                fir_length=33,
                segment_length=32,
            )
        with self.assertRaisesRegex(TypeError, "complex dtype"):
            fit_spline_hammerstein_pa(
                signal,
                signal,
                knot_count=4,
                segment_length=32,
                coefficient_dtype=np.float64,
            )


if __name__ == "__main__":
    unittest.main()
