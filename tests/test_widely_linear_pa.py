import tempfile
from pathlib import Path
import unittest

import numpy as np

from baseline.widely_linear_pa import (
    WidelyLinearResidualCorrection,
    WidelyLinearStreamingState,
    fit_widely_linear_residual_correction,
    widely_linear_design_matrix,
    widely_linear_segmented_design_matrix,
)


class WidelyLinearInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(731)
        self.signal = (
            self.rng.normal(size=79) + 1j * self.rng.normal(size=79)
        )
        self.model = WidelyLinearResidualCorrection(
            delays=(0, 1, 3),
            coefficients=np.asarray(
                [0.08 + 0.03j, -0.02 + 0.01j, 0.01 - 0.04j]
            ),
        )

    def test_prediction_matches_explicit_conjugate_design(self) -> None:
        expected = (
            widely_linear_design_matrix(self.signal, self.model.delays)
            @ self.model.coefficients
        )
        np.testing.assert_allclose(
            self.model.predict(self.signal),
            expected,
            rtol=2e-14,
            atol=2e-14,
        )

    def test_causal_prefix_does_not_use_future_samples(self) -> None:
        modified = self.signal.copy()
        modified[25:] += 3.0 - 7.0j
        np.testing.assert_array_equal(
            self.model.predict(self.signal)[:25],
            self.model.predict(modified)[:25],
        )

    def test_continuous_chunks_match_full_record(self) -> None:
        expected = self.model.predict(self.signal)
        state = self.model.initial_state()
        chunks: list[np.ndarray] = []
        start = 0
        for length in (1, 7, 2, 31, 38):
            stop = start + length
            output, state = self.model.predict_chunk(
                self.signal[start:stop],
                state,
            )
            chunks.append(output)
            start = stop
        self.assertEqual(start, self.signal.size)
        np.testing.assert_allclose(
            np.concatenate(chunks),
            expected,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_array_equal(
            state.history,
            self.signal[-self.model.maximum_delay :],
        )

    def test_segment_reset_matches_independent_frames(self) -> None:
        segmented = self.model.predict_segments(self.signal, 17)
        expected = np.concatenate(
            [
                self.model.predict(self.signal[start : start + 17])
                for start in range(0, self.signal.size, 17)
            ]
        )
        np.testing.assert_allclose(segmented, expected)
        self.assertGreater(
            abs(segmented[17] - self.model.predict(self.signal)[17]),
            1e-6,
        )

    def test_component_is_conjugate_linear_not_phase_equivariant(self) -> None:
        rotation = np.exp(0.63j)
        rotated = self.model.predict(rotation * self.signal)
        np.testing.assert_allclose(
            rotated,
            np.conjugate(rotation) * self.model.predict(self.signal),
            rtol=2e-14,
            atol=2e-14,
        )
        self.assertFalse(
            np.allclose(
                rotated,
                rotation * self.model.predict(self.signal),
            )
        )

    def test_save_load_and_complex64_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wl.npz"
            self.model.save(path)
            restored = WidelyLinearResidualCorrection.load(path)
        self.assertEqual(restored.delays, self.model.delays)
        np.testing.assert_array_equal(
            restored.coefficients,
            self.model.coefficients,
        )
        self.assertEqual(
            restored.predict(self.signal.astype(np.complex64)).dtype,
            np.complex64,
        )

    def test_operation_count_can_reuse_gmp_input_state(self) -> None:
        cost = self.model.operation_count(reuse_input_delay_state=True)
        self.assertEqual(cost.real_multiplications, 12)
        self.assertEqual(cost.real_additions, 12)
        self.assertEqual(cost.stored_real_coefficients, 6)
        self.assertEqual(cost.state_real_values, 0)

    def test_rejects_invalid_state_and_coefficients(self) -> None:
        with self.assertRaisesRegex(ValueError, "one complex value"):
            WidelyLinearResidualCorrection((0, 1), np.ones(1))
        with self.assertRaisesRegex(ValueError, "history length"):
            self.model.predict_chunk(
                self.signal,
                WidelyLinearStreamingState(np.zeros(1)),
            )


class WidelyLinearFitTests(unittest.TestCase):
    def test_complex_fit_recovers_segmented_residual_correction(self) -> None:
        rng = np.random.default_rng(947)
        signal = rng.normal(size=1200) + 1j * rng.normal(size=1200)
        truth = WidelyLinearResidualCorrection(
            delays=(0, 1, 2, 4),
            coefficients=np.asarray(
                [
                    0.07 + 0.01j,
                    -0.04 + 0.02j,
                    0.01 - 0.03j,
                    0.005 + 0.004j,
                ]
            ),
        )
        target = truth.predict_segments(signal, 300)
        fitted, diagnostics = fit_widely_linear_residual_correction(
            signal,
            target,
            delays=truth.delays,
            ridge=0.0,
            segment_length=300,
        )
        self.assertEqual(diagnostics.solver_rank, truth.tap_count)
        self.assertLess(diagnostics.training_correction_nmse_db, -250.0)
        np.testing.assert_allclose(
            fitted.coefficients,
            truth.coefficients,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_two_stage_correction_reconstructs_base_plus_residual(self) -> None:
        rng = np.random.default_rng(1013)
        signal = rng.normal(size=512) + 1j * rng.normal(size=512)
        base = (1.7 - 0.2j) * signal
        truth = WidelyLinearResidualCorrection(
            delays=(0, 1),
            coefficients=np.asarray([0.05 + 0.02j, -0.01 + 0.03j]),
        )
        measured = base + truth.predict_segments(signal, 128)
        fitted, _ = fit_widely_linear_residual_correction(
            signal,
            measured - base,
            delays=(0, 1),
            ridge=0.0,
            segment_length=128,
        )
        np.testing.assert_allclose(
            fitted.correct_segments(signal, base, 128),
            measured,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_complex64_coefficient_storage_is_not_silently_promoted(self) -> None:
        rng = np.random.default_rng(1091)
        signal = rng.normal(size=256) + 1j * rng.normal(size=256)
        truth = WidelyLinearResidualCorrection(
            (0,),
            np.asarray([0.03 - 0.01j], dtype=np.complex64),
        )
        target = truth.predict_segments(signal, 64)
        fitted, diagnostics = fit_widely_linear_residual_correction(
            signal,
            target,
            delays=(0,),
            ridge=0.0,
            segment_length=64,
            coefficient_dtype=np.complex64,
        )
        self.assertEqual(fitted.coefficients.dtype, np.dtype(np.complex64))
        self.assertEqual(diagnostics.coefficient_dtype, "complex64")

    def test_segmented_design_resets_each_frame(self) -> None:
        signal = np.arange(1, 9, dtype=np.float64).astype(np.complex128)
        design = widely_linear_segmented_design_matrix(
            signal,
            delays=(0, 1),
            segment_length=4,
        )
        self.assertEqual(design[0, 1], 0.0)
        self.assertEqual(design[4, 1], 0.0)
        self.assertEqual(design[5, 1], np.conjugate(signal[4]))

    def test_fit_rejects_zero_target_and_noncomplex_coefficients(self) -> None:
        signal = np.ones(32, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "non-zero power"):
            fit_widely_linear_residual_correction(
                signal,
                np.zeros_like(signal),
                delays=(0,),
                segment_length=16,
            )
        with self.assertRaisesRegex(TypeError, "complex dtype"):
            fit_widely_linear_residual_correction(
                signal,
                signal,
                delays=(0,),
                segment_length=16,
                coefficient_dtype=np.float64,
            )


if __name__ == "__main__":
    unittest.main()
