import tempfile
from pathlib import Path
import unittest

import numpy as np

from baseline.complex_fir_pa import (
    ComplexFIRResidualCorrection,
    ComplexFIRStreamingState,
    complex_fir_design_matrix,
    complex_fir_segmented_design_matrix,
    fit_complex_fir_residual_correction,
)


class ComplexFIRInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(1409)
        self.signal = self.rng.normal(size=157) + 1j * self.rng.normal(size=157)
        self.model = ComplexFIRResidualCorrection(
            delays=(0, 3, 11),
            coefficients=np.asarray(
                [0.08 + 0.03j, -0.02 + 0.01j, 0.01 - 0.04j]
            ),
        )

    def test_prediction_matches_explicit_design_and_is_phase_equivariant(
        self,
    ) -> None:
        expected = (
            complex_fir_design_matrix(self.signal, self.model.delays)
            @ self.model.coefficients
        )
        np.testing.assert_allclose(
            self.model.predict(self.signal),
            expected,
            rtol=2e-14,
            atol=2e-14,
        )
        rotation = np.exp(0.71j)
        np.testing.assert_allclose(
            self.model.predict(rotation * self.signal),
            rotation * expected,
            rtol=2e-14,
            atol=2e-14,
        )

    def test_causal_prefix_and_arbitrary_chunks(self) -> None:
        modified = self.signal.copy()
        modified[80:] += 4.0 - 6.0j
        np.testing.assert_array_equal(
            self.model.predict(self.signal)[:80],
            self.model.predict(modified)[:80],
        )

        expected = self.model.predict(self.signal)
        state = self.model.initial_state()
        chunks: list[np.ndarray] = []
        start = 0
        for length in (1, 9, 2, 41, 104):
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

    def test_segment_reset_matches_explicit_partial_frames(self) -> None:
        segmented = self.model.predict_segments(self.signal, 48)
        expected = np.concatenate(
            [
                self.model.predict(self.signal[start : start + 48])
                for start in range(0, self.signal.size, 48)
            ]
        )
        np.testing.assert_allclose(segmented, expected)
        self.assertGreater(
            abs(segmented[48] - self.model.predict(self.signal)[48]),
            1e-6,
        )
        design = complex_fir_segmented_design_matrix(
            self.signal,
            self.model.delays,
            segment_length=48,
        )
        np.testing.assert_allclose(segmented, design @ self.model.coefficients)

    def test_save_load_dtype_and_integrated_operation_count(self) -> None:
        model = ComplexFIRResidualCorrection(
            (42, 43, 44, 45, 46, 47, 48, 49),
            np.ones(8, dtype=np.complex64),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proper_fir.npz"
            model.save(path)
            restored = ComplexFIRResidualCorrection.load(path)
        self.assertEqual(restored.delays, model.delays)
        self.assertEqual(restored.coefficients.dtype, np.dtype(np.complex64))
        self.assertEqual(
            restored.predict(self.signal.astype(np.complex64)).dtype,
            np.dtype(np.complex64),
        )
        cost = restored.operation_count(existing_maximum_input_delay=29)
        self.assertEqual(cost.real_multiplications, 32)
        self.assertEqual(cost.real_additions, 32)
        self.assertEqual(cost.stored_real_coefficients, 16)
        self.assertEqual(cost.state_real_values, 40)

    def test_rejects_invalid_state_and_coefficients(self) -> None:
        with self.assertRaisesRegex(ValueError, "one complex value"):
            ComplexFIRResidualCorrection((0, 1), np.ones(1))
        with self.assertRaisesRegex(ValueError, "history length"):
            self.model.predict_chunk(
                self.signal,
                ComplexFIRStreamingState(np.zeros(1)),
            )


class ComplexFIRFitTests(unittest.TestCase):
    def test_complex_fit_recovers_long_delay_correction(self) -> None:
        rng = np.random.default_rng(1597)
        signal = rng.normal(size=4096) + 1j * rng.normal(size=4096)
        truth = ComplexFIRResidualCorrection(
            delays=(42, 44, 45, 46, 49),
            coefficients=np.asarray(
                [
                    0.007 + 0.001j,
                    -0.004 + 0.002j,
                    0.003 - 0.003j,
                    0.002 + 0.001j,
                    -0.001 + 0.002j,
                ]
            ),
        )
        target = truth.predict_segments(signal, 1024)
        fitted, diagnostics = fit_complex_fir_residual_correction(
            signal,
            target,
            delays=truth.delays,
            ridge=0.0,
            segment_length=1024,
        )
        self.assertEqual(diagnostics.solver_rank, truth.tap_count)
        self.assertLess(diagnostics.training_correction_nmse_db, -250.0)
        np.testing.assert_allclose(
            fitted.coefficients,
            truth.coefficients,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_two_stage_correction_and_complex64_storage(self) -> None:
        rng = np.random.default_rng(1601)
        signal = rng.normal(size=1024) + 1j * rng.normal(size=1024)
        base = (1.7 - 0.2j) * signal
        truth = ComplexFIRResidualCorrection(
            delays=(0, 9),
            coefficients=np.asarray([0.05 + 0.02j, -0.01 + 0.03j]),
        )
        measured = base + truth.predict_segments(signal, 256)
        fitted, diagnostics = fit_complex_fir_residual_correction(
            signal,
            measured - base,
            delays=truth.delays,
            ridge=0.0,
            segment_length=256,
            coefficient_dtype=np.complex64,
        )
        self.assertEqual(fitted.coefficients.dtype, np.dtype(np.complex64))
        self.assertEqual(diagnostics.coefficient_dtype, "complex64")
        np.testing.assert_allclose(
            fitted.correct_segments(signal, base, 256),
            measured,
            rtol=2e-7,
            atol=2e-7,
        )

    def test_fit_rejects_invalid_target_and_memory(self) -> None:
        signal = np.ones(64, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "non-zero power"):
            fit_complex_fir_residual_correction(
                signal,
                np.zeros_like(signal),
                delays=(0,),
                segment_length=32,
            )
        with self.assertRaisesRegex(ValueError, "shorter than each frame"):
            fit_complex_fir_residual_correction(
                signal,
                signal,
                delays=(32,),
                segment_length=32,
            )
        with self.assertRaisesRegex(TypeError, "complex dtype"):
            fit_complex_fir_residual_correction(
                signal,
                signal,
                delays=(0,),
                segment_length=32,
                coefficient_dtype=np.float64,
            )


if __name__ == "__main__":
    unittest.main()
