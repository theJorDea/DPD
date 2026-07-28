"""Regression tests for the distinction between ILA reconstruction and DPD use."""

import unittest

import numpy as np

from baseline.complex_spline_dpd import fit_ila_postdistorter
from baseline.evaluate_spline import (
    desired_input_cascade_signals,
    inverse_postdistorter_signals,
)
from baseline.metrics import nmse_pooled_db
from baseline.pa_models import (
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
    memory_polynomial_design_matrix,
)


class DpdDirectionTests(unittest.TestCase):
    @staticmethod
    def _compressive_pa(signal: np.ndarray) -> np.ndarray:
        radius = np.abs(signal)
        return signal * (1.0 - 0.30 * radius**2)

    def test_circular_reconstruction_is_not_a_deployment_score(self) -> None:
        # The PA compresses its output range: |F(x)| <= 0.7 for |x| <= 1.
        # A postdistorter can reconstruct observed outputs very accurately,
        # while a deployment input near |x|=1 lies outside that inverse's
        # calibrated input range and is endpoint-clamped by the spline.
        train_radius = np.linspace(0.001, 1.0, 12000)
        test_radius = np.linspace(0.002, 1.0, 4000)
        train_phase = np.exp(1j * (0.173 * np.arange(train_radius.size)))
        test_phase = np.exp(1j * (0.271 * np.arange(test_radius.size) + 0.37))
        train_input = train_radius * train_phase
        test_input = test_radius * test_phase
        train_output = self._compressive_pa(train_input)
        test_output = self._compressive_pa(test_input)

        # Fit an explicitly supplied deterministic forward surrogate.  It is
        # exact here, but the test still exercises the real production API.
        pa_surrogate, pa_diagnostics = fit_memory_polynomial_pa(
            train_input,
            train_output,
            orders=(1, 3),
            delays=(0,),
            ridge=0.0,
        )
        self.assertEqual(
            pa_diagnostics.solver,
            "augmented_complex_lstsq",
        )
        self.assertIsNotNone(
            pa_diagnostics.augmented_design_condition_number
        )
        model, _ = fit_ila_postdistorter(
            train_input,
            train_output,
            1.0 + 0.0j,
            knot_count=64,
            knot_strategy="uniform_amplitude",
            ridge=1e-12,
        )

        inverse = inverse_postdistorter_signals(
            model,
            test_input,
            test_output,
            1.0 + 0.0j,
        )
        circular_output = pa_surrogate.predict(inverse["estimated_pa_input"])
        circular_nmse = nmse_pooled_db(circular_output, test_output)

        cascade = desired_input_cascade_signals(
            model,
            test_input,
            1.0 + 0.0j,
            pa_surrogate,
        )
        deployment_nmse = nmse_pooled_db(
            cascade["surrogate_pa_output"],
            cascade["ideal_output"],
        )

        # The observed-output reconstruction is excellent, but the desired-x
        # cascade is materially poor.  These margins make the test robust to
        # harmless BLAS/float-rounding differences.
        self.assertLess(circular_nmse, -50.0)
        self.assertGreater(deployment_nmse, -20.0)
        np.testing.assert_array_equal(cascade["dpd_input"], test_input)
        np.testing.assert_array_equal(
            inverse["postdistorter_input"],
            test_output,
        )

    def test_memory_polynomial_is_causal_and_zero_padded(self) -> None:
        signal = np.asarray([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])
        design = memory_polynomial_design_matrix(
            signal,
            orders=(1,),
            delays=(0, 1),
        )
        expected = np.asarray(
            [
                [1.0 + 0.0j, 0.0 + 0.0j],
                [2.0 + 0.0j, 1.0 + 0.0j],
                [3.0 + 0.0j, 2.0 + 0.0j],
            ]
        )
        np.testing.assert_array_equal(design, expected)
        model = MemoryPolynomialPA(
            orders=(1,),
            delays=(0, 1),
            coefficients=np.asarray([[1.0 + 0.0j], [0.5 + 0.0j]]),
        )
        np.testing.assert_allclose(
            model.predict(signal),
            np.asarray([1.0, 2.5, 4.0], dtype=np.complex128),
        )

    def test_segmented_prediction_resets_history_per_segment(self) -> None:
        signal = np.asarray(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            dtype=np.complex128,
        )
        model = MemoryPolynomialPA(
            orders=(1,),
            delays=(0, 1),
            coefficients=np.asarray([[1.0 + 0.0j], [0.5 + 0.0j]]),
        )
        segmented = model.predict_segments(signal, segment_length=3)
        expected = np.concatenate(
            (
                model.predict(signal[:3]),
                model.predict(signal[3:6]),
                model.predict(signal[6:]),
            )
        )
        np.testing.assert_array_equal(segmented, expected)
        # A continuous call would carry sample 3 into the second segment;
        # segmented evaluation must instead use zero history at index 3.
        continuous = model.predict(signal)
        self.assertNotEqual(segmented[3], continuous[3])
        self.assertEqual(
            model.metadata["segment_state_reset"],
            "zero_state_at_each_predict_segments_boundary",
        )


if __name__ == "__main__":
    unittest.main()
