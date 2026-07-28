import unittest

import numpy as np

from baseline.complexity import memory_polynomial_inference_cost
from baseline.pa_benchmark import (
    evaluate_pa_predictor,
    freeze_pa_evaluation_protocol,
    prepare_pa_split,
)


class PABenchmarkProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(12)
        self.train_x = (
            rng.normal(size=32) + 1j * rng.normal(size=32)
        ).astype(np.complex128)
        self.train_y = (1.7 - 0.25j) * self.train_x
        self.protocol = freeze_pa_evaluation_protocol(
            self.train_x,
            self.train_y,
            dataset_label="synthetic",
            sample_rate_hz=16.0,
            nperseg=8,
            main_bandwidth_hz=4.0,
            subchannel_count=2,
            alignment_max_abs_delay=3,
            characteristic_bins=8,
        )

    def test_protocol_freezes_training_only_gain_delay_and_bins(self) -> None:
        self.assertEqual(self.protocol.alignment_delay_samples, 0)
        self.assertAlmostEqual(
            self.protocol.training_complex_ls_gain,
            1.7 - 0.25j,
        )
        self.assertEqual(len(self.protocol.characteristic_bin_edges), 9)
        self.assertAlmostEqual(
            self.protocol.characteristic_bin_edges[-1],
            np.max(np.abs(self.train_x)),
        )
        self.assertFalse(self.protocol.fractional_delay_applied)

    def test_prepare_split_reuses_frozen_delay_without_refitting(self) -> None:
        x = np.arange(1, 9, dtype=float).astype(np.complex128)
        y = 9.0 * x
        x_aligned, y_aligned = prepare_pa_split(x, y, self.protocol)
        np.testing.assert_array_equal(x_aligned, x)
        np.testing.assert_array_equal(y_aligned, y)

    def test_test_split_cannot_be_used_for_model_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "test data"):
            evaluate_pa_predictor(
                lambda x: x,
                self.train_x,
                self.train_y,
                protocol=self.protocol,
                model_label="invalid",
                split="test",
                purpose="model_selection",
            )

    def test_nonzero_flattened_delay_requires_frame_safe_preprocessing(
        self,
    ) -> None:
        delayed = np.concatenate(
            (np.zeros(1, dtype=np.complex128), self.train_y[:-1])
        )
        protocol = freeze_pa_evaluation_protocol(
            self.train_x,
            delayed,
            dataset_label="delayed-synthetic",
            sample_rate_hz=16.0,
            nperseg=8,
            main_bandwidth_hz=4.0,
            subchannel_count=2,
            alignment_delay=1,
            alignment_max_abs_delay=3,
            characteristic_bins=8,
        )
        with self.assertRaisesRegex(NotImplementedError, "nperseg boundaries"):
            evaluate_pa_predictor(
                lambda x: x,
                self.train_x,
                delayed,
                protocol=protocol,
                model_label="blocked-until-frame-safe",
                split="validation",
                purpose="diagnostic",
            )


class PABenchmarkEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        phase = np.linspace(0.1, 2.7, 24)
        amplitude = np.linspace(0.1, 1.0, 24)
        self.x = amplitude * np.exp(1j * phase)
        self.gain = 2.0 + 0.5j
        self.y = self.gain * self.x
        self.protocol = freeze_pa_evaluation_protocol(
            self.x,
            self.y,
            dataset_label="synthetic-evaluation",
            sample_rate_hz=24.0,
            nperseg=8,
            main_bandwidth_hz=6.0,
            subchannel_count=2,
            alignment_delay=0,
            alignment_max_abs_delay=2,
            characteristic_bins=6,
        )

    def test_only_pa_input_is_passed_to_predictor_and_frames_reset(self) -> None:
        calls: list[np.ndarray] = []

        def predictor(frame: np.ndarray) -> np.ndarray:
            calls.append(frame.copy())
            return self.gain * frame

        result, prediction = evaluate_pa_predictor(
            predictor,
            self.x,
            self.y,
            protocol=self.protocol,
            model_label="exact-forward-model",
            split="test",
            purpose="final_report",
            operation_count=memory_polynomial_inference_cost((1,), (0,)),
            trainable_real_parameter_count=2,
            fit_seconds=0.01,
        )
        self.assertEqual([len(frame) for frame in calls], [8, 8, 8])
        np.testing.assert_allclose(np.concatenate(calls), self.x)
        np.testing.assert_allclose(prediction, self.y)
        self.assertEqual(
            result.direction,
            "x_split -> frozen PA model -> y_hat_split; compare with measured y",
        )
        self.assertEqual(result.full_record_metrics["complex_nmse_pooled_db"], -np.inf)
        self.assertEqual(
            result.opendpd_compatible_metrics["nmse_mean_segment_db"],
            -np.inf,
        )
        self.assertTrue(result.error_psd["available"])
        np.testing.assert_array_equal(
            result.error_psd["density"],
            np.zeros(8),
        )

    def test_evaluator_does_not_hide_scale_error_with_split_gain_fit(self) -> None:
        result, _ = evaluate_pa_predictor(
            lambda frame: frame,
            self.x,
            2.0 * self.x,
            protocol=self.protocol,
            model_label="wrong-scale",
            split="validation",
            purpose="model_selection",
        )
        self.assertAlmostEqual(
            result.full_record_metrics["complex_nmse_pooled_db"],
            10.0 * np.log10(0.25),
        )
        self.assertEqual(
            result.protocol["score_gain_policy"],
            "no_post_prediction_gain_fit",
        )

    def test_partial_frame_is_predicted_but_explicitly_excluded_from_open_metric(
        self,
    ) -> None:
        x = self.x[:18]
        y = self.y[:18]
        calls: list[int] = []

        def predictor(frame: np.ndarray) -> np.ndarray:
            calls.append(len(frame))
            return self.gain * frame

        result, _ = evaluate_pa_predictor(
            predictor,
            x,
            y,
            protocol=self.protocol,
            model_label="partial-tail",
            split="validation",
            purpose="diagnostic",
        )
        self.assertEqual(calls, [8, 8, 2])
        self.assertEqual(
            result.opendpd_compatible_metrics["scored_sample_count"],
            16,
        )
        self.assertEqual(
            result.opendpd_compatible_metrics[
                "discarded_partial_tail_samples"
            ],
            2,
        )

    def test_characteristic_edges_and_support_are_frozen_from_training(self) -> None:
        x = self.x.copy()
        x[-1] *= 1.5
        y = self.gain * x
        result, _ = evaluate_pa_predictor(
            lambda frame: self.gain * frame,
            x,
            y,
            protocol=self.protocol,
            model_label="support-check",
            split="validation",
            purpose="diagnostic",
        )
        np.testing.assert_array_equal(
            result.characteristic_residuals["bin_edges"],
            np.asarray(self.protocol.characteristic_bin_edges),
        )
        self.assertGreater(
            result.input_support["fraction_above_training_maximum"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
