import unittest

import numpy as np

from baseline.residual_analysis import (
    ResidualAnalysisSpec,
    analyze_pa_residuals,
    boundary_safe_lagged,
    freeze_residual_reference,
    radial_tangential_residual,
    slow_envelope_state,
)


class BoundarySafeLagTests(unittest.TestCase):
    def test_lag_never_crosses_segment_boundary(self) -> None:
        values = np.asarray([10, 11, 12, 20, 21, 22], dtype=float)
        segments = np.asarray([0, 0, 0, 1, 1, 1])
        lagged, valid = boundary_safe_lagged(values, segments, 1)
        np.testing.assert_array_equal(
            lagged,
            np.asarray([0, 10, 11, 0, 20, 21], dtype=float),
        )
        np.testing.assert_array_equal(
            valid,
            np.asarray([False, True, True, False, True, True]),
        )

    def test_negative_lag_is_boundary_safe_future_diagnostic(self) -> None:
        values = np.asarray([1, 2, 3, 4], dtype=float)
        segments = np.asarray([0, 0, 1, 1])
        lagged, valid = boundary_safe_lagged(values, segments, -1)
        np.testing.assert_array_equal(lagged, [2, 0, 4, 0])
        np.testing.assert_array_equal(valid, [True, False, True, False])


class ResidualCoordinateTests(unittest.TestCase):
    def test_radial_and_tangential_signs_follow_input_phase(self) -> None:
        x = np.exp(1j * np.asarray([0.1, 0.7, 1.4]))
        residual = (0.2 - 0.3j) * x
        radial, tangential, valid = radial_tangential_residual(
            x,
            residual,
            amplitude_floor=0.0,
        )
        np.testing.assert_allclose(radial, 0.2, atol=1e-15)
        np.testing.assert_allclose(tangential, -0.3, atol=1e-15)
        self.assertTrue(np.all(valid))

    def test_slow_state_resets_at_explicit_segments(self) -> None:
        power = np.asarray([0.0, 1.0, 1.0, 0.0])
        segments = np.asarray([0, 0, 1, 1])
        state = slow_envelope_state(
            power,
            segments,
            time_constant_samples=2.0,
            initial_power=0.5,
        )
        alpha = np.exp(-0.5)
        self.assertAlmostEqual(state[0], alpha * 0.5)
        self.assertAlmostEqual(
            state[2],
            alpha * 0.5 + (1.0 - alpha),
        )


class ResidualAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ResidualAnalysisSpec(
            sample_rate_hz=32.0,
            psd_nperseg=8,
            main_bandwidth_hz=8.0,
            adjacent_bandwidth_hz=4.0,
            lags=(-2, -1, 0, 1, 2, 3),
            envelope_lags=(0, 1, 2),
            envelope_powers=(1, 2, 3),
            slow_time_constants_samples=(1.0, 2.0),
            amplitude_quantiles=(0.5, 0.75),
            characteristic_bins=4,
            position_bins=4,
        )
        rng = np.random.default_rng(19)
        self.x = rng.normal(size=32) + 1j * rng.normal(size=32)
        self.segments = np.repeat(np.arange(4), 8)
        delayed, valid = boundary_safe_lagged(self.x, self.segments, 2)
        self.error = np.zeros(32, dtype=np.complex128)
        self.error[valid] = 0.1 * delayed[valid]
        self.y_hat = 1.5 * self.x
        self.y = self.y_hat + self.error
        self.valid = np.ones(32, dtype=bool)

    def test_known_delayed_residual_peaks_at_causal_lag(self) -> None:
        report = analyze_pa_residuals(
            self.x,
            self.y,
            self.y_hat,
            segment_id=self.segments,
            valid_mask=self.valid,
            split_role="train_oof",
            spec=self.spec,
        )
        correlations = {
            row["lag_samples"]: row["proper_complex_correlation"]["magnitude"]
            for row in report["lag_correlations"]
        }
        self.assertEqual(max(correlations, key=correlations.get), 2)
        self.assertTrue(
            all(
                not row["eligible_for_state_branch_selection"]
                for row in report["slow_state_correlations"]
            )
        )

    def test_conjugate_residual_appears_in_pseudo_correlation(self) -> None:
        lagged, valid = boundary_safe_lagged(self.x, self.segments, 1)
        error = np.zeros_like(self.x)
        error[valid] = np.conj(lagged[valid])
        report = analyze_pa_residuals(
            self.x,
            self.y_hat + error,
            self.y_hat,
            segment_id=self.segments,
            valid_mask=self.valid,
            split_role="train_oof",
            spec=self.spec,
        )
        row = next(
            item for item in report["lag_correlations"]
            if item["lag_samples"] == 1
        )
        self.assertGreater(
            row["pseudo_complex_correlation"]["magnitude"],
            0.95,
        )

    def test_validation_requires_train_frozen_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "train-frozen"):
            analyze_pa_residuals(
                self.x,
                self.y,
                self.y_hat,
                segment_id=self.segments,
                valid_mask=self.valid,
                split_role="validation_confirmation",
                spec=self.spec,
            )
        frozen = freeze_residual_reference(self.x, self.spec)
        report = analyze_pa_residuals(
            self.x,
            self.y,
            self.y_hat,
            segment_id=self.segments,
            valid_mask=self.valid,
            split_role="validation_confirmation",
            spec=self.spec,
            frozen_reference=frozen,
        )
        self.assertEqual(
            report["frozen_reference"]["amplitude_thresholds"],
            frozen["amplitude_thresholds"],
        )
        np.testing.assert_array_equal(
            report["am_am_am_pm_residuals"]["bin_edges"],
            frozen["characteristic_bin_edges"],
        )

    def test_discovery_api_rejects_test_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "test residuals"):
            analyze_pa_residuals(
                self.x,
                self.y,
                self.y_hat,
                segment_id=self.segments,
                valid_mask=self.valid,
                split_role="test",  # type: ignore[arg-type]
                spec=self.spec,
            )

    def test_error_psd_is_residual_psd_not_difference_of_psds(self) -> None:
        report = analyze_pa_residuals(
            self.x,
            self.y,
            self.y_hat,
            segment_id=self.segments,
            valid_mask=self.valid,
            split_role="train_oof",
            spec=self.spec,
        )
        psd = report["error_psd"]
        self.assertTrue(psd["available"])
        self.assertEqual(psd["included_segment_ids"], [0, 1, 2, 3])
        self.assertEqual(len(psd["error_density"]), 8)


if __name__ == "__main__":
    unittest.main()
