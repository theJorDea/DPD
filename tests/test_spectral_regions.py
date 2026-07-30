import unittest

import numpy as np

from baseline.spectral_regions import (
    SpectralRegion,
    _suppression_db,
    configured_spectral_region_report,
)


class SpectralRegionTests(unittest.TestCase):
    def test_region_definition_is_half_open_and_validated(self) -> None:
        region = SpectralRegion("right_im", 20.0, 30.0)
        self.assertEqual(region.name, "right_im")
        self.assertEqual(region.low_hz, 20.0)
        self.assertEqual(region.high_hz, 30.0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            SpectralRegion("", 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "below"):
            SpectralRegion("bad", 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            SpectralRegion("bad", 0.0, float("inf"))

    def test_synthetic_tones_have_expected_suppression_sign_and_dbc(self) -> None:
        nperseg = 128
        samples = np.arange(nperseg)

        def tone(bin_index: int, amplitude: float) -> np.ndarray:
            return amplitude * np.exp(
                2j * np.pi * bin_index * samples / nperseg
            )

        main = tone(-5, 1.0) + tone(5, 1.0)
        no_dpd_frame = main + tone(-25, 0.1) + tone(25, 0.2)
        dpd_frame = main + tone(-25, 0.05) + tone(25, 0.02)
        no_dpd = np.tile(no_dpd_frame, 2)
        with_dpd = np.tile(dpd_frame, 2)

        report = configured_spectral_region_report(
            no_dpd,
            with_dpd,
            fs=128.0,
            nperseg=nperseg,
            main_region=SpectralRegion("main", -10.0, 10.0),
            regions=(
                SpectralRegion("left", -28.0, -22.0),
                SpectralRegion("right", 22.0, 28.0),
            ),
        )

        self.assertFalse(report["definition"]["rf_harmonic_claim"])
        self.assertFalse(report["definition"]["threshold_applied"])
        self.assertEqual(report["frame_count"], 2)
        left = report["regions"]["left"]
        right = report["regions"]["right"]
        self.assertAlmostEqual(left["suppression_db"], 20.0 * np.log10(2.0))
        self.assertAlmostEqual(right["suppression_db"], 20.0, places=11)
        self.assertAlmostEqual(
            right["no_dpd_dbc"],
            10.0 * np.log10(0.2**2 / 2.0),
            places=11,
        )
        self.assertAlmostEqual(
            right["dpd_dbc"],
            10.0 * np.log10(0.02**2 / 2.0),
            places=11,
        )
        np.testing.assert_allclose(
            right["per_frame"]["suppression_db"],
            np.asarray([20.0, 20.0]),
            rtol=0.0,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            right["quantiles"]["suppression_db"],
            np.full(4, 20.0),
            rtol=0.0,
            atol=1e-11,
        )

    def test_suppression_zero_power_policy_is_explicit(self) -> None:
        self.assertEqual(_suppression_db(1.0, 0.0), float("inf"))
        self.assertEqual(_suppression_db(0.0, 1.0), float("-inf"))
        with self.assertRaisesRegex(ValueError, "both powers"):
            _suppression_db(0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            _suppression_db(-1.0, 1.0)

    def test_duplicate_or_out_of_nyquist_regions_are_rejected(self) -> None:
        samples = np.arange(128)
        signal = np.exp(2j * np.pi * 5 * samples / 128)
        main = SpectralRegion("main", -10.0, 10.0)
        duplicate = SpectralRegion("same", 20.0, 30.0)
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            configured_spectral_region_report(
                signal,
                signal,
                fs=128.0,
                nperseg=128,
                main_region=main,
                regions=(duplicate, duplicate),
            )
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            configured_spectral_region_report(
                signal,
                signal,
                fs=128.0,
                nperseg=128,
                main_region=main,
                regions=(SpectralRegion("outside", 60.0, 70.0),),
            )

    def test_signal_shapes_and_quantiles_are_frozen(self) -> None:
        signal = np.exp(2j * np.pi * 5 * np.arange(128) / 128)
        kwargs = {
            "fs": 128.0,
            "nperseg": 128,
            "main_region": SpectralRegion("main", 0.0, 10.0),
            "regions": (SpectralRegion("region", 10.0, 20.0),),
        }
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            configured_spectral_region_report(
                signal,
                signal[:-1],
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "quantiles"):
            configured_spectral_region_report(
                signal,
                signal,
                quantiles=(0.5, 0.4),
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()
