"""Unit tests for alignment and evaluator primitives.

Run from the repository root with:

    python -m unittest tests/test_alignment.py -v
"""

from __future__ import annotations

import unittest

import numpy as np

from baseline.alignment import (
    align_and_estimate_gain,
    complex_ls_gain,
    estimate_integer_delay,
    fractional_delay_diagnostic,
    overlap_for_delay,
)
from baseline.metrics import (
    bin_am_am_am_pm,
    nmse_opendpd_db,
    nmse_pooled_db,
    opendpd_aclr_db,
    opendpd_spectral_evm_db,
    papr_db,
    peak_amplitude,
    standard_aclr_db,
    time_domain_rms_evm,
    time_domain_rms_evm_db,
    welch_numpy,
)


class AlignmentTests(unittest.TestCase):
    def test_complex_ls_gain_recovers_amplitude_and_phase(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.normal(size=128) + 1j * rng.normal(size=128)
        expected_gain = 2.3 * np.exp(1j * 0.37)
        observed = expected_gain * reference

        actual_gain = complex_ls_gain(reference, observed)

        self.assertAlmostEqual(actual_gain.real, expected_gain.real, places=13)
        self.assertAlmostEqual(actual_gain.imag, expected_gain.imag, places=13)

    def test_complex_ls_gain_rejects_zero_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-energy"):
            complex_ls_gain(np.zeros(8, dtype=complex), np.ones(8, dtype=complex))

    def test_positive_delay_means_observed_lags_reference(self) -> None:
        rng = np.random.default_rng(11)
        reference = rng.normal(size=96) + 1j * rng.normal(size=96)
        gain = 0.8 * np.exp(-1j * 0.2)
        delay = 5
        observed = np.zeros_like(reference)
        observed[delay:] = gain * reference[:-delay]

        estimated = estimate_integer_delay(reference, observed, max_abs_delay=10)
        reference_aligned, observed_aligned = overlap_for_delay(
            reference,
            observed,
            estimated,
        )

        self.assertEqual(estimated, delay)
        np.testing.assert_allclose(observed_aligned, gain * reference_aligned)

    def test_negative_delay_means_observed_leads_reference(self) -> None:
        rng = np.random.default_rng(13)
        reference = rng.normal(size=96) + 1j * rng.normal(size=96)
        gain = -0.4 + 1.1j
        delay = -4
        observed = np.zeros_like(reference)
        observed[:delay] = gain * reference[-delay:]

        estimated = estimate_integer_delay(reference, observed, max_abs_delay=9)
        reference_aligned, observed_aligned, returned_delay, fitted_gain = (
            align_and_estimate_gain(
                reference,
                observed,
                max_abs_delay=9,
            )
        )

        self.assertEqual(estimated, delay)
        self.assertEqual(returned_delay, delay)
        np.testing.assert_allclose(observed_aligned, gain * reference_aligned)
        self.assertAlmostEqual(fitted_gain.real, gain.real, places=13)
        self.assertAlmostEqual(fitted_gain.imag, gain.imag, places=13)

    def test_fractional_delay_is_diagnostic_not_resampling(self) -> None:
        samples = np.arange(256, dtype=float)
        center = 120.0
        width = 13.0
        true_delay = 2.25
        reference = np.exp(-0.5 * ((samples - center) / width) ** 2).astype(complex)
        observed = (
            1.7
            * np.exp(1j * 0.4)
            * np.exp(-0.5 * ((samples - center - true_delay) / width) ** 2)
        )

        diagnostic = fractional_delay_diagnostic(
            reference,
            observed,
            max_abs_delay=8,
        )

        self.assertTrue(diagnostic.reliable)
        self.assertEqual(diagnostic.integer_delay, 2)
        self.assertAlmostEqual(diagnostic.estimated_delay, true_delay, delta=0.08)


class ScalarMetricTests(unittest.TestCase):
    def test_pooled_and_opendpd_segment_nmse_are_distinct(self) -> None:
        reference = np.vstack(
            [
                np.ones(8, dtype=complex),
                10.0 * np.ones(8, dtype=complex),
            ]
        )
        estimate = reference.copy()
        estimate[0] += 0.1
        estimate[1] += 5.0

        expected_pooled = 10.0 * np.log10(
            (8 * 0.1**2 + 8 * 5.0**2) / (8 * 1.0**2 + 8 * 10.0**2)
        )
        expected_opendpd = np.mean(
            [
                10.0 * np.log10(0.1**2 / 1.0**2),
                10.0 * np.log10(5.0**2 / 10.0**2),
            ]
        )

        self.assertAlmostEqual(nmse_pooled_db(estimate, reference), expected_pooled)
        self.assertAlmostEqual(
            nmse_opendpd_db(estimate, reference),
            expected_opendpd,
        )
        self.assertNotAlmostEqual(expected_pooled, expected_opendpd)

    def test_time_domain_rms_evm_is_explicit_and_matches_pooled_nmse_db(self) -> None:
        reference = np.array([1 + 1j, -2 + 0.5j, 0.5 - 0.25j])
        estimate = reference + np.array([0.1j, 0.2, -0.05 + 0.03j])
        linear = np.sqrt(
            np.sum(np.abs(estimate - reference) ** 2)
            / np.sum(np.abs(reference) ** 2)
        )

        self.assertAlmostEqual(time_domain_rms_evm(estimate, reference), linear)
        self.assertAlmostEqual(
            time_domain_rms_evm_db(estimate, reference),
            20.0 * np.log10(linear),
        )
        self.assertAlmostEqual(
            time_domain_rms_evm_db(estimate, reference),
            nmse_pooled_db(estimate, reference),
        )

    def test_peak_and_papr(self) -> None:
        signal = np.array([1.0, 1.0j, 2.0 + 0.0j])
        self.assertAlmostEqual(peak_amplitude(signal), 2.0)
        self.assertAlmostEqual(papr_db(signal), 10.0 * np.log10(2.0))


class SpectralMetricTests(unittest.TestCase):
    def test_numpy_welch_complex_bin_tone_has_unit_spectrum_peak(self) -> None:
        nperseg = 128
        fs = 128.0
        tone_bin = 17
        samples = np.arange(nperseg)
        signal = np.exp(2j * np.pi * tone_bin * samples / nperseg)

        frequencies, power = welch_numpy(
            signal,
            fs=fs,
            nperseg=nperseg,
            scaling="spectrum",
            detrend="constant",
        )

        peak = int(np.argmax(power))
        self.assertEqual(frequencies[peak], float(tone_bin))
        self.assertAlmostEqual(power[peak], 1.0, places=13)

    def test_opendpd_spectral_evm_matches_uniform_complex_scale_error(self) -> None:
        rng = np.random.default_rng(17)
        nperseg = 128
        reference = rng.normal(size=(3, nperseg)) + 1j * rng.normal(
            size=(3, nperseg)
        )
        estimate = 1.1 * reference

        actual = opendpd_spectral_evm_db(
            estimate,
            reference,
            fs=128.0,
            bandwidth_main=64.0,
            n_subchannels=4,
            nperseg=nperseg,
        )

        self.assertAlmostEqual(actual, 20.0 * np.log10(0.1), places=12)

    @staticmethod
    def _multitone_for_aclr() -> np.ndarray:
        nperseg = 128
        samples = np.arange(nperseg)
        tones = [
            (-15, 1.0),
            (-5, 1.0),
            (5, 1.0),
            (15, 1.0),
            (-25, 0.1),
            (25, 0.2),
        ]
        return sum(
            amplitude * np.exp(2j * np.pi * frequency * samples / nperseg)
            for frequency, amplitude in tones
        )

    def test_opendpd_aclr_uses_strongest_main_subchannel(self) -> None:
        result = opendpd_aclr_db(
            self._multitone_for_aclr(),
            fs=128.0,
            nperseg=128,
            bandwidth_main=40.0,
            n_subchannels=4,
        )

        self.assertAlmostEqual(result.left_db, -20.0, places=11)
        self.assertAlmostEqual(result.right_db, 20.0 * np.log10(0.2), places=11)
        self.assertAlmostEqual(
            result.average_db,
            (result.left_db + result.right_db) / 2.0,
        )

    def test_standard_aclr_uses_total_main_band_power(self) -> None:
        result = standard_aclr_db(
            self._multitone_for_aclr(),
            fs=128.0,
            nperseg=128,
            bandwidth_main=40.0,
            bandwidth_adjacent=10.0,
        )

        self.assertAlmostEqual(
            result.left_db,
            10.0 * np.log10(0.1**2 / 4.0),
            places=11,
        )
        self.assertAlmostEqual(
            result.right_db,
            10.0 * np.log10(0.2**2 / 4.0),
            places=11,
        )


class CharacteristicTests(unittest.TestCase):
    def test_am_am_and_circular_am_pm_binning(self) -> None:
        amplitude = np.array([0.1, 0.2, 0.6, 0.8])
        input_phase = np.array([0.0, 0.4, -0.7, 1.0])
        input_signal = amplitude * np.exp(1j * input_phase)
        phase_shift = np.array([0.1, 0.1, -0.2, -0.2])
        output_signal = 2.0 * input_signal * np.exp(1j * phase_shift)

        result = bin_am_am_am_pm(
            input_signal,
            output_signal,
            bins=np.array([0.0, 0.5, 1.0]),
        )

        np.testing.assert_array_equal(result["count"], np.array([2, 2]))
        np.testing.assert_allclose(result["am_am_gain"], np.array([2.0, 2.0]))
        np.testing.assert_allclose(result["am_pm_rad"], np.array([0.1, -0.2]))
        np.testing.assert_allclose(result["phase_concentration"], np.ones(2))


if __name__ == "__main__":
    unittest.main()
