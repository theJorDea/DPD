"""Focused tests for the versioned frame-safe sensitivity transform."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from baseline.fractional_alignment import (
    FractionalAlignmentConfig,
    apply_fractional_alignment_frame,
    apply_fractional_alignment_frames,
    freeze_fractional_alignment,
    symmetric_guard_crop,
)


def _bandlimited_signal(times: np.ndarray) -> np.ndarray:
    frequencies = np.asarray([-0.13, -0.047, 0.021, 0.089, 0.157])
    amplitudes = np.asarray(
        [
            0.31 - 0.07j,
            -0.23 + 0.19j,
            0.82 + 0.11j,
            -0.17 - 0.29j,
            0.09 + 0.16j,
        ]
    )
    return np.sum(
        amplitudes[:, None]
        * np.exp(2j * np.pi * frequencies[:, None] * times[None, :]),
        axis=0,
    )


class FractionalAlignmentAccuracyTests(unittest.TestCase):
    def test_known_bandlimited_positive_delay_and_sign(self) -> None:
        sample_index = np.arange(512, dtype=np.float64)
        delay = 0.27
        gain = 1.4 * np.exp(0.31j)
        reference = _bandlimited_signal(sample_index)
        observed = gain * _bandlimited_signal(sample_index - delay)
        frozen = freeze_fractional_alignment(
            FractionalAlignmentConfig(
                observed_delay_samples=delay,
                tap_count=65,
                kaiser_beta=8.6,
            )
        )

        reference_aligned, observed_aligned = apply_fractional_alignment_frame(
            reference,
            observed,
            frozen,
        )
        relative_error = np.sum(
            np.abs(observed_aligned / gain - reference_aligned) ** 2
        ) / np.sum(np.abs(reference_aligned) ** 2)

        wrong_sign = freeze_fractional_alignment(
            FractionalAlignmentConfig(
                observed_delay_samples=-delay,
                tap_count=65,
                kaiser_beta=8.6,
            )
        )
        wrong_reference, wrong_observed = apply_fractional_alignment_frame(
            reference,
            observed,
            wrong_sign,
        )
        wrong_error = np.sum(
            np.abs(wrong_observed / gain - wrong_reference) ** 2
        ) / np.sum(np.abs(wrong_reference) ** 2)

        self.assertLess(relative_error, 2e-10)
        self.assertGreater(wrong_error, 1e5 * relative_error)

    def test_negative_delay_uses_same_documented_convention(self) -> None:
        sample_index = np.arange(384, dtype=np.float64)
        delay = -0.34
        reference = _bandlimited_signal(sample_index)
        observed = _bandlimited_signal(sample_index - delay)
        frozen = freeze_fractional_alignment(
            FractionalAlignmentConfig(delay, tap_count=65)
        )

        reference_aligned, observed_aligned = apply_fractional_alignment_frame(
            reference,
            observed,
            frozen,
        )

        np.testing.assert_allclose(
            observed_aligned,
            reference_aligned,
            rtol=0.0,
            atol=2e-5,
        )


class FractionalAlignmentBoundaryTests(unittest.TestCase):
    def test_frames_are_independent_and_never_circularly_wrapped(self) -> None:
        frame_length = 160
        first_reference = np.zeros(frame_length, dtype=np.complex128)
        first_observed = np.zeros(frame_length, dtype=np.complex128)
        first_observed[-1] = 9.0 - 4.0j
        second_reference = np.zeros(frame_length, dtype=np.complex128)
        second_observed = np.zeros(frame_length, dtype=np.complex128)
        frozen = freeze_fractional_alignment(
            FractionalAlignmentConfig(0.31, tap_count=33)
        )

        references, observations = apply_fractional_alignment_frames(
            (first_reference, second_reference),
            (first_observed, second_observed),
            frozen,
        )

        self.assertEqual(len(references), 2)
        self.assertEqual(len(observations), 2)
        np.testing.assert_array_equal(observations[1], np.zeros_like(observations[1]))
        # A linear FIR may legitimately spread the final impulse backward by
        # half its support.  It must not wrap it to the beginning of the same
        # frame or carry it into the next frame.
        np.testing.assert_array_equal(
            observations[0][: frozen.guard_samples],
            np.zeros(frozen.guard_samples, dtype=np.complex128),
        )

    def test_zero_and_integer_delays_are_exact_on_valid_region(self) -> None:
        rng = np.random.default_rng(91)
        reference = rng.normal(size=192) + 1j * rng.normal(size=192)

        zero = freeze_fractional_alignment(
            FractionalAlignmentConfig(0.0, tap_count=17)
        )
        zero_reference, zero_observed = apply_fractional_alignment_frame(
            reference,
            reference,
            zero,
        )
        np.testing.assert_array_equal(zero_reference, reference[8:-8])
        np.testing.assert_array_equal(zero_observed, reference[8:-8])
        np.testing.assert_array_equal(
            zero.coefficient_array(),
            np.eye(1, 17, 8, dtype=np.float64).ravel(),
        )

        delay = 3
        observed = np.zeros_like(reference)
        observed[delay:] = reference[:-delay]
        integer = freeze_fractional_alignment(
            FractionalAlignmentConfig(float(delay), tap_count=17)
        )
        integer_reference, integer_observed = (
            apply_fractional_alignment_frame(reference, observed, integer)
        )
        np.testing.assert_array_equal(
            integer_reference,
            reference[8:-delay - 8],
        )
        np.testing.assert_array_equal(integer_observed, integer_reference)

    def test_invalid_guard_and_too_short_frame_are_rejected(self) -> None:
        values = np.ones(16, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            symmetric_guard_crop(values, values, -1)
        with self.assertRaisesRegex(TypeError, "integer"):
            symmetric_guard_crop(values, values, 1.5)
        with self.assertRaisesRegex(ValueError, "leaves no scored"):
            symmetric_guard_crop(values, values, 8)

        frozen = freeze_fractional_alignment(
            FractionalAlignmentConfig(0.1, tap_count=17)
        )
        with self.assertRaisesRegex(ValueError, "leaves no scored"):
            apply_fractional_alignment_frame(values, values, frozen)

    def test_even_tap_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "odd integer"):
            FractionalAlignmentConfig(0.1, tap_count=32)


class FractionalAlignmentFreezingTests(unittest.TestCase):
    def test_freezing_is_identical_and_metadata_is_hash_friendly(self) -> None:
        config = FractionalAlignmentConfig(0.0772579722, tap_count=65)
        first = freeze_fractional_alignment(config)
        second = freeze_fractional_alignment(config)

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertEqual(first.coefficient_sha256, second.coefficient_sha256)
        self.assertEqual(first.protocol_sha256, second.protocol_sha256)
        self.assertEqual(first.to_metadata(), second.to_metadata())
        self.assertEqual(len(first.to_metadata()["coefficient_float64_hex"]), 65)
        self.assertEqual(
            first.to_metadata()["purpose"],
            "sensitivity_analysis_only_not_automatic_measurement_path_truth",
        )
        self.assertEqual(
            freeze_fractional_alignment(FractionalAlignmentConfig(-0.0)),
            freeze_fractional_alignment(FractionalAlignmentConfig(0.0)),
        )

        coefficient_array = first.coefficient_array()
        self.assertFalse(coefficient_array.flags.writeable)
        with self.assertRaises(ValueError):
            coefficient_array[0] = 0.0
        with self.assertRaises(FrozenInstanceError):
            config.observed_delay_samples = 0.0

    def test_same_frozen_transform_is_bit_identical_across_calls(self) -> None:
        rng = np.random.default_rng(43)
        reference = rng.normal(size=128) + 1j * rng.normal(size=128)
        observed = rng.normal(size=128) + 1j * rng.normal(size=128)
        frozen = freeze_fractional_alignment(
            FractionalAlignmentConfig(-0.18, tap_count=33)
        )

        first = apply_fractional_alignment_frame(reference, observed, frozen)
        second = apply_fractional_alignment_frame(reference, observed, frozen)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
