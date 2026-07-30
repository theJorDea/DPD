import unittest

import numpy as np

from baseline.fixed_point_pa import (
    FixedPointGMPPA,
    FixedPointPAConfig,
)
from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA
from baseline.metrics import nmse_pooled_db


class FixedPointGMPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gmp_config = GMPConfig(
            ka=3,
            la=3,
            kb=0,
            lb=0,
            mb=0,
            kc=0,
            lc=0,
            mc=0,
        )
        coefficients = np.asarray(
            [
                0.72 + 0.03j,
                -0.11 + 0.04j,
                0.06 - 0.02j,
                0.12 - 0.03j,
                0.04 + 0.01j,
                -0.02 + 0.02j,
                0.03 + 0.01j,
                -0.01 + 0.02j,
                0.01 - 0.01j,
            ],
            dtype=np.complex128,
        )
        self.model = GeneralizedMemoryPolynomialPA(
            self.gmp_config,
            coefficients,
        )
        rng = np.random.default_rng(44)
        self.signal = (
            rng.normal(size=257) + 1j * rng.normal(size=257)
        ) * 0.22
        self.config = FixedPointPAConfig.for_activation_bits(
            16,
            accumulator_bits=56,
            scalar_accumulator_bits=56,
        )
        self.fixed = FixedPointGMPPA(self.model, self.config)

    def test_fixed_point_tracks_reference_without_saturation(self) -> None:
        result = self.fixed.predict_chunk(self.signal)
        reference = self.model.predict(self.signal)
        self.assertEqual(result.output.shape, self.signal.shape)
        self.assertEqual(result.stats.input_saturations, 0)
        self.assertEqual(result.stats.accumulator_saturations, 0)
        self.assertEqual(result.stats.output_saturations, 0)
        self.assertLess(nmse_pooled_db(result.output, reference), -55.0)

    def test_streaming_chunks_are_bit_identical(self) -> None:
        full = self.fixed.predict_chunk(self.signal)
        first = self.fixed.predict_chunk(self.signal[:73])
        second = self.fixed.predict_chunk(self.signal[73:149], first.next_state)
        third = self.fixed.predict_chunk(self.signal[149:], second.next_state)
        streamed = np.concatenate((first.output, second.output, third.output))
        np.testing.assert_array_equal(streamed, full.output)
        np.testing.assert_array_equal(
            third.next_state.real_codes,
            full.next_state.real_codes,
        )
        np.testing.assert_array_equal(
            third.next_state.imag_codes,
            full.next_state.imag_codes,
        )

    def test_segment_reset_is_not_continuous_history(self) -> None:
        segment_length = 31
        segmented = self.fixed.predict_segments(self.signal, segment_length)
        independent = np.concatenate(
            [
                self.fixed.predict_chunk(
                    self.signal[start : start + segment_length]
                ).output
                for start in range(0, self.signal.size, segment_length)
            ]
        )
        np.testing.assert_array_equal(segmented, independent)
        continuous = self.fixed.predict_chunk(self.signal).output
        self.assertGreater(
            float(np.max(np.abs(segmented - continuous))),
            0.0,
        )

    def test_causal_prefix_does_not_depend_on_future_samples(self) -> None:
        altered = self.signal.copy()
        altered[100:] += 0.7 - 0.4j
        prefix = self.fixed.predict_chunk(self.signal).output[:80]
        altered_prefix = self.fixed.predict_chunk(altered).output[:80]
        np.testing.assert_array_equal(prefix, altered_prefix)

    def test_coefficient_and_accumulator_saturation_are_reported(self) -> None:
        narrow = FixedPointPAConfig.for_activation_bits(
            12,
            coefficient_fraction_bits=10,
            accumulator_bits=12,
            scalar_accumulator_bits=12,
        )
        large_model = GeneralizedMemoryPolynomialPA(
            self.gmp_config,
            self.model.coefficients * 3.0,
        )
        result = FixedPointGMPPA(large_model, narrow).predict_chunk(self.signal)
        self.assertGreater(result.stats.coefficient_saturations, 0)
        self.assertGreater(result.stats.accumulator_saturations, 0)

    def test_invalid_noncausal_gmp_is_rejected(self) -> None:
        noncausal_config = GMPConfig(
            ka=1,
            la=1,
            kc=1,
            lc=1,
            mc=1,
            leading_policy="opendpd_exact",
        )
        model = GeneralizedMemoryPolynomialPA(
            noncausal_config,
            np.asarray([1.0 + 0j, 0.1 + 0j]),
        )
        with self.assertRaisesRegex(ValueError, "causal_leading"):
            FixedPointGMPPA(model, self.config)


if __name__ == "__main__":
    unittest.main()
