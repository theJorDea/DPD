import tempfile
from pathlib import Path
import unittest

import numpy as np

from baseline.spline_hammerstein_pa import (
    SplineHammersteinPA,
    SplineHammersteinState,
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


if __name__ == "__main__":
    unittest.main()
