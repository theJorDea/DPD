import unittest

import numpy as np

from baseline.complex_spline_dpd import (
    ComplexLinearSplineDPD,
    local_spline_coordinates,
    make_knots,
    spline_basis,
)


class SplineBasisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.knots = np.asarray([0.0, 0.25, 0.7, 1.0])

    def test_partition_of_unity_and_local_support(self) -> None:
        radius = np.linspace(0.0, 1.0, 101)
        basis = spline_basis(radius, self.knots)
        np.testing.assert_allclose(np.sum(basis, axis=1), 1.0, atol=1e-14)
        self.assertTrue(np.all(np.count_nonzero(basis, axis=1) <= 2))
        self.assertTrue(np.all(basis >= 0.0))

    def test_continuity_and_exact_control_points(self) -> None:
        coefficients = np.asarray(
            [1.0 + 0.0j, 1.1 - 0.2j, 1.4 + 0.1j, 1.8 + 0.3j]
        )
        model = ComplexLinearSplineDPD(self.knots, coefficients)
        np.testing.assert_allclose(model.correction(self.knots), coefficients)
        epsilon = 1e-10
        for knot in self.knots[1:-1]:
            left = model.correction(np.asarray([knot - epsilon]))[0]
            right = model.correction(np.asarray([knot + epsilon]))[0]
            self.assertLess(abs(left - right), 2e-8)

    def test_endpoint_clamping(self) -> None:
        left, weight = local_spline_coordinates(
            np.asarray([0.0, 1.0, 2.0]), self.knots
        )
        np.testing.assert_array_equal(left, np.asarray([0, 2, 2]))
        np.testing.assert_allclose(weight, np.asarray([0.0, 1.0, 1.0]))

    def test_phase_equivariance(self) -> None:
        model = ComplexLinearSplineDPD(
            self.knots,
            np.asarray([1.0 + 0.1j, 1.1 + 0.2j, 1.3 - 0.1j, 1.5 - 0.2j]),
        )
        signal = np.asarray([0.1 + 0.2j, -0.4 + 0.3j, 0.8 - 0.1j])
        phase = np.exp(1j * 0.731)
        np.testing.assert_allclose(
            model.predict(signal * phase),
            model.predict(signal) * phase,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_explicit_duplicate_or_negative_knots_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            ComplexLinearSplineDPD(
                np.asarray([0.0, 0.5, 0.5, 1.0]),
                np.ones(4, dtype=complex),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ComplexLinearSplineDPD(
                np.asarray([-0.1, 0.5, 1.0]),
                np.ones(3, dtype=complex),
            )

    def test_quantile_knots_report_real_unique_support(self) -> None:
        samples = np.asarray(
            [0.0] * 20 + [0.5] * 20 + [1.0] * 20,
            dtype=np.complex128,
        )
        knots = make_knots(samples, 16, "quantile")
        self.assertLess(knots.size, 16)
        self.assertTrue(np.all(np.diff(knots) > 0.0))
        self.assertEqual(knots[0], 0.0)
        self.assertEqual(knots[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
