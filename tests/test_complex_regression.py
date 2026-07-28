import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.complex_spline_dpd import (
    complex_design_matrix,
    fit_complex_linear_spline,
)


class ComplexRegressionTests(unittest.TestCase):
    def test_recovers_known_complex_control_points(self) -> None:
        rng = np.random.default_rng(20260728)
        radius = rng.uniform(0.001, 0.999, 4000)
        phase = rng.uniform(-np.pi, np.pi, radius.size)
        samples = radius * np.exp(1j * phase)
        knots = np.asarray([0.0, 0.2, 0.47, 0.73, 1.0])
        coefficients = np.asarray(
            [0.9 + 0.1j, 1.0 - 0.05j, 1.2 - 0.2j, 1.45 - 0.1j, 1.7 + 0.2j]
        )
        target = complex_design_matrix(samples, knots) @ coefficients
        model, diagnostics = fit_complex_linear_spline(
            samples,
            target,
            knots=knots,
            ridge=0.0,
        )
        np.testing.assert_allclose(
            model.coefficients, coefficients, rtol=2e-12, atol=2e-12
        )
        self.assertLess(diagnostics.training_relative_error_power, 1e-24)

    def test_complex_solution_matches_real_block_form(self) -> None:
        rng = np.random.default_rng(44)
        samples = (
            rng.normal(size=300) + 1j * rng.normal(size=300)
        ) / np.sqrt(8.0)
        knots = np.linspace(0.0, np.max(np.abs(samples)), 8)
        phi = complex_design_matrix(samples, knots)
        target = rng.normal(size=300) + 1j * rng.normal(size=300)
        ridge = 1e-4
        model, _ = fit_complex_linear_spline(
            samples,
            target,
            knots=knots,
            ridge=ridge,
        )

        real_design = np.block(
            [[phi.real, -phi.imag], [phi.imag, phi.real]]
        )
        real_target = np.concatenate([target.real, target.imag])
        count = samples.size
        real_system = real_design.T @ real_design / count
        real_system += ridge * np.eye(2 * knots.size)
        real_rhs = real_design.T @ real_target / count
        solution = np.linalg.solve(real_system, real_rhs)
        block_coefficients = solution[: knots.size] + 1j * solution[knots.size :]
        np.testing.assert_allclose(
            model.coefficients, block_coefficients, rtol=1e-10, atol=1e-10
        )

    def test_npz_round_trip(self) -> None:
        samples = np.linspace(0.01, 1.0, 100).astype(np.complex128)
        target = samples * (1.0 + 0.2 * np.abs(samples) + 0.1j)
        model, _ = fit_complex_linear_spline(
            samples,
            target,
            knot_count=8,
            knot_strategy="uniform_power",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = type(model).load(path)
            np.testing.assert_array_equal(loaded.knots, model.knots)
            np.testing.assert_array_equal(loaded.coefficients, model.coefficients)
            np.testing.assert_allclose(loaded.predict(samples), model.predict(samples))

    def test_nonfinite_regularization_is_rejected_before_lapack(self) -> None:
        samples = np.linspace(0.01, 1.0, 32).astype(np.complex128)
        for keyword in ("ridge", "smoothness"):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, "finite"):
                    fit_complex_linear_spline(
                        samples,
                        samples,
                        knot_count=4,
                        **{keyword: np.nan},
                    )


if __name__ == "__main__":
    unittest.main()
