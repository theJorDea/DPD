import tempfile
from pathlib import Path
import unittest

import numpy as np

from baseline.complexity import (
    gmp_inference_cost,
    gmp_leading_coefficient_count,
)
from baseline.gmp_pa import (
    GMPConfig,
    GeneralizedMemoryPolynomialPA,
    fit_gmp_pa,
    gmp_design_matrix,
    gmp_terms,
)


class GMPTermTests(unittest.TestCase):
    def test_exact_and_causal_leading_coefficient_counts(self) -> None:
        self.assertEqual(
            gmp_leading_coefficient_count(
                kc=2,
                lc=4,
                mc=2,
                leading_policy="opendpd_exact",
            ),
            16,
        )
        self.assertEqual(
            gmp_leading_coefficient_count(
                kc=2,
                lc=4,
                mc=2,
                leading_policy="causal_leading",
            ),
            10,
        )

    def test_column_order_matches_opendpd_definition(self) -> None:
        config = GMPConfig(
            ka=2,
            la=2,
            kb=1,
            lb=1,
            mb=1,
            kc=1,
            lc=2,
            mc=1,
            leading_policy="opendpd_exact",
        )
        terms = gmp_terms(config)
        self.assertEqual(
            [(term.branch, term.exponent, term.signal_delay, term.envelope_delay)
             for term in terms],
            [
                ("aligned", 0, 0, 0),
                ("aligned", 0, 1, 1),
                ("aligned", 1, 0, 0),
                ("aligned", 1, 1, 1),
                ("lagging", 1, 0, 1),
                ("leading", 1, 0, -1),
                ("leading", 1, 1, 0),
            ],
        )


class GMPInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(23)
        self.signal = (
            self.rng.normal(size=64)
            + 1j * self.rng.normal(size=64)
        )

    def _model(self, policy: str) -> GeneralizedMemoryPolynomialPA:
        config = GMPConfig(
            ka=3,
            la=4,
            kb=2,
            lb=4,
            mb=2,
            kc=2,
            lc=4,
            mc=2,
            leading_policy=policy,
        )
        coefficients = (
            self.rng.normal(size=config.coefficient_count)
            + 1j * self.rng.normal(size=config.coefficient_count)
        ) * 0.02
        return GeneralizedMemoryPolynomialPA(config, coefficients)

    def test_factorized_prediction_matches_dense_basis(self) -> None:
        for policy in ("causal_leading", "opendpd_exact"):
            with self.subTest(policy=policy):
                model = self._model(policy)
                dense = (
                    gmp_design_matrix(self.signal, model.config)
                    @ model.coefficients
                )
                np.testing.assert_allclose(
                    model.predict(self.signal),
                    dense,
                    rtol=2e-13,
                    atol=2e-13,
                )

    def test_phase_equivariance(self) -> None:
        model = self._model("causal_leading")
        rotation = np.exp(0.73j)
        np.testing.assert_allclose(
            model.predict(rotation * self.signal),
            rotation * model.predict(self.signal),
            rtol=3e-13,
            atol=3e-13,
        )

    def test_causal_prefix_is_invariant_but_exact_mode_uses_future(self) -> None:
        causal = self._model("causal_leading")
        exact = self._model("opendpd_exact")
        modified = self.signal.copy()
        modified[20:] += 4.0 - 2.0j
        np.testing.assert_allclose(
            causal.predict(self.signal)[:20],
            causal.predict(modified)[:20],
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(
            np.allclose(
                exact.predict(self.signal)[18:20],
                exact.predict(modified)[18:20],
            )
        )

    def test_segment_reset_and_streaming_chunk_equivalence(self) -> None:
        model = self._model("causal_leading")
        segmented = model.predict_segments(self.signal, 17)
        expected = np.concatenate(
            [
                model.predict(self.signal[start:start + 17])
                for start in range(0, self.signal.size, 17)
            ]
        )
        np.testing.assert_allclose(segmented, expected)

        state = None
        outputs = []
        for start, stop in ((0, 7), (7, 25), (25, 26), (26, 64)):
            output, state = model.predict_streaming_chunk(
                self.signal[start:stop],
                state,
            )
            outputs.append(output)
        np.testing.assert_allclose(
            np.concatenate(outputs),
            model.predict(self.signal),
            rtol=2e-13,
            atol=2e-13,
        )

    def test_save_load_preserves_policy_and_coefficients(self) -> None:
        model = self._model("causal_leading")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gmp.npz"
            model.save(path)
            restored = GeneralizedMemoryPolynomialPA.load(path)
        self.assertEqual(restored.config, model.config)
        np.testing.assert_array_equal(
            restored.coefficients,
            model.coefficients,
        )


class GMPFitTests(unittest.TestCase):
    def test_complex_fit_recovers_known_segmented_model(self) -> None:
        rng = np.random.default_rng(29)
        config = GMPConfig(
            ka=3,
            la=3,
            kb=1,
            lb=3,
            mb=1,
            kc=1,
            lc=3,
            mc=1,
            leading_policy="causal_leading",
        )
        truth = GeneralizedMemoryPolynomialPA(
            config,
            (
                rng.normal(size=config.coefficient_count)
                + 1j * rng.normal(size=config.coefficient_count)
            ) * 0.1,
        )
        x = rng.normal(size=512) + 1j * rng.normal(size=512)
        y = truth.predict_segments(x, 128)
        fitted, diagnostics = fit_gmp_pa(
            x,
            y,
            config=config,
            ridge=0.0,
            segment_length=128,
        )
        np.testing.assert_allclose(
            fitted.predict_segments(x, 128),
            y,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertEqual(diagnostics.solver_rank, config.coefficient_count)


class GMPCostTests(unittest.TestCase):
    def test_budgeted_and_full_opendpd_counts(self) -> None:
        aligned = gmp_inference_cost(ka=5, la=30)
        self.assertEqual(aligned.real_multiplications, 364)
        self.assertEqual(aligned.real_additions, 359)
        self.assertEqual(aligned.state_real_values, 174)

        causal = gmp_inference_cost(
            ka=5,
            la=30,
            kb=2,
            lb=30,
            mb=2,
            kc=2,
            lc=30,
            mc=2,
            leading_policy="causal_leading",
        )
        self.assertEqual(causal.real_multiplications, 832)
        self.assertEqual(causal.real_additions, 827)
        self.assertEqual(causal.state_real_values, 178)

        opendpd = gmp_inference_cost(
            ka=5,
            la=30,
            kb=4,
            lb=30,
            mb=5,
            kc=4,
            lc=30,
            mc=5,
            leading_policy="opendpd_exact",
        )
        self.assertEqual(opendpd.stored_real_coefficients, 2700)
        self.assertEqual(opendpd.real_multiplications, 2764)
        self.assertEqual(opendpd.real_additions, 2759)
        self.assertEqual(opendpd.state_real_values, 224)
        self.assertGreaterEqual(opendpd.real_multiplications, 1000)


if __name__ == "__main__":
    unittest.main()
