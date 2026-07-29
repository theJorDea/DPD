import json
from pathlib import Path
import unittest

import numpy as np

from baseline.complex_fir_pa import ComplexFIRResidualCorrection
from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA
from experiments.audit_widely_linear_pa import (
    _correction_kind,
    _load_config,
    _parse_candidates,
    _streaming_checks,
)


class ComplexFIRAuditHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(
            "experiments/configs/pa_long_fir_residual_apa200.json"
        )
        cls.config = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_preregistered_candidate_counts_include_extended_state(self) -> None:
        base = GeneralizedMemoryPolynomialPA(
            GMPConfig(**self.config["base_model"]["gmp_config"]),
            np.ones(444, dtype=np.complex128),
        )
        candidates = _parse_candidates(
            self.config,
            base_cost=base.operation_count,
        )
        self.assertEqual(
            [candidate.operation_count.real_multiplications for candidate in candidates],
            [954, 958, 966, 978, 986],
        )
        self.assertEqual(
            [candidate.operation_count.state_real_values for candidate in candidates],
            [236, 268, 270, 274, 276],
        )
        self.assertEqual(candidates[-1].delays, tuple(range(42, 50)))

    def test_config_dispatches_only_to_proper_causal_mode(self) -> None:
        loaded = _load_config(self.path.resolve())
        self.assertEqual(_correction_kind(loaded), "proper")
        invalid = dict(loaded)
        invalid["task"] = "unknown_task"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            _correction_kind(invalid)

    def test_generic_streaming_checks_accept_proper_fir(self) -> None:
        rng = np.random.default_rng(2293)
        signal = rng.normal(size=117) + 1j * rng.normal(size=117)
        model = ComplexFIRResidualCorrection(
            (42, 45, 49),
            np.asarray([0.01 + 0.02j, -0.02j, 0.005 - 0.003j]),
        )
        checks = _streaming_checks(model, signal, segment_length=64)
        self.assertTrue(checks["streaming_chunk_equivalence_passed"])
        self.assertTrue(checks["reset_at_frame_equivalence_passed"])
        self.assertEqual(checks["maximum_streaming_error"], 0.0)
        self.assertEqual(checks["maximum_reset_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
