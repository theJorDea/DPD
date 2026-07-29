import json
from pathlib import Path
import unittest

import numpy as np

from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA
from baseline.widely_linear_pa import WidelyLinearResidualCorrection
from experiments.audit_widely_linear_pa import (
    _metrics,
    _parse_candidates,
    _select_candidate,
    _streaming_checks,
)


class WidelyLinearAuditHelperTests(unittest.TestCase):
    def test_preregistered_candidate_counts_match_frozen_apa_base(self) -> None:
        config = json.loads(
            Path(
                "experiments/configs/pa_widely_linear_residual_apa200.json"
            ).read_text(encoding="utf-8")
        )
        base = GeneralizedMemoryPolynomialPA(
            GMPConfig(**config["base_model"]["gmp_config"]),
            np.ones(444, dtype=np.complex128),
        )
        candidates = _parse_candidates(
            config,
            base_cost=base.operation_count,
        )
        self.assertEqual(
            [candidate.operation_count.real_multiplications for candidate in candidates],
            [954, 958, 962, 966, 974],
        )
        self.assertEqual(candidates[-1].delays, (0, 1, 2, 3, 4))

    def test_metrics_keep_pooled_partial_frame_and_mark_opendpd_unavailable(self) -> None:
        prediction = np.asarray([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        reference = np.asarray([1.0 + 0j, 2.0 + 0j, 4.0 + 0j])
        common = np.asarray([False, True, True])
        result = _metrics(
            prediction,
            reference,
            nperseg=4,
            common_mask=common,
        )
        self.assertIsNone(result["opendpd_compatible_nmse_db"])
        self.assertEqual(result["scored_samples_full"], 3)
        self.assertEqual(result["per_frame_nmse_db"], [])
        self.assertAlmostEqual(
            result["full_record_nmse_db"],
            10.0 * np.log10(1.0 / 21.0),
        )

    def test_streaming_and_reset_checks_are_exact(self) -> None:
        rng = np.random.default_rng(2207)
        signal = rng.normal(size=37) + 1j * rng.normal(size=37)
        model = WidelyLinearResidualCorrection(
            (0, 1, 4),
            np.asarray([0.03 + 0.01j, -0.02j, 0.01 - 0.03j]),
        )
        checks = _streaming_checks(model, signal, segment_length=8)
        self.assertTrue(checks["streaming_chunk_equivalence_passed"])
        self.assertTrue(checks["reset_at_frame_equivalence_passed"])
        self.assertEqual(checks["maximum_streaming_error"], 0.0)
        self.assertEqual(checks["maximum_reset_error"], 0.0)

    def test_selection_falls_back_when_no_candidate_meets_threshold(self) -> None:
        rows = {
            "no_correction": {"eligible": False, "tap_count": 0, "score_db": 0.0},
            "conj_d0": {"eligible": False, "tap_count": 1, "score_db": 0.08},
            "conj_d0_d1": {"eligible": False, "tap_count": 2, "score_db": 0.09},
        }
        candidates = (
            type("Candidate", (), {"name": "no_correction"})(),
            type("Candidate", (), {"name": "conj_d0"})(),
            type("Candidate", (), {"name": "conj_d0_d1"})(),
        )
        self.assertEqual(
            _select_candidate(rows, candidates, tie_tolerance_db=0.02),
            "no_correction",
        )

    def test_selection_prefers_fewer_taps_inside_tie_tolerance(self) -> None:
        rows = {
            "no_correction": {"eligible": False, "tap_count": 0, "score_db": 0.0},
            "conj_d0": {
                "candidate": "conj_d0",
                "eligible": True,
                "tap_count": 1,
                "score_db": 0.20,
            },
            "conj_d0_d1": {
                "candidate": "conj_d0_d1",
                "eligible": True,
                "tap_count": 2,
                "score_db": 0.21,
            },
        }
        candidates = (
            type("Candidate", (), {"name": "no_correction"})(),
            type("Candidate", (), {"name": "conj_d0"})(),
            type("Candidate", (), {"name": "conj_d0_d1"})(),
        )
        self.assertEqual(
            _select_candidate(rows, candidates, tie_tolerance_db=0.02),
            "conj_d0",
        )


if __name__ == "__main__":
    unittest.main()
