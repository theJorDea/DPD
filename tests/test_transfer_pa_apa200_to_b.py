import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA
from baseline.sparse_spline_memory_pa import SparseSplineMemoryPA
from experiments.transfer_pa_apa200_to_b import (
    _load_source_models,
    _select_calibration_record,
    _streaming_check,
    extract_prefix_segments,
    load_config,
    metric_summary,
    run_from_config,
)
from experiments.verify_pa_transfer_bundle import verify_bundle


CONFIG = Path("experiments/configs/pa_transfer_apa200_to_b.json")


class TransferContractTests(unittest.TestCase):
    def test_prefixes_are_frame_local_and_copied(self) -> None:
        frames = (
            np.arange(8, dtype=float) + 1j,
            np.arange(8, 16, dtype=float) + 2j,
        )
        prefixes = extract_prefix_segments(frames, 3)
        self.assertEqual([prefix.tolist() for prefix in prefixes], [
            [0 + 1j, 1 + 1j, 2 + 1j],
            [8 + 2j, 9 + 2j, 10 + 2j],
        ])
        frames[0][0] = 100
        self.assertEqual(prefixes[0][0], 1j)
        with self.assertRaisesRegex(ValueError, "shorter"):
            extract_prefix_segments(frames, 9)

    def test_target_held_out_split_is_not_an_addressable_split(self) -> None:
        with patch(
            "experiments.transfer_pa_apa200_to_b.load_split_pair"
        ) as loader:
            from experiments.transfer_pa_apa200_to_b import _load_allowed_pair

            with self.assertRaisesRegex(ValueError, "only preregistered"):
                _load_allowed_pair(Path("."), "test")
            loader.assert_not_called()

    def test_source_artifact_operation_contract_is_verified(self) -> None:
        config = load_config(CONFIG)
        models = _load_source_models(config)
        self.assertIsInstance(models["causal_gmp"], GeneralizedMemoryPolynomialPA)
        self.assertIsInstance(
            models["lag9_sparse_spline_memory"],
            SparseSplineMemoryPA,
        )
        self.assertEqual(
            models["causal_gmp"].operation_count.real_multiplications,
            954,
        )
        self.assertEqual(
            models["lag9_sparse_spline_memory"].operation_count().real_multiplications,
            72,
        )

    def test_streaming_equivalence_for_frozen_models(self) -> None:
        config = load_config(CONFIG)
        models = _load_source_models(config)
        rng = np.random.default_rng(17)
        signal = (
            rng.normal(size=701) + 1j * rng.normal(size=701)
        ).astype(np.complex128) * 0.2
        for model in models.values():
            result = _streaming_check(model, (signal,))
            self.assertTrue(result["streaming_chunk_equivalence_passed"])
            self.assertTrue(result["segmented_reset_equivalence_passed"])
            self.assertLessEqual(result["streaming_chunk_max_abs_error"], 1e-11)

    def test_metric_summary_has_pooled_and_complete_frame_scores(self) -> None:
        target = np.ones(10, dtype=np.complex128)
        prediction = target.copy()
        prediction[5] += 0.1
        summary = metric_summary(prediction, target, (6, 4), warmup=1)
        self.assertIn("full_record_nmse_db", summary)
        self.assertIn("common_interior_nmse_db", summary)
        self.assertIn("opendpd_compatible_nmse_db", summary)
        self.assertEqual(len(summary["per_frame_nmse_db"]), 2)

    def test_selection_excludes_zero_sample_no_update_record(self) -> None:
        def row(count: int, score: float) -> dict[str, object]:
            return {
                "sample_count_per_frame": count,
                "status": "feasible",
                "validation": {
                    "full_record_nmse_db": score,
                    "common_interior_nmse_db": score,
                    "support": {"fraction_above_support": 0.0},
                    "streaming": {
                        "streaming_chunk_equivalence_passed": True,
                        "segmented_reset_equivalence_passed": True,
                    },
                },
                "fit": {"fit_wall_clock_seconds": 1.0},
            }

        selected = _select_calibration_record(
            [row(0, -30.0), row(256, -30.1), row(512, -30.2)]
        )
        self.assertEqual(selected["sample_count_per_frame"], 256)

    def test_runner_seals_tampered_config_before_waveform_load(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["dataset_contract"]["target_val_output_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch(
                "experiments.transfer_pa_apa200_to_b.load_split_pair"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "dataset hash mismatch"):
                    run_from_config(path, progress=lambda _: None)
                loader.assert_not_called()

    def test_published_pretest_bundle_reproduces_and_seals_metrics(self) -> None:
        result = verify_bundle(
            Path("experiments/results/pa_transfer_apa200_to_b_pretest")
        )
        self.assertTrue(result["artifact_hashes_verified"])
        self.assertTrue(result["dataset_hashes_verified"])
        self.assertTrue(result["test_never_opened_or_hashed"])
        self.assertGreaterEqual(result["checked_metric_records"], 10)


if __name__ == "__main__":
    unittest.main()
