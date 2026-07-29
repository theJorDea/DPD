import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baseline.sparse_spline_memory_pa import (
    SparseSplineMemoryPA,
    SparseSplineMemoryPABranch,
)
from experiments.select_pa_sparse_spline_memory import verify_preregistered_inputs
from experiments.sparse_pa_benchmark_support import (
    acquire_lock,
    load_frozen_evidence,
    metric_summary,
    publish_bundle,
)


PRODUCTION_CONFIG = Path(
    "experiments/configs/pa_sparse_spline_memory_apa200.json"
)


class SparsePABenchmarkEvidenceTests(unittest.TestCase):
    def test_frozen_reference_archive_is_sealed_and_train_only(self) -> None:
        config = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
        verified = verify_preregistered_inputs(config, PRODUCTION_CONFIG)
        verified["config"] = config
        evidence = load_frozen_evidence(verified)
        self.assertEqual(evidence["gmp_train_oof_prediction"].shape, (58980,))
        self.assertTrue(np.all(np.isfinite(evidence["gmp_train_oof_prediction"])) )


class SparsePABenchmarkMetricTests(unittest.TestCase):
    def test_partial_frame_metric_and_common_mask_are_explicit(self) -> None:
        target = np.ones(11, dtype=np.complex128)
        prediction = target.copy()
        prediction[0] += 0.1
        prediction[5] += 0.1
        metrics = metric_summary(
            prediction,
            target,
            frame_lengths=(5, 6),
            common_warmup=1,
        )
        self.assertEqual(metrics["scored_sample_count_full"], 11)
        self.assertEqual(metrics["scored_sample_count_common"], 9)
        self.assertEqual(metrics["opendpd_complete_frame_count"], 1)
        self.assertEqual(len(metrics["per_frame"]), 2)


class SparsePABenchmarkPublicationTests(unittest.TestCase):
    def test_bundle_is_atomic_hashed_and_immutable(self) -> None:
        model = SparseSplineMemoryPA(
            knots=np.asarray([0.0, 1.0]),
            branches=(SparseSplineMemoryPABranch(0, 0),),
            coefficients=np.asarray([[1.0 + 0.0j, 0.8 - 0.1j]]),
        )
        signal = np.asarray([0.1 + 0.2j, -0.3j, 0.5 + 0.1j])
        prediction = model.predict(signal)
        residual = {
            "schema_version": 1,
            "split_role": "synthetic_fixture",
            "test_access_permitted": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "immutable_bundle"
            lock, payload = acquire_lock(output)
            try:
                manifest = publish_bundle(
                    output,
                    lock=lock,
                    lock_payload=payload,
                    model=model,
                    manifest_payload={
                        "schema_version": 1,
                        "test_split_accessed": False,
                        "test_file_hashes_recorded": False,
                    },
                    staged_ledger_payload={
                        "schema_version": 1,
                        "test_split_accessed": False,
                    },
                    predictions={"train_oof_prediction": prediction},
                    residual_reports={
                        "train_oof": residual,
                        "validation_reused": residual,
                    },
                    execution={
                        "schema_version": 1,
                        "test_split_accessed": False,
                    },
                )
                self.assertTrue(output.is_dir())
                self.assertFalse(manifest["test_split_accessed"])
                self.assertTrue((output / "selection_manifest.json").is_file())
                for artifact in manifest["artifacts"].values():
                    self.assertTrue((output / artifact["path"]).is_file())
                restored = SparseSplineMemoryPA.load(
                    output / "selected_sparse_pa.npz"
                )
                np.testing.assert_array_equal(
                    restored.coefficients, model.coefficients
                )
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    acquire_lock(output)
            finally:
                if lock.exists():
                    lock.unlink()


if __name__ == "__main__":
    unittest.main()
