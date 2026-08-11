"""Safety and metric tests for the one-shot BlackBox DPD release."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.release_blackbox_direct_dpd import (
    _load_preregistration,
    _model_metrics,
    _nmse_db,
    release,
)


class BlackBoxDirectDPDReleaseTests(unittest.TestCase):
    def test_release_flag_is_checked_before_preregistration_access(self) -> None:
        with self.assertRaisesRegex(PermissionError, "--release-test"):
            release(
                Path("/definitely/not/a/preregistration.json"),
                Path("/definitely/not/an/output"),
                release_test=False,
            )

    def test_existing_output_is_rejected_before_preregistration_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                release(
                    Path(directory) / "missing.json",
                    Path(directory),
                    release_test=True,
                )

    def test_nmse_and_chronological_segments(self) -> None:
        reference = np.ones(12, dtype=np.complex128)
        estimate = reference + 0.1
        self.assertAlmostEqual(_nmse_db(estimate, reference), -20.0)
        metrics = _model_metrics(
            estimate,
            reference,
            warmup=4,
            segment_count=4,
        )
        self.assertEqual(metrics["scored_sample_count"], 8)
        self.assertEqual([row["sample_count"] for row in metrics["segments"]],
                         [2, 2, 2, 2])
        self.assertTrue(all(
            abs(row["nmse_db"] + 20.0) < 1e-12
            for row in metrics["segments"]
        ))

    def test_preregistration_requires_frozen_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prereg.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "task": "blackbox_direct_dpd_one_shot_release",
                    "status": "draft",
                    "test_output_opened": False,
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not frozen"):
                _load_preregistration(path)


if __name__ == "__main__":
    unittest.main()
