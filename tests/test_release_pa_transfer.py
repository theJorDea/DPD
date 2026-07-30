import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.release_pa_transfer_apa200_to_b import (
    _load_pretest_manifest,
    _load_selected_coefficients,
    _verify_pretest_source_code,
    _verify_prior_release_incident,
    load_release_config,
    run_from_config,
)


RELEASE_CONFIG = Path(
    "experiments/configs/pa_transfer_apa200_to_b_release.json"
)


class TransferReleaseGuardTests(unittest.TestCase):
    def test_release_requires_explicit_acknowledgement_before_any_loader(self) -> None:
        with patch(
            "experiments.release_pa_transfer_apa200_to_b.load_split_pair"
        ) as loader:
            with self.assertRaisesRegex(PermissionError, "--release-test"):
                run_from_config(RELEASE_CONFIG, release_test=False, progress=lambda _: None)
            loader.assert_not_called()

    def test_release_config_and_frozen_selection_are_consistent(self) -> None:
        config = load_release_config(RELEASE_CONFIG)
        pretest_config, manifest, info = _load_pretest_manifest(
            config,
            RELEASE_CONFIG,
        )
        selected = _load_selected_coefficients(
            config,
            manifest,
            info["pretest_bundle"],
        )
        source_hashes = _verify_pretest_source_code(manifest)
        self.assertIn("experiments/transfer_pa_apa200_to_b.py", source_hashes)
        incident = _verify_prior_release_incident()
        self.assertEqual(incident["prior_held_out_access_count"], 1)
        self.assertFalse(incident["prior_test_metric_computed"])
        self.assertEqual(set(selected), {"causal_gmp", "lag9_sparse_spline_memory"})
        self.assertEqual(selected["causal_gmp"].shape, (444,))
        self.assertEqual(selected["lag9_sparse_spline_memory"].shape, (9, 12))
        self.assertEqual(
            manifest["target_transfer"]["causal_gmp"][
                "selected_calibration"
            ]["sample_count_per_frame"],
            16384,
        )
        self.assertEqual(
            manifest["target_transfer"]["lag9_sparse_spline_memory"][
                "selected_calibration"
            ]["sample_count_per_frame"],
            16384,
        )

    def test_release_config_does_not_prefill_target_test_hashes(self) -> None:
        config = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
        self.assertIsNone(config["target_test_files"]["input_sha256"])
        self.assertIsNone(config["target_test_files"]["output_sha256"])


if __name__ == "__main__":
    unittest.main()
