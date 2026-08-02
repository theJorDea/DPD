from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import run_surrogate_demo as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "experiments/configs/surrogate_demo.json"


class SurrogateDemoTests(unittest.TestCase):
    def _config(self) -> dict[str, object]:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def test_frozen_config_and_file_hashes_are_valid(self) -> None:
        config = self._config()
        runner.validate_config(config)
        verified = runner._verify_frozen_files(config)
        self.assertEqual(set(verified), set(config["files_sha256"]))
        self.assertEqual(
            config["claims_scope"],
            {
                "surrogate_only": True,
                "physical_pa_result": False,
                "rf_harmonic_claim": False,
                "test_split_accessed": False,
                "model_selection_performed": False,
                "validation_reused_after_historical_model_selection": True,
            },
        )

    def test_config_rejects_claim_expansion_and_unbound_child_config(self) -> None:
        config = self._config()
        physical = copy.deepcopy(config)
        physical["claims_scope"]["physical_pa_result"] = True
        with self.assertRaisesRegex(ValueError, "claims_scope"):
            runner.validate_config(physical)

        unbound = copy.deepcopy(config)
        del unbound["files_sha256"][
            "experiments/configs/dpd_fixed_point_dpa200_validation.json"
        ]
        with self.assertRaisesRegex(ValueError, "not hash-bound"):
            runner.validate_config(unbound)

    def test_metric_tolerance_is_explicit_and_enforced(self) -> None:
        self.assertEqual(
            runner._assert_close(
                1.0 + 5e-9,
                1.0,
                tolerance=1e-8,
                field="within",
            ),
            1.0 + 5e-9,
        )
        with self.assertRaisesRegex(ValueError, "reference metric mismatch"):
            runner._assert_close(
                1.0 + 2e-8,
                1.0,
                tolerance=1e-8,
                field="outside",
            )

    def test_fixed_integrity_rejects_saturation_and_streaming_failure(self) -> None:
        base = {
            split: {
                "stats": {
                    "output_saturations": 0,
                    "knot_code_collision_count": 0,
                },
                "streaming": {"streaming_chunk_equivalence_passed": True},
            }
            for split in ("train", "validation")
        }
        base["selection_or_tuning"] = {
            "precision_candidates_preregistered": [16, 14, 12],
            "precision_selected_by_runner": False,
            "scales_frozen_before_validation": True,
            "used_for_selection": False,
            "validation_used_to_modify_model": False,
        }
        base["validation"]["phase_equivariance"] = {
            "bit_exact": True,
            "rotated_input_stats": {
                "output_saturations": 0,
                "knot_code_collision_count": 0,
            },
        }
        runner._assert_fixed_integrity(base, field="valid")

        saturated = copy.deepcopy(base)
        saturated["validation"]["stats"]["output_saturations"] = 1
        with self.assertRaisesRegex(ValueError, "output_saturations"):
            runner._assert_fixed_integrity(saturated, field="saturated")

        discontinuous = copy.deepcopy(base)
        discontinuous["train"]["streaming"][
            "streaming_chunk_equivalence_passed"
        ] = False
        with self.assertRaisesRegex(ValueError, "streaming equivalence"):
            runner._assert_fixed_integrity(discontinuous, field="streaming")

        rotated_saturation = copy.deepcopy(base)
        rotated_saturation["validation"]["phase_equivariance"][
            "rotated_input_stats"
        ]["output_saturations"] = 1
        with self.assertRaisesRegex(ValueError, "phase_equivariance"):
            runner._assert_fixed_integrity(
                rotated_saturation, field="rotated_saturation"
            )

        selected_precision = copy.deepcopy(base)
        selected_precision["selection_or_tuning"][
            "precision_selected_by_runner"
        ] = True
        with self.assertRaisesRegex(ValueError, "selection contract"):
            runner._assert_fixed_integrity(
                selected_precision, field="selected_precision"
            )

    def test_failed_run_removes_owned_incomplete_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            with mock.patch.object(
                runner,
                "replay_float",
                side_effect=RuntimeError("injected failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    runner.run(CONFIG, output)
            self.assertFalse(output.exists())

    @unittest.skipUnless(
        (
            PROJECT_ROOT
            / "vendor/OpenDPD/datasets/DPA_200MHz/val_input.csv"
        ).is_file()
        and (
            PROJECT_ROOT
            / "vendor/OpenDPD/datasets/APA_200MHz/val_input.csv"
        ).is_file(),
        "OpenDPD validation submodule data are unavailable",
    )
    def test_sealed_end_to_end_demo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            result = runner.run(CONFIG, output)
            self.assertTrue(result["all_checks_passed"])
            self.assertGreater(
                result["execution"]["wall_seconds_before_summary_publication"],
                0.0,
            )
            self.assertEqual(
                result["provenance"]["config_sha256"],
                runner.sha256_file(CONFIG),
            )
            self.assertEqual(
                tuple(result["datasets"]),
                ("DPA_200MHz", "APA_200MHz"),
            )
            for dataset in result["datasets"].values():
                self.assertTrue(dataset["all_checks_passed"])
                self.assertFalse(dataset["fixed_point"]["precision_selected"])
                self.assertEqual(
                    dataset["fixed_point"]["saturation_or_collision_count"],
                    0,
                )
                self.assertTrue(
                    dataset["fixed_point"]["streaming_chunk_equivalence"]
                )
                self.assertLess(
                    dataset["float"]["metrics"]["dpd_nmse_db"],
                    dataset["float"]["metrics"]["no_dpd_nmse_db"],
                )

            summary = output / "summary.json"
            completion = output / "completion_manifest.json"
            self.assertTrue(summary.is_file())
            self.assertTrue(completion.is_file())
            manifest = json.loads(completion.read_text(encoding="utf-8"))
            self.assertTrue(manifest["all_checks_passed"])
            self.assertTrue(manifest["completion_manifest_published_last"])
            self.assertGreater(
                manifest["wall_seconds_before_completion_publication"], 0.0
            )
            self.assertFalse(manifest["fit_performed"])
            self.assertFalse(manifest["selection_performed"])
            self.assertFalse(manifest["measured_output_opened"])
            self.assertFalse(manifest["test_split_accessed"])
            self.assertFalse(manifest["physical_pa_result"])
            self.assertFalse(manifest["rf_harmonic_claim"])
            self.assertTrue(manifest["surrogate_only"])
            self.assertFalse(manifest["precision_selected"])
            self.assertTrue(
                manifest["validation_reused_after_historical_model_selection"]
            )
            self.assertEqual(
                manifest["artifacts"]["summary.json"],
                runner.sha256_file(summary),
            )
            self.assertEqual(
                len(manifest["child_completion_manifests_sha256"]), 12
            )
            self.assertLessEqual(
                summary.stat().st_mtime_ns,
                completion.stat().st_mtime_ns,
            )

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                runner.run(CONFIG, output)


if __name__ == "__main__":
    unittest.main()
