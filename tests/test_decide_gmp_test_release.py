import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from baseline.train_spline import file_sha256
from experiments import analyze_pa_residuals as residual_runner
from experiments import decide_gmp_test_release as release
from tests.test_analyze_pa_residuals import _SyntheticResidualFixture


class ReleaseGateUnitTests(unittest.TestCase):
    def test_boundary_mask_excludes_both_sides_per_frame(self) -> None:
        np.testing.assert_array_equal(
            release._boundary_mask(
                11,
                nperseg=6,
                warmup=2,
                cooldown=1,
            ),
            np.asarray(
                [False, False, True, True, True, False,
                 False, False, True, True, False]
            ),
        )

    def test_streaming_and_reset_checks_are_bit_exact_for_causal_gmp(self) -> None:
        from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA

        config = GMPConfig(
            ka=2,
            la=4,
            leading_policy="causal_leading",
        )
        model = GeneralizedMemoryPolynomialPA(
            config,
            np.asarray(
                [
                    1.1 + 0.2j,
                    0.03 - 0.01j,
                    -0.02 + 0.04j,
                    0.01 + 0.02j,
                    0.005 - 0.003j,
                    -0.004 + 0.002j,
                    0.002 + 0.001j,
                    -0.001 + 0.003j,
                ],
                dtype=np.complex128,
            ),
        )
        rng = np.random.default_rng(123)
        signal = rng.normal(size=32) + 1j * rng.normal(size=32)
        result = release._streaming_checks(model, signal, nperseg=16)
        self.assertTrue(result["streaming_equivalence_passed"])
        self.assertTrue(result["reset_at_frame_equivalence_passed"])
        self.assertEqual(result["maximum_streaming_error"], 0.0)

    def test_diagnostic_bundle_rejects_missing_required_families(self) -> None:
        report = {key: [] for key in release._REQUIRED_DIAGNOSTIC_KEYS}
        report["global_metrics"] = {}
        self.assertFalse(release._diagnostic_bundle_complete(report))


class ReleaseGateIntegrationTests(unittest.TestCase):
    """Exercise the whole release path on train/validation-only synthetic data."""

    def setUp(self) -> None:
        self.fixture = _SyntheticResidualFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_release_gate_never_loads_test_and_publishes_pass(self) -> None:
        mp_selection_config, mp_selection_manifest, mp_selection = (
            self.fixture.build_selection("mp")
        )
        mp_residual_config, mp_output = self.fixture.write_residual_config(
            "mp",
            selection_config=mp_selection_config,
            selection_manifest=mp_selection_manifest,
            selection=mp_selection,
            output_name="mp_gate_residual",
        )
        residual_runner.analyze_from_config(mp_residual_config)

        gmp_selection_config, gmp_selection_manifest, gmp_selection = (
            self.fixture.build_selection("gmp")
        )
        gmp_residual_config, gmp_output = self.fixture.write_residual_config(
            "gmp",
            selection_config=gmp_selection_config,
            selection_manifest=gmp_selection_manifest,
            selection=gmp_selection,
            output_name="gmp_gate_residual",
        )
        config = json.loads(gmp_residual_config.read_text(encoding="utf-8"))
        config["dataset_label"] = "synthetic_gmp"
        config["amplitude_quantiles"] = [0.9, 0.95, 0.99]

        alignment_path = self.fixture.root / "alignment_decision.json"
        alignment = {
            "schema_version": 1,
            "test_split_accessed": False,
            "datasets": {
                "synthetic_gmp": {
                    "frozen_protocol_variant": "a0",
                    "manual_override_of_runner_recommendation": False,
                    "formal_protocol_variant": (
                        "A0_integer_only_no_fractional_transform"
                    ),
                    "formal_gmp_config_sha256": file_sha256(
                        gmp_selection_config
                    ),
                }
            },
        }
        alignment_path.write_text(
            json.dumps(alignment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config["protocol_lock"] = {
            "alignment_decision": self.fixture.relative(alignment_path),
            "alignment_decision_sha256": file_sha256(alignment_path),
            "formal_variant": "A0_integer_only_no_fractional_transform",
            "integer_delay_samples": 0,
            "fractional_transform_applied": False,
        }

        mp_manifest_path = mp_output / "residual_manifest.json"
        mp_manifest = json.loads(mp_manifest_path.read_text(encoding="utf-8"))
        with np.load(
            mp_output / "residual_predictions.npz", allow_pickle=False
        ) as data:
            train_mp = np.asarray(data["train_oof_prediction"])
            val_mp = np.asarray(data["validation_prediction"])
        train_input, train_output = residual_runner.load_split_pair(
            self.fixture.dataset, "train"
        )
        val_input, val_output = residual_runner.load_split_pair(
            self.fixture.dataset, "val"
        )
        warmup = int(gmp_selection["common_warmup_samples_per_frame"])
        train_mask = release._boundary_mask(
            train_input.size,
            nperseg=64,
            warmup=warmup,
            cooldown=0,
        )
        val_mask = release._boundary_mask(
            val_input.size,
            nperseg=64,
            warmup=warmup,
            cooldown=0,
        )
        mp_train = release._metric_pair(train_mp, train_output, train_mask)
        mp_val = release._metric_pair(val_mp, val_output, val_mask)
        selected_trial = gmp_selection["selected_trial"]
        gmp_val_full = float(
            selected_trial["validation_full_record"]["complex_nmse_pooled_db"]
        )
        gmp_val_interior = float(
            selected_trial["validation_common_interior"][
                "complex_nmse_pooled_db"
            ]
        )
        minimum_full = max(0.10, 0.25 * (mp_val["full_nmse_db"] - gmp_val_full))
        minimum_interior = max(
            0.10,
            0.25 * (mp_val["common_interior_nmse_db"] - gmp_val_interior),
        )
        config["pretest_release_gate"] = {
            "status": "preregistered_before_gmp_residual_run",
            "scope": "synthetic unit-test authorization only",
            "matched_boundary": {
                "warmup_samples_per_frame": warmup,
                "future_cooldown_samples_per_frame": 0,
            },
            "mp_reference": {
                "residual_manifest": self.fixture.relative(mp_manifest_path),
                "residual_manifest_sha256": file_sha256(mp_manifest_path),
                "predictions_sha256": mp_manifest["predictions_sha256"],
                "train_oof_full_nmse_db": mp_train["full_nmse_db"],
                "train_oof_matched_interior_nmse_db": mp_train[
                    "common_interior_nmse_db"
                ],
                "validation_full_nmse_db": mp_val["full_nmse_db"],
                "validation_matched_interior_nmse_db": mp_val[
                    "common_interior_nmse_db"
                ],
            },
            "gmp_validation_reference": {
                "full_nmse_db": gmp_val_full,
                "common_interior_nmse_db": gmp_val_interior,
                "validation_fraction_above_training_maximum": float(
                    selected_trial["validation_input_support"][
                        "fraction_above_training_maximum"
                    ]
                ),
            },
            "threshold_definition": {
                "validation_metric_absolute_tolerance_db": 1e-9
            },
            "thresholds": {
                "minimum_oof_gain_full_db": minimum_full,
                "minimum_oof_gain_common_interior_db": minimum_interior,
                "maximum_gmp_oof_full_nmse_db": (
                    mp_train["full_nmse_db"] - minimum_full
                ),
                "maximum_gmp_oof_common_interior_nmse_db": (
                    mp_train["common_interior_nmse_db"] - minimum_interior
                ),
                "maximum_oof_to_validation_degradation_db": 100.0,
                "maximum_absolute_full_minus_common_interior_db": 100.0,
                "maximum_fold_condition_ratio": 100.0,
                "maximum_fold_coefficient_l2_norm_ratio": 100.0,
                "maximum_held_to_fit_amplitude_ratio": 2.0,
                "real_multiplication_limit_exclusive": 1000,
            },
        }
        gmp_residual_config.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        residual_runner.analyze_from_config(gmp_residual_config)

        accessed: list[str] = []
        original_loader = release.load_split_pair

        def tracking_loader(dataset: Path, split: str):
            accessed.append(split)
            return original_loader(dataset, split)

        with mock.patch.object(
            release,
            "load_split_pair",
            side_effect=tracking_loader,
        ):
            report = release.decide_from_config(gmp_residual_config)

        self.assertEqual(accessed, ["train", "val"])
        self.assertTrue(report["decision"]["may_open_gmp_test_once"])
        self.assertFalse(report["test_split_accessed"])
        self.assertEqual(report["decision"]["failed_predicates"], [])
        self.assertTrue(
            (gmp_output / "test_release_gate.json").is_file()
        )
        self.assertFalse((self.fixture.dataset / "test_input.csv").exists())
        self.assertFalse((self.fixture.dataset / "test_output.csv").exists())


if __name__ == "__main__":
    unittest.main()
