import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baseline.gmp_pa import (
    GMPConfig,
    GeneralizedMemoryPolynomialPA,
)
from experiments.select_pa_gmp import (
    enumerate_architecture_candidates,
    segmented_interior_mask,
    select_from_config,
)


class GMPSelectionConfigurationTests(unittest.TestCase):
    @staticmethod
    def _base_config() -> dict:
        return {
            "ka_values": [5],
            "memory_lengths": [24, 30],
            "topologies": [
                {
                    "name": "aligned",
                    "kb": 0,
                    "mb": 0,
                    "kc": 0,
                    "mc": 0,
                    "leading_policy": "causal_leading",
                }
            ],
            "architecture_solver_mode": "ridge_lstsq",
            "architecture_ridge": 1e-8,
            "refinement_ridges": [1e-8],
            "max_real_multiplications_per_sample": 1000,
        }

    def test_real_multiplication_budget_is_strictly_exclusive(self) -> None:
        config = self._base_config()
        # The L=30 aligned candidate costs exactly 364 real MUL/sample.
        config["max_real_multiplications_per_sample"] = 364
        candidates = enumerate_architecture_candidates(config)
        self.assertEqual(
            [candidate["gmp_config"].la for candidate in candidates],
            [24],
        )
        self.assertEqual(
            candidates[0]["operation_count"].real_multiplications,
            292,
        )

    def test_candidate_order_and_truncated_svd_recipe_are_deterministic(
        self,
    ) -> None:
        config = self._base_config()
        config.update(
            {
                "ka_values": [3, 5],
                "memory_lengths": [2],
                "topologies": [
                    {
                        "name": "aligned",
                        "kb": 0,
                        "mb": 0,
                        "kc": 0,
                        "mc": 0,
                    },
                    {
                        "name": "lag",
                        "kb": 1,
                        "mb": 1,
                        "kc": 0,
                        "mc": 0,
                    },
                ],
                "architecture_solver_mode": "truncated_svd",
                "architecture_ridge": 0.0,
                "architecture_svd_rcond": 1e-4,
            }
        )
        candidates = enumerate_architecture_candidates(config)
        self.assertEqual(
            [
                (
                    candidate["gmp_config"].ka,
                    candidate["topology"],
                )
                for candidate in candidates
            ],
            [(3, "aligned"), (3, "lag"), (5, "aligned"), (5, "lag")],
        )
        self.assertTrue(
            all(
                candidate["solver_mode"] == "truncated_svd"
                and candidate["ridge"] == 0.0
                and candidate["svd_rcond"] == 1e-4
                for candidate in candidates
            )
        )

    def test_duplicate_generated_configuration_is_rejected(self) -> None:
        config = self._base_config()
        config["memory_lengths"] = [2]
        config["topologies"] = [
            {"name": "aligned_a", "kb": 0, "mb": 0, "kc": 0, "mc": 0},
            {"name": "aligned_b", "kb": 0, "mb": 0, "kc": 0, "mc": 0},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate GMP configuration"):
            enumerate_architecture_candidates(config)

    def test_truncated_svd_requires_explicit_valid_cutoff(self) -> None:
        config = self._base_config()
        config["architecture_solver_mode"] = "truncated_svd"
        config["architecture_ridge"] = 0.0
        with self.assertRaisesRegex(ValueError, "architecture_svd_rcond"):
            enumerate_architecture_candidates(config)


class GMPCommonInteriorTests(unittest.TestCase):
    def test_each_frame_discards_the_same_warmup_and_cooldown(self) -> None:
        mask = segmented_interior_mask(
            16,
            segment_length=8,
            warmup_samples=2,
            cooldown_samples=1,
        )
        expected = np.zeros(16, dtype=bool)
        expected[2:7] = True
        expected[10:15] = True
        np.testing.assert_array_equal(mask, expected)

    def test_empty_common_support_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "consumes every"):
            segmented_interior_mask(
                8,
                segment_length=4,
                warmup_samples=2,
                cooldown_samples=2,
            )


class GMPSelectionIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_iq(path: Path, signal: np.ndarray) -> None:
        rows = ["I,Q"]
        rows.extend(
            f"{value.real:.17g},{value.imag:.17g}" for value in signal
        )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_selection_uses_only_train_validation_and_common_interior(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "selection"
            dataset.mkdir()
            rng = np.random.default_rng(41)
            truth_config = GMPConfig(ka=2, la=2)
            truth = GeneralizedMemoryPolynomialPA(
                truth_config,
                np.asarray(
                    [
                        1.2 + 0.1j,
                        0.04 - 0.02j,
                        0.03 + 0.01j,
                        -0.01 + 0.005j,
                    ],
                    dtype=np.complex128,
                ),
            )
            train_x = rng.normal(size=64) + 1j * rng.normal(size=64)
            val_x = rng.normal(size=32) + 1j * rng.normal(size=32)
            train_y = truth.predict_segments(train_x, 16)
            val_y = truth.predict_segments(val_x, 16)
            # Keep all JSON metrics finite and prevent an exact validation tie.
            train_y += 1e-5 * (
                rng.normal(size=train_y.size)
                + 1j * rng.normal(size=train_y.size)
            )
            val_y += 1e-5 * (
                rng.normal(size=val_y.size)
                + 1j * rng.normal(size=val_y.size)
            )
            self._write_iq(dataset / "train_input.csv", train_x)
            self._write_iq(dataset / "train_output.csv", train_y)
            self._write_iq(dataset / "val_input.csv", val_x)
            self._write_iq(dataset / "val_output.csv", val_y)
            (dataset / "spec.json").write_text(
                json.dumps(
                    {
                        "input_signal_fs": 16.0,
                        "nperseg": 16,
                        "bw_main_ch": 4.0,
                        "n_sub_ch": 1,
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "gmp_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": str(dataset),
                        "dataset_label": "synthetic-gmp",
                        "output_dir": str(output),
                        "alignment_delay": 0,
                        "characteristic_bins": 8,
                        "ka_values": [2],
                        "memory_lengths": [2],
                        "topologies": [
                            {
                                "name": "aligned",
                                "kb": 0,
                                "mb": 0,
                                "kc": 0,
                                "mc": 0,
                                "leading_policy": "causal_leading",
                                "selection_eligible": True,
                            },
                            {
                                "name": "future_diagnostic",
                                "kb": 0,
                                "mb": 0,
                                "kc": 1,
                                "mc": 1,
                                "leading_policy": "opendpd_exact",
                                "selection_eligible": False,
                            },
                        ],
                        "architecture_solver_mode": "truncated_svd",
                        "architecture_ridge": 0.0,
                        "architecture_svd_rcond": 1e-6,
                        "refinement_ridges": [0.0, 1e-8],
                        "max_real_multiplications_per_sample": 1000,
                    }
                ),
                encoding="utf-8",
            )

            manifest = select_from_config(config_path)

            self.assertEqual(
                manifest["model_class"],
                "complex_generalized_memory_polynomial",
            )
            self.assertFalse(manifest["test_split_accessed"])
            self.assertEqual(
                manifest["common_warmup_samples_per_frame"],
                1,
            )
            self.assertEqual(
                manifest["common_future_cooldown_samples_per_frame"],
                1,
            )
            self.assertFalse((dataset / "test_input.csv").exists())
            self.assertFalse((dataset / "test_output.csv").exists())
            self.assertNotIn(
                "test_input.csv",
                manifest["dataset_files_sha256"],
            )
            self.assertTrue((output / "selected_gmp_pa.npz").is_file())
            self.assertTrue((output / "selection_manifest.json").is_file())
            self.assertTrue((output / "validation_trials.json").is_file())
            self.assertEqual(
                manifest["selected_trial"]["gmp_config"]["leading_policy"],
                "causal_leading",
            )
            self.assertLess(
                manifest["operation_budget"]["selected_value"],
                manifest["operation_budget"]["maximum_exclusive"],
            )

            ledger = json.loads(
                (output / "validation_trials.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(ledger["trials"]), 4)
            self.assertTrue(
                all(
                    trial["validation_common_interior"][
                        "scored_sample_count"
                    ]
                    == 28
                    for trial in ledger["trials"]
                )
            )
            self.assertTrue(
                any(
                    trial["solver_mode"] == "truncated_svd"
                    for trial in ledger["trials"]
                )
            )
            restored = GeneralizedMemoryPolynomialPA.load(
                output / "selected_gmp_pa.npz"
            )
            dataclass_config = manifest["selected_trial"]["gmp_config"]
            self.assertEqual(
                dataclass_config,
                {
                    "ka": restored.config.ka,
                    "la": restored.config.la,
                    "kb": restored.config.kb,
                    "lb": restored.config.lb,
                    "mb": restored.config.mb,
                    "kc": restored.config.kc,
                    "lc": restored.config.lc,
                    "mc": restored.config.mc,
                    "leading_policy": restored.config.leading_policy,
                },
            )
            self.assertEqual(dataclass_config["ka"], 2)


if __name__ == "__main__":
    unittest.main()
