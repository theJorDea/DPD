import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.select_pa_mp import (
    enumerate_architecture_candidates,
    select_from_config,
)


class MPSelectionConfigurationTests(unittest.TestCase):
    def test_operation_budget_filters_large_full_order_candidate(self) -> None:
        config = {
            "order_families": [
                {
                    "name": "full",
                    "order_sets": [
                        list(range(1, 10)),
                    ],
                }
            ],
            "delay_counts": [16, 24],
            "architecture_ridge": 1e-8,
            "max_real_multiplications_per_sample": 1000,
        }
        candidates = enumerate_architecture_candidates(config)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["delay_count"], 16)
        self.assertEqual(
            candidates[0]["operation_count"].real_multiplications,
            960,
        )

    def test_duplicate_candidate_across_families_is_rejected(self) -> None:
        config = {
            "order_families": [
                {"name": "a", "order_sets": [[1, 3]]},
                {"name": "b", "order_sets": [[1, 3]]},
            ],
            "delay_counts": [1],
            "architecture_ridge": 1e-8,
            "max_real_multiplications_per_sample": 1000,
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            enumerate_architecture_candidates(config)


class MPSelectionIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_iq(path: Path, signal: np.ndarray) -> None:
        rows = ["I,Q"]
        rows.extend(f"{value.real:.17g},{value.imag:.17g}" for value in signal)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_selection_succeeds_without_any_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "result"
            dataset.mkdir()
            rng = np.random.default_rng(4)
            train_x = rng.normal(size=24) + 1j * rng.normal(size=24)
            val_x = rng.normal(size=8) + 1j * rng.normal(size=8)
            train_y = (1.3 - 0.2j) * train_x + 0.01 * train_x * np.abs(train_x) ** 2
            val_y = (1.3 - 0.2j) * val_x + 0.01 * val_x * np.abs(val_x) ** 2
            self._write_iq(dataset / "train_input.csv", train_x)
            self._write_iq(dataset / "train_output.csv", train_y)
            self._write_iq(dataset / "val_input.csv", val_x)
            self._write_iq(dataset / "val_output.csv", val_y)
            (dataset / "spec.json").write_text(
                json.dumps(
                    {
                        "input_signal_fs": 8.0,
                        "nperseg": 8,
                        "bw_main_ch": 2.0,
                        "n_sub_ch": 1,
                    }
                ),
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": str(dataset),
                        "dataset_label": "synthetic",
                        "output_dir": str(output),
                        "alignment_delay": 0,
                        "order_families": [
                            {
                                "name": "odd",
                                "order_sets": [[1, 3]],
                            }
                        ],
                        "delay_counts": [1],
                        "architecture_ridge": 1e-8,
                        "refinement_ridges": [1e-8],
                        "max_real_multiplications_per_sample": 1000,
                    }
                ),
                encoding="utf-8",
            )

            manifest = select_from_config(config)

            self.assertFalse(manifest["test_split_accessed"])
            self.assertEqual(manifest["test_evaluation_status"], "not_run_by_design")
            self.assertFalse((dataset / "test_input.csv").exists())
            self.assertTrue((output / "selected_mp_pa.npz").is_file())
            self.assertTrue((output / "selection_manifest.json").is_file())
            self.assertTrue((output / "validation_trials.json").is_file())


if __name__ == "__main__":
    unittest.main()
