import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.evaluate_frozen_pa import (
    evaluate_from_manifest,
    verify_selection_before_test_access,
)
from experiments.select_pa_mp import select_from_config


class FrozenPATestRunnerTests(unittest.TestCase):
    @staticmethod
    def _write_iq(path: Path, signal: np.ndarray) -> None:
        rows = ["I,Q"]
        rows.extend(f"{value.real:.17g},{value.imag:.17g}" for value in signal)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _make_selection(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "dataset"
        selection_output = root / "selection"
        dataset.mkdir()
        rng = np.random.default_rng(8)
        for split, sample_count in (("train", 24), ("val", 8), ("test", 8)):
            x = rng.normal(size=sample_count) + 1j * rng.normal(size=sample_count)
            y = (1.2 + 0.1j) * x + 0.02 * x * np.abs(x) ** 2
            self._write_iq(dataset / f"{split}_input.csv", x)
            self._write_iq(dataset / f"{split}_output.csv", y)
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
                    "output_dir": str(selection_output),
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
        select_from_config(config)
        return selection_output / "selection_manifest.json", dataset

    def test_hash_mismatch_is_rejected_before_test_loader_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = self._make_selection(Path(temporary))
            value = json.loads(manifest.read_text(encoding="utf-8"))
            model = Path(value["selected_model"])
            with model.open("ab") as stream:
                stream.write(b"corruption")

            with mock.patch(
                "experiments.evaluate_frozen_pa.load_split_pair"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verify_selection_before_test_access(manifest)
                loader.assert_not_called()

    def test_frozen_runner_writes_test_artifacts_without_refit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self._make_selection(root)
            output = root / "test_result"

            report = evaluate_from_manifest(
                manifest,
                output_directory=output,
            )

            self.assertTrue(
                report["integrity_checks_completed_before_test_access"]
            )
            self.assertFalse(report["refit_performed"])
            self.assertFalse(report["post_prediction_gain_fit"])
            self.assertFalse(report["post_prediction_delay_fit"])
            self.assertTrue((output / "test_evaluation.json").is_file())
            self.assertTrue((output / "test_prediction.npz").is_file())
            self.assertTrue((output / "test_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
