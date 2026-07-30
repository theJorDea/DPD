import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments import train_opendpd_pa as runner


def _base_config(dataset_dir: str) -> dict:
    return {
        "schema_version": 1,
        "task": runner.TASK,
        "status": runner.CONFIG_STATUS,
        "dataset_dir": dataset_dir,
        "dataset_files": list(runner.ALLOWED_DATASET_FILES),
        "scope": {
            "test_split_access_permitted": False,
            "selection_split": "validation",
        },
        "framing": {
            "train_mode": "upstream_flat_windows",
            "validation_mode": "upstream_segments",
            "frame_length": 4,
            "frame_stride": 1,
            "nperseg": 8,
        },
        "training": {
            "seed": 0,
            "device": "cpu",
            "deterministic": True,
            "n_epochs": 2,
            "batch_size": 2,
            "batch_size_eval": 1,
            "lr": 5e-3,
            "lr_end": 5e-5,
            "decay_factor": 0.5,
            "patience": 5,
            "grad_clip_val": 200,
            "optimizer": "adamw",
            "loss": "mse",
        },
        "candidates": [
            {
                "name": "gru_h2",
                "backbone": "gru",
                "hidden_size": 2,
            }
        ],
        "selection_metric": "validation_opendpd_nmse_db",
    }


class OpenDPDSealedRunnerTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        (dataset / "spec.json").write_text("{}", encoding="utf-8")
        rows = "I,Q\n0.1,0.0\n0.2,0.1\n0.3,0.0\n0.4,-0.1\n"
        for name in (
            "train_input.csv",
            "train_output.csv",
            "val_input.csv",
            "val_output.csv",
        ):
            (dataset / name).write_text(rows, encoding="utf-8")
        # A sentinel proves that the runner does not need or hash test data.
        (dataset / "test_input.csv").write_text("this must never be read\n", encoding="utf-8")
        (dataset / "test_output.csv").write_text("this must never be read\n", encoding="utf-8")
        return dataset

    def test_config_rejects_forbidden_test_filename_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _base_config("dataset")
            config["dataset_files"][-1] = "test_output.csv"
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden test split filename"):
                runner.load_config(path)

    def test_config_requires_validation_only_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _base_config("dataset")
            config["selection_metric"] = "test_nmse"
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection metric"):
                runner.load_config(path)

    def test_test_split_is_rejected_before_path_construction(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "forbidden split"):
            runner.load_allowed_split("/does/not/exist", "test")

    def test_hash_manifest_contains_only_train_validation_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            config = _base_config(str(dataset))
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = runner.load_config(config_path)
            resolved, hashes = runner.verify_allowed_inputs(loaded, config_path)
            self.assertEqual(resolved, dataset.resolve())
            self.assertEqual(set(hashes), set(runner.ALLOWED_DATASET_FILES))
            self.assertFalse(any("test" in name for name in hashes))

    def test_allowed_loader_reads_train_and_val_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            train_x, train_y = runner.load_allowed_split(dataset, "train")
            val_x, val_y = runner.load_allowed_split(dataset, "val")
            self.assertEqual(train_x.shape, (4,))
            self.assertEqual(train_y.shape, (4,))
            self.assertEqual(val_x.shape, (4,))
            self.assertEqual(val_y.shape, (4,))

    def test_opendpd_nmse_matches_pooled_for_one_segment(self) -> None:
        target = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
        prediction = target * 0.5
        expected = 10.0 * np.log10(0.25)
        self.assertAlmostEqual(runner.opendpd_nmse_db(prediction, target), expected)
        self.assertAlmostEqual(runner.pooled_nmse_db(prediction, target), expected)

    def test_candidate_validation_rejects_unsupported_backbone(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported PA backbone"):
            runner._validate_candidate(
                {"name": "bad", "backbone": "lstm", "hidden_size": 2}
            )


if __name__ == "__main__":
    unittest.main()
