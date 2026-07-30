import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from experiments import train_opendpd_pa as runner


def _base_config(dataset_dir: str) -> dict:
    config = {
        "schema_version": 1,
        "task": runner.TASK,
        "status": runner.CONFIG_STATUS,
        "dataset_dir": dataset_dir,
        "dataset_files": list(runner.ALLOWED_DATASET_FILES),
        "dataset_files_sha256": {
            name: "0" * 64 for name in runner.ALLOWED_DATASET_FILES
        },
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
    config["source"] = {
        "opendpd_commit": "0" * 40,
        "files_sha256": {
            name: "0" * 64 for name in runner._required_source_names(config)
        },
    }
    return config


def _bind_dataset_hashes(config: dict, dataset: Path) -> None:
    config["dataset_files_sha256"] = {
        name: runner.sha256_file(dataset / name)
        for name in runner.ALLOWED_DATASET_FILES
    }


def _bind_source_hashes(config: dict) -> None:
    config["source"] = {
        "opendpd_commit": subprocess.check_output(
            ["git", "-C", str(runner.OPENDPD_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "files_sha256": {
            name: runner.sha256_file(runner.PROJECT_ROOT / name)
            for name in runner._required_source_names(config)
        },
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

    def test_config_rejects_duplicate_candidate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _base_config("dataset")
            config["candidates"].append(dict(config["candidates"][0]))
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate names"):
                runner.load_config(path)

    def test_config_rejects_unsafe_candidate_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _base_config("dataset")
            config["candidates"][0]["name"] = "../outside"
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe directory"):
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
            _bind_dataset_hashes(config, dataset)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = runner.load_config(config_path)
            resolved, hashes = runner.verify_allowed_inputs(loaded, config_path)
            self.assertEqual(resolved, dataset.resolve())
            self.assertEqual(set(hashes), set(runner.ALLOWED_DATASET_FILES))
            self.assertFalse(any("test" in name for name in hashes))

    def test_dataset_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            (dataset / "train_input.csv").unlink()
            (dataset / "train_input.csv").symlink_to(dataset / "test_input.csv")
            config = _base_config(str(dataset))
            _bind_dataset_hashes(config, dataset)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                runner.verify_allowed_inputs(
                    runner.load_config(config_path),
                    config_path,
                )

    def test_dataset_hash_mismatch_stops_before_waveform_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            config = _base_config(str(dataset))
            _bind_dataset_hashes(config, dataset)
            _bind_source_hashes(config)
            config["dataset_files_sha256"]["train_input.csv"] = "f" * 64
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = runner.load_config(config_path)
            with mock.patch.object(runner, "load_allowed_split") as loader:
                with self.assertRaisesRegex(RuntimeError, "dataset SHA-256 mismatch"):
                    runner.run_candidate(
                        loaded,
                        config_path,
                        loaded["candidates"][0],
                        output_dir=root / "output",
                    )
                loader.assert_not_called()

    def test_source_hash_mismatch_stops_before_waveform_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            config = _base_config(str(dataset))
            _bind_dataset_hashes(config, dataset)
            _bind_source_hashes(config)
            config["source"]["files_sha256"][
                "experiments/train_opendpd_pa.py"
            ] = "f" * 64
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            loaded = runner.load_config(config_path)
            with mock.patch.object(runner, "load_allowed_split") as loader:
                with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
                    runner.run_candidate(
                        loaded,
                        config_path,
                        loaded["candidates"][0],
                        output_dir=root / "output",
                    )
                loader.assert_not_called()

    def test_existing_output_is_rejected_before_source_or_waveform_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            config = _base_config(str(dataset))
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"
            output.mkdir()
            with mock.patch.object(runner, "verify_source_inputs") as source:
                with self.assertRaisesRegex(FileExistsError, "reuse"):
                    runner.run_candidate(
                        runner.load_config(config_path),
                        config_path,
                        config["candidates"][0],
                        output_dir=output,
                    )
                source.assert_not_called()

    def test_vendored_import_rejects_sys_modules_contamination(self) -> None:
        fake_models = types.ModuleType("models")
        fake_models.__file__ = "/tmp/not-opendpd/models.py"
        fake_modules = types.ModuleType("modules")
        fake_modules.__path__ = []
        fake_data = types.ModuleType("modules.data_collector")
        fake_data.__file__ = "/tmp/not-opendpd/data_collector.py"
        with mock.patch.dict(
            __import__("sys").modules,
            {
                "models": fake_models,
                "modules": fake_modules,
                "modules.data_collector": fake_data,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "outside vendored"):
                runner._import_opendpd()

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

    def test_flat_framing_reports_cross_boundary_windows(self) -> None:
        self.assertEqual(
            runner._count_cross_boundary_windows(
                [8, 8],
                total_samples=16,
                frame_length=4,
                stride=1,
            ),
            3,
        )
        with self.assertRaisesRegex(ValueError, "must sum"):
            runner._count_cross_boundary_windows(
                [7, 8],
                total_samples=16,
                frame_length=4,
                stride=1,
            )

    def test_candidate_validation_rejects_unsupported_backbone(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported PA backbone"):
            runner._validate_candidate(
                {"name": "bad", "backbone": "lstm", "hidden_size": 2}
            )


if __name__ == "__main__":
    unittest.main()
