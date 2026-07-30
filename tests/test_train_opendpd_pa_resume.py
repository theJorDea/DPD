import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from experiments import train_opendpd_pa as runner

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _contract(output: Path) -> dict:
    candidate = {
        "name": "tres_gru_h2",
        "backbone": "tres_gru",
        "hidden_size": 2,
    }
    return {
        "schema_version": runner.RESUME_SCHEMA_VERSION,
        "task": runner.TASK,
        "config": {
            "path": "experiments/configs/test.json",
            "sha256": "1" * 64,
        },
        "candidate": candidate,
        "candidate_sha256": runner.sha256_json(candidate),
        "output_directory": str(output.resolve()),
        "dataset": {
            "directory": "vendor/OpenDPD/datasets/APA_200MHz",
            "files_sha256": {
                "spec.json": "2" * 64,
                "train_input.csv": "3" * 64,
                "train_output.csv": "4" * 64,
                "val_input.csv": "5" * 64,
                "val_output.csv": "6" * 64,
            },
            "test_file_hashes_recorded": False,
        },
        "source": {
            "vendored_commit": "7" * 40,
            "vendored_worktree_clean": True,
            "files": {
                "experiments/train_opendpd_pa.py": "8" * 64,
            },
            "verified_before_waveform_access": True,
        },
        "recipe": {
            "requested_epochs": 3,
            "effective_epochs": 3,
            "max_epochs_argument": None,
            "max_train_batches": None,
            "max_validation_batches": None,
            "training": {"seed": 0},
            "framing": {"train_mode": "upstream_flat_windows"},
        },
        "runtime_signature": {
            "python": "test-python",
            "numpy": "test-numpy",
            "torch": "test-torch",
            "device": "cpu",
            "torch_threads": 1,
            "platform": "test-platform",
        },
        "scope": {
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
            "selection_split": "validation",
        },
    }


def _state_summary(completed_epochs: int) -> dict:
    history = [
        {
            "epoch": epoch,
            "validation_opendpd_nmse_db": -10.0 - epoch,
        }
        for epoch in range(completed_epochs)
    ]
    return {
        "history": history,
        "best_epoch": None if completed_epochs == 0 else completed_epochs - 1,
        "best_validation_opendpd_nmse_db": (
            None if completed_epochs == 0 else -10.0 - (completed_epochs - 1)
        ),
        "productive_fit_seconds": float(completed_epochs),
    }


def _write_dummy_state(
    output: Path,
    completed_epochs: int,
    *,
    payload: bytes | None = None,
) -> tuple[Path, str]:
    _, states_dir, _ = runner._ensure_resume_directories(output)
    content = (
        payload
        if payload is not None
        else f"dummy-state-{completed_epochs}".encode("ascii")
    )
    temporary = states_dir / f"payload-{completed_epochs}.tmp"
    temporary.write_bytes(content)
    digest = runner.sha256_file(temporary)
    path = states_dir / (
        f"state_epoch_{completed_epochs:06d}_{digest[:16]}.pt"
    )
    temporary.replace(path)
    return path, digest


def _publish_epoch(
    output: Path,
    contract: dict,
    completed_epochs: int,
    *,
    previous_journal_sha256: str | None,
) -> tuple[dict, Path, str]:
    state_path, state_digest = _write_dummy_state(output, completed_epochs)
    state = _state_summary(completed_epochs)
    record = runner._journal_record(
        contract=contract,
        contract_hash=runner.sha256_json(contract),
        completed_epochs=completed_epochs,
        state_path=state_path,
        state_sha256=state_digest,
        state=state,
        previous_journal_sha256=previous_journal_sha256,
        session_id="test-session",
        recovered_orphan=False,
    )
    journal_path, journal_digest = runner._publish_journal_record(output, record)
    return record, journal_path, journal_digest


class OpenDPDResumeLayoutTests(unittest.TestCase):
    def test_exclusive_json_is_immutable_and_returns_file_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "immutable.json"
            first = {"value": 1}
            digest = runner._write_json_exclusive_atomic(path, first)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), first)
            self.assertEqual(digest, runner.sha256_file(path))
            original_bytes = path.read_bytes()

            with self.assertRaisesRegex(FileExistsError, "immutable artifact"):
                runner._write_json_exclusive_atomic(path, {"value": 2})

            self.assertEqual(path.read_bytes(), original_bytes)

    def test_run_manifest_allows_exact_resume_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            output.mkdir()
            contract = _contract(output)

            created = runner._initialize_or_verify_run_manifest(
                output,
                contract,
                resume=False,
            )
            observed = runner._initialize_or_verify_run_manifest(
                output,
                copy.deepcopy(contract),
                resume=True,
            )
            self.assertEqual(observed, created)

            changed = copy.deepcopy(contract)
            changed["runtime_signature"]["torch_threads"] = 2
            with self.assertRaisesRegex(RuntimeError, "resume contract mismatch"):
                runner._initialize_or_verify_run_manifest(
                    output,
                    changed,
                    resume=True,
                )

    def test_contiguous_hash_chained_journal_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            output.mkdir()
            contract = _contract(output)
            runner._ensure_resume_directories(output)

            _, _, first_digest = _publish_epoch(
                output,
                contract,
                0,
                previous_journal_sha256=None,
            )
            _publish_epoch(
                output,
                contract,
                1,
                previous_journal_sha256=first_digest,
            )

            records, orphan = runner._load_resume_layout(
                output,
                contract=contract,
                contract_hash=runner.sha256_json(contract),
            )
            self.assertEqual(
                [record["completed_epochs"] for record in records],
                [0, 1],
            )
            self.assertEqual(
                records[1]["previous_journal_sha256"],
                first_digest,
            )
            self.assertIsNone(orphan)

    def test_journal_gap_and_broken_hash_chain_are_rejected(self) -> None:
        with self.subTest("gap"):
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "candidate"
                output.mkdir()
                contract = _contract(output)
                runner._ensure_resume_directories(output)
                state_path, state_digest = _write_dummy_state(output, 1)
                record = runner._journal_record(
                    contract=contract,
                    contract_hash=runner.sha256_json(contract),
                    completed_epochs=1,
                    state_path=state_path,
                    state_sha256=state_digest,
                    state=_state_summary(1),
                    previous_journal_sha256=None,
                    session_id="test-session",
                    recovered_orphan=False,
                )
                runner._publish_journal_record(output, record)

                with self.assertRaisesRegex(RuntimeError, "not contiguous"):
                    runner._load_resume_layout(
                        output,
                        contract=contract,
                        contract_hash=runner.sha256_json(contract),
                    )

        with self.subTest("broken chain"):
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "candidate"
                output.mkdir()
                contract = _contract(output)
                runner._ensure_resume_directories(output)
                _publish_epoch(
                    output,
                    contract,
                    0,
                    previous_journal_sha256=None,
                )
                _publish_epoch(
                    output,
                    contract,
                    1,
                    previous_journal_sha256="f" * 64,
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    "contract validation",
                ):
                    runner._load_resume_layout(
                        output,
                        contract=contract,
                        contract_hash=runner.sha256_json(contract),
                    )

    def test_corrupt_journaled_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            output.mkdir()
            contract = _contract(output)
            record, _, _ = _publish_epoch(
                output,
                contract,
                0,
                previous_journal_sha256=None,
            )
            state_path = output / record["state_path"]
            state_path.write_bytes(b"corrupted-after-journal")

            with self.assertRaisesRegex(RuntimeError, "hash/name mismatch"):
                runner._load_resume_layout(
                    output,
                    contract=contract,
                    contract_hash=runner.sha256_json(contract),
                )

    def test_symlinked_journaled_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            output.mkdir()
            contract = _contract(output)
            record, _, _ = _publish_epoch(
                output,
                contract,
                0,
                previous_journal_sha256=None,
            )
            state_path = output / record["state_path"]
            target = output / "outside-state.pt"
            target.write_bytes(state_path.read_bytes())
            state_path.unlink()
            state_path.symlink_to(target)

            with self.assertRaisesRegex(RuntimeError, "missing or a symlink"):
                runner._load_resume_layout(
                    output,
                    contract=contract,
                    contract_hash=runner.sha256_json(contract),
                )

    def test_journal_state_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            output.mkdir()
            contract = _contract(output)
            runner._ensure_resume_directories(output)
            state_path, state_digest = _write_dummy_state(output, 0)
            record = runner._journal_record(
                contract=contract,
                contract_hash=runner.sha256_json(contract),
                completed_epochs=0,
                state_path=state_path,
                state_sha256=state_digest,
                state=_state_summary(0),
                previous_journal_sha256=None,
                session_id="test-session",
                recovered_orphan=False,
            )
            record["state_path"] = "../outside.pt"
            runner._publish_journal_record(output, record)

            with self.assertRaisesRegex(RuntimeError, "escapes its directory"):
                runner._load_resume_layout(
                    output,
                    contract=contract,
                    contract_hash=runner.sha256_json(contract),
                )

    def test_exact_next_epoch_orphan_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            output.mkdir()
            contract = _contract(output)
            runner._ensure_resume_directories(output)
            _, _, journal_digest = _publish_epoch(
                output,
                contract,
                0,
                previous_journal_sha256=None,
            )
            expected_orphan, _ = _write_dummy_state(output, 1)

            records, orphan = runner._load_resume_layout(
                output,
                contract=contract,
                contract_hash=runner.sha256_json(contract),
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(
                runner.sha256_file(
                    output
                    / runner.RESUME_DIRECTORY
                    / runner.RESUME_JOURNAL_DIRECTORY
                    / "epoch_000000.json"
                ),
                journal_digest,
            )
            self.assertEqual(orphan, expected_orphan)

    def test_invalid_or_multiple_orphans_are_rejected(self) -> None:
        with self.subTest("wrong next epoch"):
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "candidate"
                output.mkdir()
                contract = _contract(output)
                runner._ensure_resume_directories(output)
                _write_dummy_state(output, 2)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "not the next epoch",
                ):
                    runner._load_resume_layout(
                        output,
                        contract=contract,
                        contract_hash=runner.sha256_json(contract),
                    )

        with self.subTest("multiple"):
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "candidate"
                output.mkdir()
                contract = _contract(output)
                runner._ensure_resume_directories(output)
                _write_dummy_state(output, 0, payload=b"first")
                _write_dummy_state(output, 0, payload=b"second")

                with self.assertRaisesRegex(
                    RuntimeError,
                    "multiple unjournaled",
                ):
                    runner._load_resume_layout(
                        output,
                        contract=contract,
                        contract_hash=runner.sha256_json(contract),
                    )

    def test_legacy_empty_output_is_not_resumable_before_provenance_access(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "legacy-empty" / "tres_gru_h2"
            output.mkdir(parents=True)
            config = {"training": {}, "framing": {}}
            candidate = {
                "name": "tres_gru_h2",
                "backbone": "tres_gru",
                "hidden_size": 2,
            }
            with (
                mock.patch.object(runner, "verify_source_inputs") as source,
                mock.patch.object(runner, "verify_allowed_inputs") as dataset,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "legacy empty run cannot be recovered",
                ):
                    runner.run_candidate(
                        config,
                        root / "config.json",
                        candidate,
                        output_dir=output,
                        resume=True,
                    )
                source.assert_not_called()
                dataset.assert_not_called()

    def test_parser_accepts_explicit_resume_flag(self) -> None:
        args = runner._build_parser().parse_args(
            [
                "--config",
                "experiments/configs/test.json",
                "--candidate",
                "tres_gru_h2",
                "--output-root",
                "experiments/results/test",
                "--resume",
            ]
        )
        self.assertTrue(args.resume)
        self.assertEqual(args.candidates, ["tres_gru_h2"])

    def test_second_process_cannot_enter_same_candidate_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            descriptor, _ = runner._acquire_run_lock(output.resolve())
            try:
                with mock.patch.object(runner, "_run_candidate_locked") as inner:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "already owns the run lock",
                    ):
                        runner.run_candidate(
                            {},
                            Path(temporary) / "config.json",
                            {},
                            output_dir=output,
                        )
                    inner.assert_not_called()
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "exact training-resume integration requires the locked OpenDPD environment",
)
class OpenDPDResumeIntegrationTests(unittest.TestCase):
    def _training_case(self, root: Path) -> tuple[dict, Path]:
        dataset = root / "dataset"
        dataset.mkdir()
        (dataset / "spec.json").write_text("{}", encoding="utf-8")
        rows = ["I,Q"]
        for index in range(16):
            rows.append(
                f"{0.05 + 0.02 * index:.8f},"
                f"{-0.03 + 0.01 * (index % 5):.8f}"
            )
        payload = "\n".join(rows) + "\n"
        for name in runner.ALLOWED_DATASET_FILES[1:]:
            (dataset / name).write_text(payload, encoding="utf-8")
        # Sentinels exist, but are neither named in the config nor opened.
        (dataset / "test_input.csv").write_text(
            "forbidden test sentinel\n",
            encoding="utf-8",
        )
        (dataset / "test_output.csv").write_text(
            "forbidden test sentinel\n",
            encoding="utf-8",
        )
        candidate = {"name": "gru_h2", "backbone": "gru", "hidden_size": 2}
        config = {
            "schema_version": runner.SCHEMA_VERSION,
            "task": runner.TASK,
            "status": runner.CONFIG_STATUS,
            "dataset_dir": str(dataset),
            "dataset_files": list(runner.ALLOWED_DATASET_FILES),
            "dataset_files_sha256": {
                name: runner.sha256_file(dataset / name)
                for name in runner.ALLOWED_DATASET_FILES
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
                "torch_num_threads": 1,
                "n_epochs": 2,
                "batch_size": 2,
                "batch_size_eval": 1,
                "lr": 5e-3,
                "lr_end": 5e-5,
                "decay_factor": 0.5,
                "patience": 5,
                "grad_clip_val": 200,
                "optimizer": "adamw",
                "weight_decay": 0.01,
                "loss": "mse",
            },
            "candidates": [candidate],
            "selection_metric": "validation_opendpd_nmse_db",
        }
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
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return config, config_path

    def assert_nested_equal(self, left, right) -> None:
        import torch

        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertIsInstance(right, dict)
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertIsInstance(right, type(left))
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_nested_equal(left_item, right_item)
        else:
            self.assertEqual(left, right)

    def test_interrupted_epoch_resume_matches_uninterrupted_training(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, config_path = self._training_case(root)
            candidate = config["candidates"][0]
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"

            uninterrupted_report = runner.run_candidate(
                config,
                config_path,
                candidate,
                output_dir=uninterrupted,
            )

            original_train_epoch = runner._train_epoch
            calls = 0

            def interrupt_after_training(*args, **kwargs):
                nonlocal calls
                value = original_train_epoch(*args, **kwargs)
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption inside epoch")
                return value

            with mock.patch.object(
                runner,
                "_train_epoch",
                side_effect=interrupt_after_training,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated interruption",
                ):
                    runner.run_candidate(
                        config,
                        config_path,
                        candidate,
                        output_dir=resumed,
                    )

            self.assertTrue(
                (
                    resumed
                    / runner.RESUME_DIRECTORY
                    / runner.RESUME_JOURNAL_DIRECTORY
                    / "epoch_000001.json"
                ).is_file()
            )
            self.assertFalse((resumed / "training_report.json").exists())

            resumed_report = runner.run_candidate(
                config,
                config_path,
                candidate,
                output_dir=resumed,
                resume=True,
            )
            self.assertEqual(
                uninterrupted_report["history"],
                resumed_report["history"],
            )
            self.assertEqual(
                resumed_report["resume"]["resumed_from_completed_epochs"],
                1,
            )
            self.assertEqual(
                resumed_report["resume"]["session_count_observed_in_journal"],
                2,
            )

            uninterrupted_state = torch.load(
                uninterrupted / uninterrupted_report["resume"]["final_state"],
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                resumed / resumed_report["resume"]["final_state"],
                map_location="cpu",
                weights_only=True,
            )
            for key in (
                "completed_epochs",
                "current_model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "best_validation_opendpd_nmse_db",
                "best_epoch",
                "best_model_state_dict",
                "history",
                "rng_state",
            ):
                self.assert_nested_equal(
                    uninterrupted_state[key],
                    resumed_state[key],
                )

            uninterrupted_best = torch.load(
                uninterrupted / f"{candidate['name']}.pt",
                map_location="cpu",
                weights_only=True,
            )
            resumed_best = torch.load(
                resumed / f"{candidate['name']}.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assert_nested_equal(uninterrupted_best, resumed_best)
            completion = json.loads(
                (resumed / "completion_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(completion["published_last"])
            self.assertFalse(completion["scope"]["test_split_accessed"])

    def test_checkpoint_orphan_is_adopted_after_journal_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, config_path = self._training_case(root)
            candidate = config["candidates"][0]
            uninterrupted = root / "uninterrupted"
            recovered = root / "recovered"

            reference = runner.run_candidate(
                config,
                config_path,
                candidate,
                output_dir=uninterrupted,
            )
            original_publish = runner._publish_journal_record
            interrupted = False

            def interrupt_before_epoch_journal(output, record):
                nonlocal interrupted
                if record["completed_epochs"] == 1 and not interrupted:
                    interrupted = True
                    raise RuntimeError("simulated journal interruption")
                return original_publish(output, record)

            with mock.patch.object(
                runner,
                "_publish_journal_record",
                side_effect=interrupt_before_epoch_journal,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated journal interruption",
                ):
                    runner.run_candidate(
                        config,
                        config_path,
                        candidate,
                        output_dir=recovered,
                    )

            _, states_dir, journal_dir = runner._resume_paths(recovered)
            self.assertEqual(
                len(list(states_dir.glob("state_epoch_000001_*.pt"))),
                1,
            )
            self.assertFalse((journal_dir / "epoch_000001.json").exists())

            result = runner.run_candidate(
                config,
                config_path,
                candidate,
                output_dir=recovered,
                resume=True,
            )
            self.assertEqual(reference["history"], result["history"])
            recovered_record = json.loads(
                (journal_dir / "epoch_000001.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                recovered_record[
                    "recovered_after_interrupted_journal_publication"
                ]
            )
            self.assertEqual(
                result["resume"]["resumed_from_completed_epochs"],
                1,
            )

    def test_config_change_aborts_publication_until_original_is_restored(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, config_path = self._training_case(root)
            candidate = config["candidates"][0]
            output = root / "config-change"
            original_config = config_path.read_bytes()
            original_train_epoch = runner._train_epoch
            changed = False

            def change_config_after_training(*args, **kwargs):
                nonlocal changed
                value = original_train_epoch(*args, **kwargs)
                if not changed:
                    config_path.write_text(
                        '{"changed_during_training": true}\n',
                        encoding="utf-8",
                    )
                    changed = True
                return value

            with mock.patch.object(
                runner,
                "_train_epoch",
                side_effect=change_config_after_training,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "config changed during OpenDPD training",
                ):
                    runner.run_candidate(
                        config,
                        config_path,
                        candidate,
                        output_dir=output,
                    )

            self.assertFalse((output / "training_report.json").exists())
            self.assertTrue(
                (
                    output
                    / runner.RESUME_DIRECTORY
                    / runner.RESUME_JOURNAL_DIRECTORY
                    / "epoch_000002.json"
                ).is_file()
            )
            config_path.write_bytes(original_config)
            report = runner.run_candidate(
                config,
                config_path,
                candidate,
                output_dir=output,
                resume=True,
            )
            self.assertEqual(
                report["resume"]["resumed_from_completed_epochs"],
                2,
            )
            self.assertTrue((output / "completion_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
