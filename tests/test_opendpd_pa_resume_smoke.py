import copy
import json
from pathlib import Path
import signal
import tempfile
import unittest

from experiments import run_opendpd_pa_resume_smoke as smoke_runner
from experiments import verify_opendpd_pa_resume_smoke as verifier


try:
    import torch
except ImportError:  # main NumPy-only test environment
    torch = None


class OpenDPDResumeSmokePureTests(unittest.TestCase):
    def test_trainer_command_freezes_two_epochs_and_full_loaders(self) -> None:
        command = smoke_runner._trainer_command(
            Path("experiments/configs/config.json"),
            Path("experiments/results/smoke/control"),
            resume=False,
        )
        self.assertEqual(command.count("--max-epochs"), 1)
        self.assertEqual(command[command.index("--max-epochs") + 1], "2")
        self.assertNotIn("--max-train-batches", command)
        self.assertNotIn("--max-val-batches", command)
        self.assertNotIn("--resume", command)
        resumed = smoke_runner._trainer_command(
            Path("experiments/configs/config.json"),
            Path("experiments/results/smoke/interrupted"),
            resume=True,
        )
        self.assertEqual(resumed[-1], "--resume")

    def test_contract_normalization_removes_only_output_directory(self) -> None:
        first = {
            "output_directory": "control",
            "recipe": {"max_epochs_argument": 2},
        }
        second = {
            "output_directory": "interrupted",
            "recipe": {"max_epochs_argument": 2},
        }
        verifier._assert_nested_exact(
            verifier._normalized_contract(first),
            verifier._normalized_contract(second),
        )
        second["recipe"]["max_epochs_argument"] = 3
        with self.assertRaisesRegex(RuntimeError, "value/type mismatch"):
            verifier._assert_nested_exact(
                verifier._normalized_contract(first),
                verifier._normalized_contract(second),
            )

    def test_python_float_comparison_preserves_signed_zero(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "float bit-pattern mismatch"):
            verifier._assert_nested_exact(0.0, -0.0)
        self.assertNotEqual(
            verifier._logical_sha256(0.0),
            verifier._logical_sha256(-0.0),
        )

    def test_exclusive_json_publication_refuses_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            first_hash = smoke_runner._write_json_exclusive(path, {"value": 1})
            self.assertEqual(first_hash, smoke_runner._sha256_file(path))
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                smoke_runner._write_json_exclusive(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_invalid_runtime_limits_are_rejected_before_execution(self) -> None:
        for value in (float("nan"), float("inf"), 0.0, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    smoke_runner.run_smoke(
                        "unused.json",
                        "unused-output",
                        timeout_seconds=value,
                    )


@unittest.skipUnless(torch is not None, "smoke verifier requires locked PyTorch")
class OpenDPDResumeSmokeVerifierTests(unittest.TestCase):
    def _state(
        self,
        contract_hash: str,
        productive: float,
        completed_epochs: int,
    ) -> dict:
        assert torch is not None
        history = [
            {
                "epoch": 0,
                "train_loss": 0.1,
                "validation_loss": 0.2,
                "validation_opendpd_nmse_db": -20.0,
                "validation_pooled_nmse_db": -20.0,
                "learning_rate": 0.005,
            },
            {
                "epoch": 1,
                "train_loss": 0.05,
                "validation_loss": 0.1,
                "validation_opendpd_nmse_db": -23.0,
                "validation_pooled_nmse_db": -23.0,
                "learning_rate": 0.005,
            },
        ]
        history = history[:completed_epochs]
        model = {"weight": torch.tensor([[1.0, -2.0]], dtype=torch.float32)}
        best_epoch = None if completed_epochs == 0 else completed_epochs - 1
        best_metric = (
            None
            if completed_epochs == 0
            else history[-1]["validation_opendpd_nmse_db"]
        )
        return {
            "schema_version": trainer_resume_schema(),
            "artifact_type": "opendpd_pa_epoch_resume_state",
            "task": verifier.trainer.TASK,
            "resume_contract_sha256": contract_hash,
            "completed_epochs": completed_epochs,
            "current_model_state_dict": copy.deepcopy(model),
            "optimizer_state_dict": {
                "state": {
                    0: {
                        "step": torch.tensor(2.0),
                        "exp_avg": torch.tensor([0.1]),
                    }
                },
                "param_groups": [{"params": [0], "lr": 0.005}],
            },
            "scheduler_state_dict": {"best": -23.0, "num_bad_epochs": 0},
            "best_validation_opendpd_nmse_db": best_metric,
            "best_epoch": best_epoch,
            "best_model_state_dict": (
                None if completed_epochs == 0 else copy.deepcopy(model)
            ),
            "history": history,
            "productive_fit_seconds": productive,
            "rng_state": {
                "python": {"version": 3, "state": [1, 2], "gauss_next": None},
                "numpy": {
                    "bit_generator": "MT19937",
                    "state": [3, 4],
                    "position": 2,
                    "has_gauss": 0,
                    "cached_gaussian": 0.0,
                },
                "torch_cpu": torch.tensor([1, 2], dtype=torch.uint8),
                "torch_cuda": [],
                "train_loader_generator": torch.tensor([3, 4], dtype=torch.uint8),
            },
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
        }

    def _contract(self, output: str) -> dict:
        return {
            "schema_version": trainer_resume_schema(),
            "task": verifier.trainer.TASK,
            "config": {
                "path": "experiments/configs/dummy.json",
                "sha256": "0" * 64,
                "canonical_sha256": "1" * 64,
            },
            "candidate": dict(verifier.EXPECTED_CANDIDATE),
            "candidate_sha256": verifier.trainer.sha256_json(
                verifier.EXPECTED_CANDIDATE
            ),
            "output_directory": output,
            "dataset": {
                "directory": "dataset",
                "files_sha256": {},
                "test_file_hashes_recorded": False,
            },
            "environment_lock": {"sha256": "3" * 64},
            "source": {"files": {}},
            "recipe": {
                "requested_epochs": 300,
                "effective_epochs": 2,
                "max_epochs_argument": 2,
                "max_train_batches": None,
                "max_validation_batches": None,
                "training": {
                    "deterministic": True,
                    "device": "cpu",
                    "seed": 0,
                    "n_epochs": 300,
                },
                "framing": {},
            },
            "runtime_signature": {"device": "cpu", "torch_threads": 8},
            "scope": {
                "test_split_accessed": False,
                "test_path_resolved": False,
                "test_file_hashes_recorded": False,
                "selection_split": "validation",
            },
        }

    def _write_candidate(
        self,
        root: Path,
        *,
        resumed: bool,
        killed_pid: int,
        resume_pid: int,
    ) -> None:
        assert torch is not None
        root.mkdir(parents=True)
        states = root / "resume" / "states"
        journals = root / "resume" / "journal"
        states.mkdir(parents=True)
        journals.mkdir()
        contract = self._contract(str(root))
        contract_hash = verifier.trainer.sha256_json(contract)
        run_manifest = {
            "schema_version": trainer_resume_schema(),
            "artifact_type": "opendpd_pa_resumable_run_manifest",
            "task": verifier.trainer.TASK,
            "status": "in_progress_until_completion_manifest",
            "resume_contract": contract,
            "resume_contract_sha256": contract_hash,
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
        }
        _write_json(root / verifier.trainer.RESUME_MANIFEST, run_manifest)
        previous = None
        journal_records = []
        final_state = None
        final_state_path = None
        for epoch in range(3):
            epoch_state = self._state(
                contract_hash,
                productive=(2.0 if resumed else 1.0) * epoch / 2.0,
                completed_epochs=epoch,
            )
            temporary_state = states / f"payload_{epoch:06d}.pt"
            torch.save(epoch_state, temporary_state)
            state_sha256 = verifier._sha256_file(temporary_state)
            state_path = states / (
                f"state_epoch_{epoch:06d}_{state_sha256[:16]}.pt"
            )
            temporary_state.rename(state_path)
            if resumed:
                session = f"{killed_pid}-first" if epoch < 2 else f"{resume_pid}-second"
            else:
                session = "333-control"
            record = {
                "schema_version": trainer_resume_schema(),
                "artifact_type": "opendpd_pa_append_only_epoch_journal",
                "task": verifier.trainer.TASK,
                "status": "initial_state" if epoch == 0 else "completed_epoch",
                "completed_epochs": epoch,
                "resume_contract_sha256": contract_hash,
                "config_sha256": contract["config"]["sha256"],
                "candidate_sha256": contract["candidate_sha256"],
                "dataset_manifest_sha256": verifier.trainer.sha256_json(
                    contract["dataset"]["files_sha256"]
                ),
                "source_manifest_sha256": verifier.trainer.sha256_json(
                    contract["source"]
                ),
                "configured_epochs": 300,
                "contracted_epochs": 2,
                "history_length": len(epoch_state["history"]),
                "last_history_row": (
                    None
                    if not epoch_state["history"]
                    else epoch_state["history"][-1]
                ),
                "best_epoch": epoch_state["best_epoch"],
                "best_validation_opendpd_nmse_db": epoch_state[
                    "best_validation_opendpd_nmse_db"
                ],
                "productive_fit_seconds": epoch_state[
                    "productive_fit_seconds"
                ],
                "state_path": state_path.relative_to(root).as_posix(),
                "state_sha256": state_sha256,
                "previous_journal_sha256": previous,
                "session_id": session,
                "recovered_after_interrupted_journal_publication": False,
                "test_split_accessed": False,
                "test_path_resolved": False,
                "test_file_hashes_recorded": False,
            }
            journal_path = journals / f"epoch_{epoch:06d}.json"
            _write_json(journal_path, record)
            previous = verifier._sha256_file(journal_path)
            journal_records.append(record)
            final_state = epoch_state
            final_state_path = state_path

        assert final_state is not None
        assert final_state_path is not None
        checkpoint = root / f"{verifier.CANDIDATE_NAME}.pt"
        torch.save(final_state["best_model_state_dict"], checkpoint)
        scope = {
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
        }
        report = {
            "schema_version": 1,
            "task": verifier.trainer.TASK,
            "status": "runtime_preflight_not_quality",
            "scope": scope,
            "environment_lock": contract["environment_lock"],
            "candidate": dict(verifier.EXPECTED_CANDIDATE),
            "recipe": {
                "epochs_executed": 2,
                "max_epochs_argument": 2,
                "max_train_batches": None,
                "max_validation_batches": None,
                "device": "cpu",
                "deterministic": True,
            },
            "history": final_state["history"],
            "selection": {
                "best_epoch": final_state["best_epoch"],
                "best_validation_opendpd_nmse_db": final_state[
                    "best_validation_opendpd_nmse_db"
                ],
                "test_used_for_selection": False,
            },
            "model": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": verifier._sha256_file(checkpoint),
            },
            "resume": {
                "invocation_used_resume_flag": resumed,
                "resumed_from_completed_epochs": 1 if resumed else 0,
                "session_count_observed_in_journal": 2 if resumed else 1,
                "journal_entry_count": 3,
                "final_state": journal_records[-1]["state_path"],
                "final_state_sha256": journal_records[-1]["state_sha256"],
                "resume_contract_sha256": contract_hash,
                "run_manifest": str(
                    root / verifier.trainer.RESUME_MANIFEST
                ),
                "run_manifest_sha256": verifier._sha256_file(
                    root / verifier.trainer.RESUME_MANIFEST
                ),
                "final_journal": "resume/journal/epoch_000002.json",
                "final_journal_sha256": verifier._sha256_file(
                    journals / "epoch_000002.json"
                ),
            },
        }
        report_path = root / "training_report.json"
        _write_json(report_path, report)
        completion = {
            "schema_version": trainer_resume_schema(),
            "artifact_type": "opendpd_pa_training_completion_manifest",
            "task": verifier.trainer.TASK,
            "status": "runtime_preflight_not_quality",
            "quality_result": False,
            "resume_contract_sha256": contract_hash,
            "published_last": True,
            "scope": scope,
            "artifacts": {
                "run_manifest": _artifact(root / verifier.trainer.RESUME_MANIFEST),
                "final_resume_state": _artifact(
                    final_state_path,
                    displayed_path=final_state_path.relative_to(root).as_posix(),
                ),
                "final_journal": _artifact(
                    journals / "epoch_000002.json",
                    displayed_path="resume/journal/epoch_000002.json",
                ),
                "selected_checkpoint": _artifact(checkpoint),
                "training_report": _artifact(report_path),
            },
        }
        _write_json(root / "completion_manifest.json", completion)

    def _bundle(self, root: Path) -> Path:
        bundle = root / "bundle"
        killed_pid = 111
        resume_pid = 222
        self._write_candidate(
            bundle / "control" / verifier.CANDIDATE_NAME,
            resumed=False,
            killed_pid=killed_pid,
            resume_pid=resume_pid,
        )
        self._write_candidate(
            bundle / "interrupted" / verifier.CANDIDATE_NAME,
            resumed=True,
            killed_pid=killed_pid,
            resume_pid=resume_pid,
        )
        journal = (
            bundle
            / "interrupted"
            / verifier.CANDIDATE_NAME
            / "resume"
            / "journal"
            / "epoch_000001.json"
        )
        logs = bundle / "logs"
        logs.mkdir()
        (logs / "control.log").write_text("control\n", encoding="utf-8")
        (logs / "interrupted.log").write_text("killed\n", encoding="utf-8")
        (logs / "resume.log").write_text("resumed\n", encoding="utf-8")
        interruption = {
            "schema_version": 1,
            "artifact_type": "opendpd_pa_real_process_interruption",
            "status": "abrupt_process_interruption_observed",
            "quality_result": False,
            "candidate": verifier.CANDIDATE_NAME,
            "effective_epochs": 2,
            "interrupt_after_completed_epochs": 1,
            "killed_process": {
                "pid": killed_pid,
                "signal_name": "SIGKILL",
                "signal_number": signal.SIGKILL,
                "returncode": -signal.SIGKILL,
                "log": "logs/interrupted.log",
                "log_sha256": verifier._sha256_file(
                    logs / "interrupted.log"
                ),
            },
            "resume_process": {
                "pid": resume_pid,
                "returncode": 0,
                "log": "logs/resume.log",
                "log_sha256": verifier._sha256_file(logs / "resume.log"),
            },
            "observed_before_kill": {
                "process_alive": True,
                "post_checkpoint_alive_seconds": 1.0,
                "next_epoch_journal_absent": True,
                "next_epoch_state_absent": True,
                "training_report_absent": True,
                "completion_manifest_absent": True,
                "journal_session_id": f"{killed_pid}-first",
                "journal_sha256": verifier._sha256_file(journal),
            },
            "observed_after_kill": {
                "next_epoch_journal_absent": True,
                "next_epoch_state_absent": True,
                "training_report_absent": True,
                "completion_manifest_absent": True,
            },
            "scope": {
                "test_split_accessed": False,
                "test_path_resolved": False,
                "test_file_hashes_recorded": False,
                "quality_claim": False,
                "physical_pa_accessed": False,
                "dpd_evaluated": False,
            },
        }
        _write_json(bundle / "interruption_record.json", interruption)
        base_command = [
            "/locked/python",
            "experiments/train_opendpd_pa.py",
            "--config",
            "experiments/configs/config.json",
            "--candidate",
            verifier.CANDIDATE_NAME,
            "--max-epochs",
            "2",
            "--output-root",
        ]
        execution = {
            "schema_version": 1,
            "artifact_type": "opendpd_pa_resume_smoke_execution",
            "status": "commands_completed_pending_equivalence_verification",
            "quality_result": False,
            "commands": {
                "control": base_command + ["control"],
                "interrupted": base_command + ["interrupted"],
                "resume": base_command + ["interrupted", "--resume"],
            },
            "processes": {
                "control": {
                    "pid": 333,
                    "returncode": 0,
                    "log": "logs/control.log",
                    "log_sha256": verifier._sha256_file(
                        logs / "control.log"
                    ),
                },
                "interrupted": interruption["killed_process"],
                "resume": interruption["resume_process"],
            },
            "provenance": {
                "config": {
                    "path": "experiments/configs/config.json",
                    "sha256": "0" * 64,
                },
                "bound_files_sha256": {"dummy": "0" * 64},
            },
            "scope": {
                "test_split_accessed": False,
                "test_path_resolved": False,
                "test_file_hashes_recorded": False,
                "quality_claim": False,
                "physical_pa_accessed": False,
                "dpd_evaluated": False,
            },
        }
        _write_json(bundle / "execution_record.json", execution)
        return bundle

    def test_synthetic_bundle_passes_exact_state_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(Path(temporary))
            report = verifier.verify_smoke_bundle(
                bundle,
                verify_live_inputs=False,
            )
            self.assertTrue(report["passed"])
            self.assertFalse(report["quality_result"])
            self.assertTrue(
                all(report["exact_comparisons"].values())
            )

    def test_completion_manifest_must_hash_every_bundle_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(Path(temporary))
            artifacts = smoke_runner._artifact_hashes(bundle)
            completion = {
                "schema_version": 1,
                "artifact_type": "opendpd_pa_resume_smoke_completion_manifest",
                "status": "passed_runtime_resume_equivalence",
                "quality_result": False,
                "published_last": True,
                "artifacts_sha256": artifacts,
                "scope": {
                    "test_split_accessed": False,
                    "test_path_resolved": False,
                    "test_file_hashes_recorded": False,
                    "quality_claim": False,
                    "physical_pa_accessed": False,
                    "dpd_evaluated": False,
                },
            }
            _write_json(bundle / "completion_manifest.json", completion)
            verifier.verify_smoke_bundle(bundle, verify_live_inputs=False)
            (bundle / "unbound.txt").write_text("not in manifest", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "artifact set is not closed"):
                verifier.verify_smoke_bundle(bundle, verify_live_inputs=False)

    def test_tensor_difference_is_not_hidden_by_container_equality(self) -> None:
        assert torch is not None
        left = {"model": {"weight": torch.tensor([1.0, 2.0])}}
        right = {"model": {"weight": torch.tensor([1.0, 3.0])}}
        with self.assertRaisesRegex(RuntimeError, "tensor mismatch"):
            verifier._assert_nested_exact(left, right)
        self.assertNotEqual(
            verifier._logical_sha256(left),
            verifier._logical_sha256(right),
        )

    def test_tensor_comparison_preserves_dtype_and_signed_zero(self) -> None:
        assert torch is not None
        with self.assertRaisesRegex(RuntimeError, "tensor mismatch"):
            verifier._assert_nested_exact(
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([1.0], dtype=torch.float32),
            )
        with self.assertRaisesRegex(RuntimeError, "tensor mismatch"):
            verifier._assert_nested_exact(
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([-0.0], dtype=torch.float32),
            )


def trainer_resume_schema() -> int:
    return verifier.trainer.RESUME_SCHEMA_VERSION


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact(file_path: Path, *, displayed_path: str | None = None) -> dict:
    return {
        "path": str(displayed_path if displayed_path is not None else file_path),
        "sha256": verifier._sha256_file(file_path),
    }


if __name__ == "__main__":
    unittest.main()
