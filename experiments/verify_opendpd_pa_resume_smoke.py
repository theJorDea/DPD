"""Verify a real-process OpenDPD PA interrupt/resume smoke bundle.

The verifier is intentionally read-only.  It validates the append-only epoch
journals, the abrupt-process interruption evidence and the exact logical
training trajectory of a two-epoch uninterrupted control against a killed and
resumed TRes-GRU run.  Runtime measurements and output-directory-bound
contract hashes are expected to differ and are never used as equality proxies.

This is runtime/reproducibility evidence only.  It does not access a test split,
claim convergence, evaluate a DPD, or provide physical-PA evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_opendpd_pa as trainer  # noqa: E402


SCHEMA_VERSION = 1
CANDIDATE_NAME = "tres_gru_h27"
EXPECTED_CANDIDATE = {
    "name": CANDIDATE_NAME,
    "backbone": "tres_gru",
    "hidden_size": 27,
}
EFFECTIVE_EPOCHS = 2
INTERRUPT_AFTER_COMPLETED_EPOCHS = 1
COMPARABLE_STATE_KEYS = (
    "schema_version",
    "artifact_type",
    "task",
    "completed_epochs",
    "current_model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "best_validation_opendpd_nmse_db",
    "best_epoch",
    "best_model_state_dict",
    "history",
    "rng_state",
    "test_split_accessed",
    "test_path_resolved",
    "test_file_hashes_recorded",
)
EXPECTED_DIFFERENT_STATE_KEYS = (
    "resume_contract_sha256",
    "productive_fit_seconds",
)
ORCHESTRATOR_SOURCE = "experiments/run_opendpd_pa_resume_smoke.py"
VERIFIER_SOURCE = "experiments/verify_opendpd_pa_resume_smoke.py"
TRAINER_SOURCE = "experiments/train_opendpd_pa.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be one regular non-symlink file: {path}")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value


def _path_beneath(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label} must be one non-empty relative path")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError(f"{label} must not escape its artifact directory")
    target = root / value
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its artifact directory") from error
    current = root
    for component in value.parts:
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f"{label} contains a symlink component")
    return target


def _assert_nested_exact(left: Any, right: Any, *, path: str = "root") -> None:
    """Raise with a precise location unless two logical states are exact."""

    try:
        import torch
    except ImportError:  # pragma: no cover - the locked verifier has torch
        torch = None
    if torch is not None and isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise RuntimeError(f"state type mismatch at {path}")
        if (
            left.dtype != right.dtype
            or left.layout != right.layout
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(
                left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            )
        ):
            raise RuntimeError(f"tensor mismatch at {path}")
        return
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or set(left) != set(right):
            raise RuntimeError(f"mapping keys/type mismatch at {path}")
        for key in sorted(left, key=lambda item: (type(item).__name__, repr(item))):
            _assert_nested_exact(
                left[key],
                right[key],
                path=f"{path}[{key!r}]",
            )
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise RuntimeError(f"sequence type/length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_exact(
                left_item,
                right_item,
                path=f"{path}[{index}]",
            )
        return
    if type(left) is not type(right):
        raise RuntimeError(f"value/type mismatch at {path}: {left!r} != {right!r}")
    if isinstance(left, float):
        if struct.pack(">d", left) != struct.pack(">d", right):
            raise RuntimeError(
                f"float bit-pattern mismatch at {path}: {left!r} != {right!r}"
            )
        return
    if left != right:
        raise RuntimeError(f"value/type mismatch at {path}: {left!r} != {right!r}")


def _update_logical_digest(digest: Any, value: Any) -> None:
    """Hash tensor content and Python containers without pickle metadata."""

    try:
        import torch
    except ImportError:  # pure-container unit tests use the NumPy-only env
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        # Reshape first because PyTorch rejects dtype-changing ``view`` on a
        # zero-dimensional optimizer-step tensor.
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        )
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_logical_digest(digest, key)
            _update_logical_digest(digest, value[key])
        digest.update(b"end-mapping\0")
        return
    if isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            _update_logical_digest(digest, item)
        digest.update(b"end-list\0")
        return
    if isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            _update_logical_digest(digest, item)
        digest.update(b"end-tuple\0")
        return
    if value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif isinstance(value, int):
        payload = str(value).encode("ascii")
        digest.update(b"int\0" + payload + b"\0")
    elif isinstance(value, float):
        digest.update(b"float\0" + struct.pack(">d", value))
    elif isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"str\0" + str(len(payload)).encode("ascii") + b"\0" + payload)
    else:
        raise TypeError(f"unsupported logical digest value: {type(value).__name__}")


def _logical_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_logical_digest(digest, value)
    return digest.hexdigest()


def _normalized_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the sole expected control/resume contract difference."""

    normalized = copy.deepcopy(dict(contract))
    if "output_directory" not in normalized:
        raise RuntimeError("resume contract is missing output_directory")
    del normalized["output_directory"]
    return normalized


def _verify_recorded_path(
    candidate_root: Path,
    displayed: Any,
    expected: Path,
    *,
    label: str,
) -> None:
    if not isinstance(displayed, str) or not displayed:
        raise RuntimeError(f"{label} path is invalid")
    displayed_path = Path(displayed)
    if ".." in displayed_path.parts:
        raise RuntimeError(f"{label} path escapes")
    if displayed_path.is_absolute():
        candidates = [displayed_path.resolve()]
    else:
        candidates = [
            (PROJECT_ROOT / displayed_path).resolve(),
            (candidate_root / displayed_path).resolve(),
        ]
    if expected.resolve() not in candidates:
        raise RuntimeError(f"{label} path mismatch")


def _verify_completion_artifacts(
    candidate_root: Path,
    completion: Mapping[str, Any],
    *,
    final_state_path: Path,
    final_journal_path: Path,
) -> None:
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("training completion manifest is missing artifacts")
    expected = {
        "run_manifest": candidate_root / trainer.RESUME_MANIFEST,
        "final_resume_state": final_state_path,
        "final_journal": final_journal_path,
        "selected_checkpoint": candidate_root / f"{CANDIDATE_NAME}.pt",
        "training_report": candidate_root / "training_report.json",
    }
    if set(artifacts) != set(expected):
        raise RuntimeError("training completion artifact set is not exact")
    for name, path in expected.items():
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"training completion manifest is missing {name}")
        _verify_recorded_path(
            candidate_root,
            record.get("path"),
            path,
            label=f"training completion {name}",
        )
        _regular_file(path, label=f"training completion {name}")
        if record.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"training completion {name} SHA-256 mismatch")


def _verify_candidate_output(
    candidate_root: Path,
    *,
    resumed: bool,
) -> dict[str, Any]:
    import torch

    if candidate_root.is_symlink() or not candidate_root.is_dir():
        raise RuntimeError(f"candidate output is missing or a symlink: {candidate_root}")
    run_manifest = _read_json(
        candidate_root / trainer.RESUME_MANIFEST,
        label="resume run manifest",
    )
    report = _read_json(candidate_root / "training_report.json", label="training report")
    completion = _read_json(
        candidate_root / "completion_manifest.json",
        label="training completion manifest",
    )

    if (
        run_manifest.get("schema_version") != trainer.RESUME_SCHEMA_VERSION
        or run_manifest.get("artifact_type")
        != "opendpd_pa_resumable_run_manifest"
        or run_manifest.get("task") != trainer.TASK
        or run_manifest.get("status")
        != "in_progress_until_completion_manifest"
    ):
        raise RuntimeError("unexpected resume run manifest header")
    contract = run_manifest.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("resume run manifest is missing its contract")
    if (
        contract.get("schema_version") != trainer.RESUME_SCHEMA_VERSION
        or contract.get("task") != trainer.TASK
        or contract.get("candidate") != EXPECTED_CANDIDATE
        or contract.get("candidate_sha256")
        != trainer.sha256_json(EXPECTED_CANDIDATE)
    ):
        raise RuntimeError("resume contract schema/task/candidate mismatch")
    _verify_recorded_path(
        candidate_root,
        contract.get("output_directory"),
        candidate_root,
        label="resume contract output directory",
    )
    contract_environment = contract.get("environment_lock")
    if not isinstance(contract_environment, Mapping):
        raise RuntimeError("resume contract must bind an exact environment lock")
    contract_hash = trainer.sha256_json(contract)
    if run_manifest.get("resume_contract_sha256") != contract_hash:
        raise RuntimeError("resume run manifest contract SHA-256 mismatch")
    if completion.get("resume_contract_sha256") != contract_hash:
        raise RuntimeError("training completion resume contract SHA-256 mismatch")
    if (
        report.get("schema_version") != trainer.SCHEMA_VERSION
        or report.get("task") != trainer.TASK
        or report.get("status") != "runtime_preflight_not_quality"
    ):
        raise RuntimeError("smoke training report header mismatch")
    if (
        completion.get("schema_version") != trainer.RESUME_SCHEMA_VERSION
        or completion.get("artifact_type")
        != "opendpd_pa_training_completion_manifest"
        or completion.get("task") != trainer.TASK
        or completion.get("status") != "runtime_preflight_not_quality"
    ):
        raise RuntimeError("smoke training completion header mismatch")
    if completion.get("quality_result") is not False:
        raise RuntimeError("smoke completion must explicitly reject a quality claim")
    if completion.get("published_last") is not True:
        raise RuntimeError("training completion manifest was not published last")
    for scope in (run_manifest, report.get("scope"), completion.get("scope")):
        if not isinstance(scope, Mapping):
            raise RuntimeError("smoke artifact is missing its access scope")
        for key in (
            "test_split_accessed",
            "test_path_resolved",
            "test_file_hashes_recorded",
        ):
            if scope.get(key) is not False:
                raise RuntimeError(f"smoke artifact does not seal {key}=false")
    if report.get("candidate") != EXPECTED_CANDIDATE:
        raise RuntimeError("smoke candidate is not exactly TRes-GRU H27")
    if report.get("environment_lock") != contract_environment:
        raise RuntimeError("training report environment differs from resume contract")
    recipe = report.get("recipe")
    if not isinstance(recipe, Mapping):
        raise RuntimeError("training report is missing recipe")
    expected_recipe = {
        "epochs_executed": EFFECTIVE_EPOCHS,
        "max_epochs_argument": EFFECTIVE_EPOCHS,
        "max_train_batches": None,
        "max_validation_batches": None,
        "device": "cpu",
        "deterministic": True,
    }
    for key, expected_value in expected_recipe.items():
        if recipe.get(key) != expected_value:
            raise RuntimeError(f"unexpected smoke recipe {key}")
    contract_recipe = contract.get("recipe")
    if not isinstance(contract_recipe, Mapping):
        raise RuntimeError("resume contract is missing recipe")
    if (
        contract_recipe.get("requested_epochs") != 300
        or contract_recipe.get("effective_epochs") != EFFECTIVE_EPOCHS
        or contract_recipe.get("max_epochs_argument") != EFFECTIVE_EPOCHS
        or contract_recipe.get("max_train_batches") is not None
        or contract_recipe.get("max_validation_batches") is not None
    ):
        raise RuntimeError("resume contract does not bind the full-loader smoke")
    contract_training = contract_recipe.get("training")
    if (
        not isinstance(contract_training, Mapping)
        or contract_training.get("device") != "cpu"
        or contract_training.get("seed") != 0
        or contract_training.get("deterministic") is not True
        or contract_training.get("n_epochs") != 300
    ):
        raise RuntimeError("resume contract does not bind the production recipe")

    resume = report.get("resume")
    if not isinstance(resume, Mapping):
        raise RuntimeError("training report is missing resume evidence")
    expected_from = INTERRUPT_AFTER_COMPLETED_EPOCHS if resumed else 0
    expected_sessions = 2 if resumed else 1
    if resume.get("invocation_used_resume_flag") is not resumed:
        raise RuntimeError("training report resume flag does not match the run")
    if resume.get("resumed_from_completed_epochs") != expected_from:
        raise RuntimeError("training report resumed from the wrong epoch")
    if resume.get("session_count_observed_in_journal") != expected_sessions:
        raise RuntimeError("training report contains an unexpected session count")
    if resume.get("journal_entry_count") != EFFECTIVE_EPOCHS + 1:
        raise RuntimeError("training report contains an unexpected journal count")

    journal_dir = candidate_root / "resume" / "journal"
    state_dir = candidate_root / "resume" / "states"
    journal_paths = sorted(journal_dir.glob("epoch_*.json"))
    expected_names = [f"epoch_{epoch:06d}.json" for epoch in range(3)]
    if [path.name for path in journal_paths] != expected_names:
        raise RuntimeError("resume journals are not exactly contiguous epochs 0..2")
    previous_digest: str | None = None
    records: list[dict[str, Any]] = []
    journaled_state_paths: set[Path] = set()
    validated_states: list[dict[str, Any]] = []
    for epoch, journal_path in enumerate(journal_paths):
        record = _read_json(journal_path, label=f"resume journal epoch {epoch}")
        if (
            record.get("schema_version") != trainer.RESUME_SCHEMA_VERSION
            or record.get("task") != trainer.TASK
            or record.get("artifact_type")
            != "opendpd_pa_append_only_epoch_journal"
            or record.get("completed_epochs") != epoch
            or record.get("status")
            != ("initial_state" if epoch == 0 else "completed_epoch")
            or record.get("resume_contract_sha256") != contract_hash
            or record.get("previous_journal_sha256") != previous_digest
            or record.get("configured_epochs") != 300
            or record.get("contracted_epochs") != EFFECTIVE_EPOCHS
            or record.get("config_sha256") != contract["config"]["sha256"]
            or record.get("candidate_sha256") != contract["candidate_sha256"]
            or record.get("dataset_manifest_sha256")
            != trainer.sha256_json(contract["dataset"]["files_sha256"])
            or record.get("source_manifest_sha256")
            != trainer.sha256_json(contract["source"])
            or record.get("recovered_after_interrupted_journal_publication")
            is not False
        ):
            raise RuntimeError(f"resume journal contract/chain mismatch at epoch {epoch}")
        state_path = _path_beneath(
            candidate_root,
            record.get("state_path"),
            label=f"resume journal state path at epoch {epoch}",
        )
        try:
            state_path.resolve().relative_to(state_dir.resolve())
        except ValueError as error:
            raise RuntimeError("resume journal state is outside states directory") from error
        _regular_file(state_path, label=f"resume state epoch {epoch}")
        state_sha256 = _sha256_file(state_path)
        state_name = trainer.STATE_FILE_PATTERN.fullmatch(state_path.name)
        if (
            record.get("state_sha256") != state_sha256
            or state_name is None
            or int(state_name.group("epoch")) != epoch
            or state_name.group("digest") != state_sha256[:16]
        ):
            raise RuntimeError(f"resume state SHA-256 mismatch at epoch {epoch}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        state = trainer._validate_resume_state(
            state,
            contract_hash=contract_hash,
            completed_epochs=epoch,
        )
        if (
            record.get("history_length") != len(state["history"])
            or record.get("last_history_row")
            != (None if not state["history"] else state["history"][-1])
            or record.get("best_epoch") != state["best_epoch"]
            or record.get("best_validation_opendpd_nmse_db")
            != state["best_validation_opendpd_nmse_db"]
            or record.get("productive_fit_seconds")
            != state["productive_fit_seconds"]
            or any(
                record.get(key) is not False
                for key in (
                    "test_split_accessed",
                    "test_path_resolved",
                    "test_file_hashes_recorded",
                )
            )
        ):
            raise RuntimeError(f"resume journal/state metadata mismatch at epoch {epoch}")
        journaled_state_paths.add(state_path)
        validated_states.append(state)
        previous_digest = _sha256_file(journal_path)
        records.append(record)

    state_entries = set(state_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in state_entries):
        raise RuntimeError("resume states directory contains a non-regular entry")
    if state_entries != journaled_state_paths:
        raise RuntimeError("resume states directory is not exactly journal-closed")

    final_relative = resume.get("final_state")
    final_state_path = _path_beneath(
        candidate_root,
        final_relative,
        label="training report final resume state",
    )
    if final_state_path != _path_beneath(
        candidate_root,
        records[-1]["state_path"],
        label="final journal state path",
    ):
        raise RuntimeError("training report and final journal disagree on state")
    if resume.get("final_state_sha256") != _sha256_file(final_state_path):
        raise RuntimeError("training report final resume state SHA-256 mismatch")
    if resume.get("resume_contract_sha256") != contract_hash:
        raise RuntimeError("training report resume contract SHA-256 mismatch")
    run_manifest_path = candidate_root / trainer.RESUME_MANIFEST
    _verify_recorded_path(
        candidate_root,
        resume.get("run_manifest"),
        run_manifest_path,
        label="training report run manifest",
    )
    if resume.get("run_manifest_sha256") != _sha256_file(run_manifest_path):
        raise RuntimeError("training report run manifest SHA-256 mismatch")
    final_journal_path = journal_paths[-1]
    if resume.get("final_journal_sha256") != _sha256_file(final_journal_path):
        raise RuntimeError("training report final journal SHA-256 mismatch")
    _verify_recorded_path(
        candidate_root,
        resume.get("final_journal"),
        final_journal_path,
        label="training report final journal",
    )
    model_record = report.get("model")
    if not isinstance(model_record, Mapping):
        raise RuntimeError("training report is missing model artifact")
    checkpoint_path = candidate_root / f"{CANDIDATE_NAME}.pt"
    _verify_recorded_path(
        candidate_root,
        model_record.get("checkpoint"),
        checkpoint_path,
        label="training report checkpoint",
    )
    _verify_completion_artifacts(
        candidate_root,
        completion,
        final_state_path=final_state_path,
        final_journal_path=final_journal_path,
    )

    return {
        "root": candidate_root,
        "run_manifest": run_manifest,
        "contract": dict(contract),
        "contract_sha256": contract_hash,
        "report": report,
        "completion": completion,
        "journal_records": records,
        "final_state_path": final_state_path,
        "checkpoint_path": checkpoint_path,
        "final_state": validated_states[-1],
        "states": validated_states,
    }


def _verify_interruption(
    bundle: Path,
    interrupted: Mapping[str, Any],
) -> dict[str, Any]:
    record = _read_json(bundle / "interruption_record.json", label="interruption record")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("artifact_type")
        != "opendpd_pa_real_process_interruption"
        or record.get("status") != "abrupt_process_interruption_observed"
        or record.get("quality_result") is not False
        or record.get("candidate") != CANDIDATE_NAME
        or record.get("effective_epochs") != EFFECTIVE_EPOCHS
        or record.get("interrupt_after_completed_epochs")
        != INTERRUPT_AFTER_COMPLETED_EPOCHS
    ):
        raise RuntimeError("interruption record contract mismatch")
    scope = record.get("scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("interruption record is missing scope")
    for key in (
        "test_split_accessed",
        "test_path_resolved",
        "test_file_hashes_recorded",
        "quality_claim",
        "physical_pa_accessed",
        "dpd_evaluated",
    ):
        if scope.get(key) is not False:
            raise RuntimeError(f"interruption record does not seal {key}=false")
    killed = record.get("killed_process")
    resumed = record.get("resume_process")
    before = record.get("observed_before_kill")
    after = record.get("observed_after_kill")
    if not all(isinstance(value, Mapping) for value in (killed, resumed, before, after)):
        raise RuntimeError("interruption record is missing process evidence")
    assert isinstance(killed, Mapping)
    assert isinstance(resumed, Mapping)
    assert isinstance(before, Mapping)
    assert isinstance(after, Mapping)
    killed_pid = killed.get("pid")
    resume_pid = resumed.get("pid")
    if not isinstance(killed_pid, int) or killed_pid < 1:
        raise RuntimeError("interruption record killed PID is invalid")
    if not isinstance(resume_pid, int) or resume_pid < 1 or resume_pid == killed_pid:
        raise RuntimeError("interruption record resume PID is invalid")
    if (
        killed.get("signal_name") != "SIGKILL"
        or killed.get("signal_number") != signal.SIGKILL
        or killed.get("returncode") != -signal.SIGKILL
        or resumed.get("returncode") != 0
    ):
        raise RuntimeError("interruption record does not prove a SIGKILL and clean resume")
    for key in (
        "process_alive",
        "next_epoch_journal_absent",
        "next_epoch_state_absent",
        "training_report_absent",
        "completion_manifest_absent",
    ):
        if before.get(key) is not True:
            raise RuntimeError(f"pre-kill interruption evidence is missing {key}")
    for key in (
        "next_epoch_journal_absent",
        "next_epoch_state_absent",
        "training_report_absent",
        "completion_manifest_absent",
    ):
        if after.get(key) is not True:
            raise RuntimeError(f"post-kill interruption evidence is missing {key}")
    delay = before.get("post_checkpoint_alive_seconds")
    if not isinstance(delay, (int, float)) or isinstance(delay, bool) or delay < 0.5:
        raise RuntimeError("killed process was not observed alive after its checkpoint")

    records = interrupted["journal_records"]
    first_session = records[0].get("session_id")
    if records[1].get("session_id") != first_session:
        raise RuntimeError("initial and epoch-1 journals must share the killed session")
    second_session = records[2].get("session_id")
    if second_session == first_session:
        raise RuntimeError("resumed epoch must use a distinct process session")
    if not isinstance(first_session, str) or not first_session.startswith(f"{killed_pid}-"):
        raise RuntimeError("killed PID does not match epoch-1 journal session")
    if not isinstance(second_session, str) or not second_session.startswith(f"{resume_pid}-"):
        raise RuntimeError("resume PID does not match epoch-2 journal session")
    if before.get("journal_session_id") != first_session:
        raise RuntimeError("interruption record journal session mismatch")
    epoch_one_path = bundle / "interrupted" / CANDIDATE_NAME / "resume" / "journal" / "epoch_000001.json"
    if before.get("journal_sha256") != _sha256_file(epoch_one_path):
        raise RuntimeError("interruption record epoch-1 journal SHA-256 mismatch")
    for process_record, name in (
        (killed, "interrupted.log"),
        (resumed, "resume.log"),
    ):
        log_path = bundle / "logs" / name
        _regular_file(log_path, label=f"{name} process log")
        _verify_recorded_path(
            bundle,
            process_record.get("log"),
            log_path,
            label=f"{name} process log",
        )
        if (
            process_record.get("log_sha256") != _sha256_file(log_path)
        ):
            raise RuntimeError(f"interruption record {name} hash/path mismatch")
    return record


def _normalized_smoke_command(value: Any, *, expect_resume: bool) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) for item in value
    ):
        raise RuntimeError("smoke execution command must be a non-empty string list")
    command = list(value)
    if len(command) < 2 or Path(command[1]).as_posix() != TRAINER_SOURCE:
        raise RuntimeError("smoke execution command does not invoke the sealed trainer")
    has_resume = "--resume" in command
    if has_resume is not expect_resume or command.count("--resume") != int(expect_resume):
        raise RuntimeError("smoke execution command has an unexpected resume flag")
    if has_resume:
        command.remove("--resume")
    for option, expected in (
        ("--candidate", CANDIDATE_NAME),
        ("--max-epochs", str(EFFECTIVE_EPOCHS)),
    ):
        if command.count(option) != 1:
            raise RuntimeError(f"smoke execution command must contain one {option}")
        index = command.index(option)
        if index + 1 >= len(command) or command[index + 1] != expected:
            raise RuntimeError(f"smoke execution command has unexpected {option}")
    if "--max-train-batches" in command or "--max-val-batches" in command:
        raise RuntimeError("smoke execution command must use both full loaders")
    if command.count("--config") != 1:
        raise RuntimeError("smoke execution command must contain one config")
    config_index = command.index("--config")
    if config_index + 1 >= len(command) or not command[config_index + 1]:
        raise RuntimeError("smoke execution command config path is missing")
    if command.count("--output-root") != 1:
        raise RuntimeError("smoke execution command must contain one output root")
    output_index = command.index("--output-root")
    if output_index + 1 >= len(command):
        raise RuntimeError("smoke execution command output root is missing")
    command[output_index + 1] = "<normalized-output-root>"
    return command


def _verify_execution_record(
    bundle: Path,
    interruption: Mapping[str, Any],
    *,
    verify_bound_files: bool,
) -> dict[str, Any]:
    record = _read_json(bundle / "execution_record.json", label="execution record")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("artifact_type") != "opendpd_pa_resume_smoke_execution"
        or record.get("status")
        != "commands_completed_pending_equivalence_verification"
        or record.get("quality_result") is not False
    ):
        raise RuntimeError("smoke execution record contract mismatch")
    scope = record.get("scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("smoke execution record is missing scope")
    for key in (
        "test_split_accessed",
        "test_path_resolved",
        "test_file_hashes_recorded",
        "quality_claim",
        "physical_pa_accessed",
        "dpd_evaluated",
    ):
        if scope.get(key) is not False:
            raise RuntimeError(f"smoke execution record does not seal {key}=false")
    commands = record.get("commands")
    if not isinstance(commands, Mapping):
        raise RuntimeError("smoke execution record is missing commands")
    normalized_control = _normalized_smoke_command(
        commands.get("control"),
        expect_resume=False,
    )
    normalized_interrupted = _normalized_smoke_command(
        commands.get("interrupted"),
        expect_resume=False,
    )
    normalized_resume = _normalized_smoke_command(
        commands.get("resume"),
        expect_resume=True,
    )
    if not (
        normalized_control == normalized_interrupted == normalized_resume
    ):
        raise RuntimeError("control/interrupted/resume commands differ beyond output/resume")
    processes = record.get("processes")
    if not isinstance(processes, Mapping):
        raise RuntimeError("smoke execution record is missing process results")
    control_process = processes.get("control")
    killed_process = processes.get("interrupted")
    resume_process = processes.get("resume")
    if not all(
        isinstance(value, Mapping)
        for value in (control_process, killed_process, resume_process)
    ):
        raise RuntimeError("smoke execution process records are invalid")
    assert isinstance(control_process, Mapping)
    assert isinstance(killed_process, Mapping)
    assert isinstance(resume_process, Mapping)
    if control_process.get("returncode") != 0:
        raise RuntimeError("uninterrupted control process did not complete cleanly")
    if dict(killed_process) != interruption.get("killed_process"):
        raise RuntimeError("execution/interruption killed-process records differ")
    if dict(resume_process) != interruption.get("resume_process"):
        raise RuntimeError("execution/interruption resume-process records differ")
    for process_record, name in (
        (control_process, "control.log"),
        (killed_process, "interrupted.log"),
        (resume_process, "resume.log"),
    ):
        log_path = bundle / "logs" / name
        _regular_file(log_path, label=f"execution {name}")
        _verify_recorded_path(
            bundle,
            process_record.get("log"),
            log_path,
            label=f"execution {name}",
        )
        if process_record.get("log_sha256") != _sha256_file(log_path):
            raise RuntimeError(f"execution {name} SHA-256 mismatch")
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("smoke execution record is missing provenance")
    bound_files = provenance.get("bound_files_sha256")
    if not isinstance(bound_files, Mapping) or not bound_files:
        raise RuntimeError("smoke execution record is missing bound source files")
    config_provenance = provenance.get("config")
    if not isinstance(config_provenance, Mapping):
        raise RuntimeError("smoke execution record is missing config provenance")
    commands_by_name = {
        "control": commands["control"],
        "interrupted": commands["interrupted"],
        "resume": commands["resume"],
    }
    expected_output_roots = {
        "control": bundle / "control",
        "interrupted": bundle / "interrupted",
        "resume": bundle / "interrupted",
    }
    displayed_config = config_provenance.get("path")
    if not isinstance(displayed_config, str):
        raise RuntimeError("smoke execution config provenance path is invalid")
    config_path = Path(displayed_config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    for name, command_value in commands_by_name.items():
        command = list(command_value)
        config_index = command.index("--config")
        _verify_recorded_path(
            bundle,
            command[config_index + 1],
            config_path,
            label=f"{name} command config",
        )
        output_index = command.index("--output-root")
        _verify_recorded_path(
            bundle,
            command[output_index + 1],
            expected_output_roots[name],
            label=f"{name} command output root",
        )
    if verify_bound_files:
        for displayed, expected_hash in bound_files.items():
            if not isinstance(displayed, str) or not isinstance(expected_hash, str):
                raise RuntimeError("smoke execution bound source entry is invalid")
            path = Path(displayed)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            _regular_file(path, label="smoke execution bound source")
            if _sha256_file(path) != expected_hash:
                raise RuntimeError(f"smoke execution bound source changed: {displayed}")
    return record


def _verify_bundle_completion_if_present(bundle: Path) -> dict[str, Any] | None:
    path = bundle / "completion_manifest.json"
    if not path.exists():
        return None
    manifest = _read_json(path, label="smoke bundle completion manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type")
        != "opendpd_pa_resume_smoke_completion_manifest"
        or manifest.get("status") != "passed_runtime_resume_equivalence"
        or manifest.get("quality_result") is not False
        or manifest.get("published_last") is not True
    ):
        raise RuntimeError("smoke bundle completion contract mismatch")
    scope = manifest.get("scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("smoke bundle completion is missing scope")
    for key in (
        "test_split_accessed",
        "test_path_resolved",
        "test_file_hashes_recorded",
        "quality_claim",
        "physical_pa_accessed",
        "dpd_evaluated",
    ):
        if scope.get(key) is not False:
            raise RuntimeError(f"smoke bundle completion does not seal {key}=false")
    artifacts = manifest.get("artifacts_sha256")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError("smoke bundle completion is missing artifact hashes")
    entries = list(bundle.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise RuntimeError("completed smoke bundle must not contain symlinks")
    observed_paths = {
        path.relative_to(bundle).as_posix()
        for path in entries
        if path.is_file() and path != bundle / "completion_manifest.json"
    }
    if set(artifacts) != observed_paths:
        raise RuntimeError("smoke bundle completion artifact set is not closed")
    for relative, expected_hash in artifacts.items():
        artifact = _path_beneath(bundle, relative, label="bundle artifact path")
        _regular_file(artifact, label="bundle artifact")
        if _sha256_file(artifact) != expected_hash:
            raise RuntimeError(f"smoke bundle artifact SHA-256 mismatch: {relative}")
    return manifest


def verify_smoke_bundle(
    bundle_path: str | Path,
    *,
    verify_live_inputs: bool = True,
) -> dict[str, Any]:
    """Return a JSON-safe report or raise on any evidence mismatch."""

    import torch

    bundle_argument = Path(bundle_path)
    if bundle_argument.is_symlink():
        raise RuntimeError("smoke bundle must be one regular directory")
    bundle = bundle_argument.resolve()
    if not bundle.is_dir():
        raise RuntimeError("smoke bundle must be one regular directory")
    control = _verify_candidate_output(
        bundle / "control" / CANDIDATE_NAME,
        resumed=False,
    )
    interrupted = _verify_candidate_output(
        bundle / "interrupted" / CANDIDATE_NAME,
        resumed=True,
    )
    _assert_nested_exact(
        _normalized_contract(control["contract"]),
        _normalized_contract(interrupted["contract"]),
        path="normalized_resume_contract",
    )
    if control["contract"]["output_directory"] == interrupted["contract"]["output_directory"]:
        raise RuntimeError("control and interrupted outputs must be distinct")
    interruption = _verify_interruption(bundle, interrupted)
    execution = _verify_execution_record(
        bundle,
        interruption,
        verify_bound_files=verify_live_inputs,
    )

    if verify_live_inputs:
        config_record = control["contract"].get("config")
        if not isinstance(config_record, Mapping):
            raise RuntimeError("resume contract is missing config provenance")
        displayed = config_record.get("path")
        if not isinstance(displayed, str):
            raise RuntimeError("resume contract config path is invalid")
        config_path = Path(displayed)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        _regular_file(config_path, label="bound smoke config")
        if _sha256_file(config_path) != config_record.get("sha256"):
            raise RuntimeError("bound smoke config SHA-256 mismatch")
        provenance = execution.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("execution record is missing live provenance")
        bound_files = provenance.get("bound_files_sha256")
        if not isinstance(bound_files, Mapping):
            raise RuntimeError("execution record is missing bound file hashes")
        required_bound_files = {
            str(config_record["path"]),
            ORCHESTRATOR_SOURCE,
            VERIFIER_SOURCE,
            TRAINER_SOURCE,
        }
        if not required_bound_files.issubset(set(bound_files)):
            raise RuntimeError("execution record does not bind every smoke source")
        if bound_files.get(str(config_record["path"])) != config_record.get("sha256"):
            raise RuntimeError("execution/config resume hashes disagree")
        config = trainer.load_config(config_path)
        if trainer.sha256_json(config) != config_record.get("canonical_sha256"):
            raise RuntimeError("bound smoke canonical config SHA-256 mismatch")
        config_training = config.get("training")
        if (
            config.get("candidates") != [EXPECTED_CANDIDATE]
            or not isinstance(config_training, Mapping)
            or config_training.get("n_epochs") != 300
            or config_training.get("device") != "cpu"
            or config_training.get("seed") != 0
            or config_training.get("deterministic") is not True
            or "environment_lock" not in config
        ):
            raise RuntimeError("bound smoke config is not the preregistered recipe")
        if trainer._contains_forbidden_dataset_path(config):
            raise RuntimeError("bound smoke config contains a forbidden test path")
        live_environment = trainer.verify_environment_lock(config)
        live_source = trainer.verify_source_inputs(config)
        live_dataset_dir, live_dataset_hashes = trainer.verify_allowed_inputs(
            config,
            config_path,
        )
        if live_environment != control["contract"].get("environment_lock"):
            raise RuntimeError("live environment evidence differs from resume contract")
        if live_source != control["contract"].get("source"):
            raise RuntimeError("live source evidence differs from resume contract")
        dataset_record = control["contract"].get("dataset")
        if not isinstance(dataset_record, Mapping):
            raise RuntimeError("resume contract is missing dataset provenance")
        displayed_dataset = dataset_record.get("directory")
        if not isinstance(displayed_dataset, str):
            raise RuntimeError("resume contract dataset directory is invalid")
        displayed_dataset_path = Path(displayed_dataset)
        if not displayed_dataset_path.is_absolute():
            displayed_dataset_path = PROJECT_ROOT / displayed_dataset_path
        if (
            displayed_dataset_path.resolve() != live_dataset_dir.resolve()
            or dataset_record.get("files_sha256") != live_dataset_hashes
            or dataset_record.get("test_file_hashes_recorded") is not False
        ):
            raise RuntimeError("live dataset evidence differs from resume contract")

    epoch_state_digests: dict[str, dict[str, str]] = {}
    for epoch, (control_epoch_state, resumed_epoch_state) in enumerate(
        zip(control["states"], interrupted["states"])
    ):
        comparable_control_epoch = {
            key: control_epoch_state[key] for key in COMPARABLE_STATE_KEYS
        }
        comparable_resumed_epoch = {
            key: resumed_epoch_state[key] for key in COMPARABLE_STATE_KEYS
        }
        _assert_nested_exact(
            comparable_control_epoch,
            comparable_resumed_epoch,
            path=f"epoch_{epoch}_training_state",
        )
        control_epoch_digest = _logical_sha256(comparable_control_epoch)
        resumed_epoch_digest = _logical_sha256(comparable_resumed_epoch)
        if control_epoch_digest != resumed_epoch_digest:
            raise RuntimeError(f"epoch-{epoch} logical training-state hashes differ")
        epoch_state_digests[str(epoch)] = {
            "control": control_epoch_digest,
            "resumed": resumed_epoch_digest,
        }

    control_state = control["final_state"]
    resumed_state = interrupted["final_state"]
    if not isinstance(control_state, Mapping) or not isinstance(resumed_state, Mapping):
        raise RuntimeError("final resume state must be a mapping")
    expected_state_keys = set(COMPARABLE_STATE_KEYS) | set(EXPECTED_DIFFERENT_STATE_KEYS)
    if set(control_state) != expected_state_keys or set(resumed_state) != expected_state_keys:
        raise RuntimeError("final resume state contains unexpected or missing keys")
    comparable_control = {key: control_state[key] for key in COMPARABLE_STATE_KEYS}
    comparable_resumed = {key: resumed_state[key] for key in COMPARABLE_STATE_KEYS}
    _assert_nested_exact(
        comparable_control,
        comparable_resumed,
        path="final_training_state",
    )
    control_checkpoint = torch.load(
        _regular_file(control["checkpoint_path"], label="control checkpoint"),
        map_location="cpu",
        weights_only=True,
    )
    resumed_checkpoint = torch.load(
        _regular_file(interrupted["checkpoint_path"], label="resumed checkpoint"),
        map_location="cpu",
        weights_only=True,
    )
    _assert_nested_exact(
        control_checkpoint,
        resumed_checkpoint,
        path="validation_selected_checkpoint",
    )
    for label, run, state, checkpoint in (
        ("control", control, control_state, control_checkpoint),
        ("resumed", interrupted, resumed_state, resumed_checkpoint),
    ):
        if state["resume_contract_sha256"] != run["contract_sha256"]:
            raise RuntimeError(f"{label} final state contract SHA-256 mismatch")
        productive = state["productive_fit_seconds"]
        if (
            isinstance(productive, bool)
            or not isinstance(productive, (int, float))
            or not math.isfinite(float(productive))
            or productive < 0
        ):
            raise RuntimeError(f"{label} productive fit time is invalid")
        for key in (
            "test_split_accessed",
            "test_path_resolved",
            "test_file_hashes_recorded",
        ):
            if state[key] is not False:
                raise RuntimeError(f"{label} final state violates {key}=false")
        _assert_nested_exact(
            run["report"]["history"],
            state["history"],
            path=f"{label}_report_vs_state_history",
        )
        selection = run["report"].get("selection")
        if not isinstance(selection, Mapping):
            raise RuntimeError(f"{label} report is missing validation selection")
        if (
            selection.get("best_epoch") != state["best_epoch"]
            or selection.get("best_validation_opendpd_nmse_db")
            != state["best_validation_opendpd_nmse_db"]
            or selection.get("test_used_for_selection") is not False
        ):
            raise RuntimeError(f"{label} report selection differs from final state")
        _assert_nested_exact(
            checkpoint,
            state["best_model_state_dict"],
            path=f"{label}_checkpoint_vs_best_state",
        )
        model_record = run["report"].get("model")
        if (
            not isinstance(model_record, Mapping)
            or model_record.get("checkpoint_sha256")
            != _sha256_file(run["checkpoint_path"])
        ):
            raise RuntimeError(f"{label} report checkpoint SHA-256 mismatch")
    _assert_nested_exact(
        control["report"]["history"],
        interrupted["report"]["history"],
        path="training_report_history",
    )
    control_state_digest = _logical_sha256(comparable_control)
    resumed_state_digest = _logical_sha256(comparable_resumed)
    control_checkpoint_digest = _logical_sha256(control_checkpoint)
    resumed_checkpoint_digest = _logical_sha256(resumed_checkpoint)
    if control_state_digest != resumed_state_digest:
        raise RuntimeError("control/resumed logical training-state hashes differ")
    if control_checkpoint_digest != resumed_checkpoint_digest:
        raise RuntimeError("control/resumed logical checkpoint hashes differ")
    bundle_completion = _verify_bundle_completion_if_present(bundle)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_resume_smoke_verification",
        "status": "passed_runtime_resume_equivalence",
        "passed": True,
        "quality_result": False,
        "physical_pa_accessed": False,
        "dpd_evaluated": False,
        "test_split_accessed": False,
        "test_path_resolved": False,
        "test_file_hashes_recorded": False,
        "candidate": EXPECTED_CANDIDATE,
        "effective_epochs": EFFECTIVE_EPOCHS,
        "full_train_loader_used": True,
        "full_validation_loader_used": True,
        "actual_process_signal": {
            "name": "SIGKILL",
            "number": signal.SIGKILL,
            "returncode": -signal.SIGKILL,
        },
        "resume": {
            "resumed_from_completed_epochs": INTERRUPT_AFTER_COMPLETED_EPOCHS,
            "distinct_process_sessions": 2,
            "killed_before_next_epoch_publication": True,
        },
        "exact_comparisons": {
            "normalized_contract": True,
            "history": True,
            "current_model_state": True,
            "optimizer_state": True,
            "scheduler_state": True,
            "best_model_state": True,
            "rng_and_shuffle_state": True,
            "validation_selected_checkpoint": True,
        },
        "logical_sha256": {
            "control_comparable_final_training_state": control_state_digest,
            "resumed_comparable_final_training_state": resumed_state_digest,
            "control_validation_selected_checkpoint": control_checkpoint_digest,
            "resumed_validation_selected_checkpoint": resumed_checkpoint_digest,
            "journaled_epoch_states": epoch_state_digests,
        },
        "expected_differences_excluded": list(EXPECTED_DIFFERENT_STATE_KEYS),
        "interruption_record_sha256": _sha256_file(bundle / "interruption_record.json"),
        "interruption_record_status": interruption["status"],
        "execution_record_sha256": _sha256_file(bundle / "execution_record.json"),
        "execution_record_status": execution["status"],
        "bundle_completion_present_and_verified": bundle_completion is not None,
        "scope_note": (
            "Two-epoch CPU full-loader runtime/restart evidence only; not a "
            "300-epoch convergence, CUDA, DPD-quality, physical-PA or Huawei claim."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only verifier for an OpenDPD PA resume smoke bundle"
    )
    parser.add_argument("--bundle", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = verify_smoke_bundle(args.bundle)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
