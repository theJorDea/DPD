"""Run a sealed real-process interrupt/resume smoke for OpenDPD TRes-GRU.

The experiment executes the same two-epoch, full-loader recipe twice:

1. an uninterrupted control;
2. a second process that is observed alive after publishing epoch 1, killed
   abruptly with SIGKILL, and resumed in a new process from epoch 1.

The result directory is transactional: it must not exist beforehand, every
metadata file is written exclusively, and the bundle completion manifest is
published last.  A directory without that final manifest is incomplete and
must never be interpreted as evidence.

This is a runtime/reproducibility smoke only.  It does not access test data,
select a PA model, evaluate a DPD, or make a physical-PA/quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import train_opendpd_pa as trainer  # noqa: E402
from experiments.verify_opendpd_pa_resume_smoke import (  # noqa: E402
    CANDIDATE_NAME,
    EFFECTIVE_EPOCHS,
    EXPECTED_CANDIDATE,
    INTERRUPT_AFTER_COMPLETED_EPOCHS,
    SCHEMA_VERSION,
    verify_smoke_bundle,
)


RUNNER_SOURCE = "experiments/run_opendpd_pa_resume_smoke.py"
VERIFIER_SOURCE = "experiments/verify_opendpd_pa_resume_smoke.py"
TRAINER_SOURCE = "experiments/train_opendpd_pa.py"
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_POST_CHECKPOINT_DELAY_SECONDS = 1.0
POLL_SECONDS = 0.05


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically publish one immutable JSON artifact and return its hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace immutable artifact: {path}"
            ) from error
        _fsync_directory(path.parent)
        return _sha256_file(path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
            _fsync_directory(path.parent)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _bound_orchestration_files(config: Path) -> dict[str, str]:
    files = {
        _display_path(config): config,
        RUNNER_SOURCE: PROJECT_ROOT / RUNNER_SOURCE,
        VERIFIER_SOURCE: PROJECT_ROOT / VERIFIER_SOURCE,
        TRAINER_SOURCE: PROJECT_ROOT / TRAINER_SOURCE,
    }
    hashes: dict[str, str] = {}
    for name, path in files.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"bound orchestration source is missing/symlink: {path}")
        hashes[name] = _sha256_file(path)
    return hashes


def _require_orchestration_files_unchanged(
    config: Path,
    expected: Mapping[str, str],
) -> None:
    if _bound_orchestration_files(config) != dict(expected):
        raise RuntimeError(
            "config/orchestrator/verifier/trainer changed during resume smoke"
        )


def _trainer_command(
    config: Path,
    output_root: Path,
    *,
    resume: bool,
) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()),
        TRAINER_SOURCE,
        "--config",
        _display_path(config),
        "--candidate",
        CANDIDATE_NAME,
        "--max-epochs",
        str(EFFECTIVE_EPOCHS),
        "--output-root",
        _display_path(output_root),
    ]
    if resume:
        command.append("--resume")
    return command


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # A concurrent natural exit is handled by the caller's required
            # return-code check; cleanup itself remains race-safe.
            pass


def _run_to_completion(
    command: Sequence[str],
    *,
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(f"refusing to replace process log: {log_path}")
    started_ns = time.time_ns()
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            _kill_process_group(process)
            process.wait(timeout=30)
            raise RuntimeError(
                f"OpenDPD subprocess timed out after {timeout_seconds:.3f} s"
            ) from error
        finally:
            log.flush()
            os.fsync(log.fileno())
    _fsync_directory(log_path.parent)
    if returncode != 0:
        raise RuntimeError(
            f"OpenDPD subprocess failed with return code {returncode}; "
            f"see {log_path}"
        )
    return {
        "pid": process.pid,
        "returncode": returncode,
        "started_unix_time_ns": started_ns,
        "finished_unix_time_ns": time.time_ns(),
        "wall_seconds": time.monotonic() - started,
        "log": _display_path(log_path),
        "log_sha256": _sha256_file(log_path),
    }


def _absence_evidence(candidate_root: Path) -> dict[str, bool]:
    return {
        "next_epoch_journal_absent": not (
            candidate_root
            / "resume"
            / "journal"
            / f"epoch_{EFFECTIVE_EPOCHS:06d}.json"
        ).exists(),
        "next_epoch_state_absent": not any(
            (
                candidate_root
                / "resume"
                / "states"
            ).glob(f"state_epoch_{EFFECTIVE_EPOCHS:06d}_*.pt")
        ),
        "training_report_absent": not (candidate_root / "training_report.json").exists(),
        "completion_manifest_absent": not (
            candidate_root / "completion_manifest.json"
        ).exists(),
    }


def _run_and_kill_after_epoch_one(
    command: Sequence[str],
    *,
    candidate_root: Path,
    log_path: Path,
    timeout_seconds: float,
    post_checkpoint_delay_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(f"refusing to replace process log: {log_path}")
    journal_path = (
        candidate_root
        / "resume"
        / "journal"
        / f"epoch_{INTERRUPT_AFTER_COMPLETED_EPOCHS:06d}.json"
    )
    started_ns = time.time_ns()
    started = time.monotonic()
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = started + timeout_seconds
        try:
            while not journal_path.is_file():
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        "interrupted OpenDPD process exited before epoch-1 journal "
                        f"with return code {returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "timed out waiting for the immutable epoch-1 journal"
                    )
                time.sleep(POLL_SECONDS)
            if journal_path.is_symlink():
                raise RuntimeError("epoch-1 journal must not be a symlink")
            journal = _read_json(journal_path, label="epoch-1 interruption journal")
            if journal.get("completed_epochs") != INTERRUPT_AFTER_COMPLETED_EPOCHS:
                raise RuntimeError("interruption journal completed-epoch mismatch")
            checkpoint_observed = time.monotonic()
            delay_deadline = checkpoint_observed + post_checkpoint_delay_seconds
            while time.monotonic() < delay_deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "OpenDPD process was not alive after its epoch-1 checkpoint"
                    )
                if not all(_absence_evidence(candidate_root).values()):
                    raise RuntimeError(
                        "OpenDPD run completed epoch 2 before an in-progress kill"
                    )
                time.sleep(POLL_SECONDS)
            before = {
                "process_alive": process.poll() is None,
                "post_checkpoint_alive_seconds": time.monotonic()
                - checkpoint_observed,
                "journal": _display_path(journal_path),
                "journal_sha256": _sha256_file(journal_path),
                "journal_session_id": journal.get("session_id"),
                **_absence_evidence(candidate_root),
            }
            if not before["process_alive"] or not all(
                before[key]
                for key in (
                    "next_epoch_journal_absent",
                    "next_epoch_state_absent",
                    "training_report_absent",
                    "completion_manifest_absent",
                )
            ):
                raise RuntimeError("pre-kill evidence contract failed")
            _kill_process_group(process)
            returncode = process.wait(timeout=30)
        except BaseException:
            _kill_process_group(process)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            raise
        finally:
            log.flush()
            os.fsync(log.fileno())
    _fsync_directory(log_path.parent)
    if returncode != -signal.SIGKILL:
        raise RuntimeError(
            f"interrupted process return code {returncode} is not -SIGKILL"
        )
    after = _absence_evidence(candidate_root)
    if not all(after.values()):
        raise RuntimeError("partial OpenDPD output changed after SIGKILL")
    process_record = {
        "pid": process.pid,
        "returncode": returncode,
        "signal_name": "SIGKILL",
        "signal_number": signal.SIGKILL,
        "started_unix_time_ns": started_ns,
        "killed_unix_time_ns": time.time_ns(),
        "wall_seconds_until_kill": time.monotonic() - started,
        "log": _display_path(log_path),
        "log_sha256": _sha256_file(log_path),
    }
    return process_record, {"before": before, "after": after}


def _validate_preregistered_config(config_path: Path) -> dict[str, Any]:
    config = trainer.load_config(config_path)
    if config.get("candidates") != [EXPECTED_CANDIDATE]:
        raise RuntimeError("smoke config must contain only TRes-GRU H27")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise RuntimeError("smoke config is missing training recipe")
    if (
        training.get("device") != "cpu"
        or training.get("deterministic") is not True
        or training.get("seed") != 0
        or training.get("n_epochs") != 300
    ):
        raise RuntimeError(
            "smoke must use the seed-0 deterministic 300-epoch production config"
        )
    environment = trainer.verify_environment_lock(config)
    if environment is None:
        raise RuntimeError("smoke config must bind an exact environment lock")
    source = trainer.verify_source_inputs(config)
    dataset_dir, dataset_hashes = trainer.verify_allowed_inputs(config, config_path)
    return {
        "config": config,
        "environment": environment,
        "source": source,
        "dataset_dir": dataset_dir,
        "dataset_hashes": dataset_hashes,
    }


def _artifact_hashes(bundle: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(bundle.rglob("*")):
        if path == bundle / "completion_manifest.json":
            continue
        if path.is_symlink():
            raise RuntimeError(f"smoke bundle must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"smoke bundle contains a non-regular entry: {path}")
        hashes[path.relative_to(bundle).as_posix()] = _sha256_file(path)
    if not hashes:
        raise RuntimeError("cannot complete an empty smoke bundle")
    return hashes


def run_smoke(
    config_path: str | Path,
    output_root: str | Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    post_checkpoint_delay_seconds: float = DEFAULT_POST_CHECKPOINT_DELAY_SECONDS,
) -> dict[str, Any]:
    """Execute, verify and transactionally publish one smoke bundle."""

    for name, value, minimum in (
        ("timeout_seconds", timeout_seconds, 1.0),
        ("post_checkpoint_delay_seconds", post_checkpoint_delay_seconds, 0.5),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < minimum
        ):
            raise ValueError(f"{name} must be finite and >= {minimum}")
    config_argument = Path(config_path)
    if config_argument.is_symlink():
        raise RuntimeError("smoke config must be one regular non-symlink file")
    config = config_argument.resolve()
    if not config.is_file():
        raise RuntimeError("smoke config must be one regular non-symlink file")
    preflight = _validate_preregistered_config(config)
    bound_orchestration_files = _bound_orchestration_files(config)

    bundle_argument = Path(output_root)
    if bundle_argument.is_symlink():
        raise RuntimeError("smoke output root must not be a symlink")
    bundle = bundle_argument.resolve()
    try:
        bundle.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to reuse immutable smoke bundle: {bundle}"
        ) from error
    _fsync_directory(bundle.parent)
    logs = bundle / "logs"
    logs.mkdir()
    _fsync_directory(bundle)
    control_root = bundle / "control"
    interrupted_root = bundle / "interrupted"
    control_command = _trainer_command(config, control_root, resume=False)
    interrupted_command = _trainer_command(config, interrupted_root, resume=False)
    resume_command = _trainer_command(config, interrupted_root, resume=True)

    control_process = _run_to_completion(
        control_command,
        log_path=logs / "control.log",
        timeout_seconds=float(timeout_seconds),
    )
    killed_process, observations = _run_and_kill_after_epoch_one(
        interrupted_command,
        candidate_root=interrupted_root / CANDIDATE_NAME,
        log_path=logs / "interrupted.log",
        timeout_seconds=float(timeout_seconds),
        post_checkpoint_delay_seconds=float(post_checkpoint_delay_seconds),
    )
    resume_process = _run_to_completion(
        resume_command,
        log_path=logs / "resume.log",
        timeout_seconds=float(timeout_seconds),
    )
    _require_orchestration_files_unchanged(
        config,
        bound_orchestration_files,
    )

    interruption_record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_real_process_interruption",
        "status": "abrupt_process_interruption_observed",
        "quality_result": False,
        "candidate": CANDIDATE_NAME,
        "effective_epochs": EFFECTIVE_EPOCHS,
        "interrupt_after_completed_epochs": INTERRUPT_AFTER_COMPLETED_EPOCHS,
        "killed_process": killed_process,
        "resume_process": resume_process,
        "observed_before_kill": observations["before"],
        "observed_after_kill": observations["after"],
        "scope": {
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
            "quality_claim": False,
            "physical_pa_accessed": False,
            "dpd_evaluated": False,
        },
    }
    _write_json_exclusive(bundle / "interruption_record.json", interruption_record)

    execution_record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_resume_smoke_execution",
        "status": "commands_completed_pending_equivalence_verification",
        "quality_result": False,
        "commands": {
            "control": control_command,
            "interrupted": interrupted_command,
            "resume": resume_command,
        },
        "processes": {
            "control": control_process,
            "interrupted": killed_process,
            "resume": resume_process,
        },
        "recipe": {
            "candidate": EXPECTED_CANDIDATE,
            "effective_epochs": EFFECTIVE_EPOCHS,
            "full_train_loader": True,
            "full_validation_loader": True,
            "post_checkpoint_delay_seconds": float(
                post_checkpoint_delay_seconds
            ),
            "timeout_seconds_per_process": float(timeout_seconds),
        },
        "provenance": {
            "config": {
                "path": _display_path(config),
                "sha256": _sha256_file(config),
            },
            "environment_lock": preflight["environment"],
            "source": preflight["source"],
            "dataset": {
                "directory": _display_path(preflight["dataset_dir"]),
                "files_sha256": preflight["dataset_hashes"],
                "test_file_hashes_recorded": False,
            },
            "bound_files_sha256": bound_orchestration_files,
            "python_executable": str(Path(sys.executable).resolve()),
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
    _write_json_exclusive(bundle / "execution_record.json", execution_record)

    verification = verify_smoke_bundle(bundle)
    _require_orchestration_files_unchanged(
        config,
        bound_orchestration_files,
    )
    _write_json_exclusive(bundle / "verification_report.json", verification)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_resume_smoke_completion_manifest",
        "status": "passed_runtime_resume_equivalence",
        "quality_result": False,
        "published_last": True,
        "artifacts_sha256": _artifact_hashes(bundle),
        "scope": {
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
            "quality_claim": False,
            "physical_pa_accessed": False,
            "dpd_evaluated": False,
        },
        "scope_note": verification["scope_note"],
    }
    _write_json_exclusive(bundle / "completion_manifest.json", completion)
    # Verify the just-published hash closure without changing the bundle.
    final_verification = verify_smoke_bundle(bundle)
    return {
        "bundle": _display_path(bundle),
        "completion_manifest_sha256": _sha256_file(
            bundle / "completion_manifest.json"
        ),
        "verification_status": final_verification["status"],
        "quality_result": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-process OpenDPD TRes-GRU resume smoke"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--post-checkpoint-delay-seconds",
        type=float,
        default=DEFAULT_POST_CHECKPOINT_DELAY_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_smoke(
        args.config,
        args.output_root,
        timeout_seconds=args.timeout_seconds,
        post_checkpoint_delay_seconds=args.post_checkpoint_delay_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
