"""One-shot held-out release for the frozen APA capture-transfer pre-test.

This command is intentionally separate from the pre-test runner.  It consumes
the immutable validation-selected bundle, verifies the selected calibration
sample count and coefficient payload, and only then opens the target held-out
pair.  The required ``--release-test`` acknowledgement is an operational
guard, not a tuning option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import sys
import time
from typing import Any, Callable

import numpy as np

from baseline.pa_benchmark import (
    evaluate_pa_predictor,
    freeze_pa_evaluation_protocol,
)
from baseline.train_spline import load_split_pair
from experiments.transfer_pa_apa200_to_b import (
    PROJECT_ROOT,
    _array_sha256,
    _load_source_models,
    _operation_count,
    _project_path,
    _streaming_check,
    file_sha256,
    load_config as load_transfer_config,
)
from experiments.verify_pa_transfer_bundle import verify_bundle


Progress = Callable[[str], None]
RELEASE_SCHEMA_VERSION = 1
PRIOR_RELEASE_INCIDENT = (
    "experiments/results/pa_transfer_apa200_to_b_release_incident_001.json"
)
PRIOR_RELEASE_INCIDENT_SHA256 = (
    "d03217f7ec74f49fbcd3f8619c528d7737b907e281d609b90363339ecacb2a34"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {"real": float(number.real), "imag": float(number.imag)}
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_release_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("release config must contain one JSON object")
    if config.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError("release config schema mismatch")
    if config.get("status") != "preregistered_before_held_out_release":
        raise ValueError("release config is not preregistered")
    if config.get("task") != "frozen_apa_capture_transfer_held_out_release":
        raise ValueError("unexpected release task")
    if config.get("release_guard", {}).get("required_cli_acknowledgement") != (
        "--release-test"
    ):
        raise ValueError("release acknowledgement guard is missing")
    if config.get("target_test_files", {}).get("hashes_recorded_only_at_release") is not True:
        raise ValueError("target test hashes must be recorded only at release")
    for record in config.get("frozen_models", ()):
        if int(record["selected_sample_count_per_frame"]) <= 0:
            raise ValueError("selected calibration sample count must be positive")
    return config


def _load_pretest_manifest(
    release_config: dict[str, Any],
    release_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pretest_config_path = _project_path(
        release_config["pretest_config"],
        name="pretest config",
    )
    if file_sha256(pretest_config_path) != release_config["pretest_config_sha256"]:
        raise RuntimeError("pretest config hash differs from release contract")
    pretest_config = load_transfer_config(pretest_config_path)
    bundle = _project_path(release_config["pretest_bundle"], name="pretest bundle")
    manifest_path = bundle / "transfer_manifest.json"
    if file_sha256(manifest_path) != release_config["pretest_manifest_sha256"]:
        raise RuntimeError("pretest manifest hash differs from release contract")
    # Reproduce all pre-test metric/hash checks before the held-out load.
    verified_bundle = verify_bundle(bundle)
    if not verified_bundle["test_never_opened_or_hashed"]:
        raise RuntimeError("pretest verifier did not seal held-out access")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != release_config["pretest_config_sha256"]:
        raise RuntimeError("pretest manifest config hash mismatch")
    return pretest_config, manifest, {
        "pretest_config_path": pretest_config_path,
        "pretest_bundle": bundle,
        "verified_bundle": verified_bundle,
    }


def _load_selected_coefficients(
    release_config: dict[str, Any],
    manifest: dict[str, Any],
    bundle: Path,
) -> dict[str, np.ndarray]:
    coefficient_path = bundle / "calibration_coefficients.npz"
    with np.load(coefficient_path, allow_pickle=False) as archive:
        coefficients = {
            name: np.asarray(archive[name], dtype=np.complex128).copy()
            for name in archive.files
            if name != "schema_version"
        }
    selected: dict[str, np.ndarray] = {}
    for record in release_config["frozen_models"]:
        name = record["name"]
        expected_n = int(record["selected_sample_count_per_frame"])
        transfer_record = manifest["target_transfer"][name]["selected_calibration"]
        if int(transfer_record["sample_count_per_frame"]) != expected_n:
            raise RuntimeError(f"{name} selected N changed after pre-test")
        if transfer_record.get("status") != "feasible":
            raise RuntimeError(f"{name} selected calibration is not feasible")
        key = record["calibration_coefficient_key"]
        if key not in coefficients:
            raise RuntimeError(f"missing frozen coefficient payload: {key}")
        expected_hash = transfer_record["fit"]["coefficient_hash"]
        if _array_sha256(coefficients[key]) != expected_hash:
            raise RuntimeError(f"frozen coefficient hash mismatch: {key}")
        selected[name] = coefficients[key]
    return selected


def _verify_pretest_source_code(manifest: dict[str, Any]) -> dict[str, str]:
    source_code_paths = {
        name: _project_path(name, name="release source")
        for name in manifest["source_code_sha256"]
    }
    actual = {
        name: file_sha256(path) for name, path in source_code_paths.items()
    }
    for name, expected in manifest["source_code_sha256"].items():
        if actual[name] != expected:
            raise RuntimeError(f"source code changed since pre-test: {name}")
    return actual


def _verify_prior_release_incident() -> dict[str, Any]:
    path = _project_path(PRIOR_RELEASE_INCIDENT, name="prior release incident")
    if file_sha256(path) != PRIOR_RELEASE_INCIDENT_SHA256:
        raise RuntimeError("prior release incident record hash mismatch")
    incident = json.loads(path.read_text(encoding="utf-8"))
    if incident.get("status") != (
        "failed_after_first_held_out_load_before_model_inference"
    ):
        raise RuntimeError("prior release incident status changed")
    if incident.get("not_completed", {}).get("target_test_metric_computed") is not False:
        raise RuntimeError("prior release incident unexpectedly computed a metric")
    if incident.get("retry_policy", {}).get("required_final_access_count") != 2:
        raise RuntimeError("prior release retry policy changed")
    return {
        "path": PRIOR_RELEASE_INCIDENT,
        "sha256": PRIOR_RELEASE_INCIDENT_SHA256,
        "prior_held_out_access_count": 1,
        "prior_model_inference_started": False,
        "prior_test_metric_computed": False,
    }


def _model_with_coefficients(model: Any, coefficients: np.ndarray) -> Any:
    if model.__class__.__name__ == "GeneralizedMemoryPolynomialPA":
        return model.__class__(model.config, coefficients)
    return model.__class__(
        knots=model.knots,
        branches=model.branches,
        coefficients=coefficients,
        knot_strategy=model.knot_strategy,
    )


def _acquire_lock(output: Path) -> tuple[Path, bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable release output already exists: {output}")
    lock = output.parent / f".{output.name}.lock"
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f"release output lock already exists: {lock}")
    payload = json.dumps(
        {"pid": os.getpid(), "token": secrets.token_hex(24)},
        sort_keys=True,
    ).encode("utf-8")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock, payload


def _verify_lock(lock: Path, payload: bytes) -> None:
    if lock.is_symlink() or not lock.is_file() or lock.read_bytes() != payload:
        raise RuntimeError("release publication lock was replaced")


def _publish(
    output: Path,
    *,
    lock: Path,
    lock_payload: bytes,
    manifest: dict[str, Any],
    predictions: dict[str, np.ndarray],
    reports: dict[str, dict[str, Any]],
    execution: dict[str, Any],
) -> dict[str, Any]:
    _verify_lock(lock, lock_payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError("release output appeared before publication")
    temporary = output.parent / f".{output.name}.tmp-{secrets.token_hex(12)}"
    temporary.mkdir()
    try:
        np.savez_compressed(
            temporary / "test_predictions.npz",
            schema_version=np.asarray(1, dtype=np.int64),
            **predictions,
        )
        _write_json(temporary / "test_evaluations.json", reports)
        _write_json(temporary / "execution_record.json", execution)
        names = (
            "test_predictions.npz",
            "test_evaluations.json",
            "execution_record.json",
        )
        final_manifest = {
            **manifest,
            "artifacts": {
                name: {
                    "path": name,
                    "sha256": file_sha256(temporary / name),
                }
                for name in names
            },
            "publication": {
                "immutable_bundle": True,
                "atomic_directory_rename": True,
                "manifest_written_last_inside_temporary_bundle": True,
            },
        }
        _write_json(temporary / "release_manifest.json", final_manifest)
        _verify_lock(lock, lock_payload)
        if output.exists() or output.is_symlink():
            raise FileExistsError("release output appeared during publication")
        os.replace(temporary, output)
        temporary = None  # type: ignore[assignment]
        return final_manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def run_from_config(
    config_path: str | Path,
    *,
    release_test: bool = False,
    progress: Progress = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Release the held-out target report only with explicit acknowledgement."""

    if not release_test:
        raise PermissionError("held-out release requires the --release-test acknowledgement")
    started = time.perf_counter()
    release_config_path = Path(config_path).resolve()
    release_config_hash = file_sha256(release_config_path)
    release_config = load_release_config(release_config_path)
    pretest_config, pretest_manifest, pretest_info = _load_pretest_manifest(
        release_config,
        release_config_path,
    )
    source_models = _load_source_models(pretest_config)
    selected_coefficients = _load_selected_coefficients(
        release_config,
        pretest_manifest,
        pretest_info["pretest_bundle"],
    )
    source_code_hashes = _verify_pretest_source_code(pretest_manifest)
    prior_incident = _verify_prior_release_incident()

    target_dataset = _project_path(
        release_config["target_dataset"],
        name="target dataset",
    )
    test_input_path = target_dataset / release_config["target_test_files"]["input"]
    test_output_path = target_dataset / release_config["target_test_files"]["output"]
    target_train_input, target_train_output = load_split_pair(
        target_dataset,
        "train",
    )
    expected_train_lengths = tuple(
        int(value)
        for value in pretest_config["dataset_contract"]["target_frame_lengths"]
    )
    nperseg = int(release_config["protocol"]["nperseg"])
    if (
        target_train_input.size != sum(expected_train_lengths)
        or target_train_output.size != sum(expected_train_lengths)
        or expected_train_lengths != (19662, 19662, 19656)
    ):
        raise ValueError("target train frame lengths disagree with frozen contract")
    protocol = freeze_pa_evaluation_protocol(
        target_train_input,
        target_train_output,
        dataset_label="OpenDPD APA_200MHz_b measurement B",
        sample_rate_hz=983.04e6,
        nperseg=nperseg,
        main_bandwidth_hz=200e6,
        subchannel_count=5,
        alignment_delay=int(release_config["protocol"]["alignment_delay_samples"]),
        characteristic_bins=32,
    )

    # The release config deliberately has null expected hashes.  Hashing is
    # performed here only after the pre-test/source/train/incident gates and
    # immediately before the retry held-out waveform load.
    if release_config["target_test_files"]["input_sha256"] is not None:
        raise RuntimeError("target test input hash was prefilled before release")
    if release_config["target_test_files"]["output_sha256"] is not None:
        raise RuntimeError("target test output hash was prefilled before release")
    target_test_hashes = {
        "input_sha256": file_sha256(test_input_path),
        "output_sha256": file_sha256(test_output_path),
    }
    progress("[release gate] pretest bundle and selected N verified; opening target held-out pair")

    test_input, test_output = load_split_pair(target_dataset, "test")
    if test_input.size != nperseg or test_output.size != nperseg:
        raise ValueError("target held-out length is incompatible with release protocol")
    if file_sha256(release_config_path) != release_config_hash:
        raise RuntimeError("release config changed after held-out load")

    reports: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    for record in release_config["frozen_models"]:
        name = record["name"]
        source_model = source_models[name]
        warmup = int(release_config["protocol"]["common_warmup_samples"][name])
        operation = _operation_count(source_model)
        zero_result, zero_prediction = evaluate_pa_predictor(
            source_model.predict,
            test_input,
            test_output,
            protocol=protocol,
            model_label=f"{name}/zero_shot",
            split="test",
            purpose="final_report",
            common_warmup_samples=warmup,
            operation_count=operation,
            trainable_real_parameter_count=int(
                operation.stored_real_coefficients
            ),
            fit_seconds=None,
        )
        selected_model = _model_with_coefficients(
            source_model,
            selected_coefficients[name],
        )
        selected_validation = pretest_manifest["target_transfer"][name][
            "selected_calibration"
        ]
        selected_result, selected_prediction = evaluate_pa_predictor(
            selected_model.predict,
            test_input,
            test_output,
            protocol=protocol,
            model_label=(
                f"{name}/coefficient_only_N"
                f"{record['selected_sample_count_per_frame']}"
            ),
            split="test",
            purpose="final_report",
            common_warmup_samples=warmup,
            operation_count=_operation_count(selected_model),
            trainable_real_parameter_count=int(
                _operation_count(selected_model).stored_real_coefficients
            ),
            fit_seconds=float(
                selected_validation["fit"]["fit_wall_clock_seconds"]
            ),
        )
        zero_stream = _streaming_check(source_model, (test_input,))
        selected_stream = _streaming_check(selected_model, (test_input,))
        reports[name] = {
            "zero_shot": {
                "evaluation": zero_result.to_dict(),
                "streaming": zero_stream,
            },
            "coefficient_only_selected_N": {
                "selected_sample_count_per_frame": int(
                    record["selected_sample_count_per_frame"]
                ),
                "validation_selection_record": selected_validation,
                "evaluation": selected_result.to_dict(),
                "streaming": selected_stream,
                "coefficient_hash": _array_sha256(selected_coefficients[name]),
            },
        }
        predictions[f"{name}__zero_shot_test"] = zero_prediction
        predictions[f"{name}__selected_N_test"] = selected_prediction
        progress(
            f"[test release] {name}: zero-shot "
            f"{zero_result.full_record_metrics['complex_nmse_pooled_db']:.6f} dB; "
            f"selected-N "
            f"{selected_result.full_record_metrics['complex_nmse_pooled_db']:.6f} dB"
        )

    output = _project_path(release_config["output_dir"], name="release output")
    lock, lock_payload = _acquire_lock(output)
    try:
        execution = {
            "schema_version": 1,
            "command": " ".join(sys.argv),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "runtime_seconds_before_publication": time.perf_counter() - started,
            "explicit_release_acknowledged": True,
            "target_test_accessed": True,
            "current_process_held_out_access_number": 2,
            "strict_single_open_execution": False,
            "prior_release_incident": prior_incident,
            "target_test_hashes_recorded_after_gate": True,
            "selection_source": "frozen pretest target validation only",
        }
        manifest = {
            "schema_version": 1,
            "task": release_config["task"],
            "status": "held_out_release_completed_after_metric_free_failed_access",
            "config": str(release_config_path.relative_to(PROJECT_ROOT)),
            "config_sha256": release_config_hash,
            "pretest_config": release_config["pretest_config"],
            "pretest_config_sha256": release_config["pretest_config_sha256"],
            "pretest_manifest_sha256": release_config["pretest_manifest_sha256"],
            "scope": release_config["scope"],
            "target_test_hashes": target_test_hashes,
            "target_test_sample_count": int(test_input.size),
            "target_evaluation_protocol": protocol.to_dict(),
            "source_code_sha256": source_code_hashes,
            "release_access_audit": {
                **prior_incident,
                "current_held_out_access_count": 2,
                "strict_single_open_execution": False,
                "selection_changed_after_first_access": False,
            },
            "reports": reports,
            "input_integrity": {
                "pretest_bundle_verified_before_test_load": True,
                "target_test_hashes_recorded_before_waveform_load": True,
                "target_test_loaded_after_selected_N_freeze": True,
                "target_test_used_for_selection": False,
                "target_test_used_for_coefficient_fit": False,
                "target_test_used_for_delay_gain_or_bin_fit": False,
            },
            "dpd_gate": "closed",
        }
        final_manifest = _publish(
            output,
            lock=lock,
            lock_payload=lock_payload,
            manifest=manifest,
            predictions=predictions,
            reports=reports,
            execution=execution,
        )
        progress(f"[publish] immutable held-out release: {output}")
        return final_manifest
    finally:
        if lock.exists() and not lock.is_symlink():
            lock.unlink()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Release frozen APA transfer models on the held-out split once."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--release-test",
        action="store_true",
        help="explicitly acknowledge one-time held-out access",
    )
    return parser


def main() -> None:
    args = _argument_parser().parse_args()
    manifest = run_from_config(args.config, release_test=args.release_test)
    print(
        "APA held-out release complete:",
        manifest["status"],
        "target_test_used_for_selection=False",
    )


if __name__ == "__main__":
    main()
