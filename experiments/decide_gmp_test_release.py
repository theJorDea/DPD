"""Decide whether one frozen GMP model may be evaluated on test once.

This command is a train/validation-only release gate.  It verifies the
validation selection, coefficient-OOF residual bundle, matched MP reference,
causal streaming contract, and preregistered thresholds.  It never opens,
hashes, or scans test files.  Passing this gate authorizes one frozen test
evaluation; it is not Gate A->B, a physical-PA result, or Huawei acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import time
from typing import Any

import numpy as np

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA
from baseline.metrics import nmse_pooled_db
from baseline.train_spline import file_sha256, load_split_pair, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_DATASET_FILES = {
    "spec.json",
    "train_input.csv",
    "train_output.csv",
    "val_input.csv",
    "val_output.csv",
}
_REQUIRED_DIAGNOSTIC_KEYS = {
    "global_metrics",
    "lag_correlations",
    "envelope_correlations",
    "slow_state_correlations",
    "amplitude_regions",
    "am_am_am_pm_residuals",
    "segment_position",
    "error_psd",
}


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _resolve_project_relative(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty project-relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{name} must be project-relative")
    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{name} escapes the project root") from error
    return resolved


def _verify_hash(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numeric_finite(item) for item in value.values())
    return False


def _boundary_mask(
    sample_count: int,
    *,
    nperseg: int,
    warmup: int,
    cooldown: int,
) -> np.ndarray:
    if sample_count <= 0 or nperseg <= 0:
        raise ValueError("sample_count and nperseg must be positive")
    if warmup < 0 or cooldown < 0 or warmup + cooldown >= nperseg:
        raise ValueError("invalid matched boundary support")
    mask = np.ones(sample_count, dtype=bool)
    for start in range(0, sample_count, nperseg):
        stop = min(start + nperseg, sample_count)
        if warmup + cooldown >= stop - start:
            raise ValueError("matched boundary consumes a partial frame")
        mask[start : start + warmup] = False
        if cooldown:
            mask[stop - cooldown : stop] = False
    return mask


def _metric_pair(
    prediction: np.ndarray,
    reference: np.ndarray,
    interior: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if prediction.shape != reference.shape or prediction.ndim != 1:
        raise ValueError("prediction/reference shape mismatch")
    if interior.shape != reference.shape or interior.dtype != np.bool_:
        raise ValueError("interior mask shape/type mismatch")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction contains non-finite samples")
    return {
        "full_nmse_db": nmse_pooled_db(prediction, reference),
        "common_interior_nmse_db": nmse_pooled_db(
            prediction[interior],
            reference[interior],
        ),
    }


def _diagnostic_bundle_complete(report: dict[str, Any]) -> bool:
    if not _REQUIRED_DIAGNOSTIC_KEYS <= set(report):
        return False
    if not _all_numeric_finite(report):
        return False
    if not report.get("lag_correlations"):
        return False
    if not report.get("envelope_correlations"):
        return False
    if not report.get("slow_state_correlations"):
        return False
    thresholds = {
        str(row.get("threshold_name"))
        for row in report.get("amplitude_regions", [])
        if isinstance(row, dict)
    }
    if not {"q90", "q95", "q99"} <= thresholds:
        return False
    psd = report.get("error_psd")
    if not isinstance(psd, dict) or psd.get("available") is not True:
        return False
    integrated = psd.get("integrated_bands")
    return isinstance(integrated, dict) and {
        "left_adjacent",
        "main",
        "right_adjacent",
    } <= set(integrated)


def _streaming_checks(
    model: GeneralizedMemoryPolynomialPA,
    train_input: np.ndarray,
    *,
    nperseg: int,
) -> dict[str, Any]:
    first_frame = np.asarray(train_input[:nperseg], dtype=np.complex128)
    if first_frame.size != nperseg:
        raise ValueError("training record has no complete frame")
    full = model.predict(first_frame)
    chunk_sizes = (1, 7, 31, 257, 1024)
    pieces: list[np.ndarray] = []
    state = None
    start = 0
    chunk_index = 0
    while start < first_frame.size:
        size = chunk_sizes[chunk_index % len(chunk_sizes)]
        stop = min(start + size, first_frame.size)
        prediction, state = model.predict_streaming_chunk(
            first_frame[start:stop],
            state,
        )
        pieces.append(prediction)
        start = stop
        chunk_index += 1
    streamed = np.concatenate(pieces)
    stream_error = float(np.max(np.abs(streamed - full), initial=0.0))
    streaming_passed = bool(
        np.allclose(streamed, full, rtol=1e-12, atol=1e-12)
    )

    segmented = model.predict_segments(train_input, nperseg)
    reset = np.concatenate(
        [
            model.predict(train_input[start : min(start + nperseg, train_input.size)])
            for start in range(0, train_input.size, nperseg)
        ]
    )
    reset_error = float(np.max(np.abs(segmented - reset), initial=0.0))
    reset_passed = bool(
        np.allclose(segmented, reset, rtol=1e-12, atol=1e-12)
    )
    return {
        "rtol": 1e-12,
        "atol": 1e-12,
        "chunk_schedule": list(chunk_sizes),
        "streamed_sample_count": int(first_frame.size),
        "maximum_streaming_error": stream_error,
        "streaming_equivalence_passed": streaming_passed,
        "maximum_reset_frame_error": reset_error,
        "reset_at_frame_equivalence_passed": reset_passed,
    }


def _acquire_lock(path: Path) -> bytes:
    payload = (
        json.dumps(
            {"pid": os.getpid(), "token": secrets.token_hex(32)},
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"release-gate lock already exists: {path}") from error
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _publish_report(path: Path, report: dict[str, Any], lock: Path, payload: bytes) -> None:
    temporary = path.parent / ".test_release_gate.publishing.json"
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError("immutable release-gate artifact already exists")
    if not lock.is_file() or lock.read_bytes() != payload:
        raise RuntimeError("release-gate lock was replaced")
    write_json(temporary, report)
    if path.exists() or path.is_symlink():
        raise FileExistsError("release-gate artifact appeared during publication")
    os.replace(temporary, path)
    if not lock.is_file() or lock.read_bytes() != payload:
        raise RuntimeError("release-gate lock changed after publication")
    lock.unlink()


def decide_from_config(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config_hash = file_sha256(config_path)
    config = _load_object(config_path, name="GMP residual config")
    if config.get("schema_version") != 2:
        raise ValueError("release gate requires residual config schema_version=2")
    gate = config.get("pretest_release_gate")
    if not isinstance(gate, dict):
        raise ValueError("residual config has no preregistered pretest_release_gate")
    if gate.get("status") != "preregistered_before_gmp_residual_run":
        raise ValueError("release thresholds were not preregistered")

    dataset = _resolve_project_relative(config["dataset"], name="dataset")
    selection_path = _resolve_project_relative(
        config["selection_manifest"], name="selection_manifest"
    )
    selection_config_path = _resolve_project_relative(
        config["selection_config"], name="selection_config"
    )
    output = _resolve_project_relative(config["output_dir"], name="output_dir")
    residual_manifest_path = output / "residual_manifest.json"
    result_path = output / "test_release_gate.json"
    temporary_path = output / ".test_release_gate.publishing.json"
    lock_path = output / ".test_release_gate.lock"
    for owned in (result_path, temporary_path, lock_path):
        if owned.exists() or owned.is_symlink():
            raise FileExistsError(f"immutable release-gate path exists: {owned}")

    _verify_hash(
        selection_path,
        str(config["selection_manifest_sha256"]),
        label="GMP selection manifest",
    )
    _verify_hash(
        selection_config_path,
        str(config["selection_config_sha256"]),
        label="GMP selection config",
    )
    selection = _load_object(selection_path, name="GMP selection manifest")
    if selection.get("model_class") != "complex_generalized_memory_polynomial":
        raise ValueError("release gate requires a selected GMP model")
    if selection.get("selection_split") != "validation":
        raise ValueError("GMP must be selected on validation")
    if selection.get("test_split_accessed") is not False:
        raise ValueError("GMP selection did not preserve the workflow test seal")
    if selection.get("test_evaluation_status") != "not_run_by_design":
        raise ValueError("GMP selection manifest indicates test evaluation")
    if selection.get("config_sha256") != config["selection_config_sha256"]:
        raise ValueError("selection config hash chain is inconsistent")

    model_name = Path(str(selection["selected_model"])).name
    model_path = selection_path.parent / model_name
    _verify_hash(
        model_path,
        str(selection["selected_model_sha256"]),
        label="selected GMP model",
    )
    model = GeneralizedMemoryPolynomialPA.load(model_path)

    protocol_lock = config.get("protocol_lock")
    if not isinstance(protocol_lock, dict):
        raise ValueError("residual config has no protocol_lock")
    alignment_path = _resolve_project_relative(
        protocol_lock["alignment_decision"], name="alignment_decision"
    )
    _verify_hash(
        alignment_path,
        str(protocol_lock["alignment_decision_sha256"]),
        label="alignment protocol decision",
    )
    alignment = _load_object(alignment_path, name="alignment decision")
    dataset_key = str(config["dataset_label"]).removeprefix("OpenDPD ")
    alignment_entry = alignment.get("datasets", {}).get(dataset_key)
    if not isinstance(alignment_entry, dict):
        raise ValueError("alignment decision has no matching dataset entry")
    protocol_a0 = bool(
        alignment.get("test_split_accessed") is False
        and alignment_entry.get("frozen_protocol_variant") == "a0"
        and alignment_entry.get("manual_override_of_runner_recommendation") is False
        and alignment_entry.get("formal_protocol_variant")
        == protocol_lock.get("formal_variant")
        and alignment_entry.get("formal_gmp_config_sha256")
        == config["selection_config_sha256"]
        and protocol_lock.get("integer_delay_samples") == 0
        and protocol_lock.get("fractional_transform_applied") is False
    )

    residual = _load_object(residual_manifest_path, name="GMP residual manifest")
    if residual.get("schema_version") != 2:
        raise ValueError("GMP residual manifest must use schema_version=2")
    if residual.get("task") != "forward_pa_residual_analysis":
        raise ValueError("unexpected GMP residual task")
    if residual.get("model_class") != selection.get("model_class"):
        raise ValueError("GMP residual model class mismatch")
    if residual.get("config_sha256") != config_hash:
        raise ValueError("GMP residual bundle was not produced by this config")
    if residual.get("selection_manifest_sha256") != config["selection_manifest_sha256"]:
        raise ValueError("GMP residual/selection hash chain mismatch")
    if residual.get("selected_model_sha256") != selection["selected_model_sha256"]:
        raise ValueError("GMP residual/model hash chain mismatch")
    if residual.get("accessed_splits") != ["train", "validation"]:
        raise ValueError("GMP residual accessed unexpected splits")
    if residual.get("test_split_accessed") is not False:
        raise ValueError("GMP residual manifest indicates test access")
    if residual.get("test_file_hashes_recorded") is not False:
        raise ValueError("GMP residual manifest records test hashes")

    residual_prediction_path = output / Path(str(residual["predictions"])).name
    train_report_path = output / Path(str(residual["train_oof_report"])).name
    validation_report_path = output / Path(str(residual["validation_report"])).name
    _verify_hash(
        residual_prediction_path,
        str(residual["predictions_sha256"]),
        label="GMP residual predictions",
    )
    _verify_hash(
        train_report_path,
        str(residual["train_oof_report_sha256"]),
        label="GMP train residual report",
    )
    _verify_hash(
        validation_report_path,
        str(residual["validation_report_sha256"]),
        label="GMP validation residual report",
    )
    train_report = _load_object(train_report_path, name="GMP train residual report")
    validation_report = _load_object(
        validation_report_path, name="GMP validation residual report"
    )

    mp_reference = gate.get("mp_reference")
    if not isinstance(mp_reference, dict):
        raise ValueError("pretest release gate has no MP reference")
    mp_manifest_path = _resolve_project_relative(
        mp_reference["residual_manifest"], name="MP residual manifest"
    )
    _verify_hash(
        mp_manifest_path,
        str(mp_reference["residual_manifest_sha256"]),
        label="MP residual manifest",
    )
    mp_manifest = _load_object(mp_manifest_path, name="MP residual manifest")
    if mp_manifest.get("test_split_accessed") is not False:
        raise ValueError("MP residual reference indicates test access")
    if mp_manifest.get("accessed_splits") != ["train", "validation"]:
        raise ValueError("MP residual reference accessed unexpected splits")
    if Path(str(mp_manifest.get("dataset"))).name != dataset.name:
        raise ValueError("MP residual reference belongs to another dataset")
    mp_prediction_path = mp_manifest_path.parent / Path(
        str(mp_manifest["predictions"])
    ).name
    expected_mp_prediction_hash = str(mp_reference["predictions_sha256"])
    if mp_manifest.get("predictions_sha256") != expected_mp_prediction_hash:
        raise ValueError("MP prediction hash chain is inconsistent")
    _verify_hash(
        mp_prediction_path,
        expected_mp_prediction_hash,
        label="MP residual predictions",
    )

    dataset_hashes = selection.get("dataset_files_sha256")
    if not isinstance(dataset_hashes, dict) or set(dataset_hashes) != _REQUIRED_DATASET_FILES:
        raise ValueError("selection dataset hashes must contain exactly train/val/spec")
    if any(Path(name).name.startswith("test_") for name in dataset_hashes):
        raise ValueError("test hashes are forbidden in the release gate")
    dataset_paths = {name: dataset / name for name in sorted(_REQUIRED_DATASET_FILES)}
    for name, path in dataset_paths.items():
        _verify_hash(path, str(dataset_hashes[name]), label=f"dataset {name}")

    tracked_paths = {
        "config": config_path,
        "selection_manifest": selection_path,
        "selection_config": selection_config_path,
        "selected_model": model_path,
        "alignment_decision": alignment_path,
        "gmp_residual_manifest": residual_manifest_path,
        "gmp_predictions": residual_prediction_path,
        "gmp_train_report": train_report_path,
        "gmp_validation_report": validation_report_path,
        "mp_residual_manifest": mp_manifest_path,
        "mp_predictions": mp_prediction_path,
        **{f"dataset:{name}": path for name, path in dataset_paths.items()},
    }
    frozen_hashes = {name: file_sha256(path) for name, path in tracked_paths.items()}
    lock_payload = _acquire_lock(lock_path)

    # The complete waveform access list.  Test is intentionally absent.
    train_input, train_output = load_split_pair(dataset, "train")
    validation_input, validation_output = load_split_pair(dataset, "val")
    nperseg = _nonnegative_integer(
        selection["protocol"]["nperseg"], name="selection nperseg"
    )
    matched = gate.get("matched_boundary")
    if not isinstance(matched, dict):
        raise ValueError("release gate has no matched_boundary")
    warmup = _nonnegative_integer(
        matched["warmup_samples_per_frame"], name="matched warmup"
    )
    cooldown = _nonnegative_integer(
        matched["future_cooldown_samples_per_frame"], name="matched cooldown"
    )
    train_mask = _boundary_mask(
        train_input.size,
        nperseg=nperseg,
        warmup=warmup,
        cooldown=cooldown,
    )
    validation_mask = _boundary_mask(
        validation_input.size,
        nperseg=nperseg,
        warmup=warmup,
        cooldown=cooldown,
    )

    with np.load(residual_prediction_path, allow_pickle=False) as data:
        if int(data["schema_version"]) != 2:
            raise ValueError("GMP residual prediction schema mismatch")
        gmp_train_prediction = np.asarray(data["train_oof_prediction"])
        gmp_validation_prediction = np.asarray(data["validation_prediction"])
    with np.load(mp_prediction_path, allow_pickle=False) as data:
        mp_train_prediction = np.asarray(data["train_oof_prediction"])
        mp_validation_prediction = np.asarray(data["validation_prediction"])

    metrics = {
        "mp_train_oof": _metric_pair(mp_train_prediction, train_output, train_mask),
        "mp_validation": _metric_pair(
            mp_validation_prediction, validation_output, validation_mask
        ),
        "gmp_train_oof": _metric_pair(
            gmp_train_prediction, train_output, train_mask
        ),
        "gmp_validation": _metric_pair(
            gmp_validation_prediction, validation_output, validation_mask
        ),
    }
    metrics["oof_gain_db"] = {
        "full": (
            metrics["mp_train_oof"]["full_nmse_db"]
            - metrics["gmp_train_oof"]["full_nmse_db"]
        ),
        "common_interior": (
            metrics["mp_train_oof"]["common_interior_nmse_db"]
            - metrics["gmp_train_oof"]["common_interior_nmse_db"]
        ),
    }
    metrics["gmp_oof_to_validation_degradation_db"] = {
        "full": (
            metrics["gmp_train_oof"]["full_nmse_db"]
            - metrics["gmp_validation"]["full_nmse_db"]
        ),
        "common_interior": (
            metrics["gmp_train_oof"]["common_interior_nmse_db"]
            - metrics["gmp_validation"]["common_interior_nmse_db"]
        ),
    }

    definition = gate.get("threshold_definition", {})
    tolerance = _finite_number(
        definition.get("validation_metric_absolute_tolerance_db"),
        name="validation metric tolerance",
    )
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("release gate has no thresholds")
    numeric_thresholds = {
        key: _finite_number(value, name=f"threshold {key}")
        for key, value in thresholds.items()
    }

    expected_mp = {
        "train_oof_full_nmse_db": metrics["mp_train_oof"]["full_nmse_db"],
        "train_oof_matched_interior_nmse_db": metrics["mp_train_oof"][
            "common_interior_nmse_db"
        ],
        "validation_full_nmse_db": metrics["mp_validation"]["full_nmse_db"],
        "validation_matched_interior_nmse_db": metrics["mp_validation"][
            "common_interior_nmse_db"
        ],
    }
    mp_reference_reproduced = all(
        abs(_finite_number(mp_reference[key], name=f"MP reference {key}") - actual)
        <= tolerance
        for key, actual in expected_mp.items()
    )
    expected_gmp = gate.get("gmp_validation_reference")
    if not isinstance(expected_gmp, dict):
        raise ValueError("release gate has no GMP validation reference")
    selection_trial = selection.get("selected_trial")
    if not isinstance(selection_trial, dict):
        raise ValueError("selection has no selected_trial")
    validation_reproduced = bool(
        abs(
            metrics["gmp_validation"]["full_nmse_db"]
            - _finite_number(expected_gmp["full_nmse_db"], name="GMP validation full")
        )
        <= tolerance
        and abs(
            metrics["gmp_validation"]["common_interior_nmse_db"]
            - _finite_number(
                expected_gmp["common_interior_nmse_db"],
                name="GMP validation interior",
            )
        )
        <= tolerance
        and abs(
            metrics["gmp_validation"]["full_nmse_db"]
            - float(selection_trial["validation_full_record"]["complex_nmse_pooled_db"])
        )
        <= tolerance
        and abs(
            metrics["gmp_validation"]["common_interior_nmse_db"]
            - float(
                selection_trial["validation_common_interior"][
                    "complex_nmse_pooled_db"
                ]
            )
        )
        <= tolerance
    )

    validation_gain_full = (
        expected_mp["validation_full_nmse_db"]
        - metrics["gmp_validation"]["full_nmse_db"]
    )
    validation_gain_interior = (
        expected_mp["validation_matched_interior_nmse_db"]
        - metrics["gmp_validation"]["common_interior_nmse_db"]
    )
    threshold_formula_consistent = bool(
        abs(
            numeric_thresholds["minimum_oof_gain_full_db"]
            - max(0.10, 0.25 * validation_gain_full)
        )
        <= 1e-12
        and abs(
            numeric_thresholds["minimum_oof_gain_common_interior_db"]
            - max(0.10, 0.25 * validation_gain_interior)
        )
        <= 1e-12
        and abs(
            numeric_thresholds["maximum_gmp_oof_full_nmse_db"]
            - (
                expected_mp["train_oof_full_nmse_db"]
                - numeric_thresholds["minimum_oof_gain_full_db"]
            )
        )
        <= 1e-12
        and abs(
            numeric_thresholds["maximum_gmp_oof_common_interior_nmse_db"]
            - (
                expected_mp["train_oof_matched_interior_nmse_db"]
                - numeric_thresholds["minimum_oof_gain_common_interior_db"]
            )
        )
        <= 1e-12
    )

    selected_recipe = residual.get("selected_recipe", {})
    recipe_matches = bool(
        selected_recipe.get("gmp_config") == selection_trial.get("gmp_config")
        and selected_recipe.get("ridge") == selection_trial.get("ridge")
        and selected_recipe.get("solver_mode") == selection_trial.get("solver_mode")
        and selected_recipe.get("svd_rcond") == selection_trial.get("svd_rcond")
    )
    folds = residual.get("oof_folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("GMP residual manifest has no OOF folds")
    full_diagnostics = residual.get("full_training_refit", {}).get(
        "fit_diagnostics", {}
    )
    full_condition = _finite_number(
        full_diagnostics.get("scaled_augmented_condition_number"),
        name="full-train condition number",
    )
    full_norm = _finite_number(
        full_diagnostics.get("coefficient_l2_norm"),
        name="full-train coefficient norm",
    )
    fold_recipe_exact = True
    fold_rank = True
    fold_finite = True
    condition_ratios: list[float] = []
    coefficient_ratios: list[float] = []
    support_ratios: list[float] = []
    for fold in folds:
        diagnostics = fold.get("fit_diagnostics", {})
        numerical = fold.get("fit_numerical_diagnostics", {})
        support = fold.get("input_support", {})
        fold_recipe_exact &= bool(
            diagnostics.get("solver_mode") == selection_trial.get("solver_mode")
            and diagnostics.get("ridge") == selection_trial.get("ridge")
            and diagnostics.get("svd_rcond") == selection_trial.get("svd_rcond")
        )
        fold_rank &= bool(
            numerical.get("solver_rank") == numerical.get("feature_count")
            and numerical.get("feature_count")
            == selection_trial.get("fit_diagnostics", {}).get("feature_count")
        )
        fold_finite &= _all_numeric_finite(fold)
        condition_ratios.append(
            _finite_number(numerical.get("condition_number"), name="fold condition")
            / full_condition
        )
        coefficient_ratios.append(
            _finite_number(
                numerical.get("coefficient_l2_norm"), name="fold coefficient norm"
            )
            / full_norm
        )
        support_ratios.append(
            _finite_number(
                support.get("held_to_fit_maximum_amplitude_ratio"),
                name="fold amplitude support ratio",
            )
        )

    operation = residual.get("operation_count_verification", {}).get("recomputed", {})
    selected_operations = selection_trial.get("operation_count_per_complex_sample", {})
    operation_exact = bool(
        isinstance(operation, dict)
        and all(operation.get(key) == selected_operations.get(key) for key in selected_operations)
    )
    real_multiplications = int(operation.get("real_multiplications", -1))
    streaming = _streaming_checks(model, train_input, nperseg=nperseg)
    validation_support = selection_trial.get("validation_input_support", {})
    residual_complete = bool(
        _diagnostic_bundle_complete(train_report)
        and _diagnostic_bundle_complete(validation_report)
    )

    max_gap = numeric_thresholds["maximum_oof_to_validation_degradation_db"]
    max_boundary_difference = numeric_thresholds[
        "maximum_absolute_full_minus_common_interior_db"
    ]
    predicates = {
        "provenance_and_workflow_seal": True,
        "protocol_a0_frozen": protocol_a0,
        "threshold_formula_consistent": threshold_formula_consistent,
        "mp_reference_metrics_reproduced": mp_reference_reproduced,
        "full_train_reproduction": bool(
            residual.get("full_training_refit", {})
            .get("frozen_npz_reproduction", {})
            .get("passed")
        ),
        "selected_recipe_frozen": recipe_matches and fold_recipe_exact,
        "fold_rank_and_finiteness": fold_rank and fold_finite,
        "fold_condition_stability": max(condition_ratios)
        <= numeric_thresholds["maximum_fold_condition_ratio"],
        "fold_coefficient_stability": max(coefficient_ratios)
        <= numeric_thresholds["maximum_fold_coefficient_l2_norm_ratio"],
        "validation_metrics_reproduced": validation_reproduced,
        "oof_full_gain": metrics["oof_gain_db"]["full"]
        >= numeric_thresholds["minimum_oof_gain_full_db"],
        "oof_common_interior_gain": metrics["oof_gain_db"]["common_interior"]
        >= numeric_thresholds["minimum_oof_gain_common_interior_db"],
        "oof_to_validation_gap": max(
            metrics["gmp_oof_to_validation_degradation_db"].values()
        )
        <= max_gap,
        "boundary_contract": bool(
            residual.get("analysis_common_boundary", {}).get(
                "warmup_samples_per_frame"
            )
            == warmup
            and residual.get("analysis_common_boundary", {}).get(
                "future_cooldown_samples_per_frame"
            )
            == cooldown
            and abs(
                metrics["gmp_train_oof"]["full_nmse_db"]
                - metrics["gmp_train_oof"]["common_interior_nmse_db"]
            )
            <= max_boundary_difference
            and abs(
                metrics["gmp_validation"]["full_nmse_db"]
                - metrics["gmp_validation"]["common_interior_nmse_db"]
            )
            <= max_boundary_difference
        ),
        "fold_input_support": max(support_ratios)
        <= numeric_thresholds["maximum_held_to_fit_amplitude_ratio"],
        "validation_input_support": bool(
            validation_support.get("fraction_above_training_maximum") == 0.0
            and expected_gmp.get("validation_fraction_above_training_maximum")
            == 0.0
        ),
        "operation_budget": bool(
            operation_exact
            and real_multiplications
            < int(numeric_thresholds["real_multiplication_limit_exclusive"])
        ),
        "causal_model": bool(
            model.config.leading_policy == "causal_leading"
            and model.config.lookahead_samples == 0
        ),
        "selected_model_streaming_equivalence": bool(
            streaming["streaming_equivalence_passed"]
        ),
        "reset_at_frame_equivalence": bool(
            streaming["reset_at_frame_equivalence_passed"]
        ),
        "residual_bundle_complete": residual_complete,
    }
    failed = [name for name, passed in predicates.items() if not passed]

    for name, path in tracked_paths.items():
        _verify_hash(path, frozen_hashes[name], label=f"pre-publication {name}")
    source_path = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "task": "gmp_sealed_test_release_gate",
        "dataset_label": config["dataset_label"],
        "config": str(config_path),
        "config_sha256": config_hash,
        "inputs": {
            "gmp_selection_manifest": {
                "path": str(selection_path),
                "sha256": file_sha256(selection_path),
            },
            "gmp_residual_manifest": {
                "path": str(residual_manifest_path),
                "sha256": file_sha256(residual_manifest_path),
            },
            "mp_residual_manifest": {
                "path": str(mp_manifest_path),
                "sha256": file_sha256(mp_manifest_path),
            },
            "selected_gmp_model": {
                "path": str(model_path),
                "sha256": file_sha256(model_path),
            },
        },
        "accessed_splits": ["train", "validation"],
        "test_split_accessed": False,
        "test_file_hashes_recorded": False,
        "workflow_specific_seal": (
            "the dataset test split was used by historical MP experiments; "
            "sealed here means never opened by GMP selection, residual audit, "
            "or this release command"
        ),
        "thresholds": numeric_thresholds,
        "threshold_definition": definition,
        "metrics": metrics,
        "fold_diagnostics": {
            "fold_count": len(folds),
            "condition_ratios_to_full_train": condition_ratios,
            "coefficient_l2_norm_ratios_to_full_train": coefficient_ratios,
            "held_to_fit_amplitude_ratios": support_ratios,
        },
        "streaming_checks": streaming,
        "predicates": predicates,
        "diagnostics": {
            "residual_correlation_is_hard_rejection": False,
            "slow_state_supported": (
                train_report.get("slow_state_branch_eligible") is True
                and validation_report.get("slow_state_branch_eligible") is True
            ),
            "fixed_point_pending": True,
        },
        "decision": {
            "may_open_gmp_test_once": not failed,
            "failed_predicates": failed,
            "authorization_scope": (
                "one evaluation of the already frozen GMP on the workflow-"
                "sealed test split; no refit, retuning, or retry"
            ),
            "does_not_establish": [
                "Gate A-to-B",
                "fixed-point readiness",
                "physical-PA validation",
                "Huawei acceptance",
                "superiority over OpenDPD",
            ],
        },
        "evaluation_seconds": time.perf_counter() - started,
        "source_sha256": {
            "experiments/decide_gmp_test_release.py": file_sha256(source_path),
            "baseline/gmp_pa.py": file_sha256(PROJECT_ROOT / "baseline/gmp_pa.py"),
            "baseline/metrics.py": file_sha256(PROJECT_ROOT / "baseline/metrics.py"),
        },
        "publication": {
            "immutable": True,
            "atomic_same_directory_replace": True,
            "failure_lock_remains": True,
        },
    }
    _publish_report(result_path, report, lock_path, lock_payload)
    return report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide whether one frozen GMP test evaluation is released"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    report = decide_from_config(args.config)
    decision = report["decision"]
    print(
        "GMP test release:",
        "PASS" if decision["may_open_gmp_test_once"] else "HOLD",
        "failed=" + ",".join(decision["failed_predicates"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
