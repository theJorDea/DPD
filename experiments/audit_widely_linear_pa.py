"""Run the preregistered APA widely-linear residual PA audit.

This runner is intentionally narrower than the general MP/GMP residual
workflow.  It consumes a hash-bound frozen GMP recipe, refits only GMP
coefficients in leave-one-frame-out training folds, and fits the conjugate
residual branch on each fold's fit residual.  The validation split is loaded
only after the OOF candidate is frozen and is labelled descriptive/reused;
test files are never opened or hashed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import sys
import time
from typing import Any

import numpy as np

from baseline.complexity import (
    OperationCount,
    widely_linear_residual_correction_cost,
)
from baseline.gmp_pa import GeneralizedMemoryPolynomialPA
from baseline.metrics import (
    nmse_opendpd_db,
    nmse_pooled_db,
    time_domain_rms_evm_db,
)
from baseline.residual_analysis import (
    ResidualAnalysisSpec,
    analyze_pa_residuals,
)
from baseline.train_spline import (
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)
from baseline.widely_linear_pa import (
    WidelyLinearResidualCorrection,
    fit_widely_linear_residual_correction,
)
from experiments.analyze_pa_residuals import (
    SelectedGMPRecipe,
    _fit_selected_recipe,
    _load_json_object,
    _parse_and_verify_recipe,
    _resolve_project_relative,
    _segmented_interior_mask,
    explicit_frame_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
REQUIRED_DATASET_FILES = (
    "spec.json",
    "train_input.csv",
    "train_output.csv",
    "val_input.csv",
    "val_output.csv",
)
OPERATION_FIELDS = (
    "real_multiplications",
    "real_additions",
    "stored_real_coefficients",
)


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    name: str
    delays: tuple[int, ...]
    operation_count: OperationCount

    @property
    def tap_count(self) -> int:
        return len(self.delays)


def _verify_file(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )


def _resolve_manifest_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (PROJECT_ROOT / candidate).resolve()
    )
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{name} must remain inside the project root") from error
    return resolved


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_json_object(path, name="widely-linear audit config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("widely-linear audit config schema_version mismatch")
    if config.get("task") != "forward_pa_model_widely_linear_residual_audit":
        raise ValueError("unexpected widely-linear audit task")
    if config.get("scope", {}).get("test_split_access_permitted") is not False:
        raise ValueError("widely-linear audit must prohibit test access")
    for key in (
        "dataset",
        "output_dir",
        "discovery_evidence",
        "base_model",
        "correction_fit",
        "candidate_supports",
        "selection_rule",
    ):
        if key not in config:
            raise ValueError(f"audit config is missing {key}")
    return config


def _verify_discovery_evidence(
    config: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    evidence = config["discovery_evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("discovery_evidence must be an object")
    required_entries = (
        "base_selection_manifest",
        "base_selected_model",
        "base_residual_manifest",
        "train_oof_residual_report",
        "validation_residual_report",
    )
    paths: dict[str, Path] = {}
    for key in required_entries:
        entry = evidence.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"discovery evidence entry {key} is missing")
        path = _resolve_project_relative(
            entry.get("path"),
            project_root=PROJECT_ROOT,
            name=f"discovery_evidence.{key}.path",
        )
        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"discovery evidence {key} has invalid SHA-256")
        _verify_file(path, expected, label=f"discovery evidence {key}")
        paths[key] = path

    selection_path = paths["base_selection_manifest"]
    model_path = paths["base_selected_model"]
    residual_manifest_path = paths["base_residual_manifest"]
    selection_manifest = _load_json_object(
        selection_path,
        name="base selection manifest",
    )
    residual_manifest = _load_json_object(
        residual_manifest_path,
        name="base residual manifest",
    )
    if selection_manifest.get("test_split_accessed") is not False:
        raise ValueError("base selection manifest does not seal test")
    if residual_manifest.get("test_split_accessed") is not False:
        raise ValueError("base residual manifest does not seal test")
    if residual_manifest.get("test_file_hashes_recorded") is not False:
        raise ValueError("base residual manifest must not hash test files")
    if selection_manifest.get("selected_model_sha256") != file_sha256(model_path):
        raise ValueError("base selection manifest model hash disagrees")
    if residual_manifest.get("selected_model_sha256") != file_sha256(model_path):
        raise ValueError("base residual manifest model hash disagrees")
    if residual_manifest.get("selection_manifest_sha256") != file_sha256(
        selection_path
    ):
        raise ValueError("base residual manifest selection hash disagrees")
    if selection_manifest.get("model_class") != (
        "complex_generalized_memory_polynomial"
    ):
        raise ValueError("this audit requires a selected GMP base model")

    source_status: dict[str, Any] = {}
    source_files = evidence.get("source_files", {})
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("discovery_evidence.source_files is missing")
    for raw_label, expected in source_files.items():
        label = str(raw_label)
        path = _resolve_project_relative(
            label,
            project_root=PROJECT_ROOT,
            name="discovery source path",
        )
        actual = file_sha256(path)
        source_status[label] = {
            "preregistered_sha256": str(expected),
            "current_sha256": actual,
            "match": actual == str(expected),
            "role": "provenance_only; numerical evidence hashes are authoritative",
        }
    return {
        "paths": paths,
        "selection": selection_manifest,
        "residual_manifest": residual_manifest,
        "source_status": source_status,
    }


def _verify_dataset_before_load(
    config: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[Path, dict[str, str]]:
    dataset = _resolve_project_relative(
        config["dataset"],
        project_root=PROJECT_ROOT,
        name="dataset",
    )
    hashes = selection.get("dataset_files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_DATASET_FILES):
        raise ValueError(
            "selection dataset hashes must contain exactly spec and train/val files"
        )
    if any(Path(name).name.startswith("test_") for name in hashes):
        raise ValueError("test file hash appears in dataset evidence")
    frozen = {name: str(hashes[name]) for name in REQUIRED_DATASET_FILES}
    for name, expected in frozen.items():
        _verify_file(dataset / name, expected, label=f"pre-load dataset {name}")
    return dataset, frozen


def _verify_dataset_after_load(
    dataset: Path,
    frozen: dict[str, str],
    *,
    scope: str,
) -> None:
    for name, expected in frozen.items():
        _verify_file(dataset / name, expected, label=f"{scope} dataset {name}")


def _parse_candidates(
    config: dict[str, Any],
    *,
    base_cost: OperationCount,
) -> tuple[CandidateSpec, ...]:
    raw_candidates = config["candidate_supports"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate_supports must be a non-empty list")
    limit = int(
        config["operation_count_convention"][
            "real_multiplication_limit_exclusive"
        ]
    )
    result: list[CandidateSpec] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ValueError("every candidate support must be an object")
        name = raw.get("name")
        delays_raw = raw.get("delays")
        if not isinstance(name, str) or not name:
            raise ValueError("candidate support name must be non-empty")
        if not isinstance(delays_raw, list):
            raise ValueError(f"candidate {name} delays must be a JSON list")
        delays = tuple(int(delay) for delay in delays_raw)
        if any(delay < 0 for delay in delays):
            raise ValueError(f"candidate {name} contains a non-causal delay")
        if len(set(delays)) != len(delays):
            raise ValueError(f"candidate {name} delays are not unique")
        if index == 0 and delays:
            raise ValueError("first candidate must be no_correction")
        if index > 0 and not delays:
            raise ValueError("only first candidate may have no delays")
        incremental = (
            None
            if not delays
            else widely_linear_residual_correction_cost(
                delays,
                convention="4m2a",
                reuse_input_delay_state=True,
            )
        )
        actual = base_cost if incremental is None else base_cost + incremental
        for field in OPERATION_FIELDS:
            recorded = raw.get(field)
            if not isinstance(recorded, (int, np.integer)) or isinstance(
                recorded,
                (bool, np.bool_),
            ):
                raise ValueError(f"candidate {name} lacks integer {field}")
            if int(recorded) != int(getattr(actual, field)):
                raise ValueError(
                    f"candidate {name} {field} disagrees with analytical count"
                )
        if actual.real_multiplications >= limit:
            raise ValueError(
                f"candidate {name} violates strict MUL limit {limit}"
            )
        result.append(
            CandidateSpec(
                name=name,
                delays=delays,
                operation_count=actual,
            )
        )
    if len({candidate.name for candidate in result}) != len(result):
        raise ValueError("candidate support names must be unique")
    return tuple(result)


def _validate_base_contract(
    config: dict[str, Any],
    recipe: SelectedGMPRecipe,
    base_model: GeneralizedMemoryPolynomialPA,
    selection: dict[str, Any],
) -> tuple[OperationCount, int, int]:
    base_cfg = config["base_model"]
    expected_gmp = base_cfg.get("gmp_config")
    if not isinstance(expected_gmp, dict):
        raise ValueError("base_model.gmp_config is missing")
    if expected_gmp != dataclasses.asdict(recipe.config):
        raise ValueError("audit base GMP config disagrees with selection")
    if float(base_cfg.get("ridge")) != recipe.ridge:
        raise ValueError("audit base GMP ridge disagrees with selection")
    if base_cfg.get("solver_mode") != recipe.solver_mode:
        raise ValueError("audit base GMP solver disagrees with selection")
    if base_cfg.get("common_warmup_samples_per_frame") != selection.get(
        "common_warmup_samples_per_frame"
    ):
        raise ValueError("audit common warmup disagrees with selection")
    if base_cfg.get("common_future_cooldown_samples_per_frame") != selection.get(
        "common_future_cooldown_samples_per_frame"
    ):
        raise ValueError("audit common cooldown disagrees with selection")
    operation = base_model.operation_count
    recorded_operation = base_cfg.get("operation_count_per_complex_sample")
    if not isinstance(recorded_operation, dict):
        raise ValueError("base_model operation count is missing")
    for field in OPERATION_FIELDS:
        if int(recorded_operation.get(field, -1)) != int(
            getattr(operation, field)
        ):
            raise ValueError(f"base GMP {field} disagrees with implementation")
    selected_trial = selection.get("selected_trial")
    if not isinstance(selected_trial, dict):
        raise ValueError("selection selected_trial is missing")
    selected_operation = selected_trial.get("operation_count_per_complex_sample")
    if not isinstance(selected_operation, dict):
        raise ValueError("selection operation count is missing")
    for field in OPERATION_FIELDS:
        if int(selected_operation.get(field, -1)) != int(
            getattr(operation, field)
        ):
            raise ValueError(f"selection GMP {field} disagrees with implementation")
    selection_rule = config["selection_rule"]
    eligibility = selection_rule.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError("selection_rule.eligibility is missing")
    limit = int(
        config["operation_count_convention"][
            "real_multiplication_limit_exclusive"
        ]
    )
    if int(eligibility.get("real_multiplications_strictly_below", -1)) != limit:
        raise ValueError("selection MUL limit is not bound to operation convention")
    return operation, int(base_cfg["common_warmup_samples_per_frame"]), int(
        base_cfg["common_future_cooldown_samples_per_frame"]
    )


def _complex_pairs(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"real": float(value.real), "imag": float(value.imag)}
        for value in np.asarray(values).reshape(-1)
    ]


def _metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    nperseg: int,
    common_mask: np.ndarray,
) -> dict[str, Any]:
    if prediction.shape != reference.shape or prediction.ndim != 1:
        raise ValueError("prediction/reference must be aligned 1-D arrays")
    complete = (prediction.size // nperseg) * nperseg
    if complete:
        frames_prediction = prediction[:complete].reshape(-1, nperseg)
        frames_reference = reference[:complete].reshape(-1, nperseg)
        per_frame = [
            nmse_pooled_db(frame_prediction, frame_reference)
            for frame_prediction, frame_reference in zip(
                frames_prediction,
                frames_reference,
                strict=True,
            )
        ]
        opendpd_metric = nmse_opendpd_db(
            frames_prediction,
            frames_reference,
        )
    else:
        frames_prediction = np.empty((0, nperseg), dtype=np.complex128)
        frames_reference = np.empty((0, nperseg), dtype=np.complex128)
        per_frame = []
        opendpd_metric = None
    common_prediction = prediction[common_mask]
    common_reference = reference[common_mask]
    return {
        "full_record_nmse_db": nmse_pooled_db(prediction, reference),
        "common_interior_nmse_db": nmse_pooled_db(
            common_prediction,
            common_reference,
        ),
        "opendpd_compatible_nmse_db": opendpd_metric,
        "common_interior_evm_db": time_domain_rms_evm_db(
            common_prediction,
            common_reference,
        ),
        "per_frame_nmse_db": per_frame,
        "scored_samples_full": int(prediction.size),
        "scored_samples_common_interior": int(np.count_nonzero(common_mask)),
    }


def _streaming_checks(
    model: WidelyLinearResidualCorrection,
    signal: np.ndarray,
    *,
    segment_length: int,
) -> dict[str, Any]:
    expected = model.predict(signal)
    boundaries = sorted(
        set(
            [
                0,
                min(signal.size, 1),
                min(signal.size, signal.size // 3),
                min(signal.size, 2 * signal.size // 3),
                signal.size,
            ]
        )
    )
    state = model.initial_state()
    chunks: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop <= start:
            continue
        chunk, state = model.predict_chunk(signal[start:stop], state)
        chunks.append(chunk)
    streamed = np.concatenate(chunks) if chunks else np.asarray([], dtype=complex)
    reset = model.predict_segments(signal, segment_length)
    expected_reset = np.concatenate(
        [
            model.predict(signal[start : min(start + segment_length, signal.size)])
            for start in range(0, signal.size, segment_length)
        ]
    )
    streaming_error = float(
        np.max(np.abs(streamed - expected), initial=0.0)
    )
    reset_error = float(np.max(np.abs(reset - expected_reset), initial=0.0))
    return {
        "streaming_chunk_equivalence_passed": bool(
            np.array_equal(streamed, expected)
        ),
        "reset_at_frame_equivalence_passed": bool(
            np.array_equal(reset, expected_reset)
        ),
        "maximum_streaming_error": streaming_error,
        "maximum_reset_error": reset_error,
    }


def _fit_oof_candidates(
    train_input: np.ndarray,
    train_output: np.ndarray,
    *,
    nperseg: int,
    common_warmup: int,
    common_cooldown: int,
    recipe: SelectedGMPRecipe,
    candidates: tuple[CandidateSpec, ...],
    correction_ridge: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    frame_ids = explicit_frame_ids(train_input.size, nperseg)
    frame_values = tuple(int(value) for value in np.unique(frame_ids))
    frame_count = len(frame_values)
    if frame_count < 2:
        raise ValueError("OOF audit requires at least two train frames")
    predictions = {
        candidate.name: np.empty(train_input.shape, dtype=np.complex128)
        for candidate in candidates
    }
    fold_reports: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    for held_frame in frame_values:
        held_indices = np.flatnonzero(frame_ids == held_frame)
        fit_indices = np.concatenate(
            [
                np.flatnonzero(frame_ids == frame)
                for frame in frame_values
                if frame != held_frame
            ]
        )
        x_fit = train_input[fit_indices]
        y_fit = train_output[fit_indices]
        x_held = train_input[held_indices]
        y_held = train_output[held_indices]
        base_started = time.perf_counter()
        base_model, base_diagnostics = _fit_selected_recipe(
            recipe,
            x_fit,
            y_fit,
            nperseg=nperseg,
        )
        base_fit_seconds = time.perf_counter() - base_started
        total_fit_seconds += base_fit_seconds
        base_fit_prediction = base_model.predict_segments(x_fit, nperseg)
        fit_residual = y_fit - base_fit_prediction
        base_held_prediction = base_model.predict(x_held)
        predictions[candidates[0].name][held_indices] = base_held_prediction
        candidate_fold_reports: list[dict[str, Any]] = []
        held_common_mask = _segmented_interior_mask(
            x_held.size,
            nperseg=nperseg,
            warmup_samples=common_warmup,
            cooldown_samples=common_cooldown,
        )
        base_fold_metrics = _metrics(
            base_held_prediction,
            y_held,
            nperseg=nperseg,
            common_mask=held_common_mask,
        )
        for candidate in candidates[1:]:
            correction_started = time.perf_counter()
            correction, correction_diagnostics = (
                fit_widely_linear_residual_correction(
                    x_fit,
                    fit_residual,
                    delays=candidate.delays,
                    ridge=correction_ridge,
                    segment_length=nperseg,
                    coefficient_dtype=np.complex128,
                )
            )
            correction_fit_seconds = time.perf_counter() - correction_started
            total_fit_seconds += correction_fit_seconds
            corrected_held_prediction = base_held_prediction + correction.predict(
                x_held
            )
            predictions[candidate.name][held_indices] = corrected_held_prediction
            corrected_metrics = _metrics(
                corrected_held_prediction,
                y_held,
                nperseg=nperseg,
                common_mask=held_common_mask,
            )
            stream_checks = _streaming_checks(
                correction,
                x_held,
                segment_length=nperseg,
            )
            candidate_fold_reports.append(
                {
                    "candidate": candidate.name,
                    "delays": list(candidate.delays),
                    "correction_fit_seconds": correction_fit_seconds,
                    "correction_diagnostics": dataclasses.asdict(
                        correction_diagnostics
                    ),
                    "coefficients": _complex_pairs(correction.coefficients),
                    "metrics": corrected_metrics,
                    "fold_full_gain_db": (
                        base_fold_metrics["full_record_nmse_db"]
                        - corrected_metrics["full_record_nmse_db"]
                    ),
                    "fold_common_gain_db": (
                        base_fold_metrics["common_interior_nmse_db"]
                        - corrected_metrics["common_interior_nmse_db"]
                    ),
                    "streaming_checks": stream_checks,
                }
            )
        fold_reports.append(
            {
                "held_frame_id": held_frame,
                "fit_frame_ids": [
                    frame for frame in frame_values if frame != held_frame
                ],
                "fit_sample_count": int(fit_indices.size),
                "held_sample_count": int(held_indices.size),
                "frame_state_policy": "zero/reset at each explicit frame",
                "base_fit_seconds": base_fit_seconds,
                "base_fit_diagnostics": base_diagnostics,
                "base_metrics": base_fold_metrics,
                "correction_candidates": candidate_fold_reports,
            }
        )
    if not all(np.all(np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("OOF candidate prediction contains non-finite values")
    return predictions, fold_reports, total_fit_seconds


def _aggregate_candidate_results(
    predictions: dict[str, np.ndarray],
    reference: np.ndarray,
    *,
    candidates: tuple[CandidateSpec, ...],
    fold_reports: list[dict[str, Any]],
    nperseg: int,
    common_mask: np.ndarray,
    limit: int,
    minimum_full_gain: float,
    minimum_common_gain: float,
    maximum_fold_degradation: float,
) -> dict[str, dict[str, Any]]:
    baseline_metrics = _metrics(
        predictions[candidates[0].name],
        reference,
        nperseg=nperseg,
        common_mask=common_mask,
    )
    results: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        metrics = _metrics(
            predictions[candidate.name],
            reference,
            nperseg=nperseg,
            common_mask=common_mask,
        )
        if index == 0:
            results[candidate.name] = {
                "candidate": candidate.name,
                "delays": [],
                "tap_count": 0,
                "operation_count": candidate.operation_count.to_dict(),
                "metrics": metrics,
                "gain_full_db": 0.0,
                "gain_common_db": 0.0,
                "score_db": 0.0,
                "eligible": False,
                "fallback": True,
                "reason": "no_correction_fallback_only",
            }
            continue
        fold_rows = [
            row
            for fold in fold_reports
            for row in fold["correction_candidates"]
            if row["candidate"] == candidate.name
        ]
        full_gains = [float(row["fold_full_gain_db"]) for row in fold_rows]
        common_gains = [float(row["fold_common_gain_db"]) for row in fold_rows]
        stream_pass = all(
            row["streaming_checks"]["streaming_chunk_equivalence_passed"]
            and row["streaming_checks"]["reset_at_frame_equivalence_passed"]
            for row in fold_rows
        )
        rank_pass = all(
            int(row["correction_diagnostics"]["solver_rank"])
            == candidate.tap_count
            for row in fold_rows
        )
        gain_full = float(
            baseline_metrics["full_record_nmse_db"]
            - metrics["full_record_nmse_db"]
        )
        gain_common = float(
            baseline_metrics["common_interior_nmse_db"]
            - metrics["common_interior_nmse_db"]
        )
        eligible = bool(
            gain_full >= minimum_full_gain
            and gain_common >= minimum_common_gain
            and min(full_gains, default=-np.inf) >= -maximum_fold_degradation
            and min(common_gains, default=-np.inf)
            >= -maximum_fold_degradation
            and stream_pass
            and rank_pass
            and candidate.operation_count.real_multiplications < limit
        )
        results[candidate.name] = {
            "candidate": candidate.name,
            "delays": list(candidate.delays),
            "tap_count": candidate.tap_count,
            "operation_count": candidate.operation_count.to_dict(),
            "metrics": metrics,
            "gain_full_db": gain_full,
            "gain_common_db": gain_common,
            "score_db": min(gain_full, gain_common),
            "fold_full_gain_db": full_gains,
            "fold_common_gain_db": common_gains,
            "minimum_fold_full_gain_db": min(full_gains, default=float("nan")),
            "minimum_fold_common_gain_db": min(
                common_gains,
                default=float("nan"),
            ),
            "all_folds_full_rank": rank_pass,
            "all_folds_streaming_and_reset_equivalent": stream_pass,
            "eligible": eligible,
            "fallback": False,
        }
    return results


def _select_candidate(
    results: dict[str, dict[str, Any]],
    candidates: tuple[CandidateSpec, ...],
    *,
    tie_tolerance_db: float,
) -> str:
    eligible = [
        results[candidate.name]
        for candidate in candidates[1:]
        if results[candidate.name]["eligible"]
    ]
    if not eligible:
        return candidates[0].name
    best_score = max(float(row["score_db"]) for row in eligible)
    near_best = [
        row
        for row in eligible
        if float(row["score_db"]) >= best_score - tie_tolerance_db
    ]
    near_best.sort(key=lambda row: (int(row["tap_count"]), row["candidate"]))
    return str(near_best[0]["candidate"])


def _make_residual_spec(
    train_report: dict[str, Any],
) -> tuple[ResidualAnalysisSpec, dict[str, object]]:
    raw_spec = train_report.get("spec")
    frozen_reference = train_report.get("frozen_reference")
    if not isinstance(raw_spec, dict) or not isinstance(frozen_reference, dict):
        raise ValueError("frozen residual report lacks spec/reference")
    return ResidualAnalysisSpec(**raw_spec), frozen_reference


def _write_bundle(
    output: Path,
    *,
    config_path: Path,
    config_sha256: str,
    source_hashes: dict[str, str],
    evidence_status: dict[str, Any],
    dataset_hashes: dict[str, str],
    selected_name: str,
    selected_candidate: CandidateSpec,
    candidate_results: dict[str, dict[str, Any]],
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    train_base_prediction: np.ndarray,
    train_selected_prediction: np.ndarray,
    validation_base_prediction: np.ndarray,
    validation_selected_prediction: np.ndarray,
    train_common_mask: np.ndarray,
    validation_common_mask: np.ndarray,
    correction: WidelyLinearResidualCorrection | None,
    correction_diagnostics: dict[str, Any] | None,
    train_analysis: dict[str, Any],
    validation_analysis: dict[str, Any],
    runtime_seconds: float,
    test_split_accessed: bool,
    input_reverify: dict[str, bool],
) -> dict[str, Any]:
    if test_split_accessed:
        raise RuntimeError("internal error: test access flag is true")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable audit output already exists: {output}")
    lock_path = output.parent / f".{output.name}.lock"
    if lock_path.exists() or lock_path.is_symlink():
        raise FileExistsError(f"audit lock already exists: {lock_path}")
    payload = json.dumps(
        {"pid": os.getpid(), "token": secrets.token_hex(24)},
        sort_keys=True,
    ).encode("utf-8")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary = output.parent / f".{output.name}.tmp-{secrets.token_hex(12)}"
    temporary.mkdir()
    try:
        predictions_path = temporary / "predictions.npz"
        np.savez_compressed(
            predictions_path,
            schema_version=np.asarray(1, dtype=np.int64),
            train_base_prediction=train_base_prediction,
            train_selected_prediction=train_selected_prediction,
            validation_base_prediction=validation_base_prediction,
            validation_selected_prediction=validation_selected_prediction,
            train_common_mask=train_common_mask,
            validation_common_mask=validation_common_mask,
        )
        write_json(temporary / "train_oof_residual_analysis.json", train_analysis)
        write_json(
            temporary / "validation_reused_residual_analysis.json",
            validation_analysis,
        )
        if correction is not None:
            correction.save(temporary / "selected_correction.npz")
        execution = {
            "schema_version": 1,
            "task": "forward_pa_model_widely_linear_residual_audit",
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "runtime_seconds": runtime_seconds,
            "test_split_accessed": False,
            "input_reverification": input_reverify,
        }
        write_json(temporary / "execution_record.json", execution)
        relative = lambda path: str(path.relative_to(PROJECT_ROOT))
        manifest = {
            "schema_version": 1,
            "task": "forward_pa_model_widely_linear_residual_audit",
            "status": "post_discovery_internal_resampling_only",
            "dataset": str(config_path),
            "config": relative(config_path),
            "config_sha256": config_sha256,
            "source_sha256": source_hashes,
            "discovery_evidence_source_status": evidence_status,
            "dataset_files_sha256": dataset_hashes,
            "accessed_splits": ["train", "validation"],
            "test_split_accessed": False,
            "test_file_hashes_recorded": False,
            "base_model": "frozen selected causal GMP; OOF coefficients refit only",
            "candidate_results": candidate_results,
            "selected_candidate": selected_name,
            "selected_delays": list(selected_candidate.delays),
            "selected_operation_count": selected_candidate.operation_count.to_dict(),
            "selected_correction_diagnostics": correction_diagnostics,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "validation_role": "already-viewed descriptive reused split; not independent confirmation",
            "independent_acceptance_required": "new capture or operating point plus measurement-path IQ audit",
            "negative_lag_policy": "future-input diagnostics are not deployable features",
            "slow_state_policy": "locked because independent capture count is zero",
            "predictions": "predictions.npz",
            "predictions_sha256": file_sha256(predictions_path),
            "train_oof_residual_analysis": "train_oof_residual_analysis.json",
            "train_oof_residual_analysis_sha256": file_sha256(
                temporary / "train_oof_residual_analysis.json"
            ),
            "validation_reused_residual_analysis": "validation_reused_residual_analysis.json",
            "validation_reused_residual_analysis_sha256": file_sha256(
                temporary / "validation_reused_residual_analysis.json"
            ),
            "execution_record": "execution_record.json",
            "execution_record_sha256": file_sha256(
                temporary / "execution_record.json"
            ),
            "selected_correction": (
                None if correction is None else "selected_correction.npz"
            ),
            "selected_correction_sha256": (
                None
                if correction is None
                else file_sha256(temporary / "selected_correction.npz")
            ),
            "input_integrity": {
                "all_hashes_verified_before_waveform_load": True,
                "all_inputs_reverified_after_waveform_load": all(
                    input_reverify.values()
                ),
                "test_never_opened_or_hashed": True,
            },
        }
        write_json(temporary / "audit_manifest.json", manifest)
        os.replace(temporary, output)
        temporary = None  # type: ignore[assignment]
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        if lock_path.exists() and not lock_path.is_symlink():
            lock_path.unlink()


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    source_config = Path(config_path).resolve()
    config_sha256 = file_sha256(source_config)
    config = _load_config(source_config)
    if file_sha256(source_config) != config_sha256:
        raise RuntimeError("audit config changed during parsing")
    evidence_bundle = _verify_discovery_evidence(config, {})
    selection = evidence_bundle["selection"]
    residual_manifest = evidence_bundle["residual_manifest"]
    model_path = evidence_bundle["paths"]["base_selected_model"]
    recipe, frozen_model, _ = _parse_and_verify_recipe(selection, model_path)
    if not isinstance(recipe, SelectedGMPRecipe):
        raise ValueError("widely-linear audit requires SelectedGMPRecipe")
    if not isinstance(frozen_model, GeneralizedMemoryPolynomialPA):
        raise ValueError("selected base model is not GMP")
    base_cost, common_warmup, common_cooldown = _validate_base_contract(
        config,
        recipe,
        frozen_model,
        selection,
    )
    candidates = _parse_candidates(config, base_cost=base_cost)
    dataset, dataset_hashes = _verify_dataset_before_load(config, selection)
    dataset_spec = load_dataset_spec(dataset)
    protocol = selection.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("selection protocol is missing")
    nperseg = int(protocol.get("nperseg", -1))
    if nperseg <= 1 or int(dataset_spec.get("nperseg", -1)) != nperseg:
        raise ValueError("dataset and selected protocol nperseg disagree")
    if protocol.get("alignment_delay_samples") != 0 or protocol.get(
        "fractional_delay_applied"
    ) is not False:
        raise ValueError("audit requires frozen A0 integer/no-fractional alignment")
    train_input, train_output = load_split_pair(dataset, "train")
    validation_input, validation_output = load_split_pair(dataset, "val")
    _verify_dataset_after_load(dataset, dataset_hashes, scope="post-load")
    train_common_mask = _segmented_interior_mask(
        train_input.size,
        nperseg=nperseg,
        warmup_samples=common_warmup,
        cooldown_samples=common_cooldown,
    )
    validation_common_mask = _segmented_interior_mask(
        validation_input.size,
        nperseg=nperseg,
        warmup_samples=common_warmup,
        cooldown_samples=common_cooldown,
    )
    source_paths = {
        "experiments/audit_widely_linear_pa.py": Path(__file__).resolve(),
        "baseline/widely_linear_pa.py": PROJECT_ROOT / "baseline/widely_linear_pa.py",
        "baseline/complexity.py": PROJECT_ROOT / "baseline/complexity.py",
        "baseline/gmp_pa.py": PROJECT_ROOT / "baseline/gmp_pa.py",
        "baseline/residual_analysis.py": PROJECT_ROOT / "baseline/residual_analysis.py",
    }
    source_hashes = {label: file_sha256(path) for label, path in source_paths.items()}
    input_reverify = {
        name: file_sha256(dataset / name) == expected
        for name, expected in dataset_hashes.items()
    }
    if not all(input_reverify.values()):
        raise RuntimeError("input changed after waveform load")
    oof_predictions, fold_reports, oof_fit_seconds = _fit_oof_candidates(
        train_input,
        train_output,
        nperseg=nperseg,
        common_warmup=common_warmup,
        common_cooldown=common_cooldown,
        recipe=recipe,
        candidates=candidates,
        correction_ridge=float(config["correction_fit"]["ridge"]),
    )
    rule = config["selection_rule"]
    eligibility = rule["eligibility"]
    candidate_results = _aggregate_candidate_results(
        oof_predictions,
        train_output,
        candidates=candidates,
        fold_reports=fold_reports,
        nperseg=nperseg,
        common_mask=train_common_mask,
        limit=int(
            config["operation_count_convention"][
                "real_multiplication_limit_exclusive"
            ]
        ),
        minimum_full_gain=float(eligibility["minimum_oof_full_record_gain_db"]),
        minimum_common_gain=float(eligibility["minimum_oof_common_interior_gain_db"]),
        maximum_fold_degradation=float(
            eligibility["maximum_worst_fold_degradation_db"]
        ),
    )
    selected_name = _select_candidate(
        candidate_results,
        candidates,
        tie_tolerance_db=float(rule["complexity_tie_break_db"]),
    )
    selected_candidate = next(
        candidate for candidate in candidates if candidate.name == selected_name
    )
    selected_delays = selected_candidate.delays
    base_train_prediction = frozen_model.predict_segments(train_input, nperseg)
    base_validation_prediction = frozen_model.predict_segments(
        validation_input,
        nperseg,
    )
    correction: WidelyLinearResidualCorrection | None = None
    correction_diagnostics: dict[str, Any] | None = None
    if selected_delays:
        correction, diagnostics = fit_widely_linear_residual_correction(
            train_input,
            train_output - base_train_prediction,
            delays=selected_delays,
            ridge=float(config["correction_fit"]["ridge"]),
            segment_length=nperseg,
            coefficient_dtype=np.complex128,
        )
        correction_diagnostics = dataclasses.asdict(diagnostics)
        selected_train_prediction = base_train_prediction + correction.predict_segments(
            train_input,
            nperseg,
        )
        selected_validation_prediction = base_validation_prediction + correction.predict_segments(
            validation_input,
            nperseg,
        )
        full_stream_checks = _streaming_checks(
            correction,
            train_input,
            segment_length=nperseg,
        )
    else:
        selected_train_prediction = base_train_prediction.copy()
        selected_validation_prediction = base_validation_prediction.copy()
        full_stream_checks = {
            "streaming_chunk_equivalence_passed": True,
            "reset_at_frame_equivalence_passed": True,
            "maximum_streaming_error": 0.0,
            "maximum_reset_error": 0.0,
        }
    candidate_results[selected_name]["selected_full_train_streaming_checks"] = (
        full_stream_checks
    )
    train_metrics = _metrics(
        selected_train_prediction,
        train_output,
        nperseg=nperseg,
        common_mask=train_common_mask,
    )
    validation_metrics = _metrics(
        selected_validation_prediction,
        validation_output,
        nperseg=nperseg,
        common_mask=validation_common_mask,
    )
    base_oof_metrics = _metrics(
        oof_predictions[candidates[0].name],
        train_output,
        nperseg=nperseg,
        common_mask=train_common_mask,
    )
    historical_oof = residual_manifest["reset_boundary_diagnostics"]["train_oof"]
    historical_validation = residual_manifest["reset_boundary_diagnostics"][
        "validation"
    ]
    if abs(
        base_oof_metrics["full_record_nmse_db"]
        - float(historical_oof["full_record"]["pooled_complex_nmse_db"])
    ) > 1e-9:
        raise ValueError("base OOF reproduction disagrees with frozen residual evidence")
    if abs(
        base_oof_metrics["common_interior_nmse_db"]
        - float(historical_oof["common_boundary_interior"]["pooled_complex_nmse_db"])
    ) > 1e-9:
        raise ValueError("base OOF common reproduction disagrees with evidence")
    base_validation_metrics = _metrics(
        base_validation_prediction,
        validation_output,
        nperseg=nperseg,
        common_mask=validation_common_mask,
    )
    if abs(
        base_validation_metrics["full_record_nmse_db"]
        - float(historical_validation["full_record"]["pooled_complex_nmse_db"])
    ) > 1e-9:
        raise ValueError("base validation reproduction disagrees with evidence")
    if abs(
        base_validation_metrics["common_interior_nmse_db"]
        - float(
            historical_validation["common_boundary_interior"][
                "pooled_complex_nmse_db"
            ]
        )
    ) > 1e-9:
        raise ValueError("base validation common reproduction disagrees with evidence")
    train_report_path = evidence_bundle["paths"]["train_oof_residual_report"]
    validation_report_path = evidence_bundle["paths"]["validation_residual_report"]
    historical_train_report = _load_json_object(
        train_report_path,
        name="historical train residual report",
    )
    _load_json_object(
        validation_report_path,
        name="historical validation residual report",
    )
    residual_spec, frozen_reference = _make_residual_spec(historical_train_report)
    train_analysis = analyze_pa_residuals(
        train_input,
        train_output,
        oof_predictions[selected_name],
        segment_id=explicit_frame_ids(train_input.size, nperseg),
        valid_mask=train_common_mask,
        split_role="train_oof",
        spec=residual_spec,
        frozen_reference=frozen_reference,
    )
    validation_analysis = analyze_pa_residuals(
        validation_input,
        validation_output,
        selected_validation_prediction,
        segment_id=explicit_frame_ids(validation_input.size, nperseg),
        valid_mask=validation_common_mask,
        split_role="validation_reused_descriptive",
        spec=residual_spec,
        frozen_reference=frozen_reference,
    )
    train_analysis["runner_scope"] = {
        "status": "post_discovery_internal_resampling_only",
        "candidate_selection": "train coefficient-OOF only",
        "validation_used_for_selection": False,
        "test_accessed": False,
    }
    validation_analysis["runner_scope"] = {
        "status": "validation_reused_descriptive",
        "already_viewed_during_family_discovery": True,
        "independent_confirmation": False,
        "test_accessed": False,
    }
    runtime = time.perf_counter() - started
    output = _resolve_project_relative(
        config["output_dir"],
        project_root=PROJECT_ROOT,
        name="output_dir",
    )
    return _write_bundle(
        output,
        config_path=source_config,
        config_sha256=config_sha256,
        source_hashes=source_hashes,
        evidence_status=evidence_bundle["source_status"],
        dataset_hashes=dataset_hashes,
        selected_name=selected_name,
        selected_candidate=selected_candidate,
        candidate_results=candidate_results,
        train_metrics={
            "selected": train_metrics,
            "base_oof": base_oof_metrics,
            "oof_fit_seconds": oof_fit_seconds,
        },
        validation_metrics={
            "selected": validation_metrics,
            "base": base_validation_metrics,
        },
        train_base_prediction=oof_predictions[candidates[0].name],
        train_selected_prediction=oof_predictions[selected_name],
        validation_base_prediction=base_validation_prediction,
        validation_selected_prediction=selected_validation_prediction,
        train_common_mask=train_common_mask,
        validation_common_mask=validation_common_mask,
        correction=correction,
        correction_diagnostics=correction_diagnostics,
        train_analysis=train_analysis,
        validation_analysis=validation_analysis,
        runtime_seconds=runtime,
        test_split_accessed=False,
        input_reverify=input_reverify,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the APA widely-linear residual audit without test access."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    run_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
