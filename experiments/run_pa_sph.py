"""Run and atomically publish the preregistered APA SPH PA experiment.

Waveform access order is fixed: verify hashes, load train, freeze the S0--S3
recipe on train OOF, and only then load the reused descriptive validation
split.  Test files are never opened, hashed, or named as artifacts.
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

from baseline.residual_analysis import (
    ResidualAnalysisSpec,
    analyze_pa_residuals,
)
from baseline.spline_hammerstein_pa import (
    SplineHammersteinPA,
    fit_spline_hammerstein_pa,
    sph_coordinate_values,
)
from baseline.train_spline import load_split_pair, write_json
from experiments.select_pa_sph import (
    PROJECT_ROOT,
    SPHOOFProtocol,
    SPHRecipe,
    _array_sha256,
    _common_mask,
    _file_sha256,
    _frame_ids,
    _metric_summary,
    _project_path,
    _streaming_checks,
    load_verified_gmp_oof_prediction,
    run_staged_oof_search,
    verify_sph_preregistered_inputs,
)


REFERENCE_REPRODUCTION_TOLERANCE_DB = 1e-9


def _reverify_inputs(verified: dict[str, Any], *, scope: str) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    config_path = verified["config_path"]
    checks["config"] = _file_sha256(config_path) == verified["config_sha256"]
    for name, expected in verified["dataset_hashes"].items():
        checks[f"dataset/{name}"] = (
            _file_sha256(verified["dataset"] / name) == expected
        )
    for name, path in verified["evidence_paths"].items():
        entry = verified["config"]["evidence"]
        if name.startswith("negative_linear_ablation_"):
            index = int(name.rsplit("_", 1)[1])
            expected = entry["negative_linear_ablations"][index]["sha256"]
        else:
            expected = entry[name]["sha256"]
        checks[f"evidence/{name}"] = _file_sha256(path) == expected
    for name, status in verified["preimplementation_source_status"].items():
        checks[f"preimplementation_source/{name}"] = (
            _file_sha256(_project_path(name, name="source reverify path"))
            == status["expected_sha256"]
        )
    checks["reference_predictions"] = (
        _file_sha256(verified["reference_predictions_path"])
        == verified["reference_predictions_sha256"]
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"{scope} input reverification failed: {failed}")
    return checks


def _reference_metrics_or_raise(
    prediction: np.ndarray,
    train_output: np.ndarray,
    protocol: SPHOOFProtocol,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics = _metric_summary(prediction, train_output, protocol)
    expected = config["reference_models"]["matched_gmp_oof"]
    comparisons = {
        "full_record_nmse_db": (
            float(metrics["full_record_nmse_db"]),
            float(expected["full_record_nmse_db"]),
        ),
        "common_interior_nmse_db": (
            float(metrics["common_interior_nmse_db"]),
            float(expected["common_interior_nmse_db"]),
        ),
    }
    deltas = {
        name: actual - frozen for name, (actual, frozen) in comparisons.items()
    }
    if any(
        abs(delta) > REFERENCE_REPRODUCTION_TOLERANCE_DB
        for delta in deltas.values()
    ):
        raise ValueError(f"frozen GMP OOF reference reproduction failed: {deltas}")
    return {
        "metrics": metrics,
        "frozen_expected": expected,
        "delta_db": deltas,
        "tolerance_db": REFERENCE_REPRODUCTION_TOLERANCE_DB,
        "passed": True,
    }


def _reset_and_streaming_checks(
    model: SplineHammersteinPA,
    signal: np.ndarray,
    *,
    segment_length: int,
) -> dict[str, Any]:
    continuous = _streaming_checks(model, signal)
    segmented = model.predict_segments(signal, segment_length)
    manual = np.concatenate(
        [
            model.predict(signal[start : start + segment_length])
            for start in range(0, signal.size, segment_length)
        ]
    )
    reset_error = float(np.max(np.abs(segmented - manual), initial=0.0))
    return {
        **continuous,
        "segmented_reset_equivalence_passed": bool(
            np.array_equal(segmented, manual)
        ),
        "maximum_segmented_reset_error": reset_error,
    }


def _support_summary(
    model: SplineHammersteinPA,
    signal: np.ndarray,
) -> dict[str, Any]:
    coordinate = sph_coordinate_values(signal, model.coordinate)
    above = coordinate > model.knots[-1]
    return {
        "coordinate": model.coordinate,
        "fit_maximum": float(model.knots[-1]),
        "observed_maximum": float(np.max(coordinate)),
        "count_above_fit_maximum": int(np.count_nonzero(above)),
        "fraction_above_fit_maximum": float(np.mean(above)),
    }


def _serializable_trial(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in row.items()
        if key not in {"recipe", "oof_prediction"}
    }
    recipe = row.get("recipe")
    if isinstance(recipe, SPHRecipe):
        result["recipe"] = recipe.to_dict()
    return result


def _staged_ledger(search: dict[str, Any]) -> dict[str, Any]:
    unique_trials = {
        identity: _serializable_trial(row)
        for identity, row in search["cache"].items()
    }
    stages = {
        stage: [row["recipe_sha256"] for row in rows]
        for stage, rows in search["stage_results"].items()
    }
    return {
        "schema_version": 1,
        "task": "forward_pa_model_spline_hammerstein_selection",
        "selection_samples": "train leave-one-explicit-frame-out only",
        "validation_loaded": False,
        "test_split_accessed": False,
        "budget_summary": search["budget_summary"],
        "stages": stages,
        "unique_trials": unique_trials,
        "selections": search["selections"],
        "decision": search["decision"],
        "counts": {
            key: search[key]
            for key in (
                "stage_recipe_associations",
                "unique_recipe_evaluations",
                "cache_hits",
                "completed_unique_oof_fit_calls",
                "evaluated_recipe_oof_fit_call_upper_bound",
                "stage_association_oof_fit_call_upper_bound_without_cache",
                "failed_unique_recipe_count",
            )
        },
    }


def _load_residual_contract(
    verified: dict[str, Any],
) -> tuple[ResidualAnalysisSpec, dict[str, object]]:
    path = verified["evidence_paths"]["train_oof_residual_report"]
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("frozen residual report must be an object")
    raw_spec = report.get("spec")
    frozen_reference = report.get("frozen_reference")
    if not isinstance(raw_spec, dict) or not isinstance(frozen_reference, dict):
        raise ValueError("frozen residual report lacks spec/reference")
    return ResidualAnalysisSpec(**raw_spec), frozen_reference


def _acquire_lock(output: Path) -> tuple[Path, bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable SPH output already exists: {output}")
    lock = output.parent / f".{output.name}.lock"
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f"SPH output lock already exists: {lock}")
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
        raise RuntimeError("SPH publication lock was replaced or modified")


def _publish_bundle(
    output: Path,
    *,
    lock: Path,
    lock_payload: bytes,
    verified: dict[str, Any],
    source_hashes: dict[str, str],
    staged_ledger: dict[str, Any],
    final_recipe: SPHRecipe,
    decision: dict[str, Any],
    model: SplineHammersteinPA,
    fit_diagnostics: dict[str, Any],
    reference_reproduction: dict[str, Any],
    oof_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    train_support: dict[str, Any],
    validation_support: dict[str, Any],
    stream_checks: dict[str, Any],
    train_oof_prediction: np.ndarray,
    reference_gmp_oof_prediction: np.ndarray,
    train_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    train_segment_id: np.ndarray,
    validation_segment_id: np.ndarray,
    train_common_mask: np.ndarray,
    validation_common_mask: np.ndarray,
    train_residual_analysis: dict[str, Any],
    validation_residual_analysis: dict[str, Any],
    execution: dict[str, Any],
    input_reverification: dict[str, bool],
) -> dict[str, Any]:
    _verify_lock(lock, lock_payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError("SPH output appeared before publication")
    temporary = output.parent / f".{output.name}.tmp-{secrets.token_hex(12)}"
    temporary.mkdir()
    try:
        model_path = temporary / "selected_sph_pa.npz"
        model.save(model_path)
        predictions_path = temporary / "predictions.npz"
        np.savez_compressed(
            predictions_path,
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray("phase_equivariant_spline_hammerstein_pa"),
            train_oof_prediction=train_oof_prediction,
            reference_gmp_train_oof_prediction=reference_gmp_oof_prediction,
            train_full_prediction=train_prediction,
            validation_reused_prediction=validation_prediction,
            train_segment_id=train_segment_id,
            validation_segment_id=validation_segment_id,
            train_common_mask=train_common_mask,
            validation_common_mask=validation_common_mask,
        )
        write_json(temporary / "staged_trials.json", staged_ledger)
        write_json(
            temporary / "train_oof_residual_analysis.json",
            train_residual_analysis,
        )
        write_json(
            temporary / "validation_reused_residual_analysis.json",
            validation_residual_analysis,
        )
        write_json(temporary / "execution_record.json", execution)
        artifact_names = (
            "selected_sph_pa.npz",
            "predictions.npz",
            "staged_trials.json",
            "train_oof_residual_analysis.json",
            "validation_reused_residual_analysis.json",
            "execution_record.json",
        )
        artifacts = {
            name: {"path": name, "sha256": _file_sha256(temporary / name)}
            for name in artifact_names
        }
        manifest = {
            "schema_version": 1,
            "task": "forward_pa_model_spline_hammerstein_selection",
            "status": "post_discovery_internal_resampling_and_reused_validation",
            "config": str(verified["config_path"].relative_to(PROJECT_ROOT)),
            "config_sha256": verified["config_sha256"],
            "dataset": verified["config"]["dataset"],
            "dataset_files_sha256": verified["dataset_hashes"],
            "accessed_splits": ["train", "validation"],
            "test_split_accessed": False,
            "test_file_hashes_recorded": False,
            "selection_samples": "train leave-one-explicit-frame-out only",
            "validation_role": (
                "loaded after recipe freeze; already-viewed descriptive evidence, "
                "not independent confirmation and not used for selection"
            ),
            "final_recipe": final_recipe.to_dict(),
            "final_recipe_sha256": final_recipe.canonical_sha256,
            "selected_operation_count": model.operation_count().to_dict(),
            "selected_fit_diagnostics": fit_diagnostics,
            "reference_gmp_reproduction": reference_reproduction,
            "train_oof_metrics": oof_metrics,
            "train_full_refit_metrics": train_metrics,
            "validation_reused_metrics": validation_metrics,
            "train_input_support": train_support,
            "validation_input_support": validation_support,
            "streaming_and_reset_checks": stream_checks,
            "decision": decision,
            "gate_a_to_b_opened": False,
            "dpd_optimization_status": "paused",
            "independent_validation_required": (
                "new capture or verified APA_200MHz_b protocol plus a second "
                "evaluator ranking before any Gate A-to-B claim"
            ),
            "artifacts": artifacts,
            "source_sha256": source_hashes,
            "preimplementation_source_status": verified[
                "preimplementation_source_status"
            ],
            "input_integrity": {
                "all_hashes_verified_before_waveform_load": True,
                "all_inputs_reverified_before_publication": all(
                    input_reverification.values()
                ),
                "test_never_opened_or_hashed": True,
            },
            "publication": {
                "immutable_bundle": True,
                "atomic_directory_rename": True,
                "completion_manifest_written_last_inside_temporary_bundle": True,
            },
        }
        write_json(temporary / "selection_manifest.json", manifest)
        _verify_lock(lock, lock_payload)
        if output.exists() or output.is_symlink():
            raise FileExistsError("SPH output appeared during publication")
        os.replace(temporary, output)
        temporary = None  # type: ignore[assignment]
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    verified = verify_sph_preregistered_inputs(config_path)
    config = verified["config"]
    output = _project_path(config["output_dir"], name="output_dir")
    lock, lock_payload = _acquire_lock(output)
    try:
        reference_gmp_oof = load_verified_gmp_oof_prediction(verified)
        train_input, train_output = load_split_pair(verified["dataset"], "train")
        contract = config["dataset_contract"]
        if train_input.size != int(contract["train_sample_count"]):
            raise ValueError("train sample count disagrees with SPH contract")
        _reverify_inputs(verified, scope="post-train-load")
        protocol = SPHOOFProtocol(
            segment_length=int(contract["nperseg"]),
            common_warmup_samples=int(
                contract["common_warmup_samples_per_frame"]
            ),
            common_cooldown_samples=int(
                contract["common_future_cooldown_samples_per_frame"]
            ),
            maximum_alternations=int(config["fit"]["maximum_full_alternations"]),
            minimum_alternations=int(config["fit"]["minimum_full_alternations"]),
            convergence_tolerance=float(
                config["fit"]["relative_objective_convergence_tolerance"]
            ),
            objective_increase_tolerance=float(
                config["fit"]["relative_objective_increase_tolerance"]
            ),
            real_multiplication_limit_exclusive=int(
                config["operation_count_convention"][
                    "real_multiplication_limit_exclusive"
                ]
            ),
        )
        reference_reproduction = _reference_metrics_or_raise(
            reference_gmp_oof,
            train_output,
            protocol,
            config,
        )
        search = run_staged_oof_search(
            config,
            train_input,
            train_output,
            protocol=protocol,
            reference_gmp_oof_prediction=reference_gmp_oof,
            progress=lambda message: print(message, flush=True),
        )
        final_recipe = search["final_recipe"]
        final_trial = search["final_trial"]
        frozen_recipe_sha256 = final_recipe.canonical_sha256
        staged_ledger = _staged_ledger(search)
        model, fit_diagnostics = fit_spline_hammerstein_pa(
            train_input,
            train_output,
            knot_count=final_recipe.knot_count,
            knot_variant=final_recipe.variant,  # type: ignore[arg-type]
            fir_length=final_recipe.fir_length,
            segment_length=protocol.segment_length,
            control_ridge=final_recipe.control_ridge,
            smoothness=final_recipe.smoothness,
            fir_ridge=final_recipe.fir_ridge,
            maximum_alternations=protocol.maximum_alternations,
            minimum_alternations=protocol.minimum_alternations,
            convergence_tolerance=protocol.convergence_tolerance,
            objective_increase_tolerance=protocol.objective_increase_tolerance,
            coefficient_dtype=np.complex128,
        )
        if (
            not fit_diagnostics.all_updates_monotonic
            or not fit_diagnostics.all_data_designs_full_column_rank
        ):
            raise RuntimeError("selected full-train SPH fit failed hard validity")
        if model.operation_count().to_dict() != final_recipe.operation_count.to_dict():
            raise RuntimeError("selected model operation count changed after fit")
        train_prediction = model.predict_segments(
            train_input,
            protocol.segment_length,
        )
        train_metrics = _metric_summary(train_prediction, train_output, protocol)
        train_support = _support_summary(model, train_input)
        stream_checks = _reset_and_streaming_checks(
            model,
            train_input,
            segment_length=protocol.segment_length,
        )
        if not all(
            stream_checks[key]
            for key in (
                "streaming_chunk_equivalence_passed",
                "reset_at_frame_equivalence_passed",
                "segmented_reset_equivalence_passed",
            )
        ):
            raise RuntimeError("selected full-train SPH streaming check failed")
        frozen_model_hashes = {
            "knots": _array_sha256(model.knots),
            "control_points": _array_sha256(model.control_points),
            "fir_tail": _array_sha256(model.fir_tail),
        }

        # Neither the recipe nor full-train coefficients can change below.
        validation_input, validation_output = load_split_pair(
            verified["dataset"],
            "val",
        )
        if final_recipe.canonical_sha256 != frozen_recipe_sha256:
            raise RuntimeError("SPH recipe changed while loading validation")
        if frozen_model_hashes != {
            "knots": _array_sha256(model.knots),
            "control_points": _array_sha256(model.control_points),
            "fir_tail": _array_sha256(model.fir_tail),
        }:
            raise RuntimeError("full-train SPH model changed after validation load")
        if validation_input.size != int(contract["validation_sample_count"]):
            raise ValueError("validation sample count disagrees with contract")
        validation_prediction = model.predict_segments(
            validation_input,
            protocol.segment_length,
        )
        validation_metrics = _metric_summary(
            validation_prediction,
            validation_output,
            protocol,
        )
        validation_support = _support_summary(model, validation_input)

        train_segment_id = _frame_ids(train_input.size, protocol.segment_length)
        validation_segment_id = _frame_ids(
            validation_input.size,
            protocol.segment_length,
        )
        train_common_mask = _common_mask(train_input.size, protocol)
        validation_common_mask = _common_mask(validation_input.size, protocol)
        residual_spec, frozen_reference = _load_residual_contract(verified)
        train_residual_analysis = analyze_pa_residuals(
            train_input,
            train_output,
            final_trial["oof_prediction"],
            segment_id=train_segment_id,
            valid_mask=train_common_mask,
            split_role="train_oof",
            spec=residual_spec,
            frozen_reference=frozen_reference,
        )
        validation_residual_analysis = analyze_pa_residuals(
            validation_input,
            validation_output,
            validation_prediction,
            segment_id=validation_segment_id,
            valid_mask=validation_common_mask,
            split_role="validation_reused_descriptive",
            spec=residual_spec,
            frozen_reference=frozen_reference,
        )
        input_reverification = _reverify_inputs(
            verified,
            scope="pre-publication",
        )
        source_paths = {
            "experiments/run_pa_sph.py": Path(__file__).resolve(),
            "experiments/select_pa_sph.py": PROJECT_ROOT
            / "experiments/select_pa_sph.py",
            "baseline/spline_hammerstein_pa.py": PROJECT_ROOT
            / "baseline/spline_hammerstein_pa.py",
            "baseline/complexity.py": PROJECT_ROOT / "baseline/complexity.py",
            "baseline/residual_analysis.py": PROJECT_ROOT
            / "baseline/residual_analysis.py",
            "baseline/metrics.py": PROJECT_ROOT / "baseline/metrics.py",
        }
        source_hashes = {
            name: _file_sha256(path) for name, path in source_paths.items()
        }
        execution = {
            "schema_version": 1,
            "command": " ".join(sys.argv),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "runtime_seconds_before_publication": time.perf_counter() - started,
            "selection_recipe_frozen_before_validation_load": True,
            "full_train_model_frozen_before_validation_load": True,
            "validation_loaded_after_recipe_sha256": frozen_recipe_sha256,
            "validation_loaded_after_model_parameter_hashes": frozen_model_hashes,
            "test_split_accessed": False,
        }
        manifest = _publish_bundle(
            output,
            lock=lock,
            lock_payload=lock_payload,
            verified=verified,
            source_hashes=source_hashes,
            staged_ledger=staged_ledger,
            final_recipe=final_recipe,
            decision=search["decision"],
            model=model,
            fit_diagnostics=dataclasses.asdict(fit_diagnostics),
            reference_reproduction=reference_reproduction,
            oof_metrics=final_trial["metrics"],
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            train_support=train_support,
            validation_support=validation_support,
            stream_checks=stream_checks,
            train_oof_prediction=final_trial["oof_prediction"],
            reference_gmp_oof_prediction=reference_gmp_oof,
            train_prediction=train_prediction,
            validation_prediction=validation_prediction,
            train_segment_id=train_segment_id,
            validation_segment_id=validation_segment_id,
            train_common_mask=train_common_mask,
            validation_common_mask=validation_common_mask,
            train_residual_analysis=train_residual_analysis,
            validation_residual_analysis=validation_residual_analysis,
            execution=execution,
            input_reverification=input_reverification,
        )
        return manifest
    finally:
        if lock.exists() and not lock.is_symlink():
            lock.unlink()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run preregistered train-OOF SPH PA selection, then descriptive "
            "validation, without any test access."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    manifest = run_from_config(arguments.config)
    print(
        "SPH PA selection complete:",
        manifest["final_recipe"]["name"],
        f"OOF NMSE={manifest['train_oof_metrics']['full_record_nmse_db']:.6f} dB,",
        f"validation NMSE={manifest['validation_reused_metrics']['full_record_nmse_db']:.6f} dB,",
        f"decision={manifest['decision']['classification']}",
    )


if __name__ == "__main__":
    main()
