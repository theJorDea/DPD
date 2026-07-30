"""Run the preregistered sparse spline-memory APA PA benchmark.

Only train and validation are available to this runner.  The recipe is chosen
by leave-one-explicit-frame-out train OOF, the full-train coefficients are
hashed, and validation is loaded only after that freeze.  There is no test
loader in this module.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

from baseline.residual_analysis import analyze_pa_residuals
from baseline.sparse_spline_memory_pa import fit_sparse_spline_memory_pa_segments
from baseline.train_spline import load_split_pair
from experiments.select_pa_sparse_spline_memory import (
    PROJECT_ROOT,
    SparseRecipe,
    common_mask,
    file_sha256,
    frame_segments,
    load_config,
    project_path,
    run_staged_search,
    validate_search_budget,
    verify_preregistered_inputs,
)
from experiments.sparse_pa_benchmark_support import (
    acquire_lock,
    array_sha256,
    frame_ids,
    load_frozen_evidence,
    metric_summary,
    partition_lengths,
    publish_bundle,
    reference_reproduction,
    reverify_inputs,
    staged_ledger,
    streaming_and_reset_checks,
    support_summary,
)


Progress = Callable[[str], None]


def run_from_config(
    config_path: str | Path,
    *,
    progress: Progress = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Execute the sealed train-OOF → frozen validation workflow."""

    started = time.perf_counter()
    source_config = Path(config_path).resolve()
    if not source_config.is_file() or source_config.is_symlink():
        raise FileNotFoundError("sparse PA config must be a regular file")
    initial_config_sha256 = file_sha256(source_config)
    config = load_config(source_config)
    if file_sha256(source_config) != initial_config_sha256:
        raise RuntimeError("sparse PA config changed while being parsed")
    verified = verify_preregistered_inputs(config, source_config)
    if verified["config_sha256"] != initial_config_sha256:
        raise RuntimeError("sparse PA config snapshot disagrees with verification")
    verified["config"] = config
    validate_search_budget(config)
    output = project_path(config["output_dir"], name="output directory")
    lock, lock_payload = acquire_lock(output)
    try:
        progress("[integrity] preregistered hashes verified before waveform load")
        evidence = load_frozen_evidence(verified)
        dataset = verified["dataset"]
        train_input, train_output = load_split_pair(dataset, "train")
        contract = config["dataset_contract"]
        frame_lengths = tuple(int(value) for value in contract["frame_lengths"])
        if train_input.size != int(contract["train_sample_count"]):
            raise ValueError("train sample count disagrees with sparse PA contract")
        input_segments = frame_segments(train_input, frame_lengths)
        output_segments = frame_segments(train_output, frame_lengths)
        reverify_inputs(verified, source_config, scope="post-train-load")
        reference = reference_reproduction(
            evidence["gmp_train_oof_prediction"],
            train_output,
            frame_lengths=frame_lengths,
            common_warmup=int(contract["common_warmup_samples_per_frame"]),
            config=config,
        )
        progress("[reference] frozen GMP OOF metrics reproduced exactly")

        search = run_staged_search(
            config,
            input_segments,
            output_segments,
            evidence["gmp_train_oof_prediction"],
            progress=progress,
        )
        final_recipe: SparseRecipe = search["final_recipe"]
        final_trial = search["final_trial"]
        frozen_recipe_sha256 = final_recipe.sha256

        model, fit_diagnostics = fit_sparse_spline_memory_pa_segments(
            input_segments,
            output_segments,
            branches=final_recipe.branch_objects,
            knot_count=final_recipe.knot_count,
            knot_strategy="uniform_amplitude",
            ridge=final_recipe.ridge,
        )
        gates = config["gates"]
        if fit_diagnostics.data_design_rank != fit_diagnostics.feature_count:
            raise RuntimeError("selected full-train sparse PA design is rank deficient")
        if fit_diagnostics.minimum_nonzero_feature_samples <= 0:
            raise RuntimeError("selected full-train sparse PA has an unseen feature")
        if fit_diagnostics.maximum_absolute_coefficient > float(
            gates["maximum_absolute_coefficient"]
        ):
            raise RuntimeError("selected full-train sparse PA coefficient is unbounded")
        if (
            not np.isfinite(fit_diagnostics.augmented_design_condition_number)
            or fit_diagnostics.augmented_design_condition_number
            > float(gates["maximum_augmented_condition_number"])
        ):
            raise RuntimeError("selected full-train sparse PA fit is ill-conditioned")
        if model.operation_count().to_dict() != final_trial["operation_count"]:
            raise RuntimeError("selected sparse PA operation schedule changed after refit")
        if model.operation_count().real_multiplications >= int(
            gates["real_multiplications_strictly_below"]
        ):
            raise RuntimeError("selected full-train sparse PA exceeds MUL budget")

        train_prediction = np.concatenate(
            [model.predict(segment) for segment in input_segments]
        )
        train_metrics = metric_summary(
            train_prediction,
            train_output,
            frame_lengths=frame_lengths,
            common_warmup=int(contract["common_warmup_samples_per_frame"]),
        )
        train_support = support_summary(model, input_segments)
        stream_checks = streaming_and_reset_checks(model, input_segments)
        if not all(
            stream_checks[name]
            for name in (
                "streaming_chunk_equivalence_passed",
                "segmented_reset_equivalence_passed",
            )
        ):
            raise RuntimeError("selected sparse PA streaming/reset check failed")
        frozen_parameter_hashes = {
            "knots": array_sha256(model.knots),
            "coefficients": array_sha256(model.coefficients),
        }
        progress(
            f"[freeze] {final_recipe.name}; train NMSE="
            f"{train_metrics['full_record_nmse_db']:.6f} dB"
        )

        # Nothing below this point can alter recipe or full-train parameters.
        validation_input, validation_output = load_split_pair(dataset, "val")
        if validation_input.size != int(contract["validation_sample_count"]):
            raise ValueError("validation sample count disagrees with sparse PA contract")
        if final_recipe.sha256 != frozen_recipe_sha256:
            raise RuntimeError("sparse PA recipe changed while loading validation")
        if frozen_parameter_hashes != {
            "knots": array_sha256(model.knots),
            "coefficients": array_sha256(model.coefficients),
        }:
            raise RuntimeError("sparse PA coefficients changed after validation load")
        validation_lengths = partition_lengths(
            validation_input.size, int(contract["nperseg"])
        )
        validation_segments = frame_segments(validation_input, validation_lengths)
        validation_prediction = np.concatenate(
            [model.predict(segment) for segment in validation_segments]
        )
        validation_metrics = metric_summary(
            validation_prediction,
            validation_output,
            frame_lengths=validation_lengths,
            common_warmup=int(contract["common_warmup_samples_per_frame"]),
        )
        validation_support = support_summary(model, validation_segments)
        progress(
            "[validation] loaded only after freeze; NMSE="
            f"{validation_metrics['full_record_nmse_db']:.6f} dB"
        )

        train_ids = frame_ids(frame_lengths)
        validation_ids = frame_ids(validation_lengths)
        train_common = common_mask(
            frame_lengths, int(contract["common_warmup_samples_per_frame"])
        )
        validation_common = common_mask(
            validation_lengths, int(contract["common_warmup_samples_per_frame"])
        )
        train_residual = analyze_pa_residuals(
            train_input,
            train_output,
            final_trial["oof_prediction"],
            segment_id=train_ids,
            valid_mask=train_common,
            split_role="train_oof",
            spec=evidence["residual_spec"],
            frozen_reference=evidence["residual_frozen_reference"],
        )
        validation_residual = analyze_pa_residuals(
            validation_input,
            validation_output,
            validation_prediction,
            segment_id=validation_ids,
            valid_mask=validation_common,
            split_role="validation_reused_descriptive",
            spec=evidence["residual_spec"],
            frozen_reference=evidence["residual_frozen_reference"],
        )
        input_reverification = reverify_inputs(
            verified, source_config, scope="pre-publication"
        )
        source_paths = {
            "experiments/run_pa_sparse_spline_memory.py": Path(__file__).resolve(),
            "experiments/select_pa_sparse_spline_memory.py": PROJECT_ROOT
            / "experiments/select_pa_sparse_spline_memory.py",
            "experiments/sparse_pa_benchmark_support.py": PROJECT_ROOT
            / "experiments/sparse_pa_benchmark_support.py",
            "baseline/sparse_spline_memory_pa.py": PROJECT_ROOT
            / "baseline/sparse_spline_memory_pa.py",
            "baseline/spline_memory_dpd.py": PROJECT_ROOT / "baseline/spline_memory_dpd.py",
            "baseline/residual_analysis.py": PROJECT_ROOT / "baseline/residual_analysis.py",
            "baseline/metrics.py": PROJECT_ROOT / "baseline/metrics.py",
        }
        source_hashes = {
            name: file_sha256(path) for name, path in source_paths.items()
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
            "validation_loaded_after_parameter_hashes": frozen_parameter_hashes,
            "test_split_accessed": False,
        }
        final_trial_metric_names = [
            "full_record_nmse_db",
            "common_interior_nmse_db",
            "reference_gmp_full_record_nmse_db",
            "reference_gmp_common_interior_nmse_db",
            "gain_over_gmp_full_db",
            "gain_over_gmp_common_db",
            "minimum_fold_gain_over_gmp_full_db",
            "minimum_fold_gain_over_gmp_common_db",
            "loss_vs_mp_full_db",
            "loss_vs_mp_common_db",
        ]
        final_trial_metric_names.extend(
            name
            for name in (
                "gain_over_incremental_control_full_db",
                "gain_over_incremental_control_common_db",
                "minimum_fold_gain_over_incremental_control_full_db",
                "minimum_fold_gain_over_incremental_control_common_db",
                "incremental_control_fold_gains",
            )
            if name in final_trial
        )
        final_trial_metrics = {
            key: final_trial[key] for key in final_trial_metric_names
        }
        manifest_payload = {
            "schema_version": 1,
            "task": "forward_pa_non_factorized_sparse_spline_memory_selection",
            "status": "post_discovery_internal_resampling_and_reused_validation",
            "config": str(source_config.relative_to(PROJECT_ROOT)),
            "config_sha256": verified["config_sha256"],
            "dataset": config["dataset"],
            "dataset_files_sha256": verified["dataset_hashes"],
            "accessed_splits": ["train", "validation"],
            "test_split_accessed": False,
            "test_file_hashes_recorded": False,
            "selection_samples": "train leave-one-explicit-frame-out only",
            "validation_role": (
                "loaded after recipe and coefficient freeze; reused descriptive "
                "evidence, not independent confirmation"
            ),
            "final_recipe": final_recipe.canonical_dict
            | {"name": final_recipe.name, "sha256": final_recipe.sha256},
            "selected_operation_count": model.operation_count().to_dict(),
            "selected_fit_diagnostics": dataclasses.asdict(fit_diagnostics),
            "reference_gmp_reproduction": reference,
            "train_oof_metrics": final_trial_metrics,
            "train_full_refit_metrics": train_metrics,
            "validation_reused_metrics": validation_metrics,
            "train_input_support": train_support,
            "validation_input_support": validation_support,
            "streaming_and_reset_checks": stream_checks,
            "decision": search["decision"],
            "gate_a_to_b_opened": False,
            "dpd_optimization_status": "paused",
            "source_sha256": source_hashes,
            "preimplementation_source_hashes": verified[
                "preimplementation_source_hashes"
            ],
            "new_source_files_present_after_implementation": verified[
                "new_source_files_present_after_implementation"
            ],
            "input_integrity": {
                "all_hashes_verified_before_waveform_load": True,
                "all_inputs_reverified_before_publication": all(
                    input_reverification.values()
                ),
                "test_never_opened_or_hashed": True,
            },
        }
        manifest = publish_bundle(
            output,
            lock=lock,
            lock_payload=lock_payload,
            model=model,
            manifest_payload=manifest_payload,
            staged_ledger_payload=staged_ledger(search),
            predictions={
                "train_oof_prediction": final_trial["oof_prediction"],
                "reference_gmp_train_oof_prediction": evidence[
                    "gmp_train_oof_prediction"
                ],
                "train_full_prediction": train_prediction,
                "validation_reused_prediction": validation_prediction,
                "train_segment_id": train_ids,
                "validation_segment_id": validation_ids,
                "train_common_mask": train_common,
                "validation_common_mask": validation_common,
            },
            residual_reports={
                "train_oof": train_residual,
                "validation_reused": validation_residual,
            },
            execution=execution,
        )
        progress(f"[publish] immutable result bundle: {output}")
        return manifest
    finally:
        if lock.exists() and not lock.is_symlink():
            lock.unlink()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run preregistered train-OOF sparse spline-memory PA selection and "
            "descriptive validation without test access."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    manifest = run_from_config(arguments.config)
    print(
        "Sparse spline-memory PA selection complete:",
        manifest["final_recipe"]["name"],
        f"OOF NMSE={manifest['train_oof_metrics']['full_record_nmse_db']:.6f} dB,",
        f"validation NMSE={manifest['validation_reused_metrics']['full_record_nmse_db']:.6f} dB,",
        f"decision={manifest['decision']['classification']}",
    )


if __name__ == "__main__":
    main()
