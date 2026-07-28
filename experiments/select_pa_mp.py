"""Select a budget-constrained complex memory-polynomial PA model.

This command opens only ``train_*`` and ``val_*`` files.  It performs a
two-stage deterministic selection:

1. choose an order set and contiguous causal memory depth at one preregistered
   ridge value;
2. refine ridge for that validation-selected architecture.

The official test split is intentionally absent from this module.  Final test
evaluation is a separate command operating on the frozen model and manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from baseline.complexity import (
    OperationCount,
    memory_polynomial_inference_cost,
)
from baseline.pa_benchmark import (
    PAEvaluationProtocol,
    PAEvaluationResult,
    evaluate_pa_predictor,
    freeze_pa_evaluation_protocol,
)
from baseline.pa_models import (
    MemoryPolynomialFitDiagnostics,
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
)
from baseline.train_spline import (
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain one JSON object")
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("config schema_version must equal 1")
    required = {
        "dataset",
        "dataset_label",
        "output_dir",
        "order_families",
        "delay_counts",
        "architecture_ridge",
        "refinement_ridges",
        "max_real_multiplications_per_sample",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    return value


def _integer_tuple(values: Any, *, name: str, minimum: int) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON list")
    result: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"every {name} entry must be an integer >= {minimum}")
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return tuple(result)


def _ridge_tuple(values: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON list")
    result = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"every {name} entry must be finite and non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def enumerate_architecture_candidates(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the preregistered candidates that satisfy the real-MUL budget."""

    delay_counts = _integer_tuple(
        config["delay_counts"],
        name="delay_counts",
        minimum=1,
    )
    maximum_multiplications = int(
        config["max_real_multiplications_per_sample"]
    )
    if maximum_multiplications <= 0:
        raise ValueError(
            "max_real_multiplications_per_sample must be positive"
        )
    families = config["order_families"]
    if not isinstance(families, list) or not families:
        raise ValueError("order_families must be a non-empty JSON list")

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("every order_families entry must be an object")
        family_name = family.get("name")
        order_sets = family.get("order_sets")
        if not isinstance(family_name, str) or not family_name.strip():
            raise ValueError("every order family needs a non-empty name")
        if not isinstance(order_sets, list) or not order_sets:
            raise ValueError("every order family needs non-empty order_sets")
        for raw_orders in order_sets:
            orders = _integer_tuple(
                raw_orders,
                name=f"{family_name}.order_set",
                minimum=1,
            )
            for delay_count in delay_counts:
                identity = (orders, delay_count)
                if identity in seen:
                    raise ValueError(
                        "duplicate order-set/delay-count candidate across families"
                    )
                seen.add(identity)
                delays = tuple(range(delay_count))
                cost = memory_polynomial_inference_cost(orders, delays)
                if cost.real_multiplications > maximum_multiplications:
                    continue
                candidates.append(
                    {
                        "family": family_name.strip(),
                        "orders": orders,
                        "delays": delays,
                        "delay_count": delay_count,
                        "ridge": float(config["architecture_ridge"]),
                        "operation_count": cost,
                    }
                )
    if not candidates:
        raise ValueError("no architecture candidate satisfies the operation budget")
    return candidates


def _scalar_trial(
    *,
    stage: str,
    family: str,
    model: MemoryPolynomialPA,
    ridge: float,
    operation_count: OperationCount,
    fit_seconds: float,
    diagnostics: MemoryPolynomialFitDiagnostics,
    evaluation: PAEvaluationResult,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "family": family,
        "orders": list(model.orders),
        "delays": list(model.delays),
        "delay_count": len(model.delays),
        "ridge": ridge,
        "fit_seconds": fit_seconds,
        "fit_diagnostics": dataclasses.asdict(diagnostics),
        "operation_count_per_complex_sample": operation_count.to_dict(),
        "validation_full_record": evaluation.full_record_metrics,
        "validation_common_warmup": evaluation.steady_state_metrics,
        "validation_opendpd_compatible": (
            evaluation.opendpd_compatible_metrics
        ),
        "validation_input_support": evaluation.input_support,
        "selection_metric_name": (
            "validation_full_record.complex_nmse_pooled_db"
        ),
        "selection_score_db": evaluation.full_record_metrics[
            "complex_nmse_pooled_db"
        ],
    }


def _selection_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    score = float(trial["selection_score_db"])
    if not np.isfinite(score) and score != -np.inf:
        score = np.inf
    operations = trial["operation_count_per_complex_sample"]
    return (
        score,
        int(operations["real_multiplications"]),
        int(operations["real_additions"]),
        len(trial["orders"]) * len(trial["delays"]),
        tuple(trial["orders"]),
        tuple(trial["delays"]),
        float(trial["ridge"]),
    )


def _fit_and_score(
    *,
    stage: str,
    family: str,
    orders: tuple[int, ...],
    delays: tuple[int, ...],
    ridge: float,
    train_input: np.ndarray,
    train_output: np.ndarray,
    validation_input: np.ndarray,
    validation_output: np.ndarray,
    protocol: PAEvaluationProtocol,
    common_warmup_samples: int,
) -> tuple[
    dict[str, Any],
    MemoryPolynomialPA,
    PAEvaluationResult,
]:
    started = time.perf_counter()
    model, diagnostics = fit_memory_polynomial_pa(
        train_input,
        train_output,
        orders=orders,
        delays=delays,
        ridge=ridge,
        segment_length=protocol.nperseg,
        coefficient_dtype=np.complex128,
    )
    fit_seconds = time.perf_counter() - started
    operation_count = memory_polynomial_inference_cost(orders, delays)
    evaluation, _ = evaluate_pa_predictor(
        model.predict,
        validation_input,
        validation_output,
        protocol=protocol,
        model_label=(
            f"complex_mp_{family}_o{len(orders)}_q{len(delays)}"
        ),
        split="validation",
        purpose="model_selection",
        common_warmup_samples=common_warmup_samples,
        operation_count=operation_count,
        trainable_real_parameter_count=model.stored_real_coefficients,
        fit_seconds=fit_seconds,
        precision_label="numpy_complex128",
    )
    trial = _scalar_trial(
        stage=stage,
        family=family,
        model=model,
        ridge=ridge,
        operation_count=operation_count,
        fit_seconds=fit_seconds,
        diagnostics=diagnostics,
        evaluation=evaluation,
    )
    return trial, model, evaluation


def _trial_ledger(
    *,
    config_path: Path,
    dataset: Path,
    protocol: PAEvaluationProtocol,
    common_warmup_samples: int,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "forward_pa_identification_model_selection",
        "model_class": "complex_memory_polynomial",
        "selection_split": "validation",
        "test_split_accessed": False,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "dataset": str(dataset),
        "protocol": protocol,
        "common_warmup_samples_per_frame": common_warmup_samples,
        "trials": trials,
    }


def select_from_config(config_path: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Run train/validation-only MP selection and freeze the chosen model."""

    source_config = Path(config_path).resolve()
    config = _load_config(source_config)
    dataset = Path(config["dataset"]).resolve()
    output_directory = Path(config["output_dir"]).resolve()
    spec = load_dataset_spec(dataset)
    required_spec = {
        "input_signal_fs",
        "nperseg",
        "bw_main_ch",
        "n_sub_ch",
    }
    missing_spec = required_spec - set(spec)
    if missing_spec:
        raise ValueError(f"dataset spec is missing keys: {sorted(missing_spec)}")

    model_path = output_directory / "selected_mp_pa.npz"
    manifest_path = output_directory / "selection_manifest.json"
    trials_path = output_directory / "validation_trials.json"
    validation_path = output_directory / "selected_validation_evaluation.json"
    owned_paths = (model_path, manifest_path, trials_path, validation_path)
    existing = [path for path in owned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing selection artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    # This is the complete split access list for selection.  There is no test
    # loader call, test filename, or test hash below this point.
    train_input, train_output = load_split_pair(dataset, "train")
    validation_input, validation_output = load_split_pair(dataset, "val")
    protocol = freeze_pa_evaluation_protocol(
        train_input,
        train_output,
        dataset_label=str(config["dataset_label"]),
        sample_rate_hz=float(spec["input_signal_fs"]),
        nperseg=int(spec["nperseg"]),
        main_bandwidth_hz=float(spec["bw_main_ch"]),
        subchannel_count=int(spec["n_sub_ch"]),
        alignment_max_abs_delay=int(
            config.get("alignment_max_abs_delay", 32)
        ),
        alignment_delay=(
            None
            if config.get("alignment_delay") is None
            else int(config["alignment_delay"])
        ),
        characteristic_bins=int(config.get("characteristic_bins", 32)),
    )
    candidates = enumerate_architecture_candidates(config)
    if any(
        candidate["delay_count"] > protocol.nperseg
        for candidate in candidates
    ):
        raise ValueError("every candidate memory depth must fit inside nperseg")
    common_warmup_samples = max(
        candidate["delay_count"] - 1 for candidate in candidates
    )

    trials: list[dict[str, Any]] = []
    architecture_best: tuple[
        dict[str, Any],
        MemoryPolynomialPA,
        PAEvaluationResult,
    ] | None = None
    for candidate in candidates:
        trial, model, evaluation = _fit_and_score(
            stage="architecture",
            family=str(candidate["family"]),
            orders=candidate["orders"],
            delays=candidate["delays"],
            ridge=float(candidate["ridge"]),
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
            protocol=protocol,
            common_warmup_samples=common_warmup_samples,
        )
        trials.append(trial)
        if (
            architecture_best is None
            or _selection_key(trial) < _selection_key(architecture_best[0])
        ):
            architecture_best = (trial, model, evaluation)
        write_json(
            trials_path,
            _trial_ledger(
                config_path=source_config,
                dataset=dataset,
                protocol=protocol,
                common_warmup_samples=common_warmup_samples,
                trials=trials,
            ),
        )
    assert architecture_best is not None

    selected_architecture_trial = architecture_best[0]
    refinement_ridges = _ridge_tuple(
        config["refinement_ridges"],
        name="refinement_ridges",
    )
    final_best: tuple[
        dict[str, Any],
        MemoryPolynomialPA,
        PAEvaluationResult,
    ] | None = None
    for ridge in refinement_ridges:
        trial, model, evaluation = _fit_and_score(
            stage="ridge_refinement",
            family=str(selected_architecture_trial["family"]),
            orders=tuple(selected_architecture_trial["orders"]),
            delays=tuple(selected_architecture_trial["delays"]),
            ridge=ridge,
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
            protocol=protocol,
            common_warmup_samples=common_warmup_samples,
        )
        trials.append(trial)
        if final_best is None or _selection_key(trial) < _selection_key(final_best[0]):
            final_best = (trial, model, evaluation)
        write_json(
            trials_path,
            _trial_ledger(
                config_path=source_config,
                dataset=dataset,
                protocol=protocol,
                common_warmup_samples=common_warmup_samples,
                trials=trials,
            ),
        )
    assert final_best is not None
    selected_trial, selected_model, selected_evaluation = final_best

    selected_model.save(model_path)
    write_json(validation_path, selected_evaluation.to_dict())
    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "task": "forward_pa_identification_model_selection",
        "model_class": "complex_memory_polynomial",
        "selection_split": "validation",
        "test_split_accessed": False,
        "test_evaluation_status": "not_run_by_design",
        "dataset": dataset,
        "dataset_label": config["dataset_label"],
        "dataset_spec": spec,
        "dataset_files_sha256": {
            "train_input.csv": file_sha256(dataset / "train_input.csv"),
            "train_output.csv": file_sha256(dataset / "train_output.csv"),
            "val_input.csv": file_sha256(dataset / "val_input.csv"),
            "val_output.csv": file_sha256(dataset / "val_output.csv"),
            "spec.json": file_sha256(dataset / "spec.json"),
        },
        "config": source_config,
        "config_sha256": file_sha256(source_config),
        "protocol": protocol,
        "selection_metric": (
            "validation full-record pooled complex NMSE; no post-hoc gain/delay"
        ),
        "selection_stages": (
            "architecture at fixed ridge, then ridge for selected architecture"
        ),
        "common_warmup_samples_per_frame": common_warmup_samples,
        "selected_architecture_trial": selected_architecture_trial,
        "selected_trial": selected_trial,
        "selected_model": model_path,
        "selected_model_sha256": file_sha256(model_path),
        "validation_evaluation": validation_path,
        "validation_trials": trials_path,
        "determinism": {
            "stochastic_fitting": False,
            "seed": None,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_sha256": {
            "experiments/select_pa_mp.py": file_sha256(source_path),
            "baseline/pa_benchmark.py": file_sha256(
                source_path.parents[1] / "baseline" / "pa_benchmark.py"
            ),
            "baseline/pa_models.py": file_sha256(
                source_path.parents[1] / "baseline" / "pa_models.py"
            ),
            "baseline/complexity.py": file_sha256(
                source_path.parents[1] / "baseline" / "complexity.py"
            ),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a budget-constrained complex MP PA model using train/val "
            "only. Test evaluation is a separate command."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = select_from_config(args.config, overwrite=args.overwrite)
    selected = manifest["selected_trial"]
    print(
        "Selected MP:",
        f"orders={selected['orders']}",
        f"Q={selected['delay_count']}",
        f"ridge={selected['ridge']}",
        f"validation NMSE={selected['selection_score_db']:.6f} dB",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
