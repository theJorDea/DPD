"""Select a budget-constrained complex GMP PA model on train/validation only.

The candidate grid is fully declared in JSON as

``ka_values x memory_lengths x topologies``.

Every active lagging or leading branch uses the selected common memory length.
Topology selection is performed with one preregistered solver recipe.  Ridge
values are then refined only for the selected architecture.  The architecture
fit itself remains eligible as the final fit, which permits an OpenDPD-style
truncated-SVD fit to beat all ridge refinements honestly.

This module never opens the official test split.  Frozen test evaluation is a
separate command and must first verify the emitted model, config, source, and
train/validation hashes.
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

from baseline.gmp_pa import (
    GMPConfig,
    GMPFitDiagnostics,
    GeneralizedMemoryPolynomialPA,
    fit_gmp_pa,
)
from baseline.metrics import nmse_pooled_db, time_domain_rms_evm_db
from baseline.pa_benchmark import (
    PAEvaluationProtocol,
    PAEvaluationResult,
    evaluate_pa_predictor,
    freeze_pa_evaluation_protocol,
    prepare_pa_split,
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
        "ka_values",
        "memory_lengths",
        "topologies",
        "architecture_solver_mode",
        "architecture_ridge",
        "refinement_ridges",
        "max_real_multiplications_per_sample",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    return value


def _integer_tuple(
    values: Any,
    *,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON list")
    result: list[int] = []
    for value in values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(
                f"every {name} entry must be an integer >= {minimum}"
            )
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


def _svd_rcond_tuple(values: Any, *, name: str) -> tuple[float, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a JSON list")
    result = tuple(float(value) for value in values)
    if any(
        not np.isfinite(value) or not 0.0 < value < 1.0
        for value in result
    ):
        raise ValueError(
            f"every {name} entry must satisfy finite 0 < rcond < 1"
        )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _selection_metric(config: dict[str, Any]) -> str:
    metric = str(config.get("selection_metric", "full_record"))
    if metric not in {"full_record", "common_interior"}:
        raise ValueError(
            "selection_metric must be full_record or common_interior"
        )
    return metric


def _architecture_fit_recipe(config: dict[str, Any]) -> dict[str, Any]:
    mode = config["architecture_solver_mode"]
    if mode not in {"ridge_lstsq", "truncated_svd"}:
        raise ValueError(
            "architecture_solver_mode must be ridge_lstsq or truncated_svd"
        )
    ridge = float(config["architecture_ridge"])
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("architecture_ridge must be finite and non-negative")
    raw_rcond = config.get("architecture_svd_rcond")
    if mode == "truncated_svd":
        if ridge != 0.0:
            raise ValueError("truncated_svd architecture fit requires ridge=0")
        if raw_rcond is None:
            raise ValueError(
                "truncated_svd architecture fit requires "
                "architecture_svd_rcond"
            )
        rcond = float(raw_rcond)
        if not np.isfinite(rcond) or not 0.0 < rcond < 1.0:
            raise ValueError(
                "architecture_svd_rcond must satisfy finite 0 < rcond < 1"
            )
    else:
        if raw_rcond is not None:
            raise ValueError(
                "architecture_svd_rcond is only valid for truncated_svd"
            )
        rcond = None
    return {
        "solver_mode": mode,
        "ridge": ridge,
        "svd_rcond": rcond,
    }


def _parse_topologies(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    values = config["topologies"]
    if not isinstance(values, list) or not values:
        raise ValueError("topologies must be a non-empty JSON list")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in values:
        if not isinstance(raw, dict):
            raise ValueError("every topology must be a JSON object")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every topology requires a non-empty name")
        name = name.strip()
        if name in names:
            raise ValueError(f"duplicate topology name: {name}")
        names.add(name)
        dimensions: dict[str, int] = {}
        for key in ("kb", "mb", "kc", "mc"):
            value = raw.get(key, 0)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"topology {name}.{key} must be a non-negative integer"
                )
            dimensions[key] = value
        if (dimensions["kb"] == 0) != (dimensions["mb"] == 0):
            raise ValueError(
                f"topology {name} must enable or disable kb and mb together"
            )
        if (dimensions["kc"] == 0) != (dimensions["mc"] == 0):
            raise ValueError(
                f"topology {name} must enable or disable kc and mc together"
            )
        policy = raw.get("leading_policy", "causal_leading")
        if policy not in {"causal_leading", "opendpd_exact"}:
            raise ValueError(
                f"topology {name} has unsupported leading_policy {policy!r}"
            )
        eligible = raw.get("selection_eligible", True)
        if not isinstance(eligible, bool):
            raise ValueError(
                f"topology {name}.selection_eligible must be boolean"
            )
        result.append(
            {
                "name": name,
                **dimensions,
                "leading_policy": policy,
                "selection_eligible": eligible,
            }
        )
    if not any(item["selection_eligible"] for item in result):
        raise ValueError("at least one topology must be selection eligible")
    return tuple(result)


def enumerate_architecture_candidates(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enumerate the preregistered grid under a strict exclusive MUL ceiling."""

    ka_values = _integer_tuple(
        config["ka_values"],
        name="ka_values",
        minimum=1,
    )
    memory_lengths = _integer_tuple(
        config["memory_lengths"],
        name="memory_lengths",
        minimum=1,
    )
    topologies = _parse_topologies(config)
    recipe = _architecture_fit_recipe(config)
    budget = config["max_real_multiplications_per_sample"]
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(
            "max_real_multiplications_per_sample must be a positive integer"
        )

    candidates: list[dict[str, Any]] = []
    identities: set[GMPConfig] = set()
    for ka in ka_values:
        for memory_length in memory_lengths:
            for topology in topologies:
                gmp_config = GMPConfig(
                    ka=ka,
                    la=memory_length,
                    kb=topology["kb"],
                    lb=memory_length if topology["kb"] else 0,
                    mb=topology["mb"],
                    kc=topology["kc"],
                    lc=memory_length if topology["kc"] else 0,
                    mc=topology["mc"],
                    leading_policy=topology["leading_policy"],
                )
                if gmp_config in identities:
                    raise ValueError(
                        "duplicate GMP configuration generated by topologies"
                    )
                identities.add(gmp_config)
                operation_count = GeneralizedMemoryPolynomialPA(
                    gmp_config,
                    np.zeros(gmp_config.coefficient_count, dtype=np.complex128),
                ).operation_count
                # Huawei's slide says "less than 1000", not "at most 1000".
                if operation_count.real_multiplications >= budget:
                    continue
                candidates.append(
                    {
                        "topology": topology["name"],
                        "selection_eligible": topology["selection_eligible"],
                        "gmp_config": gmp_config,
                        "operation_count": operation_count,
                        **recipe,
                    }
                )
    if not candidates:
        raise ValueError("no GMP architecture satisfies the strict MUL budget")
    if not any(candidate["selection_eligible"] for candidate in candidates):
        raise ValueError(
            "no selection-eligible GMP architecture satisfies the strict MUL budget"
        )
    return candidates


def segmented_interior_mask(
    sample_count: int,
    *,
    segment_length: int,
    warmup_samples: int,
    cooldown_samples: int,
) -> np.ndarray:
    """Apply identical left/right boundary exclusions to every validation frame."""

    values = (sample_count, segment_length, warmup_samples, cooldown_samples)
    if any(
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        for value in values
    ):
        raise TypeError("interior-mask arguments must be integers")
    sample_count, segment_length, warmup_samples, cooldown_samples = (
        int(value) for value in values
    )
    if sample_count < 1 or segment_length < 1:
        raise ValueError("sample_count and segment_length must be positive")
    if warmup_samples < 0 or cooldown_samples < 0:
        raise ValueError("warmup and cooldown must be non-negative")
    mask = np.zeros(sample_count, dtype=bool)
    for start in range(0, sample_count, segment_length):
        stop = min(start + segment_length, sample_count)
        interior_start = min(start + warmup_samples, stop)
        interior_stop = max(start, stop - cooldown_samples)
        if interior_start < interior_stop:
            mask[interior_start:interior_stop] = True
    if not np.any(mask):
        raise ValueError("common warmup/cooldown consumes every validation sample")
    return mask


def _common_interior_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    segment_length: int,
    warmup_samples: int,
    cooldown_samples: int,
) -> dict[str, float | int]:
    estimate = np.asarray(prediction)
    target = np.asarray(reference)
    if estimate.shape != target.shape or estimate.ndim != 1:
        raise ValueError("prediction/reference must be equal-length vectors")
    mask = segmented_interior_mask(
        estimate.size,
        segment_length=segment_length,
        warmup_samples=warmup_samples,
        cooldown_samples=cooldown_samples,
    )
    error_power = float(np.mean(np.abs(estimate[mask] - target[mask]) ** 2))
    reference_power = float(np.mean(np.abs(target[mask]) ** 2))
    return {
        "complex_nmse_pooled_db": nmse_pooled_db(
            estimate[mask],
            target[mask],
        ),
        "time_domain_rms_sample_evm_db": time_domain_rms_evm_db(
            estimate[mask],
            target[mask],
        ),
        "mse": error_power,
        "reference_power": reference_power,
        "relative_error_power": error_power / reference_power,
        "warmup_samples_per_frame": warmup_samples,
        "cooldown_samples_per_frame": cooldown_samples,
        "scored_sample_count": int(np.count_nonzero(mask)),
        "discarded_sample_count": int(mask.size - np.count_nonzero(mask)),
    }


def _trial_record(
    *,
    stage: str,
    topology: str,
    selection_eligible: bool,
    model: GeneralizedMemoryPolynomialPA,
    diagnostics: GMPFitDiagnostics,
    evaluation: PAEvaluationResult,
    common_interior: dict[str, float | int],
    fit_seconds: float,
    ridge: float,
    solver_mode: str,
    svd_rcond: float | None,
    selection_metric: str,
) -> dict[str, Any]:
    full_score = evaluation.full_record_metrics[
        "complex_nmse_pooled_db"
    ]
    interior_score = common_interior["complex_nmse_pooled_db"]
    if selection_metric == "full_record":
        selection_name = (
            "validation_full_record.complex_nmse_pooled_db"
        )
        selection_score = full_score
        secondary_name = (
            "validation_common_interior.complex_nmse_pooled_db"
        )
        secondary_score = interior_score
    elif selection_metric == "common_interior":
        selection_name = (
            "validation_common_interior.complex_nmse_pooled_db"
        )
        selection_score = interior_score
        secondary_name = (
            "validation_full_record.complex_nmse_pooled_db"
        )
        secondary_score = full_score
    else:
        raise ValueError(f"unsupported selection metric: {selection_metric}")
    return {
        "stage": stage,
        "topology": topology,
        "selection_eligible": selection_eligible,
        "gmp_config": dataclasses.asdict(model.config),
        "ridge": ridge,
        "solver_mode": solver_mode,
        "svd_rcond": svd_rcond,
        "fit_seconds": fit_seconds,
        "fit_diagnostics": dataclasses.asdict(diagnostics),
        "operation_count_per_complex_sample": model.operation_count.to_dict(),
        "validation_full_record": evaluation.full_record_metrics,
        "validation_common_warmup": evaluation.steady_state_metrics,
        "validation_common_interior": common_interior,
        "validation_opendpd_compatible": (
            evaluation.opendpd_compatible_metrics
        ),
        "validation_input_support": evaluation.input_support,
        "selection_metric_name": selection_name,
        "selection_score_db": selection_score,
        "selection_secondary_metric_name": secondary_name,
        "selection_secondary_score_db": secondary_score,
    }


def _selection_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    score = float(trial["selection_score_db"])
    if not np.isfinite(score) and score != -np.inf:
        score = np.inf
    secondary_score = float(trial["selection_secondary_score_db"])
    if not np.isfinite(secondary_score) and secondary_score != -np.inf:
        secondary_score = np.inf
    operations = trial["operation_count_per_complex_sample"]
    gmp_config = trial["gmp_config"]
    return (
        score,
        secondary_score,
        int(operations["real_multiplications"]),
        int(operations["real_additions"]),
        int(operations["stored_real_coefficients"]),
        str(trial["topology"]),
        tuple(
            gmp_config[key]
            for key in ("ka", "la", "kb", "lb", "mb", "kc", "lc", "mc")
        ),
        str(gmp_config["leading_policy"]),
        str(trial["solver_mode"]),
        -1.0 if trial["svd_rcond"] is None else float(trial["svd_rcond"]),
        float(trial["ridge"]),
    )


def _fit_and_score(
    *,
    stage: str,
    topology: str,
    selection_eligible: bool,
    gmp_config: GMPConfig,
    ridge: float,
    solver_mode: str,
    svd_rcond: float | None,
    train_input: np.ndarray,
    train_output: np.ndarray,
    validation_input: np.ndarray,
    validation_output: np.ndarray,
    validation_reference: np.ndarray,
    protocol: PAEvaluationProtocol,
    common_warmup_samples: int,
    common_cooldown_samples: int,
    selection_metric: str,
) -> tuple[
    dict[str, Any],
    GeneralizedMemoryPolynomialPA,
    PAEvaluationResult,
]:
    started = time.perf_counter()
    model, diagnostics = fit_gmp_pa(
        train_input,
        train_output,
        config=gmp_config,
        ridge=ridge,
        segment_length=protocol.nperseg,
        coefficient_dtype=np.complex128,
        solver_mode=solver_mode,
        svd_rcond=svd_rcond,
    )
    fit_seconds = time.perf_counter() - started
    evaluation, prediction = evaluate_pa_predictor(
        model.predict,
        validation_input,
        validation_output,
        protocol=protocol,
        model_label=(
            f"complex_gmp_{topology}_ka{gmp_config.ka}_l{gmp_config.la}"
        ),
        split="validation",
        purpose="model_selection" if selection_eligible else "diagnostic",
        common_warmup_samples=common_warmup_samples,
        operation_count=model.operation_count,
        trainable_real_parameter_count=model.stored_real_coefficients,
        fit_seconds=fit_seconds,
        precision_label="numpy_complex128",
    )
    common_interior = _common_interior_metrics(
        prediction,
        validation_reference,
        segment_length=protocol.nperseg,
        warmup_samples=common_warmup_samples,
        cooldown_samples=common_cooldown_samples,
    )
    trial = _trial_record(
        stage=stage,
        topology=topology,
        selection_eligible=selection_eligible,
        model=model,
        diagnostics=diagnostics,
        evaluation=evaluation,
        common_interior=common_interior,
        fit_seconds=fit_seconds,
        ridge=ridge,
        solver_mode=solver_mode,
        svd_rcond=svd_rcond,
        selection_metric=selection_metric,
    )
    return trial, model, evaluation


def _trial_ledger(
    *,
    config_path: Path,
    dataset: Path,
    protocol: PAEvaluationProtocol,
    common_warmup_samples: int,
    common_cooldown_samples: int,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "forward_pa_identification_model_selection",
        "model_class": "complex_generalized_memory_polynomial",
        "selection_split": "validation",
        "test_split_accessed": False,
        "config": config_path,
        "config_sha256": file_sha256(config_path),
        "dataset": dataset,
        "protocol": protocol,
        "common_warmup_samples_per_frame": common_warmup_samples,
        "common_future_cooldown_samples_per_frame": common_cooldown_samples,
        "trials": trials,
    }


def select_from_config(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run deterministic train/validation GMP selection and freeze the winner."""

    source_config = Path(config_path).resolve()
    config = _load_config(source_config)
    selection_metric = _selection_metric(config)
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

    model_path = output_directory / "selected_gmp_pa.npz"
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

    # This is the complete split access list in this command.
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
    if protocol.alignment_delay_samples != 0:
        raise NotImplementedError(
            "non-zero flattened alignment is not frame-safe in the unified "
            "PA evaluator; preprocess frame-aligned pairs before GMP selection"
        )
    _, validation_reference = prepare_pa_split(
        validation_input,
        validation_output,
        protocol,
    )

    candidates = enumerate_architecture_candidates(config)
    if any(
        candidate["gmp_config"].causal_warmup_samples >= protocol.nperseg
        or candidate["gmp_config"].lookahead_samples >= protocol.nperseg
        for candidate in candidates
    ):
        raise ValueError("every GMP memory/lookahead must fit inside nperseg")
    common_warmup_samples = max(
        candidate["gmp_config"].causal_warmup_samples
        for candidate in candidates
    )
    common_cooldown_samples = max(
        candidate["gmp_config"].lookahead_samples
        for candidate in candidates
    )
    # Fail before the expensive sweep if the common scoring support is empty.
    segmented_interior_mask(
        validation_reference.size,
        segment_length=protocol.nperseg,
        warmup_samples=common_warmup_samples,
        cooldown_samples=common_cooldown_samples,
    )

    trials: list[dict[str, Any]] = []
    architecture_best: tuple[
        dict[str, Any],
        GeneralizedMemoryPolynomialPA,
        PAEvaluationResult,
    ] | None = None
    for candidate in candidates:
        trial, model, evaluation = _fit_and_score(
            stage="architecture",
            topology=str(candidate["topology"]),
            selection_eligible=bool(candidate["selection_eligible"]),
            gmp_config=candidate["gmp_config"],
            ridge=float(candidate["ridge"]),
            solver_mode=str(candidate["solver_mode"]),
            svd_rcond=candidate["svd_rcond"],
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
            validation_reference=validation_reference,
            protocol=protocol,
            common_warmup_samples=common_warmup_samples,
            common_cooldown_samples=common_cooldown_samples,
            selection_metric=selection_metric,
        )
        trials.append(trial)
        if trial["selection_eligible"] and (
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
                common_cooldown_samples=common_cooldown_samples,
                trials=trials,
            ),
        )
    assert architecture_best is not None
    selected_architecture_trial = architecture_best[0]

    # The topology-stage fit remains a valid final candidate.  This matters
    # when it is the OpenDPD-compatible truncated-SVD fit.
    final_best = architecture_best
    selected_config = GMPConfig(**selected_architecture_trial["gmp_config"])
    for ridge in _ridge_tuple(
        config["refinement_ridges"],
        name="refinement_ridges",
    ):
        trial, model, evaluation = _fit_and_score(
            stage="ridge_refinement",
            topology=str(selected_architecture_trial["topology"]),
            selection_eligible=True,
            gmp_config=selected_config,
            ridge=ridge,
            solver_mode="ridge_lstsq",
            svd_rcond=None,
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
            validation_reference=validation_reference,
            protocol=protocol,
            common_warmup_samples=common_warmup_samples,
            common_cooldown_samples=common_cooldown_samples,
            selection_metric=selection_metric,
        )
        trials.append(trial)
        if _selection_key(trial) < _selection_key(final_best[0]):
            final_best = (trial, model, evaluation)
        write_json(
            trials_path,
            _trial_ledger(
                config_path=source_config,
                dataset=dataset,
                protocol=protocol,
                common_warmup_samples=common_warmup_samples,
                common_cooldown_samples=common_cooldown_samples,
                trials=trials,
            ),
        )

    completed_recipes = {
        (
            str(trial["solver_mode"]),
            float(trial["ridge"]),
            (
                None
                if trial["svd_rcond"] is None
                else float(trial["svd_rcond"])
            ),
        )
        for trial in trials
        if trial["gmp_config"] == selected_architecture_trial["gmp_config"]
    }
    for svd_rcond in _svd_rcond_tuple(
        config.get("refinement_svd_rconds"),
        name="refinement_svd_rconds",
    ):
        recipe = ("truncated_svd", 0.0, svd_rcond)
        if recipe in completed_recipes:
            continue
        trial, model, evaluation = _fit_and_score(
            stage="svd_refinement",
            topology=str(selected_architecture_trial["topology"]),
            selection_eligible=True,
            gmp_config=selected_config,
            ridge=0.0,
            solver_mode="truncated_svd",
            svd_rcond=svd_rcond,
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
            validation_reference=validation_reference,
            protocol=protocol,
            common_warmup_samples=common_warmup_samples,
            common_cooldown_samples=common_cooldown_samples,
            selection_metric=selection_metric,
        )
        trials.append(trial)
        completed_recipes.add(recipe)
        if _selection_key(trial) < _selection_key(final_best[0]):
            final_best = (trial, model, evaluation)
        write_json(
            trials_path,
            _trial_ledger(
                config_path=source_config,
                dataset=dataset,
                protocol=protocol,
                common_warmup_samples=common_warmup_samples,
                common_cooldown_samples=common_cooldown_samples,
                trials=trials,
            ),
        )

    selected_trial, selected_model, selected_evaluation = final_best
    selected_model.save(model_path)
    write_json(validation_path, selected_evaluation.to_dict())
    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "task": "forward_pa_identification_model_selection",
        "model_class": "complex_generalized_memory_polynomial",
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
        "selection_metric": selected_trial["selection_metric_name"],
        "selection_secondary_metric": (
            selected_trial["selection_secondary_metric_name"]
        ),
        "selection_metric_policy": (
            f"primary={selection_metric}; full-record and common-interior "
            "pooled complex NMSE are both recorded; the secondary metric "
            "breaks exact primary ties; no post-hoc gain or delay fit"
        ),
        "selection_stages": (
            "architecture/topology at the preregistered solver recipe; ridge "
            "and optional truncated-SVD-rcond refinement for the selected "
            "architecture; architecture fit remains eligible as the final model"
        ),
        "operation_budget": {
            "metric": "factorized real multiplications per complex sample",
            "maximum_exclusive": int(
                config["max_real_multiplications_per_sample"]
            ),
            "selected_value": int(
                selected_trial["operation_count_per_complex_sample"][
                    "real_multiplications"
                ]
            ),
            "complex_multiply_convention": "4 real MUL + 2 real ADD",
        },
        "common_warmup_samples_per_frame": common_warmup_samples,
        "common_future_cooldown_samples_per_frame": common_cooldown_samples,
        "selected_architecture_trial": selected_architecture_trial,
        "selected_trial": selected_trial,
        "selected_model": model_path,
        "selected_model_sha256": file_sha256(model_path),
        "validation_evaluation": validation_path,
        "validation_trials": trials_path,
        "determinism": {
            "stochastic_fitting": False,
            "seed": None,
            "candidate_iteration_order": (
                "ka_values then memory_lengths then topologies, preserving "
                "config list order"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_sha256": {
            "experiments/select_pa_gmp.py": file_sha256(source_path),
            "baseline/pa_benchmark.py": file_sha256(
                source_path.parents[1] / "baseline" / "pa_benchmark.py"
            ),
            "baseline/gmp_pa.py": file_sha256(
                source_path.parents[1] / "baseline" / "gmp_pa.py"
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
            "Select a strict-budget complex GMP PA model with train/validation "
            "only. Frozen test evaluation is a separate command."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = select_from_config(args.config, overwrite=args.overwrite)
    selected = manifest["selected_trial"]
    selected_config = selected["gmp_config"]
    print(
        "Selected GMP:",
        f"topology={selected['topology']}",
        f"Ka={selected_config['ka']}",
        f"L={selected_config['la']}",
        f"solver={selected['solver_mode']}",
        f"ridge={selected['ridge']}",
        f"{selected['selection_metric_name']}="
        f"{selected['selection_score_db']:.6f} dB",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
