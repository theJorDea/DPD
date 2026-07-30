"""Train-only staged selection helpers for the sparse spline-memory PA.

This module contains no test loader.  It is intentionally usable by unit tests
and by the atomic runner without giving the selection code a path to the sealed
APA test split.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np

from baseline.metrics import nmse_opendpd_db, nmse_pooled_db
from baseline.sparse_spline_memory_pa import (
    SparseSplineMemoryPA,
    SparseSplineMemoryPABranch,
    fit_sparse_spline_memory_pa_segments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TASK = "forward_pa_non_factorized_sparse_spline_memory_selection"
PREREGISTERED_STATUSES = frozenset(
    {
        "preregistered_before_model_implementation_and_candidate_fit",
        "preregistered_before_candidate_fit_using_frozen_existing_implementation",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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
        return {"real": number.real, "imag": number.imag}
    if isinstance(value, Path):
        return str(value)
    return value


@dataclasses.dataclass(frozen=True)
class SparseRecipe:
    family: str
    branches: tuple[tuple[int, int], ...]
    knot_count: int
    ridge: float

    def __post_init__(self) -> None:
        if not self.family or not isinstance(self.family, str):
            raise ValueError("recipe family must be a non-empty string")
        if not self.branches:
            raise ValueError("recipe needs at least one branch")
        normalized = tuple(
            (int(signal), int(envelope)) for signal, envelope in self.branches
        )
        if any(signal < 0 or envelope < 0 for signal, envelope in normalized):
            raise ValueError("sparse PA branches must be causal")
        if len(set(normalized)) != len(normalized):
            raise ValueError("recipe contains duplicate branch pairs")
        if int(self.knot_count) < 2:
            raise ValueError("knot_count must be at least two")
        ridge = float(self.ridge)
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        object.__setattr__(self, "branches", normalized)
        object.__setattr__(self, "knot_count", int(self.knot_count))
        object.__setattr__(self, "ridge", ridge)

    @property
    def canonical_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "branches": [list(pair) for pair in self.branches],
            "knot_count": self.knot_count,
            "ridge": self.ridge,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.canonical_dict)

    @property
    def name(self) -> str:
        branch_label = ",".join(f"{m}:{d}" for m, d in self.branches)
        return f"{self.family}_K{self.knot_count}_r{self.ridge:.0e}_b{branch_label}"

    @property
    def branch_objects(self) -> tuple[SparseSplineMemoryPABranch, ...]:
        return tuple(
            SparseSplineMemoryPABranch(signal, envelope)
            for signal, envelope in self.branches
        )


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("sparse PA config must contain one object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("sparse PA config schema mismatch")
    if config.get("task") != TASK:
        raise ValueError("unexpected sparse PA task")
    if config.get("status") not in PREREGISTERED_STATUSES:
        raise ValueError("sparse PA config is not preregistered")
    if config.get("scope", {}).get("test_split_access_permitted") is not False:
        raise ValueError("sparse PA config must forbid test access")
    for key in (
        "dataset",
        "dataset_contract",
        "evidence",
        "reference_models",
        "branch_families",
        "search",
        "gates",
        "search_budget",
        "preimplementation_source_sha256",
    ):
        if key not in config:
            raise ValueError(f"sparse PA config missing {key}")
    return config


def project_path(value: str, *, name: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{name} must be repository-relative")
    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository") from error
    return resolved


def verify_preregistered_inputs(
    config: dict[str, Any],
    config_path: str | Path,
) -> dict[str, Any]:
    """Verify every non-waveform input before train/validation load."""

    dataset = project_path(str(config["dataset"]), name="dataset")
    contract = config["dataset_contract"]
    required = contract["required_files_sha256"]
    dataset_hashes: dict[str, str] = {}
    for name, expected in required.items():
        path = dataset / name
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"dataset hash mismatch for {name}")
        dataset_hashes[name] = actual

    evidence_hashes: dict[str, str] = {}
    for label, record in config["evidence"].items():
        path = project_path(str(record["path"]), name=f"evidence/{label}")
        actual = file_sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"evidence hash mismatch for {label}")
        evidence_hashes[label] = actual

    source_hashes: dict[str, str] = {}
    for name, expected in config["preimplementation_source_sha256"].items():
        path = project_path(name, name="source")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"preimplementation source hash mismatch: {name}")
        source_hashes[name] = actual

    new_sources = {
        name: file_sha256(project_path(name, name="new source"))
        for name in config.get("new_source_files_must_not_exist", [])
        if project_path(name, name="new source").is_file()
    }
    return {
        "config_sha256": file_sha256(config_path),
        "dataset": dataset,
        "dataset_hashes": dataset_hashes,
        "evidence_hashes": evidence_hashes,
        "preimplementation_source_hashes": source_hashes,
        "new_source_files_present_after_implementation": new_sources,
    }


def frame_segments(signal: np.ndarray, frame_lengths: Iterable[int]) -> tuple[np.ndarray, ...]:
    values = np.asarray(signal, dtype=np.complex128)
    lengths = tuple(int(length) for length in frame_lengths)
    if any(length <= 0 for length in lengths) or sum(lengths) != values.size:
        raise ValueError("frame lengths do not partition signal")
    result: list[np.ndarray] = []
    start = 0
    for length in lengths:
        result.append(values[start : start + length].copy())
        start += length
    return tuple(result)


def common_mask(frame_lengths: Iterable[int], warmup: int) -> np.ndarray:
    lengths = tuple(int(length) for length in frame_lengths)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    mask = np.zeros(sum(lengths), dtype=bool)
    start = 0
    for length in lengths:
        if warmup >= length:
            raise ValueError("warmup consumes a frame")
        mask[start + warmup : start + length] = True
        start += length
    return mask


def _metric_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        return nmse_pooled_db(prediction, target)
    if mask.shape != prediction.shape or not np.any(mask):
        raise ValueError("invalid metric mask")
    return nmse_pooled_db(prediction[mask], target[mask])


def _opendpd_complete_metric(
    prediction: np.ndarray,
    target: np.ndarray,
    frame_lengths: tuple[int, ...],
) -> tuple[float | None, int]:
    complete = [length for length in frame_lengths if length == max(frame_lengths)]
    if not complete or any(length != max(frame_lengths) for length in frame_lengths[:-1]):
        # The APA contract has two complete frames and one partial frame.
        pass
    segments_pred: list[np.ndarray] = []
    segments_target: list[np.ndarray] = []
    start = 0
    for length in frame_lengths:
        if length == max(frame_lengths):
            segments_pred.append(prediction[start : start + length])
            segments_target.append(target[start : start + length])
        start += length
    if not segments_pred:
        return None, 0
    return nmse_opendpd_db(
        np.stack(segments_pred),
        np.stack(segments_target),
    ), len(segments_pred)


def _fold_offsets(frame_lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    current = 0
    for length in frame_lengths:
        offsets.append(current)
        current += length
    return tuple(offsets)


def _reference_fold_metrics(
    reference_prediction: np.ndarray,
    target: np.ndarray,
    frame_lengths: tuple[int, ...],
    held_frame: int,
    warmup: int,
) -> dict[str, float]:
    offsets = _fold_offsets(frame_lengths)
    start = offsets[held_frame]
    stop = start + frame_lengths[held_frame]
    held_prediction = reference_prediction[start:stop]
    held_target = target[start:stop]
    common = slice(warmup, None)
    return {
        "full_record_nmse_db": nmse_pooled_db(held_prediction, held_target),
        "common_interior_nmse_db": nmse_pooled_db(
            held_prediction[common], held_target[common]
        ),
    }


def _recipe_record(recipe: SparseRecipe) -> dict[str, Any]:
    return {
        **recipe.canonical_dict,
        "name": recipe.name,
        "sha256": recipe.sha256,
    }


def evaluate_recipe_oof(
    recipe: SparseRecipe,
    input_segments: tuple[np.ndarray, ...],
    output_segments: tuple[np.ndarray, ...],
    reference_gmp_prediction: np.ndarray,
    *,
    frame_lengths: tuple[int, ...],
    common_warmup: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Fit one recipe leave-one-frame-out and return a JSON-ready record."""

    if len(input_segments) != len(frame_lengths):
        raise ValueError("input segments and frame lengths disagree")
    if reference_gmp_prediction.size != sum(frame_lengths):
        raise ValueError("reference GMP OOF prediction has wrong length")
    started = time.perf_counter()
    offsets = _fold_offsets(frame_lengths)
    prediction = np.empty(sum(frame_lengths), dtype=np.complex128)
    fold_records: list[dict[str, Any]] = []
    hard_checks = {
        "all_data_designs_full_column_rank": True,
        "all_active_features_observed": True,
        "all_augmented_condition_numbers_finite": True,
        "all_coefficients_finite_and_bounded": True,
        "all_predictions_finite": True,
        "all_support_exceedance_limits": True,
        "real_multiplications_strictly_below_limit": True,
    }
    for held_frame, held_input in enumerate(input_segments):
        fit_inputs = tuple(
            segment
            for index, segment in enumerate(input_segments)
            if index != held_frame
        )
        fit_outputs = tuple(
            segment
            for index, segment in enumerate(output_segments)
            if index != held_frame
        )
        model, diagnostics = fit_sparse_spline_memory_pa_segments(
            fit_inputs,
            fit_outputs,
            branches=recipe.branch_objects,
            knot_count=recipe.knot_count,
            knot_strategy="uniform_amplitude",
            ridge=recipe.ridge,
        )
        held_prediction = model.predict(held_input)
        start = offsets[held_frame]
        stop = start + held_input.size
        prediction[start:stop] = held_prediction
        operation = model.operation_count()
        fit_coordinate_max = float(model.knots[-1])
        held_radius = np.abs(held_input)
        support_fraction = float(np.mean(held_radius > fit_coordinate_max))
        hard_checks["all_data_designs_full_column_rank"] &= (
            diagnostics.data_design_rank == diagnostics.feature_count
        )
        hard_checks["all_active_features_observed"] &= (
            diagnostics.minimum_nonzero_feature_samples > 0
        )
        hard_checks["all_augmented_condition_numbers_finite"] &= bool(
            np.isfinite(diagnostics.augmented_design_condition_number)
            and diagnostics.augmented_design_condition_number
            <= float(gates["maximum_augmented_condition_number"])
        )
        hard_checks["all_coefficients_finite_and_bounded"] &= bool(
            np.isfinite(diagnostics.maximum_absolute_coefficient)
            and diagnostics.maximum_absolute_coefficient
            <= float(gates["maximum_absolute_coefficient"])
        )
        hard_checks["all_predictions_finite"] &= bool(np.all(np.isfinite(held_prediction)))
        hard_checks["all_support_exceedance_limits"] &= (
            support_fraction <= float(gates["maximum_support_exceedance_fraction"])
        )
        hard_checks["real_multiplications_strictly_below_limit"] &= (
            operation.real_multiplications
            < int(gates["real_multiplications_strictly_below"])
        )
        reference_start = offsets[held_frame]
        reference_stop = reference_start + held_input.size
        gmp_metrics = _reference_fold_metrics(
            reference_gmp_prediction[reference_start:reference_stop],
            output_segments[held_frame],
            (held_input.size,),
            0,
            common_warmup,
        )
        candidate_metrics = {
            "full_record_nmse_db": nmse_pooled_db(held_prediction, output_segments[held_frame]),
            "common_interior_nmse_db": nmse_pooled_db(
                held_prediction[common_warmup:],
                output_segments[held_frame][common_warmup:],
            ),
        }
        fold_records.append(
            {
                "held_frame_id": held_frame,
                "fit_frame_ids": [
                    index for index in range(len(input_segments)) if index != held_frame
                ],
                "held_metrics": candidate_metrics,
                "reference_gmp_metrics": gmp_metrics,
                "gain_over_gmp_full_db": (
                    gmp_metrics["full_record_nmse_db"]
                    - candidate_metrics["full_record_nmse_db"]
                ),
                "gain_over_gmp_common_db": (
                    gmp_metrics["common_interior_nmse_db"]
                    - candidate_metrics["common_interior_nmse_db"]
                ),
                "fit_diagnostics": diagnostics.to_dict(),
                "operation_count": operation.to_dict(),
                "fit_support_maximum": fit_coordinate_max,
                "held_support_maximum": float(np.max(held_radius)),
                "held_support_exceedance_fraction": support_fraction,
            }
        )
    common = common_mask(frame_lengths, common_warmup)
    full_nmse = _metric_pair(prediction, np.concatenate(output_segments))
    common_nmse = _metric_pair(prediction, np.concatenate(output_segments), common)
    reference_full_nmse = _metric_pair(
        reference_gmp_prediction,
        np.concatenate(output_segments),
    )
    reference_common_nmse = _metric_pair(
        reference_gmp_prediction,
        np.concatenate(output_segments),
        common,
    )
    operation = SparseSplineMemoryPA(
        knots=np.linspace(0.0, 1.0, recipe.knot_count),
        branches=recipe.branch_objects,
        coefficients=np.zeros(
            (len(recipe.branches), recipe.knot_count), dtype=np.complex128
        ),
    ).operation_count()
    hard_valid = bool(all(hard_checks.values()))
    return {
        "recipe": _recipe_record(recipe),
        "recipe_sha256": recipe.sha256,
        "full_record_nmse_db": full_nmse,
        "common_interior_nmse_db": common_nmse,
        "reference_gmp_full_record_nmse_db": reference_full_nmse,
        "reference_gmp_common_interior_nmse_db": reference_common_nmse,
        "gain_over_gmp_full_db": reference_full_nmse - full_nmse,
        "gain_over_gmp_common_db": reference_common_nmse - common_nmse,
        "minimum_fold_gain_over_gmp_full_db": min(
            record["gain_over_gmp_full_db"] for record in fold_records
        ),
        "minimum_fold_gain_over_gmp_common_db": min(
            record["gain_over_gmp_common_db"] for record in fold_records
        ),
        "operation_count": operation.to_dict(),
        "hard_validity_checks": hard_checks,
        "hard_valid": hard_valid,
        "fold_records": fold_records,
        "oof_prediction": prediction,
        "fit_seconds": time.perf_counter() - started,
    }


def _ranking_key(record: dict[str, Any]) -> tuple[Any, ...]:
    recipe = record["recipe"]
    return (
        float(record["full_record_nmse_db"]),
        float(record["common_interior_nmse_db"]),
        int(record["operation_count"]["real_multiplications"]),
        len(recipe["branches"]),
        int(recipe["knot_count"]),
        str(recipe["name"]),
    )


def rank_valid_records(
    records: Iterable[dict[str, Any]],
    *,
    tie_tolerance_db: float = 0.02,
) -> list[dict[str, Any]]:
    valid = [record for record in records if record.get("hard_valid")]
    if not valid:
        raise ValueError("no hard-valid sparse PA records")
    primary = min(float(record["full_record_nmse_db"]) for record in valid)
    window = [
        record
        for record in valid
        if float(record["full_record_nmse_db"]) <= primary + tie_tolerance_db
    ]
    window_hashes = {str(record["recipe_sha256"]) for record in window}
    # The preregistered tie rule treats full-record NMSE as the primary
    # criterion.  Once candidates fall inside that tolerance window, use the
    # common/interior metric and only then complexity as deterministic
    # secondary criteria.  Sorting the whole window by full NMSE would make a
    # tiny numerical difference defeat the declared tie rule.
    def _window_ranking_key(record: dict[str, Any]) -> tuple[Any, ...]:
        recipe = record["recipe"]
        return (
            float(record["common_interior_nmse_db"]),
            int(record["operation_count"]["real_multiplications"]),
            len(recipe["branches"]),
            int(recipe["knot_count"]),
            str(recipe["name"]),
        )

    return sorted(window, key=_window_ranking_key) + sorted(
        [
            record
            for record in valid
            if str(record["recipe_sha256"]) not in window_hashes
        ],
        key=_ranking_key,
    )


def retain_topologies(
    records: Iterable[dict[str, Any]],
    *,
    maximum: int,
    window_db: float,
) -> list[dict[str, Any]]:
    ranked = rank_valid_records(records, tie_tolerance_db=0.0)
    winner = float(ranked[0]["full_record_nmse_db"])
    retained = [
        record
        for record in ranked
        if float(record["full_record_nmse_db"]) <= winner + window_db
    ]
    return retained[:maximum]


def strip_prediction(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_ready(value)
        for key, value in record.items()
        if key != "oof_prediction"
    }


def branch_families(
    config: dict[str, Any],
) -> dict[str, tuple[tuple[int, int], ...]]:
    raw = config["branch_families"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("branch_families must be a non-empty object")
    families: dict[str, tuple[tuple[int, int], ...]] = {}
    for name, pairs in raw.items():
        if not isinstance(name, str) or not isinstance(pairs, list) or not pairs:
            raise ValueError("invalid sparse branch family")
        if any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
            raise ValueError("each sparse branch must be a two-element list")
        normalized = tuple((int(pair[0]), int(pair[1])) for pair in pairs)
        # SparseRecipe performs the causal and duplicate checks.
        SparseRecipe(name, normalized, 2, 0.0)
        families[name] = normalized
    return families


def validate_search_budget(config: dict[str, Any]) -> dict[str, int]:
    families = branch_families(config)
    search = config["search"]
    budget = config["search_budget"]
    retained = int(search["stage_s0_topology_screen"]["retain_topologies"])
    s0 = len(families)
    s1 = retained * len(search["stage_s1_knot_count"]["knot_counts"])
    s2 = len(search["stage_s2_ridge"]["ridges"])
    expected = {
        "maximum_s0_recipes": s0,
        "maximum_s1_recipes": s1,
        "maximum_s2_recipes": s2,
        "maximum_unique_recipes": s0 + s1 + s2,
        "maximum_oof_fit_calls_without_cache": (
            (s0 + s1 + s2) * int(budget["oof_fold_count"])
        ),
    }
    mismatches = {
        name: (int(budget[name]), value)
        for name, value in expected.items()
        if int(budget[name]) != value
    }
    if mismatches:
        raise ValueError(f"search budget disagrees with recipe axes: {mismatches}")
    if int(budget["oof_fold_count"]) != len(
        config["dataset_contract"]["frame_lengths"]
    ):
        raise ValueError("OOF fold count disagrees with explicit train frames")
    return {
        "S0": s0,
        "S1": s1,
        "S2": s2,
        "folds": int(budget["oof_fold_count"]),
    }


def _annotate_research_gates(
    record: dict[str, Any],
    config: dict[str, Any],
) -> None:
    references = config["reference_models"]
    gates = config["gates"]
    full = float(record["full_record_nmse_db"])
    common = float(record["common_interior_nmse_db"])
    mp = references["matched_mp_oof"]
    gmp = references["matched_gmp_oof"]
    loss_mp_full = full - float(mp["full_record_nmse_db"])
    loss_mp_common = common - float(mp["common_interior_nmse_db"])
    gain_gmp_full = float(gmp["full_record_nmse_db"]) - full
    gain_gmp_common = float(gmp["common_interior_nmse_db"]) - common
    record.update(
        {
            "loss_vs_mp_full_db": loss_mp_full,
            "loss_vs_mp_common_db": loss_mp_common,
            "gain_over_gmp_full_db_from_frozen_metric": gain_gmp_full,
            "gain_over_gmp_common_db_from_frozen_metric": gain_gmp_common,
            "research_gate_checks": {
                "cheap_pareto_full": loss_mp_full
                <= float(gates["cheap_pareto_max_full_loss_vs_mp_db"]),
                "cheap_pareto_common": loss_mp_common
                <= float(gates["cheap_pareto_max_common_loss_vs_mp_db"]),
                "evaluator_full": gain_gmp_full
                >= float(gates["evaluator_min_full_gain_over_gmp_db"]),
                "evaluator_common": gain_gmp_common
                >= float(gates["evaluator_min_common_gain_over_gmp_db"]),
                "evaluator_every_fold_full": float(
                    record["minimum_fold_gain_over_gmp_full_db"]
                )
                >= float(gates["evaluator_minimum_fold_gain_over_gmp_db"]),
                "evaluator_every_fold_common": float(
                    record["minimum_fold_gain_over_gmp_common_db"]
                )
                >= float(gates["evaluator_minimum_fold_gain_over_gmp_db"]),
            },
        }
    )


def run_staged_search(
    config: dict[str, Any],
    input_segments: tuple[np.ndarray, ...],
    output_segments: tuple[np.ndarray, ...],
    reference_gmp_prediction: np.ndarray,
    *,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Run the exact preregistered S0/S1/S2 train-OOF search."""

    budget_summary = validate_search_budget(config)
    families = branch_families(config)
    lengths = tuple(
        int(value) for value in config["dataset_contract"]["frame_lengths"]
    )
    warmup = int(
        config["dataset_contract"]["common_warmup_samples_per_frame"]
    )
    ranking_tolerance = float(config["search"]["ranking"]["tie_tolerance_db"])
    cache: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    completed_fit_calls = 0

    def evaluate(
        recipe: SparseRecipe,
        stage: str,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        nonlocal cache_hits, completed_fit_calls
        if recipe.sha256 in cache:
            cache_hits += 1
            progress(f"[{stage} {index}/{total}] cache {recipe.name}")
            return cache[recipe.sha256]
        progress(f"[{stage} {index}/{total}] fit {recipe.name}")
        row = evaluate_recipe_oof(
            recipe,
            input_segments,
            output_segments,
            reference_gmp_prediction,
            frame_lengths=lengths,
            common_warmup=warmup,
            gates=config["gates"],
        )
        _annotate_research_gates(row, config)
        cache[recipe.sha256] = row
        completed_fit_calls += len(lengths)
        progress(
            f"[{stage} {index}/{total}] NMSE={row['full_record_nmse_db']:.6f} dB "
            f"common={row['common_interior_nmse_db']:.6f} dB "
            f"valid={row['hard_valid']}"
        )
        return row

    s0_config = config["search"]["stage_s0_topology_screen"]
    s0_recipes = [
        SparseRecipe(
            family,
            branches,
            int(s0_config["knot_count"]),
            float(s0_config["ridge"]),
        )
        for family, branches in families.items()
    ]
    s0_rows = [
        evaluate(recipe, "S0", index, len(s0_recipes))
        for index, recipe in enumerate(s0_recipes, start=1)
    ]
    retained_rows = retain_topologies(
        s0_rows,
        maximum=int(s0_config["retain_topologies"]),
        window_db=float(s0_config["retention_window_db"]),
    )
    retained_families = tuple(row["recipe"]["family"] for row in retained_rows)
    if len(set(retained_families)) != len(retained_families):
        raise RuntimeError("S0 retained duplicate sparse topologies")
    progress(f"[S0] retained {', '.join(retained_families)}")

    s1_config = config["search"]["stage_s1_knot_count"]
    s1_recipes = [
        SparseRecipe(
            family,
            families[family],
            int(knot_count),
            float(s1_config["ridge"]),
        )
        for family in retained_families
        for knot_count in s1_config["knot_counts"]
    ]
    s1_rows = [
        evaluate(recipe, "S1", index, len(s1_recipes))
        for index, recipe in enumerate(s1_recipes, start=1)
    ]
    s1_winner = rank_valid_records(
        s1_rows, tie_tolerance_db=ranking_tolerance
    )[0]
    winner_recipe = s1_winner["recipe"]
    progress(f"[S1] selected {winner_recipe['name']}")

    s2_config = config["search"]["stage_s2_ridge"]
    s2_recipes = [
        SparseRecipe(
            str(winner_recipe["family"]),
            tuple(tuple(pair) for pair in winner_recipe["branches"]),
            int(winner_recipe["knot_count"]),
            float(ridge),
        )
        for ridge in s2_config["ridges"]
    ]
    s2_rows = [
        evaluate(recipe, "S2", index, len(s2_recipes))
        for index, recipe in enumerate(s2_recipes, start=1)
    ]
    final_row = rank_valid_records(
        s2_rows, tie_tolerance_db=ranking_tolerance
    )[0]
    final_recipe = SparseRecipe(
        str(final_row["recipe"]["family"]),
        tuple(tuple(pair) for pair in final_row["recipe"]["branches"]),
        int(final_row["recipe"]["knot_count"]),
        float(final_row["recipe"]["ridge"]),
    )
    if final_recipe.sha256 != final_row["recipe_sha256"]:
        raise RuntimeError("final sparse recipe identity changed")

    hard = bool(final_row["hard_valid"])
    gate_checks = final_row["research_gate_checks"]
    cheap = hard and bool(gate_checks["cheap_pareto_full"]) and bool(
        gate_checks["cheap_pareto_common"]
    )
    evaluator = hard and all(
        bool(gate_checks[name])
        for name in (
            "evaluator_full",
            "evaluator_common",
            "evaluator_every_fold_full",
            "evaluator_every_fold_common",
        )
    )
    if cheap and evaluator:
        classification = "cheap_pareto_and_evaluator_candidate"
    elif cheap:
        classification = "cheap_pareto_only"
    elif evaluator:
        classification = "evaluator_candidate_only"
    else:
        classification = "neither_evaluator_nor_cheap_pareto"
    decision = {
        "classification": classification,
        "hard_valid": hard,
        "cheap_pareto_gate_passed": cheap,
        "evaluator_candidate_gate_passed": evaluator,
        "gate_checks": gate_checks,
        "gate_a_to_b_opened": False,
        "reason_gate_a_to_b_remains_closed": (
            "post-discovery internal train OOF and reused validation are not "
            "independent evaluator confirmation"
        ),
    }

    if len(cache) > int(config["search_budget"]["maximum_unique_recipes"]):
        raise RuntimeError("unique sparse recipe budget exceeded")
    if completed_fit_calls > int(
        config["search_budget"]["maximum_oof_fit_calls_without_cache"]
    ):
        raise RuntimeError("sparse OOF fit-call budget exceeded")
    progress(f"[S2] selected {final_recipe.name}; decision={classification}")
    return {
        "final_recipe": final_recipe,
        "final_trial": final_row,
        "decision": decision,
        "stage_results": {"S0": s0_rows, "S1": s1_rows, "S2": s2_rows},
        "retained_families": retained_families,
        "cache": cache,
        "cache_hits": cache_hits,
        "unique_recipe_evaluations": len(cache),
        "completed_oof_fit_calls": completed_fit_calls,
        "stage_recipe_associations": len(s0_rows) + len(s1_rows) + len(s2_rows),
        "budget_summary": budget_summary,
    }
