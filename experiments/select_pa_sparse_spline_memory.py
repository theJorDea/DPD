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
from typing import Any, Iterable

import numpy as np

from baseline.complexity import OperationCount
from baseline.metrics import nmse_opendpd_db, nmse_pooled_db
from baseline.sparse_spline_memory_pa import (
    SparseSplineMemoryPA,
    SparseSplineMemoryPAFitDiagnostics,
    SparseSplineMemoryPABranch,
    fit_sparse_spline_memory_pa_segments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TASK = "forward_pa_non_factorized_sparse_spline_memory_selection"


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
    if config.get("status") != (
        "preregistered_before_model_implementation_and_candidate_fit"
    ):
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
