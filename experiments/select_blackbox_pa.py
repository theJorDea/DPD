"""Select a frozen MP/GMP PA model from a BlackBox selection view.

Only the five files named by ``selection/selection_view.json`` are opened.
The command has no release evaluator and never walks or inspects the parent
prepared-data directory.  It therefore cannot accidentally use the sealed
holdout while choosing an architecture or regularization value.

The capture metadata does not define frames.  Train and validation are each
treated as one independent chronological record, with model state reset once
at the start of that record.  Integer delay is estimated on normalized train
data and then frozen.  The fractional-delay estimate is diagnostic only.
Neither a post-prediction delay nor a complex gain is fitted while scoring.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

from baseline.alignment import (
    estimate_integer_delay,
    fractional_delay_diagnostic,
    overlap_for_delay,
)
from baseline.complexity import memory_polynomial_inference_cost
from baseline.gmp_pa import (
    GMPConfig,
    GeneralizedMemoryPolynomialPA,
    fit_gmp_pa,
)
from baseline.metrics import nmse_pooled_db
from baseline.pa_models import (
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
)
from baseline.train_spline import (
    file_sha256,
    load_complex_iq_csv,
    write_json,
)


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_FILES = (
    "train_input.csv",
    "train_output.csv",
    "val_input.csv",
    "val_output.csv",
    "spec.json",
)
OWNED_OUTPUTS = (
    "selected_pa.npz",
    "selection_manifest.json",
    "validation_trials.json",
    "completion_manifest.json",
)
CONFIG_KEYS = {
    "schema_version",
    "selection_dir",
    "output_dir",
    "dataset_label",
    "alignment_max_abs_delay",
    "ridge_values",
    "mp_candidates",
    "gmp_candidates",
    "maximum_fit_count",
    "expected_source_sha256",
    "expected_selection_view_sha256",
    "practical_ridge_tie_db",
}
MP_CANDIDATE_KEYS = {"name", "orders", "delay_count"}
GMP_CANDIDATE_KEYS = {
    "name",
    "ka",
    "la",
    "kb",
    "lb",
    "mb",
    "kc",
    "lc",
    "mc",
    "leading_policy",
}
SELECTION_VIEW_KEYS = {
    "schema_version",
    "artifact_type",
    "generator",
    "source_filename",
    "source_sha256",
    "available_splits",
    "test_split_available",
    "test_path_or_hash_included",
    "split_contract",
    "normalization_contract",
    "semantics",
    "missing_metadata",
    "files_sha256",
}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_json_object(path, label="config")
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("config schema_version must equal 1")
    required = CONFIG_KEYS - {"schema_version"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"config has unknown keys: {sorted(unknown)}")
    if not isinstance(config["dataset_label"], str) or not config[
        "dataset_label"
    ].strip():
        raise ValueError("dataset_label must be a non-empty string")
    _strict_sha256(
        config["expected_source_sha256"],
        name="expected_source_sha256",
    )
    _strict_sha256(
        config["expected_selection_view_sha256"],
        name="expected_selection_view_sha256",
    )
    tie_db = float(config["practical_ridge_tie_db"])
    if not np.isfinite(tie_db) or tie_db < 0.0:
        raise ValueError("practical_ridge_tie_db must be finite and non-negative")
    return config


def _resolve_project_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _manifest_path(path: Path) -> str:
    """Use repository-relative paths whenever the artifact is in the project."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # External absolute paths are needed by isolated tests and deliberate
        # out-of-repository captures.  Production config/artifacts are rooted
        # in PROJECT_ROOT and therefore always take the branch above.
        return str(resolved)


def _reject_symlink_components(path: Path, *, name: str) -> None:
    """Reject a configured path if any existing component is a symlink."""

    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not contain symlink components")


def _strict_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.lower()


def _strict_integer(value: Any, *, name: str, minimum: int) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _integer_tuple(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON list")
    result = tuple(
        _strict_integer(item, name=f"{name} entry", minimum=minimum)
        for item in value
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _ridge_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("ridge_values must be a non-empty JSON list")
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError("ridge_values must not contain booleans")
    result = tuple(float(item) for item in value)
    if any(not np.isfinite(item) or item < 0.0 for item in result):
        raise ValueError("ridge_values must be finite and non-negative")
    if len(set(result)) != len(result):
        raise ValueError("ridge_values must be unique")
    return result


def _named_candidates(value: Any, *, family: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{family}_candidates must be a JSON list")
    result: list[dict[str, Any]] = []
    allowed = MP_CANDIDATE_KEYS if family == "mp" else GMP_CANDIDATE_KEYS
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError(f"every {family} candidate must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"every {family} candidate needs a non-empty name")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"{family} candidate {name!r} has unknown keys: {sorted(unknown)}"
            )
        result.append({**raw, "name": name.strip()})
    return tuple(result)


def _gmp_config(raw: dict[str, Any]) -> GMPConfig:
    dimensions = {
        key: _strict_integer(
            raw.get(key, 0),
            name=f"{raw['name']}.{key}",
            minimum=(1 if key in {"ka", "la"} else 0),
        )
        for key in ("ka", "la", "kb", "lb", "mb", "kc", "lc", "mc")
    }
    policy = raw.get("leading_policy", "causal_leading")
    if policy != "causal_leading":
        raise ValueError(
            "BlackBox selection permits causal_leading GMP candidates only"
        )
    return GMPConfig(**dimensions, leading_policy=policy)


def enumerate_candidate_recipes(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the complete deterministic model-by-ridge fit ledger."""

    ridges = _ridge_tuple(config["ridge_values"])
    mp_candidates = _named_candidates(config["mp_candidates"], family="mp")
    gmp_candidates = _named_candidates(config["gmp_candidates"], family="gmp")
    if not mp_candidates or not gmp_candidates:
        raise ValueError("at least one MP and one GMP candidate are required")

    names = [item["name"] for item in (*mp_candidates, *gmp_candidates)]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique across MP and GMP")

    architectures: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for raw in mp_candidates:
        orders = _integer_tuple(
            raw.get("orders"),
            name=f"{raw['name']}.orders",
            minimum=1,
        )
        delay_count = _strict_integer(
            raw.get("delay_count"),
            name=f"{raw['name']}.delay_count",
            minimum=1,
        )
        delays = tuple(range(delay_count))
        identity = ("mp", orders, delays)
        if identity in identities:
            raise ValueError("duplicate MP architecture")
        identities.add(identity)
        architectures.append(
            {
                "name": raw["name"],
                "family": "mp",
                "orders": orders,
                "delays": delays,
                "causal_warmup_samples": max(delays),
                "operation_count": memory_polynomial_inference_cost(
                    orders,
                    delays,
                ),
            }
        )
    for raw in gmp_candidates:
        model_config = _gmp_config(raw)
        identity = ("gmp", model_config)
        if identity in identities:
            raise ValueError("duplicate GMP architecture")
        identities.add(identity)
        empty_model = GeneralizedMemoryPolynomialPA(
            model_config,
            np.zeros(model_config.coefficient_count, dtype=np.complex128),
        )
        architectures.append(
            {
                "name": raw["name"],
                "family": "gmp",
                "gmp_config": model_config,
                "causal_warmup_samples": (
                    model_config.causal_warmup_samples
                ),
                "operation_count": empty_model.operation_count,
            }
        )

    recipes = [
        {**architecture, "ridge": ridge}
        for architecture in architectures
        for ridge in ridges
    ]
    maximum_fit_count = _strict_integer(
        config["maximum_fit_count"],
        name="maximum_fit_count",
        minimum=1,
    )
    if len(recipes) > maximum_fit_count:
        raise ValueError(
            f"candidate grid has {len(recipes)} fits, exceeding "
            f"maximum_fit_count={maximum_fit_count}"
        )
    return recipes


def _validated_split_contract(view: dict[str, Any]) -> dict[str, dict[str, int]]:
    contract = view.get("split_contract")
    if not isinstance(contract, dict):
        raise ValueError("selection_view split_contract must be an object")
    if set(contract) != {"indexing", "train", "validation"}:
        raise ValueError(
            "selection_view split_contract must contain indexing/train/validation"
        )
    if contract["indexing"] != "zero_based_half_open":
        raise ValueError("selection_view indexing must be zero_based_half_open")
    validated: dict[str, dict[str, int]] = {}
    for name in ("train", "validation"):
        raw = contract[name]
        if not isinstance(raw, dict) or set(raw) != {"start", "stop", "count"}:
            raise ValueError(
                f"selection_view {name} range must contain start/stop/count"
            )
        start = _strict_integer(raw["start"], name=f"{name}.start", minimum=0)
        stop = _strict_integer(raw["stop"], name=f"{name}.stop", minimum=1)
        count = _strict_integer(raw["count"], name=f"{name}.count", minimum=1)
        if stop <= start or count != stop - start:
            raise ValueError(
                f"selection_view {name} must satisfy count == stop - start > 0"
            )
        validated[name] = {"start": start, "stop": stop, "count": count}
    if validated["train"]["stop"] != validated["validation"]["start"]:
        raise ValueError("train.stop must equal validation.start")
    return validated


def _verify_selection_view(selection_dir: Path) -> dict[str, Any]:
    """Verify the prepared-data selection contract without directory walking."""

    if not selection_dir.is_dir():
        raise FileNotFoundError(selection_dir)
    view_path = selection_dir / "selection_view.json"
    if view_path.is_symlink() or view_path.resolve().parent != selection_dir:
        raise ValueError("selection_view must be a contained regular file")
    view = _load_json_object(view_path, label="selection_view")
    if set(view) != SELECTION_VIEW_KEYS:
        missing = sorted(SELECTION_VIEW_KEYS - set(view))
        unknown = sorted(set(view) - SELECTION_VIEW_KEYS)
        raise ValueError(
            f"selection_view top-level schema mismatch: missing={missing}, "
            f"unknown={unknown}"
        )
    if int(view.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("selection_view schema_version must equal 1")
    if view.get("artifact_type") != "blackbox_selection_view":
        raise ValueError("unexpected selection_view artifact_type")
    if view.get("available_splits") != ["train", "validation"]:
        raise ValueError("selection_view must expose train and validation only")
    if view.get("test_split_available") is not False:
        raise ValueError("selection_view must not expose the sealed split")
    if view.get("test_path_or_hash_included") is not False:
        raise ValueError("selection_view must not include sealed paths or hashes")
    source_filename = view.get("source_filename")
    source_sha256 = view.get("source_sha256")
    if (
        not isinstance(source_filename, str)
        or not source_filename
        or Path(source_filename).name != source_filename
    ):
        raise ValueError("selection_view source_filename must be a basename")
    _strict_sha256(source_sha256, name="selection_view source_sha256")
    split_contract = _validated_split_contract(view)

    generator = view.get("generator")
    if not isinstance(generator, dict) or set(generator) != {
        "project_relative_path",
        "sha256",
    }:
        raise ValueError("selection_view generator binding is missing")
    if generator["project_relative_path"] != "experiments/prepare_blackbox_data.py":
        raise ValueError("unexpected selection_view generator path")
    generator_path = PROJECT_ROOT / generator["project_relative_path"]
    generator_sha256 = generator["sha256"]
    generator_sha256 = _strict_sha256(
        generator_sha256,
        name="selection_view generator SHA-256",
    )
    if file_sha256(generator_path) != generator_sha256:
        raise ValueError("selection_view generator hash mismatch")

    recorded_hashes = view.get("files_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("selection_view files_sha256 must be an object")
    if set(recorded_hashes) != set(SELECTION_FILES):
        raise ValueError(
            "selection_view files_sha256 must name exactly the selection files"
        )
    verified_hashes: dict[str, str] = {}
    for name in SELECTION_FILES:
        path = selection_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink() or path.resolve().parent != selection_dir:
            raise ValueError(f"selection file {name} must not escape its directory")
        expected = recorded_hashes[name]
        expected = _strict_sha256(expected, name=f"recorded SHA-256 for {name}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"selection-view hash mismatch for {name}")
        verified_hashes[name] = actual

    normalization = view.get("normalization_contract")
    if not isinstance(normalization, dict):
        raise ValueError("selection_view normalization_contract is missing")
    if normalization.get("csv_values_scaled") is not False:
        raise ValueError("selection CSV values must remain in source units")
    if normalization.get("scale_fitted_from") != "train_input_only":
        raise ValueError("normalization scale must be fitted from train input only")
    peak = float(normalization.get("training_input_peak", float("nan")))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("training_input_peak must be positive and finite")
    recommended_scale = float(
        normalization.get("recommended_common_scale_for_x_and_y", float("nan"))
    )
    if not np.isclose(recommended_scale, peak, rtol=0.0, atol=0.0):
        raise ValueError("recommended common scale must equal training_input_peak")
    semantics = view["semantics"]
    missing_metadata = view["missing_metadata"]
    if not isinstance(semantics, dict):
        raise ValueError("selection_view semantics must be an object")
    if (
        not isinstance(missing_metadata, list)
        or any(not isinstance(item, str) for item in missing_metadata)
        or len(set(missing_metadata)) != len(missing_metadata)
    ):
        raise ValueError("selection_view missing_metadata must be unique strings")
    return {
        "view": view,
        "view_path": view_path,
        "view_sha256": file_sha256(view_path),
        "verified_file_hashes": verified_hashes,
        "training_input_peak": peak,
        "split_contract": split_contract,
        "generator": generator,
        "semantics": semantics,
        "missing_metadata": missing_metadata,
    }


def _load_normalized_pairs(
    selection_dir: Path,
    *,
    scale: float,
    expected_counts: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    train_x_raw = load_complex_iq_csv(selection_dir / "train_input.csv")
    train_y_raw = load_complex_iq_csv(selection_dir / "train_output.csv")
    validation_x_raw = load_complex_iq_csv(selection_dir / "val_input.csv")
    validation_y_raw = load_complex_iq_csv(selection_dir / "val_output.csv")
    if train_x_raw.shape != train_y_raw.shape:
        raise ValueError("train input/output lengths differ")
    if validation_x_raw.shape != validation_y_raw.shape:
        raise ValueError("validation input/output lengths differ")
    if expected_counts is not None:
        actual_counts = {
            "train": int(train_x_raw.size),
            "validation": int(validation_x_raw.size),
        }
        for split, actual in actual_counts.items():
            expected = int(expected_counts[split]["count"])
            if actual != expected:
                raise ValueError(
                    f"{split} CSV row count {actual} does not match "
                    f"selection_view count {expected}"
                )
    recomputed_peak = float(np.max(np.abs(train_x_raw)))
    if not np.isclose(recomputed_peak, scale, rtol=1e-12, atol=0.0):
        raise ValueError(
            "selection_view training_input_peak does not match train_input.csv"
        )
    train_x = np.asarray(train_x_raw / scale, dtype=np.complex128)
    train_y = np.asarray(train_y_raw / scale, dtype=np.complex128)
    validation_x = np.asarray(validation_x_raw / scale, dtype=np.complex128)
    validation_y = np.asarray(validation_y_raw / scale, dtype=np.complex128)
    return train_x, train_y, validation_x, validation_y, {
        "common_train_only_scale": scale,
        "recomputed_raw_train_input_peak": recomputed_peak,
        "normalized_train_input_peak": float(np.max(np.abs(train_x))),
        "normalized_train_output_peak": float(np.max(np.abs(train_y))),
        "normalized_validation_input_peak": float(
            np.max(np.abs(validation_x))
        ),
        "normalized_validation_output_peak": float(
            np.max(np.abs(validation_y))
        ),
    }


def _pooled_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    warmup_samples: int,
) -> dict[str, float | int]:
    if prediction.shape != reference.shape or prediction.ndim != 1:
        raise ValueError("prediction and reference must be equal-length vectors")
    if warmup_samples < 0 or warmup_samples >= prediction.size:
        raise ValueError("warmup must leave at least one scored sample")
    estimate = prediction[warmup_samples:]
    target = reference[warmup_samples:]
    error_power = float(np.mean(np.abs(estimate - target) ** 2))
    reference_power = float(np.mean(np.abs(target) ** 2))
    if reference_power <= 0.0:
        raise ValueError("validation target must have positive power")
    return {
        "complex_nmse_pooled_db": nmse_pooled_db(estimate, target),
        "mse": error_power,
        "reference_power": reference_power,
        "relative_error_power": error_power / reference_power,
        "warmup_samples_at_record_start": int(warmup_samples),
        "scored_sample_count": int(estimate.size),
        "discarded_sample_count": int(warmup_samples),
    }


def _fit_recipe(
    recipe: dict[str, Any],
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    common_warmup_samples: int,
) -> tuple[dict[str, Any], MemoryPolynomialPA | GeneralizedMemoryPolynomialPA]:
    started = time.perf_counter()
    ridge = float(recipe["ridge"])
    if recipe["family"] == "mp":
        model, diagnostics = fit_memory_polynomial_pa(
            train_x,
            train_y,
            orders=recipe["orders"],
            delays=recipe["delays"],
            ridge=ridge,
            segment_length=None,
            coefficient_dtype=np.complex128,
        )
        model_specification: dict[str, Any] = {
            "orders": list(model.orders),
            "delays": list(model.delays),
            "complex_coefficient_count": model.stored_complex_coefficients,
        }
    elif recipe["family"] == "gmp":
        model, diagnostics = fit_gmp_pa(
            train_x,
            train_y,
            config=recipe["gmp_config"],
            ridge=ridge,
            segment_length=int(train_x.size),
            coefficient_dtype=np.complex128,
            solver_mode="ridge_lstsq",
            svd_rcond=None,
        )
        model_specification = {
            "gmp_config": dataclasses.asdict(model.config),
            "complex_coefficient_count": model.stored_complex_coefficients,
        }
    else:  # pragma: no cover - internal enumeration invariant
        raise RuntimeError(f"unsupported family: {recipe['family']}")
    fit_seconds = time.perf_counter() - started
    prediction = np.asarray(model.predict(validation_x), dtype=np.complex128)
    full_metrics = _pooled_metrics(
        prediction,
        validation_y,
        warmup_samples=0,
    )
    common_metrics = _pooled_metrics(
        prediction,
        validation_y,
        warmup_samples=common_warmup_samples,
    )
    trial = {
        "candidate_name": recipe["name"],
        "model_family": recipe["family"],
        "model_specification": model_specification,
        "ridge": ridge,
        "solver": "augmented_complex_lstsq",
        "fit_seconds": fit_seconds,
        "fit_diagnostics": dataclasses.asdict(diagnostics),
        "operation_count_per_complex_sample": recipe[
            "operation_count"
        ].to_dict(),
        "validation_full_record": full_metrics,
        "validation_common_warmup": common_metrics,
        "selection_metric_name": (
            "validation_common_warmup.complex_nmse_pooled_db"
        ),
        "selection_score_db": common_metrics["complex_nmse_pooled_db"],
        "score_transform": (
            "direct model prediction versus measured y; no post-fit delay or gain"
        ),
    }
    return trial, model


def _selection_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    score = float(trial["selection_score_db"])
    if not np.isfinite(score) and score != -np.inf:
        score = np.inf
    cost = trial["operation_count_per_complex_sample"]
    return (
        score,
        int(cost["real_multiplications"]),
        int(cost["real_additions"]),
        int(cost["nonlinear_operations"]),
        int(cost["lookups"]),
        int(cost["real_memory_reads"]),
        int(cost["real_memory_writes"]),
        int(cost["stored_real_coefficients"]),
        str(trial["model_family"]),
        str(trial["candidate_name"]),
        float(trial["ridge"]),
    )


def _select_topology_representatives(
    fitted: list[
        tuple[
            dict[str, Any],
            MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
        ]
    ],
    *,
    practical_ridge_tie_db: float,
) -> tuple[
    tuple[dict[str, Any], MemoryPolynomialPA | GeneralizedMemoryPolynomialPA],
    list[dict[str, Any]],
    tuple[dict[str, Any], MemoryPolynomialPA | GeneralizedMemoryPolynomialPA],
]:
    """Apply the preregistered stability-aware two-level selection rule."""

    if not fitted:
        raise ValueError("at least one fitted PA candidate is required")
    tie_db = float(practical_ridge_tie_db)
    if not np.isfinite(tie_db) or tie_db < 0.0:
        raise ValueError("practical_ridge_tie_db must be finite and non-negative")
    exact_winner = min(fitted, key=lambda item: _selection_key(item[0]))
    grouped: dict[
        tuple[str, str],
        list[
            tuple[
                dict[str, Any],
                MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
            ]
        ],
    ] = {}
    for item in fitted:
        trial = item[0]
        grouped.setdefault(
            (str(trial["model_family"]), str(trial["candidate_name"])),
            [],
        ).append(item)

    representative_records: list[dict[str, Any]] = []
    representatives: list[
        tuple[dict[str, Any], MemoryPolynomialPA | GeneralizedMemoryPolynomialPA]
    ] = []
    for (family, name), topology_fits in grouped.items():
        best_score = min(float(item[0]["selection_score_db"]) for item in topology_fits)
        threshold = best_score + tie_db
        eligible = [
            item
            for item in topology_fits
            if float(item[0]["selection_score_db"]) <= threshold
        ]
        # Prefer the strongest regularization within the practical NMSE tie;
        # lower NMSE resolves only a duplicate-ridge tie.
        representative = min(
            eligible,
            key=lambda item: (
                -float(item[0]["ridge"]),
                float(item[0]["selection_score_db"]),
                _selection_key(item[0]),
            ),
        )
        representatives.append(representative)
        representative_records.append(
            {
                "model_family": family,
                "candidate_name": name,
                "exact_best_nmse_db": best_score,
                "practical_threshold_nmse_db": threshold,
                "eligible_ridges": [
                    float(item[0]["ridge"])
                    for item in eligible
                ],
                "representative_trial": representative[0],
            }
        )
    selected = min(representatives, key=lambda item: _selection_key(item[0]))
    return exact_winner, representative_records, selected


def select_from_config(config_path: str | Path) -> dict[str, Any]:
    """Fit the declared train-only grid and freeze its validation winner."""

    raw_config_path = Path(config_path)
    source_config = (
        raw_config_path
        if raw_config_path.is_absolute()
        else PROJECT_ROOT / raw_config_path
    ).resolve()
    config = _load_config(source_config)
    configured_selection_path = Path(config["selection_dir"])
    _reject_symlink_components(
        configured_selection_path,
        name="selection_dir",
    )
    selection_dir = _resolve_project_path(
        config["selection_dir"],
        name="selection_dir",
    )
    output_dir = _resolve_project_path(config["output_dir"], name="output_dir")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )

    verification = _verify_selection_view(selection_dir)
    expected_source_sha256 = _strict_sha256(
        config["expected_source_sha256"],
        name="expected_source_sha256",
    )
    expected_view_sha256 = _strict_sha256(
        config["expected_selection_view_sha256"],
        name="expected_selection_view_sha256",
    )
    if verification["view"]["source_sha256"] != expected_source_sha256:
        raise ValueError("configured source SHA-256 does not match selection_view")
    if verification["view_sha256"] != expected_view_sha256:
        raise ValueError("configured selection_view SHA-256 does not match")
    recipes = enumerate_candidate_recipes(config)
    train_x, train_y, validation_x, validation_y, scaling = (
        _load_normalized_pairs(
            selection_dir,
            scale=float(verification["training_input_peak"]),
            expected_counts=verification["split_contract"],
        )
    )

    alignment_max_abs_delay = _strict_integer(
        config["alignment_max_abs_delay"],
        name="alignment_max_abs_delay",
        minimum=0,
    )
    integer_delay = estimate_integer_delay(
        train_x,
        train_y,
        alignment_max_abs_delay,
    )
    fractional = fractional_delay_diagnostic(
        train_x,
        train_y,
        alignment_max_abs_delay,
    )
    train_x, train_y = overlap_for_delay(train_x, train_y, integer_delay)
    validation_x, validation_y = overlap_for_delay(
        validation_x,
        validation_y,
        integer_delay,
    )
    common_warmup_samples = max(
        int(recipe["causal_warmup_samples"]) for recipe in recipes
    )
    if common_warmup_samples >= min(train_x.size, validation_x.size):
        raise ValueError("common warmup consumes an aligned record")
    for recipe in recipes:
        if recipe["family"] == "mp" and max(recipe["delays"]) >= train_x.size:
            raise ValueError("MP delay must be shorter than aligned train")
        if (
            recipe["family"] == "gmp"
            and recipe["gmp_config"].causal_warmup_samples >= train_x.size
        ):
            raise ValueError("GMP memory must be shorter than aligned train")

    trials: list[dict[str, Any]] = []
    fitted: list[
        tuple[dict[str, Any], MemoryPolynomialPA | GeneralizedMemoryPolynomialPA]
    ] = []
    for recipe in recipes:
        trial, model = _fit_recipe(
            recipe,
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            common_warmup_samples=common_warmup_samples,
        )
        trials.append(trial)
        fitted.append((trial, model))
    practical_ridge_tie_db = float(config["practical_ridge_tie_db"])
    exact_winner, topology_representatives, selected = (
        _select_topology_representatives(
            fitted,
            practical_ridge_tie_db=practical_ridge_tie_db,
        )
    )
    selected_trial, selected_model = selected

    ledger = {
        "schema_version": SCHEMA_VERSION,
        "task": "blackbox_forward_pa_identification_model_selection",
        "direction": "normalized aligned x -> PA model -> y_hat; compare with measured y",
        "selection_split": "validation",
        "selection_metric": (
            "pooled complex NMSE after one common record-start warmup"
        ),
        "common_warmup_samples_at_record_start": common_warmup_samples,
        "selection_policy": {
            "revision": "blackbox_pa_v2_stability_aware_ridge",
            "practical_ridge_tie_db": practical_ridge_tie_db,
            "exact_validation_winner": exact_winner[0],
            "topology_representatives": topology_representatives,
            "selected_representative": selected_trial,
        },
        "trials": trials,
    }

    source_path = Path(__file__).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        staged_model = stage_dir / "selected_pa.npz"
        staged_trials = stage_dir / "validation_trials.json"
        staged_manifest = stage_dir / "selection_manifest.json"
        staged_completion = stage_dir / "completion_manifest.json"
        selected_model.save(staged_model)
        write_json(staged_trials, ledger)

        final_model = output_dir / "selected_pa.npz"
        final_trials = output_dir / "validation_trials.json"
        final_manifest = output_dir / "selection_manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "task": "blackbox_forward_pa_identification_model_selection",
            "scope": "train_validation_selection_only",
            "dataset_label": str(config["dataset_label"]),
            "direction": (
                "normalized aligned x -> PA model -> y_hat; compare with measured y"
            ),
            "data_provenance": {
                "selection_directory": _manifest_path(selection_dir),
                "selection_view": _manifest_path(verification["view_path"]),
                "selection_view_sha256": verification["view_sha256"],
                "configured_expected_selection_view_sha256": (
                    expected_view_sha256
                ),
                "selection_generator": verification["generator"],
                "source_filename": verification["view"].get("source_filename"),
                "source_sha256": verification["view"].get("source_sha256"),
                "configured_expected_source_sha256": expected_source_sha256,
                "semantics": verification["semantics"],
                "missing_metadata": verification["missing_metadata"],
                "split_contract": verification["split_contract"],
                "verified_selection_files_sha256": verification[
                    "verified_file_hashes"
                ],
            },
            "normalization": {
                **scaling,
                "policy": "divide x and y by one common train-input peak",
                "scale_source": "selection_view train_input_only contract",
                "validation_refit": False,
            },
            "alignment": {
                "integer_delay_samples": integer_delay,
                "integer_delay_definition": (
                    "maximum normalized complex-correlation power on normalized train only"
                ),
                "search_maximum_absolute_delay_samples": alignment_max_abs_delay,
                "fractional_delay_diagnostic": fractional._asdict(),
                "fractional_delay_applied": False,
                "validation_delay_refit": False,
                "post_prediction_delay_fit": False,
                "post_prediction_gain_fit": False,
            },
            "model_loading_contract": {
                "model_npz_is_not_standalone": True,
                "required_loader": (
                    "experiments.select_blackbox_pa."
                    "load_frozen_blackbox_pa_selection"
                ),
                "input_transform": (
                    "x_normalized = x_source_units / common_train_only_scale"
                ),
                "output_transform": (
                    "y_source_hat = y_normalized_hat * common_train_only_scale"
                ),
                "timing_contract": (
                    "integer_delay_samples maps input time to observed-output time; "
                    "it must accompany the model and is not absorbed into coefficients"
                ),
            },
            "sequence_contract": {
                "train": "one independent chronological record",
                "validation": "one independent chronological record",
                "state_reset": "once at each record start",
                "frame_length_invented": False,
                "common_warmup_samples_at_record_start": common_warmup_samples,
            },
            "candidate_grid": {
                "fit_count": len(recipes),
                "maximum_fit_count": int(config["maximum_fit_count"]),
                "model_families": ["complex_memory_polynomial", "complex_gmp"],
                "candidate_iteration_order": (
                    "MP config order, then GMP config order; ridge order inside "
                    "each architecture"
                ),
                "pa_deployment_operation_limit_applied": False,
                "reason": "PA model is an auxiliary evaluator, not the deployed DPD",
            },
            "selection": {
                "split": "validation",
                "metric": selected_trial["selection_metric_name"],
                "metric_policy": (
                    "within each topology retain the largest ridge no worse than "
                    "its exact best plus practical_ridge_tie_db; then choose the "
                    "representative with minimum NMSE; analytical cost breaks only "
                    "an exact NMSE tie"
                ),
                "protocol_revision": "blackbox_pa_v2_stability_aware_ridge",
                "practical_ridge_tie_db": practical_ridge_tie_db,
                "exact_validation_winner": exact_winner[0],
                "topology_representatives": topology_representatives,
                "no_post_fit_alignment_or_gain": True,
                "selected_trial": selected_trial,
            },
            "selected_model": {
                "model_family": selected_trial["model_family"],
                "path": _manifest_path(final_model),
                "sha256": file_sha256(staged_model),
            },
            "artifacts": {
                "validation_trials": _manifest_path(final_trials),
                "validation_trials_sha256": file_sha256(staged_trials),
                "selection_manifest": _manifest_path(final_manifest),
            },
            "config": {
                "path": _manifest_path(source_config),
                "sha256": file_sha256(source_config),
            },
            "code_sha256": {
                "experiments/select_blackbox_pa.py": file_sha256(source_path),
                "experiments/prepare_blackbox_data.py": file_sha256(
                    PROJECT_ROOT / "experiments" / "prepare_blackbox_data.py"
                ),
                "baseline/alignment.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "alignment.py"
                ),
                "baseline/pa_models.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "pa_models.py"
                ),
                "baseline/gmp_pa.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "gmp_pa.py"
                ),
                "baseline/complexity.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "complexity.py"
                ),
                "baseline/metrics.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "metrics.py"
                ),
            },
            "determinism": {
                "stochastic_fitting": False,
                "seed": None,
                "coefficient_dtype": "complex128",
                "solver": "numpy augmented complex least squares",
            },
            "unavailable_metrics": {
                "spectral_metrics": (
                    "not computed because sample rate and RF regions are unknown"
                ),
                "frame_mean_nmse": (
                    "not computed because frame boundaries are unknown"
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        }
        write_json(staged_manifest, manifest)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "blackbox_pa_selection_completion",
            "status": "complete",
            "bound_files_sha256": {
                "selection_manifest.json": file_sha256(staged_manifest),
                "selected_pa.npz": file_sha256(staged_model),
                "validation_trials.json": file_sha256(staged_trials),
            },
            "publication_contract": (
                "the bundle is valid only when every bound hash verifies"
            ),
        }
        write_json(staged_completion, completion)
        load_frozen_blackbox_pa_selection(stage_dir)
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to replace concurrently created output: {output_dir}"
            )
        os.replace(stage_dir, output_dir)
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return manifest


@dataclasses.dataclass(frozen=True)
class FrozenBlackBoxPASelection:
    model: MemoryPolynomialPA | GeneralizedMemoryPolynomialPA
    normalization_scale: float
    integer_delay_samples: int
    manifest: dict[str, Any]

    def predict_aligned_source_units(self, pa_input: np.ndarray) -> np.ndarray:
        """Predict aligned PA output while applying the frozen scale exactly.

        This method does not invent or hide a time shift.  ``pa_input`` must
        already use the input-time coordinate returned by
        :meth:`align_measured_pair` when it is compared with a measurement.
        """

        values = np.asarray(pa_input)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("pa_input must be a non-empty one-dimensional record")
        values = np.asarray(values, dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise ValueError("pa_input contains non-finite values")
        normalized_prediction = self.model.predict(
            values / self.normalization_scale
        )
        return np.asarray(
            normalized_prediction * self.normalization_scale,
            dtype=np.complex128,
        )

    def align_measured_pair(
        self,
        pa_input: np.ndarray,
        measured_pa_output: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the frozen train-only delay to a source-unit measured pair."""

        return overlap_for_delay(
            pa_input,
            measured_pa_output,
            self.integer_delay_samples,
        )


def load_frozen_blackbox_pa_selection(
    output_dir: str | Path,
) -> FrozenBlackBoxPASelection:
    """Verify all completion bindings before loading model/scale/delay."""

    bundle = _resolve_project_path(output_dir, name="output_dir")
    completion_path = bundle / "completion_manifest.json"
    completion = _load_json_object(completion_path, label="completion_manifest")
    if (
        int(completion.get("schema_version", -1)) != SCHEMA_VERSION
        or completion.get("artifact_type") != "blackbox_pa_selection_completion"
        or completion.get("status") != "complete"
    ):
        raise ValueError("invalid or incomplete BlackBox PA selection bundle")
    bindings = completion.get("bound_files_sha256")
    expected_names = {
        "selection_manifest.json",
        "selected_pa.npz",
        "validation_trials.json",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_names:
        raise ValueError("completion manifest has incomplete file bindings")
    for name in sorted(expected_names):
        recorded = bindings[name]
        if not isinstance(recorded, str) or len(recorded) != 64:
            raise ValueError(f"invalid completion hash for {name}")
        if file_sha256(bundle / name) != recorded:
            raise ValueError(f"completion hash mismatch for {name}")

    manifest = _load_json_object(
        bundle / "selection_manifest.json",
        label="selection_manifest",
    )
    model_record = manifest.get("selected_model")
    artifacts = manifest.get("artifacts")
    if not isinstance(model_record, dict) or not isinstance(artifacts, dict):
        raise ValueError("selection manifest lacks artifact bindings")
    if model_record.get("sha256") != bindings["selected_pa.npz"]:
        raise ValueError("selection/completion model hash disagreement")
    if (
        artifacts.get("validation_trials_sha256")
        != bindings["validation_trials.json"]
    ):
        raise ValueError("selection/completion trial hash disagreement")
    if Path(str(model_record.get("path"))).name != "selected_pa.npz":
        raise ValueError("selection manifest model path is inconsistent")

    family = model_record.get("model_family")
    if family == "mp":
        model: MemoryPolynomialPA | GeneralizedMemoryPolynomialPA = (
            MemoryPolynomialPA.load(bundle / "selected_pa.npz")
        )
    elif family == "gmp":
        model = GeneralizedMemoryPolynomialPA.load(bundle / "selected_pa.npz")
    else:
        raise ValueError(f"unsupported selected model family: {family!r}")
    selected_trial = manifest.get("selection", {}).get("selected_trial", {})
    if selected_trial.get("model_family") != family:
        raise ValueError("selected trial/model family disagreement")

    scale = float(
        manifest.get("normalization", {}).get(
            "common_train_only_scale",
            float("nan"),
        )
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("bundle normalization scale is invalid")
    delay = manifest.get("alignment", {}).get("integer_delay_samples")
    # Negative delays are valid under the project's alignment convention.
    if not isinstance(delay, (int, np.integer)) or isinstance(
        delay,
        (bool, np.bool_),
    ):
        raise ValueError("bundle integer delay is invalid")
    delay = int(delay)
    return FrozenBlackBoxPASelection(model, scale, delay, manifest)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select and freeze an MP/GMP BlackBox PA model using the verified "
            "train/validation selection view only."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = select_from_config(args.config)
    selected = manifest["selection"]["selected_trial"]
    print(
        "Selected BlackBox PA:",
        f"family={selected['model_family']}",
        f"candidate={selected['candidate_name']}",
        f"ridge={selected['ridge']}",
        f"validation_common_warmup_nmse_db={selected['selection_score_db']:.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
