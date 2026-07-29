"""Generate integrity-gated train-OOF/validation residual diagnostics.

The selected MP or GMP recipe is read from a hash-bound validation-selection
manifest and checked against the frozen model before waveform access.  For
discovery, coefficients are refit in leave-one-explicit-frame-out folds inside
the training split.  The validation residual uses the verified full-training
model.  It is not independent model-selection evidence because validation
already selected the recipe.  Test files are never opened or hashed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any, Literal

import numpy as np

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
    segmented_steady_state_mask,
)
from baseline.residual_analysis import (
    ResidualAnalysisSpec,
    analyze_pa_residuals,
    freeze_residual_reference,
)
from baseline.train_spline import (
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


ModelKind = Literal["mp", "gmp"]
PAModel = MemoryPolynomialPA | GeneralizedMemoryPolynomialPA
REAL_MULTIPLICATION_LIMIT = 1000
_OPERATION_NUMERIC_FIELDS = (
    "real_multiplications",
    "real_additions",
    "real_divisions",
    "nonlinear_operations",
    "comparisons",
    "lookups",
    "real_memory_reads",
    "real_memory_writes",
    "stored_real_coefficients",
    "stored_real_constants",
    "state_real_values",
)


@dataclasses.dataclass(frozen=True, slots=True)
class SelectedMPRecipe:
    orders: tuple[int, ...]
    delays: tuple[int, ...]
    ridge: float
    solver_mode: str = "augmented_complex_lstsq"

    @property
    def kind(self) -> ModelKind:
        return "mp"

    @property
    def causal_warmup_samples(self) -> int:
        return max(self.delays)

    @property
    def lookahead_samples(self) -> int:
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "model_class": "complex_memory_polynomial",
            "orders": list(self.orders),
            "delays": list(self.delays),
            "ridge": self.ridge,
            "solver_mode": self.solver_mode,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SelectedGMPRecipe:
    config: GMPConfig
    ridge: float
    solver_mode: Literal["ridge_lstsq", "truncated_svd"]
    svd_rcond: float | None

    @property
    def kind(self) -> ModelKind:
        return "gmp"

    @property
    def causal_warmup_samples(self) -> int:
        return self.config.causal_warmup_samples

    @property
    def lookahead_samples(self) -> int:
        return self.config.lookahead_samples

    def to_dict(self) -> dict[str, object]:
        return {
            "model_class": "complex_generalized_memory_polynomial",
            "gmp_config": dataclasses.asdict(self.config),
            "ridge": self.ridge,
            "solver_mode": self.solver_mode,
            "svd_rcond": self.svd_rcond,
        }


SelectedRecipe = SelectedMPRecipe | SelectedGMPRecipe


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    value = _load_json_object(path, name="residual config")
    if value.get("schema_version") != 2:
        raise ValueError("residual config schema_version must equal 2")
    required = {
        "dataset",
        "selection_manifest",
        "selection_manifest_sha256",
        "selection_config",
        "selection_config_sha256",
        "output_dir",
        "lag_grid",
        "expected_common_warmup_samples_per_frame",
        "expected_common_future_cooldown_samples_per_frame",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"residual config is missing keys: {sorted(missing)}")
    return value


def _verify_file(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )


def _snapshot_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {label: file_sha256(path) for label, path in paths.items()}


def _verify_frozen_hashes(
    paths: dict[str, Path],
    expected: dict[str, str],
    *,
    scope: str,
) -> None:
    if set(paths) != set(expected):
        raise ValueError(f"{scope} path/hash labels differ")
    for label, path in paths.items():
        _verify_file(
            path,
            expected[label],
            label=f"{scope} {label}",
        )


def _path_entry_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _resolve_project_relative(
    value: Any,
    *,
    project_root: Path,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty project-relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{name} must not be an absolute path")
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{name} must stay inside the project root") from error
    return resolved


def _acquire_bundle_lock(lock_path: Path) -> bytes:
    payload = (
        json.dumps(
            {
                "pid": os.getpid(),
                "token": secrets.token_hex(32),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"residual-analysis bundle lock already exists: {lock_path}"
        ) from error
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("bundle lock write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _verify_owned_lock(
    lock_path: Path,
    payload: bytes,
    *,
    scope: str,
) -> None:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError(f"{scope}: residual bundle lock is missing/replaced")
    if lock_path.read_bytes() != payload:
        raise RuntimeError(f"{scope}: residual bundle lock owner changed")


def _atomic_write_json(
    final_path: Path,
    temporary_path: Path,
    value: dict[str, Any],
) -> None:
    if _path_entry_exists(temporary_path):
        raise FileExistsError(f"JSON temp already exists: {temporary_path}")
    if _path_entry_exists(final_path):
        raise FileExistsError(f"immutable JSON already exists: {final_path}")
    write_json(temporary_path, value)
    if _path_entry_exists(final_path):
        raise FileExistsError(
            f"immutable JSON appeared during publication: {final_path}"
        )
    os.replace(temporary_path, final_path)


def _atomic_write_predictions(
    final_path: Path,
    temporary_path: Path,
    **arrays: np.ndarray,
) -> None:
    if temporary_path.suffix != ".npz":
        raise ValueError("prediction temp path must end in .npz")
    if _path_entry_exists(temporary_path):
        raise FileExistsError(f"NPZ temp already exists: {temporary_path}")
    if _path_entry_exists(final_path):
        raise FileExistsError(f"immutable NPZ already exists: {final_path}")
    np.savez_compressed(temporary_path, **arrays)
    if _path_entry_exists(final_path):
        raise FileExistsError(
            f"immutable NPZ appeared during publication: {final_path}"
        )
    os.replace(temporary_path, final_path)


def _positive_integer_tuple(
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
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
        ):
            raise ValueError(
                f"every {name} value must be an integer >= {minimum}"
            )
        result.append(int(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} values must be unique")
    return tuple(result)


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _operation_count_dict(
    model: PAModel,
    recipe: SelectedRecipe,
) -> dict[str, object]:
    if isinstance(recipe, SelectedMPRecipe):
        return memory_polynomial_inference_cost(
            recipe.orders,
            recipe.delays,
        ).to_dict()
    if not isinstance(model, GeneralizedMemoryPolynomialPA):
        raise TypeError("GMP recipe/model dispatch mismatch")
    return model.operation_count.to_dict()


def _verify_operation_count(
    selected_trial: dict[str, Any],
    actual: dict[str, object],
    *,
    kind: ModelKind,
    selection: dict[str, Any],
) -> dict[str, object]:
    recorded = selected_trial.get("operation_count_per_complex_sample")
    if not isinstance(recorded, dict):
        raise ValueError("selected trial has no operation-count object")
    mismatches: dict[str, dict[str, int]] = {}
    for field in _OPERATION_NUMERIC_FIELDS:
        raw = recorded.get(field)
        if (
            isinstance(raw, (bool, np.bool_))
            or not isinstance(raw, (int, np.integer))
        ):
            raise ValueError(
                f"selected operation count {field} must be an integer"
            )
        expected_value = int(actual[field])
        recorded_value = int(raw)
        if recorded_value != expected_value:
            mismatches[field] = {
                "manifest": recorded_value,
                "recomputed": expected_value,
            }

    legacy_mp_state_exception = (
        kind == "mp"
        and set(mismatches) <= {"state_real_values"}
        and (
            not mismatches
            or (
                mismatches["state_real_values"]["manifest"] == 0
                and mismatches["state_real_values"]["recomputed"] > 0
            )
        )
    )
    if mismatches and not legacy_mp_state_exception:
        raise ValueError(
            "selected operation count disagrees with current implementation: "
            f"{mismatches}"
        )
    real_multiplications = int(actual["real_multiplications"])
    if real_multiplications >= REAL_MULTIPLICATION_LIMIT:
        raise ValueError(
            f"selected {kind.upper()} requires {real_multiplications} real "
            f"multiplications/sample; required <{REAL_MULTIPLICATION_LIMIT}"
        )

    budget = selection.get("operation_budget")
    if kind == "gmp":
        if not isinstance(budget, dict):
            raise ValueError("GMP selection manifest lacks operation_budget")
        if int(budget.get("maximum_exclusive", -1)) != (
            REAL_MULTIPLICATION_LIMIT
        ):
            raise ValueError("GMP operation budget must be exclusive 1000")
        if int(budget.get("selected_value", -1)) != real_multiplications:
            raise ValueError("GMP operation budget selected_value mismatch")

    return {
        "limit_exclusive": REAL_MULTIPLICATION_LIMIT,
        "recomputed": actual,
        "manifest_numeric_fields_verified": [
            field
            for field in _OPERATION_NUMERIC_FIELDS
            if field not in mismatches
        ],
        "legacy_mp_state_accounting_exception": (
            mismatches if legacy_mp_state_exception and mismatches else None
        ),
    }


def _verify_selection_source_integrity(
    selection: dict[str, Any],
    *,
    project_root: Path,
    kind: ModelKind,
) -> tuple[dict[str, object], dict[str, Path], dict[str, str]]:
    recorded = selection.get("source_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("selection manifest has no source_sha256 object")
    paths: dict[str, Path] = {}
    actual_hashes: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for raw_label, raw_hash in recorded.items():
        label = str(raw_label)
        candidate = Path(label)
        if candidate.is_absolute():
            raise ValueError("selection source labels must be project-relative")
        resolved = (project_root / candidate).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as error:
            raise ValueError(
                "selection source label escapes the project root"
            ) from error
        if not resolved.is_file():
            raise FileNotFoundError(
                f"selection source dependency does not exist: {resolved}"
            )
        actual = file_sha256(resolved)
        expected = str(raw_hash)
        paths[label] = resolved
        actual_hashes[label] = actual
        if actual != expected:
            mismatches[label] = {
                "selection_manifest": expected,
                "current_checkout": actual,
            }

    allowed_legacy_mp_mismatches = {
        "baseline/complexity.py",
        "experiments/select_pa_mp.py",
    }
    legacy_exception = (
        kind == "mp"
        and bool(mismatches)
        and set(mismatches) <= allowed_legacy_mp_mismatches
    )
    if mismatches and not legacy_exception:
        raise ValueError(
            "selection source SHA-256 mismatch: "
            f"{mismatches}"
        )
    return (
        {
            "status": (
                "verified_exact"
                if not mismatches
                else "legacy_mp_explicit_report_only_exception"
            ),
            "recorded_hashes": {
                str(key): str(value) for key, value in recorded.items()
            },
            "mismatches": mismatches,
            "legacy_exception_scope": (
                sorted(allowed_legacy_mp_mismatches)
                if legacy_exception
                else []
            ),
            "legacy_exception_rationale": (
                "legacy MP artifacts predate corrected delay-state accounting "
                "and later selector integrity changes; model/data/recipe and "
                "arithmetic counts remain independently verified"
                if legacy_exception
                else None
            ),
        },
        paths,
        actual_hashes,
    )


def _parse_and_verify_recipe(
    selection: dict[str, Any],
    model_path: Path,
) -> tuple[SelectedRecipe, PAModel, dict[str, object]]:
    if int(selection.get("schema_version", -1)) != 1:
        raise ValueError("unsupported selection manifest schema")
    if selection.get("task") != "forward_pa_identification_model_selection":
        raise ValueError("selection manifest is not forward PA selection")
    if selection.get("selection_split") != "validation":
        raise ValueError("selected recipe must have used validation")
    if selection.get("test_split_accessed") is not False:
        raise ValueError("selection manifest must certify sealed test data")
    if selection.get("test_evaluation_status") not in {
        None,
        "not_run_by_design",
    }:
        raise ValueError("selection manifest indicates test evaluation")

    selected = selection.get("selected_trial")
    if not isinstance(selected, dict):
        raise ValueError("selection manifest has no selected_trial")
    diagnostics = selected.get("fit_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("selected trial has no fit_diagnostics")

    model_class = selection.get("model_class")
    if model_class == "complex_memory_polynomial":
        recipe = SelectedMPRecipe(
            orders=_positive_integer_tuple(
                selected.get("orders"),
                name="MP orders",
                minimum=1,
            ),
            delays=_positive_integer_tuple(
                selected.get("delays"),
                name="MP delays",
                minimum=0,
            ),
            ridge=_finite_nonnegative(
                selected.get("ridge"),
                name="MP ridge",
            ),
        )
        if diagnostics.get("solver") != recipe.solver_mode:
            raise ValueError("MP solver metadata mismatch")
        if float(diagnostics.get("ridge", np.nan)) != recipe.ridge:
            raise ValueError("MP ridge metadata mismatch")
        model: PAModel = MemoryPolynomialPA.load(model_path)
        if (
            model.orders != recipe.orders
            or model.delays != recipe.delays
        ):
            raise ValueError("loaded MP model does not match selected recipe")
        if int(diagnostics.get("feature_count", -1)) != model.feature_count:
            raise ValueError("MP feature-count metadata mismatch")
        if int(diagnostics.get("causal_warmup_samples", -1)) != (
            recipe.causal_warmup_samples
        ):
            raise ValueError("MP warmup metadata mismatch")
    elif model_class == "complex_generalized_memory_polynomial":
        raw_config = selected.get("gmp_config")
        required = {
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
        if not isinstance(raw_config, dict) or set(raw_config) != required:
            raise ValueError("selected GMP gmp_config is incomplete/extra")
        gmp_config = GMPConfig(**raw_config)
        solver_mode = selected.get("solver_mode")
        if solver_mode not in {"ridge_lstsq", "truncated_svd"}:
            raise ValueError("selected GMP solver_mode is unsupported")
        ridge = _finite_nonnegative(
            selected.get("ridge"),
            name="GMP ridge",
        )
        raw_rcond = selected.get("svd_rcond")
        if solver_mode == "ridge_lstsq":
            if raw_rcond is not None:
                raise ValueError("ridge GMP must have svd_rcond=null")
            svd_rcond = None
        else:
            if ridge != 0.0:
                raise ValueError("truncated-SVD GMP must have ridge=0")
            if isinstance(raw_rcond, (bool, np.bool_)) or not isinstance(
                raw_rcond,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError("truncated-SVD GMP requires svd_rcond")
            svd_rcond = float(raw_rcond)
            if not np.isfinite(svd_rcond) or not 0.0 < svd_rcond < 1.0:
                raise ValueError("GMP svd_rcond must satisfy 0 < rcond < 1")
        recipe = SelectedGMPRecipe(
            config=gmp_config,
            ridge=ridge,
            solver_mode=solver_mode,
            svd_rcond=svd_rcond,
        )
        if diagnostics.get("solver_mode") != recipe.solver_mode:
            raise ValueError("GMP solver metadata mismatch")
        if float(diagnostics.get("ridge", np.nan)) != recipe.ridge:
            raise ValueError("GMP ridge metadata mismatch")
        if diagnostics.get("svd_rcond") != recipe.svd_rcond:
            raise ValueError("GMP svd_rcond metadata mismatch")
        model = GeneralizedMemoryPolynomialPA.load(model_path)
        if dataclasses.asdict(model.config) != dataclasses.asdict(
            recipe.config
        ):
            raise ValueError(
                "loaded GMP model gmp_config does not match selected recipe"
            )
        if int(diagnostics.get("feature_count", -1)) != (
            model.stored_complex_coefficients
        ):
            raise ValueError("GMP feature-count metadata mismatch")
        if int(diagnostics.get("causal_warmup_samples", -1)) != (
            recipe.causal_warmup_samples
        ):
            raise ValueError("GMP warmup metadata mismatch")
        if int(diagnostics.get("future_cooldown_samples", -1)) != (
            recipe.lookahead_samples
        ):
            raise ValueError("GMP cooldown metadata mismatch")
    else:
        raise ValueError(
            "selection model_class must be complex MP or complex GMP"
        )

    actual_operations = _operation_count_dict(model, recipe)
    operation_verification = _verify_operation_count(
        selected,
        actual_operations,
        kind=recipe.kind,
        selection=selection,
    )
    if int(actual_operations["stored_real_coefficients"]) != (
        model.stored_real_coefficients
    ):
        raise ValueError("operation-count coefficient storage mismatch")
    return recipe, model, operation_verification


def _parse_lag_grid(value: Any) -> tuple[int, ...]:
    if not isinstance(value, dict):
        raise ValueError("lag_grid must be a JSON object")
    start = int(value["start"])
    stop = int(value["stop"])
    step = int(value.get("step", 1))
    if step <= 0 or stop < start:
        raise ValueError("lag_grid requires positive step and stop >= start")
    lags = tuple(range(start, stop + 1, step))
    if not lags:
        raise ValueError("lag_grid generated no lags")
    return lags


def _number_tuple(
    values: Any,
    *,
    name: str,
    integer: bool,
) -> tuple[int, ...] | tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON list")
    if integer:
        result = tuple(int(value) for value in values)
        if any(value < 0 for value in result):
            raise ValueError(f"{name} must be non-negative")
    else:
        result = tuple(float(value) for value in values)
        if any(not np.isfinite(value) or value <= 0.0 for value in result):
            raise ValueError(f"{name} must be positive and finite")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _build_spec(
    config: dict[str, Any],
    dataset_spec: dict[str, Any],
) -> ResidualAnalysisSpec:
    return ResidualAnalysisSpec(
        sample_rate_hz=float(dataset_spec["input_signal_fs"]),
        psd_nperseg=int(dataset_spec["nperseg"]),
        main_bandwidth_hz=float(dataset_spec["bw_main_ch"]),
        adjacent_bandwidth_hz=float(dataset_spec["bw_sub_ch"]),
        lags=_parse_lag_grid(config["lag_grid"]),
        envelope_lags=_number_tuple(
            config.get(
                "envelope_lags",
                [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
            ),
            name="envelope_lags",
            integer=True,
        ),
        envelope_powers=tuple(
            int(value)
            for value in _number_tuple(
                config.get("envelope_powers", [1, 2, 3]),
                name="envelope_powers",
                integer=True,
            )
        ),
        slow_time_constants_samples=_number_tuple(
            config.get(
                "slow_time_constants_samples",
                [4.0, 16.0, 64.0, 256.0, 1024.0],
            ),
            name="slow_time_constants_samples",
            integer=False,
        ),
        amplitude_quantiles=tuple(
            float(value)
            for value in config.get("amplitude_quantiles", [0.90, 0.95, 0.99])
        ),
        characteristic_bins=int(config.get("characteristic_bins", 32)),
        position_bins=int(config.get("position_bins", 10)),
        amplitude_floor_fraction=float(
            config.get("amplitude_floor_fraction", 1e-6)
        ),
        minimum_time_constants_per_segment=float(
            config.get("minimum_time_constants_per_segment", 20.0)
        ),
        independent_capture_count=int(
            config.get("independent_capture_count", 0)
        ),
    )


def explicit_frame_ids(sample_count: int, nperseg: int) -> np.ndarray:
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if not isinstance(nperseg, int) or nperseg <= 0:
        raise ValueError("nperseg must be a positive integer")
    return np.arange(sample_count, dtype=int) // nperseg


def _segmented_interior_mask(
    sample_count: int,
    *,
    nperseg: int,
    warmup_samples: int,
    cooldown_samples: int,
) -> np.ndarray:
    if cooldown_samples < 0:
        raise ValueError("cooldown_samples must be non-negative")
    selected = segmented_steady_state_mask(
        sample_count,
        segment_length=nperseg,
        warmup_samples=warmup_samples,
    )
    if cooldown_samples:
        for start in range(0, sample_count, nperseg):
            stop = min(start + nperseg, sample_count)
            selected[max(start, stop - cooldown_samples) : stop] = False
    if not np.any(selected):
        raise ValueError("common warmup/cooldown consumes all residual samples")
    return selected


def _fit_selected_recipe(
    recipe: SelectedRecipe,
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    *,
    nperseg: int,
) -> tuple[PAModel, dict[str, Any]]:
    if isinstance(recipe, SelectedMPRecipe):
        model, diagnostics = fit_memory_polynomial_pa(
            pa_input,
            measured_output,
            orders=recipe.orders,
            delays=recipe.delays,
            ridge=recipe.ridge,
            segment_length=nperseg,
            coefficient_dtype=np.complex128,
        )
    else:
        model, diagnostics = fit_gmp_pa(
            pa_input,
            measured_output,
            config=recipe.config,
            ridge=recipe.ridge,
            segment_length=nperseg,
            coefficient_dtype=np.complex128,
            solver_mode=recipe.solver_mode,
            svd_rcond=recipe.svd_rcond,
        )
    return model, dataclasses.asdict(diagnostics)


def _leave_one_frame_out_predictions(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    *,
    segment_id: np.ndarray,
    recipe: SelectedRecipe,
    nperseg: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Refit only coefficients on all other explicit model-reset frames."""

    x = np.asarray(pa_input, dtype=np.complex128)
    y = np.asarray(measured_output, dtype=np.complex128)
    segments = np.asarray(segment_id)
    if x.ndim != 1 or y.ndim != 1 or segments.ndim != 1:
        raise ValueError("OOF input, output, and segment_id must be 1-D")
    if x.shape != y.shape or x.shape != segments.shape:
        raise ValueError("OOF arrays must have equal length")
    unique_segments = np.unique(segments)
    if unique_segments.size < 2:
        raise ValueError("OOF residuals require at least two explicit frames")

    prediction = np.empty(x.shape, dtype=np.complex128)
    fold_reports: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    for held_segment in unique_segments:
        held = segments == held_segment
        fit = ~held
        started = time.perf_counter()
        model, diagnostics = _fit_selected_recipe(
            recipe,
            x[fit],
            y[fit],
            nperseg=nperseg,
        )
        fit_seconds = time.perf_counter() - started
        total_fit_seconds += fit_seconds
        prediction[held] = model.predict(x[held])
        fit_maximum_amplitude = float(np.max(np.abs(x[fit])))
        held_maximum_amplitude = float(np.max(np.abs(x[held])))
        held_above_fit = np.abs(x[held]) > fit_maximum_amplitude
        condition_number = diagnostics.get(
            "scaled_augmented_condition_number",
            diagnostics.get("augmented_design_condition_number"),
        )
        fold_reports.append(
            {
                "held_segment_id": (
                    held_segment.item()
                    if hasattr(held_segment, "item")
                    else held_segment
                ),
                "fit_sample_count": int(np.count_nonzero(fit)),
                "held_sample_count": int(np.count_nonzero(held)),
                "fit_segment_ids": [
                    (
                        value.item()
                        if hasattr(value, "item")
                        else value
                    )
                    for value in unique_segments
                    if value != held_segment
                ],
                "coefficient_fit_only": True,
                "frame_state_policy": (
                    "zero/reset state at every explicit fit and held frame"
                ),
                "input_support": {
                    "fit_maximum_amplitude": fit_maximum_amplitude,
                    "held_maximum_amplitude": held_maximum_amplitude,
                    "held_to_fit_maximum_amplitude_ratio": (
                        held_maximum_amplitude / fit_maximum_amplitude
                        if fit_maximum_amplitude > 0.0
                        else None
                    ),
                    "held_fraction_above_fit_maximum": float(
                        np.mean(held_above_fit)
                    ),
                    "held_sample_count_above_fit_maximum": int(
                        np.count_nonzero(held_above_fit)
                    ),
                },
                "fit_numerical_diagnostics": {
                    "feature_count": int(
                        diagnostics.get("feature_count", -1)
                    ),
                    "solver_rank": int(
                        diagnostics.get("solver_rank", -1)
                    ),
                    "condition_number": (
                        None
                        if condition_number is None
                        else float(condition_number)
                    ),
                    "condition_number_source": (
                        "scaled_augmented_condition_number"
                        if "scaled_augmented_condition_number" in diagnostics
                        else "augmented_design_condition_number"
                    ),
                    "coefficient_l2_norm": float(
                        np.linalg.norm(model.coefficients)
                    ),
                },
                "fit_seconds": fit_seconds,
                "fit_diagnostics": diagnostics,
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("OOF prediction contains non-finite samples")
    return prediction, fold_reports, total_fit_seconds


def leave_one_frame_out_predictions(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    *,
    segment_id: np.ndarray,
    orders: tuple[int, ...],
    delays: tuple[int, ...],
    ridge: float,
    nperseg: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Backward-compatible MP-only wrapper around generic recipe dispatch."""

    return _leave_one_frame_out_predictions(
        pa_input,
        measured_output,
        segment_id=segment_id,
        recipe=SelectedMPRecipe(
            orders=tuple(orders),
            delays=tuple(delays),
            ridge=float(ridge),
        ),
        nperseg=nperseg,
    )


def _verify_full_train_reproduction(
    frozen_model: PAModel,
    reproduced_model: PAModel,
    train_input: np.ndarray,
    *,
    nperseg: int,
) -> dict[str, object]:
    frozen = np.asarray(
        frozen_model.coefficients,
        dtype=np.complex128,
    ).reshape(-1)
    reproduced = np.asarray(
        reproduced_model.coefficients,
        dtype=np.complex128,
    ).reshape(-1)
    if frozen.shape != reproduced.shape:
        raise ValueError("full-train reproduction coefficient shape mismatch")
    difference = reproduced - frozen
    maximum_absolute_error = float(
        np.max(np.abs(difference), initial=0.0)
    )
    denominator = float(np.linalg.norm(frozen))
    relative_l2_error = float(
        np.linalg.norm(difference) / denominator
        if denominator > 0.0
        else np.linalg.norm(difference)
    )
    coefficient_passed = bool(
        np.allclose(
            reproduced,
            frozen,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    frozen_prediction = frozen_model.predict_segments(train_input, nperseg)
    reproduced_prediction = reproduced_model.predict_segments(
        train_input,
        nperseg,
    )
    prediction_difference = reproduced_prediction - frozen_prediction
    maximum_prediction_error = float(
        np.max(np.abs(prediction_difference), initial=0.0)
    )
    prediction_denominator = float(np.linalg.norm(frozen_prediction))
    relative_prediction_error = float(
        np.linalg.norm(prediction_difference) / prediction_denominator
        if prediction_denominator > 0.0
        else np.linalg.norm(prediction_difference)
    )
    prediction_passed = bool(
        np.allclose(
            reproduced_prediction,
            frozen_prediction,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    passed = coefficient_passed and prediction_passed
    report = {
        "coefficient_count_complex": int(frozen.size),
        "rtol": 1e-10,
        "atol": 1e-12,
        "maximum_absolute_coefficient_error": maximum_absolute_error,
        "relative_l2_coefficient_error": relative_l2_error,
        "coefficient_reproduction_passed": coefficient_passed,
        "maximum_absolute_training_prediction_error": (
            maximum_prediction_error
        ),
        "relative_l2_training_prediction_error": relative_prediction_error,
        "training_prediction_reproduction_passed": prediction_passed,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            "full-train selected-recipe refit does not reproduce frozen NPZ: "
            f"{report}"
        )
    return report


def _top_rows(
    rows: list[dict[str, Any]],
    *,
    value_path: tuple[str, ...],
    causal_only: bool,
    count: int = 5,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if causal_only and not row.get("causal_feature_eligible", True):
            continue
        value: Any = row
        for key in value_path:
            value = value[key]
        number = float(value)
        if np.isfinite(number):
            scored.append((abs(number), row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:count]]


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    lag_rows = report["lag_correlations"]
    envelope_rows = report["envelope_correlations"]
    amplitude_summary: list[dict[str, Any]] = []
    for row in report["amplitude_regions"]:
        high = row["high_region"]["nmse_db"]
        complement = row["complement_region"]["nmse_db"]
        amplitude_summary.append(
            {
                "threshold_name": row["threshold_name"],
                "threshold_amplitude": row["threshold_amplitude"],
                "high_nmse_db": high,
                "complement_nmse_db": complement,
                "high_minus_complement_db": (
                    None
                    if high is None or complement is None
                    else float(high - complement)
                ),
            }
        )
    return {
        "global_metrics": report["global_metrics"],
        "top_causal_proper_lag_correlations": _top_rows(
            lag_rows,
            value_path=("proper_complex_correlation", "magnitude"),
            causal_only=True,
        ),
        "top_causal_pseudo_lag_correlations": _top_rows(
            lag_rows,
            value_path=("pseudo_complex_correlation", "magnitude"),
            causal_only=True,
        ),
        "top_future_input_diagnostics": _top_rows(
            [
                row
                for row in lag_rows
                if int(row["lag_samples"]) < 0
            ],
            value_path=("proper_complex_correlation", "magnitude"),
            causal_only=False,
        ),
        "top_residual_autocorrelations": _top_rows(
            [
                row
                for row in lag_rows
                if int(row["lag_samples"]) > 0
            ],
            value_path=("residual_acf", "magnitude"),
            causal_only=False,
        ),
        "top_complex_candidate_feature_correlations": _top_rows(
            envelope_rows,
            value_path=(
                "proper_candidate_feature_correlation",
                "magnitude",
            ),
            causal_only=False,
        ),
        "top_radial_envelope_correlations": _top_rows(
            envelope_rows,
            value_path=("corr_radial_envelope",),
            causal_only=False,
        ),
        "top_tangential_envelope_correlations": _top_rows(
            envelope_rows,
            value_path=("corr_tangential_envelope",),
            causal_only=False,
        ),
        "amplitude_region_summary": amplitude_summary,
        "slow_state_branch_eligible": any(
            row["eligible_for_state_branch_selection"]
            for row in report["slow_state_correlations"]
        ),
        "error_psd_integrated_bands": report["error_psd"].get(
            "integrated_bands"
        ),
    }


def _masked_error_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int | None]:
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {
            "sample_count": 0,
            "pooled_complex_nmse_db": None,
            "error_energy": 0.0,
            "reference_energy": 0.0,
        }
    return {
        "sample_count": count,
        "pooled_complex_nmse_db": nmse_pooled_db(
            prediction[selected],
            reference[selected],
        ),
        "error_energy": float(
            np.sum(np.abs(prediction[selected] - reference[selected]) ** 2)
        ),
        "reference_energy": float(
            np.sum(np.abs(reference[selected]) ** 2)
        ),
    }


def _reset_boundary_summary(
    prediction: np.ndarray,
    reference: np.ndarray,
    common_interior_mask: np.ndarray,
    selected_model_interior_mask: np.ndarray,
) -> dict[str, object]:
    full = np.ones(prediction.size, dtype=bool)
    full_metrics = _masked_error_metrics(prediction, reference, full)
    common_interior_metrics = _masked_error_metrics(
        prediction,
        reference,
        common_interior_mask,
    )
    common_excluded_metrics = _masked_error_metrics(
        prediction,
        reference,
        ~common_interior_mask,
    )
    model_interior_metrics = _masked_error_metrics(
        prediction,
        reference,
        selected_model_interior_mask,
    )
    model_excluded_metrics = _masked_error_metrics(
        prediction,
        reference,
        ~selected_model_interior_mask,
    )
    additional_fair_comparison_exclusion = (
        selected_model_interior_mask & ~common_interior_mask
    )
    additional_metrics = _masked_error_metrics(
        prediction,
        reference,
        additional_fair_comparison_exclusion,
    )
    total_error_energy = float(full_metrics["error_energy"])
    excluded_error_energy = float(common_excluded_metrics["error_energy"])
    return {
        "full_record": full_metrics,
        "selected_model_boundary_interior": model_interior_metrics,
        "selected_model_boundary_excluded": model_excluded_metrics,
        "common_boundary_interior": common_interior_metrics,
        "common_boundary_excluded": common_excluded_metrics,
        "additional_common_fair_comparison_exclusion": additional_metrics,
        "common_excluded_fraction_of_total_error_energy": (
            excluded_error_energy / total_error_energy
            if total_error_energy > 0.0
            else None
        ),
        # Backward-compatible keys; their historical names are less precise
        # than the primary fields above.
        "common_warmup_excluded": common_interior_metrics,
        "discarded_reset_region": common_excluded_metrics,
        "discarded_region_fraction_of_total_error_energy": (
            excluded_error_energy / total_error_energy
            if total_error_energy > 0.0
            else None
        ),
        "legacy_key_caveat": (
            "discarded_reset_region is retained for schema compatibility; "
            "the region may also contain conservative common warmup and "
            "future cooldown beyond the selected model's own boundary need"
        ),
        "interpretation": (
            "boundary exclusion under evaluator framing; only the selected-"
            "model boundary subset is attributable to its reset/lookahead "
            "support, and none is automatically a physical PA-memory effect"
        ),
    }


def analyze_from_config(config_path: str | Path) -> dict[str, Any]:
    source_config = Path(config_path).resolve()
    initial_config_sha256 = file_sha256(source_config)
    config = _load_config(source_config)
    _verify_file(
        source_config,
        initial_config_sha256,
        label="residual config after parsing",
    )
    source_path = Path(__file__).resolve()
    project_root = source_path.parents[1]

    dataset = _resolve_project_relative(
        config["dataset"],
        project_root=project_root,
        name="dataset",
    )
    selection_path = _resolve_project_relative(
        config["selection_manifest"],
        project_root=project_root,
        name="selection_manifest",
    )
    selection_config_path = _resolve_project_relative(
        config["selection_config"],
        project_root=project_root,
        name="selection_config",
    )
    output = _resolve_project_relative(
        config["output_dir"],
        project_root=project_root,
        name="output_dir",
    )
    _verify_file(
        selection_path,
        str(config["selection_manifest_sha256"]),
        label="selection manifest",
    )
    initial_selection_sha256 = file_sha256(selection_path)
    _verify_file(
        selection_config_path,
        str(config["selection_config_sha256"]),
        label="selection config",
    )
    initial_selection_config_sha256 = file_sha256(selection_config_path)
    selection = _load_json_object(
        selection_path,
        name="selection manifest",
    )
    _verify_file(
        selection_path,
        initial_selection_sha256,
        label="selection manifest after parsing",
    )
    if selection.get("config_sha256") != initial_selection_config_sha256:
        raise ValueError(
            "portable selection_config hash does not match the hash frozen "
            "inside the selection manifest"
        )

    legacy_model_value = selection.get("selected_model")
    if not isinstance(legacy_model_value, str) or not legacy_model_value:
        raise ValueError("selection manifest has no selected_model path")
    if "selected_model" in config:
        model_path = _resolve_project_relative(
            config["selected_model"],
            project_root=project_root,
            name="selected_model",
        )
        model_resolution = "explicit_repo_relative_runner_config_path"
    else:
        model_basename = Path(legacy_model_value).name
        if not model_basename:
            raise ValueError("selection selected_model has no basename")
        model_path = (selection_path.parent / model_basename).resolve()
        try:
            model_path.relative_to(project_root)
        except ValueError as error:
            raise ValueError(
                "derived selected model must stay inside project root"
            ) from error
        model_resolution = "selection_manifest_parent_plus_legacy_basename"
    _verify_file(
        model_path,
        str(selection["selected_model_sha256"]),
        label="selected PA model",
    )
    initial_model_sha256 = file_sha256(model_path)
    recipe, frozen_model, operation_verification = (
        _parse_and_verify_recipe(selection, model_path)
    )
    (
        selection_source_integrity,
        selection_source_paths,
        current_selection_source_hashes,
    ) = _verify_selection_source_integrity(
        selection,
        project_root=project_root,
        kind=recipe.kind,
    )
    _verify_file(
        model_path,
        initial_model_sha256,
        label="selected PA model after loading",
    )

    protocol = selection.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("selection manifest has no protocol object")
    alignment_delay = protocol.get("alignment_delay_samples")
    if (
        isinstance(alignment_delay, (bool, np.bool_))
        or not isinstance(alignment_delay, (int, np.integer))
        or int(alignment_delay) != 0
    ):
        raise ValueError(
            "residual analysis requires zero integer alignment until "
            "frame-safe aligned data are materialized"
        )
    if protocol.get("fractional_delay_applied") is not False:
        raise ValueError("residual input must not already apply fractional delay")
    raw_nperseg = protocol.get("nperseg")
    if (
        isinstance(raw_nperseg, (bool, np.bool_))
        or not isinstance(raw_nperseg, (int, np.integer))
        or int(raw_nperseg) <= 1
    ):
        raise ValueError("selection protocol nperseg must exceed one")
    nperseg = int(raw_nperseg)
    selection_common_warmup = _nonnegative_integer(
        selection.get("common_warmup_samples_per_frame"),
        name="selection common_warmup_samples_per_frame",
    )
    selection_common_cooldown = _nonnegative_integer(
        selection.get("common_future_cooldown_samples_per_frame", 0),
        name="selection common_future_cooldown_samples_per_frame",
    )
    common_warmup = _nonnegative_integer(
        config["expected_common_warmup_samples_per_frame"],
        name="expected_common_warmup_samples_per_frame",
    )
    common_cooldown = _nonnegative_integer(
        config["expected_common_future_cooldown_samples_per_frame"],
        name="expected_common_future_cooldown_samples_per_frame",
    )
    if (
        selection_common_warmup < recipe.causal_warmup_samples
        or selection_common_cooldown < recipe.lookahead_samples
        or selection_common_warmup + selection_common_cooldown >= nperseg
    ):
        raise ValueError(
            "selection-manifest common boundary support is inconsistent with "
            "the selected model or nperseg"
        )
    if (
        common_warmup < recipe.causal_warmup_samples
        or common_cooldown < recipe.lookahead_samples
        or common_warmup + common_cooldown >= nperseg
    ):
        raise ValueError(
            "configured expected common boundary support is inconsistent "
            "with the selected model or nperseg"
        )
    if recipe.kind == "gmp" and (
        common_warmup != selection_common_warmup
        or common_cooldown != selection_common_cooldown
    ):
        raise ValueError(
            "configured expected GMP common warmup/cooldown must exactly "
            "match the selection manifest"
        )
    if recipe.kind == "mp" and (
        common_warmup < selection_common_warmup
        or common_cooldown < selection_common_cooldown
    ):
        raise ValueError(
            "configured matched MP support may conservatively extend but "
            "must not shrink selection-manifest common support"
        )

    dataset_hashes = selection.get("dataset_files_sha256")
    if not isinstance(dataset_hashes, dict):
        raise ValueError("selection manifest has no dataset file hashes")
    if any(Path(name).name.startswith("test_") for name in dataset_hashes):
        raise ValueError("selection dataset hashes must not include test files")
    required_dataset_files = {
        "train_input.csv",
        "train_output.csv",
        "val_input.csv",
        "val_output.csv",
        "spec.json",
    }
    if set(dataset_hashes) != required_dataset_files:
        raise ValueError(
            "selection dataset hashes must contain exactly train/val/spec"
        )
    dataset_paths = {
        name: dataset / name for name in sorted(required_dataset_files)
    }
    frozen_dataset_hashes = {
        name: str(dataset_hashes[name])
        for name in sorted(required_dataset_files)
    }
    _verify_frozen_hashes(
        dataset_paths,
        frozen_dataset_hashes,
        scope="pre-load dataset",
    )

    source_dependency_paths = {
        "experiments/analyze_pa_residuals.py": source_path,
        "baseline/complexity.py": (
            project_root / "baseline" / "complexity.py"
        ),
        "baseline/gmp_pa.py": project_root / "baseline" / "gmp_pa.py",
        "baseline/metrics.py": project_root / "baseline" / "metrics.py",
        "baseline/pa_models.py": project_root / "baseline" / "pa_models.py",
        "baseline/residual_analysis.py": (
            project_root / "baseline" / "residual_analysis.py"
        ),
        "baseline/train_spline.py": (
            project_root / "baseline" / "train_spline.py"
        ),
    }
    initial_source_hashes = _snapshot_hashes(source_dependency_paths)

    oof_report_path = output / "train_oof_residual_analysis.json"
    validation_report_path = output / "validation_residual_analysis.json"
    prediction_path = output / "residual_predictions.npz"
    manifest_path = output / "residual_manifest.json"
    oof_report_temp = output / ".train_oof_residual_analysis.publishing.json"
    validation_report_temp = (
        output / ".validation_residual_analysis.publishing.json"
    )
    prediction_temp = output / ".residual_predictions.publishing.npz"
    manifest_temp = output / ".residual_manifest.publishing.json"
    lock_path = output / ".residual_analysis.lock"
    owned_paths = (
        oof_report_path,
        validation_report_path,
        prediction_path,
        manifest_path,
        oof_report_temp,
        validation_report_temp,
        prediction_temp,
        manifest_temp,
    )
    existing = [
        path
        for path in (*owned_paths, lock_path)
        if _path_entry_exists(path)
    ]
    if existing:
        raise FileExistsError(
            "immutable residual bundle has an existing lock/final/temp: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    lock_payload = _acquire_bundle_lock(lock_path)
    initial_lock_sha256 = file_sha256(lock_path)
    appeared = [path for path in owned_paths if _path_entry_exists(path)]
    if appeared:
        raise FileExistsError(
            "residual artifact appeared while taking bundle lock: "
            + ", ".join(str(path) for path in appeared)
        )

    control_paths = {
        "runner_config": source_config,
        "selection_manifest": selection_path,
        "selection_config": selection_config_path,
        "selected_model": model_path,
    }
    frozen_control_hashes = {
        "runner_config": initial_config_sha256,
        "selection_manifest": initial_selection_sha256,
        "selection_config": initial_selection_config_sha256,
        "selected_model": initial_model_sha256,
    }

    def verify_all_inputs(scope: str) -> None:
        _verify_frozen_hashes(
            control_paths,
            frozen_control_hashes,
            scope=f"{scope} control input",
        )
        _verify_frozen_hashes(
            dataset_paths,
            frozen_dataset_hashes,
            scope=f"{scope} dataset",
        )
        _verify_frozen_hashes(
            source_dependency_paths,
            initial_source_hashes,
            scope=f"{scope} source dependency",
        )
        _verify_frozen_hashes(
            selection_source_paths,
            current_selection_source_hashes,
            scope=f"{scope} selection source dependency",
        )

    # Complete data access list. Test is intentionally absent.
    train_input, train_output = load_split_pair(dataset, "train")
    validation_input, validation_output = load_split_pair(dataset, "val")
    dataset_spec = load_dataset_spec(dataset)
    verify_all_inputs("post-waveform-load")
    if int(dataset_spec.get("nperseg", -1)) != nperseg:
        raise ValueError("dataset spec and selection protocol nperseg differ")
    spec = _build_spec(config, dataset_spec)
    train_segments = explicit_frame_ids(train_input.size, nperseg)
    validation_segments = explicit_frame_ids(
        validation_input.size,
        nperseg,
    )

    full_fit_started = time.perf_counter()
    reproduced_model, full_fit_diagnostics = _fit_selected_recipe(
        recipe,
        train_input,
        train_output,
        nperseg=nperseg,
    )
    full_fit_seconds = time.perf_counter() - full_fit_started
    reproduction = _verify_full_train_reproduction(
        frozen_model,
        reproduced_model,
        train_input,
        nperseg=nperseg,
    )
    validation_prediction = frozen_model.predict_segments(
        validation_input,
        nperseg,
    )
    oof_prediction, fold_reports, total_oof_fit_seconds = (
        _leave_one_frame_out_predictions(
            train_input,
            train_output,
            segment_id=train_segments,
            recipe=recipe,
            nperseg=nperseg,
        )
    )
    train_valid = _segmented_interior_mask(
        train_input.size,
        nperseg=nperseg,
        warmup_samples=common_warmup,
        cooldown_samples=common_cooldown,
    )
    validation_valid = _segmented_interior_mask(
        validation_input.size,
        nperseg=nperseg,
        warmup_samples=common_warmup,
        cooldown_samples=common_cooldown,
    )
    train_model_valid = _segmented_interior_mask(
        train_input.size,
        nperseg=nperseg,
        warmup_samples=recipe.causal_warmup_samples,
        cooldown_samples=recipe.lookahead_samples,
    )
    validation_model_valid = _segmented_interior_mask(
        validation_input.size,
        nperseg=nperseg,
        warmup_samples=recipe.causal_warmup_samples,
        cooldown_samples=recipe.lookahead_samples,
    )
    frozen_reference = freeze_residual_reference(train_input, spec)
    oof_report = analyze_pa_residuals(
        train_input,
        train_output,
        oof_prediction,
        segment_id=train_segments,
        valid_mask=train_valid,
        split_role="train_oof",
        spec=spec,
        frozen_reference=frozen_reference,
    )
    validation_report = analyze_pa_residuals(
        validation_input,
        validation_output,
        validation_prediction,
        segment_id=validation_segments,
        valid_mask=validation_valid,
        split_role="validation_confirmation",
        spec=spec,
        frozen_reference=frozen_reference,
    )
    oof_report["runner_scope"] = {
        "coefficient_estimation": (
            "leave-one-explicit-frame-out coefficient-only refit"
        ),
        "topology_and_solver_selected_on_validation": True,
        "preprocessing_nested_within_oof": False,
        "conditional_preprocessing": (
            "alignment protocol and residual diagnostic bins/reference are "
            "frozen from the complete training split"
        ),
    }
    validation_report["runner_scope"] = {
        "coefficient_estimation": (
            "full-training selected-recipe refit reproduced against frozen NPZ"
        ),
        "used_for_fit_or_tuning_in_this_runner": False,
        "independent_model_selection_confirmation": False,
    }

    verify_all_inputs("pre-artifact-publication")
    _verify_owned_lock(
        lock_path,
        lock_payload,
        scope="before residual artifact publication",
    )
    appeared = [path for path in owned_paths if _path_entry_exists(path)]
    if appeared:
        raise FileExistsError(
            "residual artifact appeared during analysis: "
            + ", ".join(str(path) for path in appeared)
        )
    _atomic_write_json(oof_report_path, oof_report_temp, oof_report)
    _atomic_write_json(
        validation_report_path,
        validation_report_temp,
        validation_report,
    )
    _atomic_write_predictions(
        prediction_path,
        prediction_temp,
        schema_version=np.asarray(2, dtype=np.int64),
        model_class=np.asarray(selection["model_class"]),
        train_oof_prediction=oof_prediction,
        validation_prediction=validation_prediction,
        train_segment_id=train_segments,
        validation_segment_id=validation_segments,
        train_valid_mask=train_valid,
        validation_valid_mask=validation_valid,
        train_selected_model_valid_mask=train_model_valid,
        validation_selected_model_valid_mask=validation_model_valid,
    )

    if isinstance(recipe, SelectedMPRecipe):
        selected_architecture: dict[str, object] = {
            "orders": list(recipe.orders),
            "delays": list(recipe.delays),
            "ridge": recipe.ridge,
        }
    else:
        selected_architecture = {
            "gmp_config": dataclasses.asdict(recipe.config),
            "ridge": recipe.ridge,
            "solver_mode": recipe.solver_mode,
            "svd_rcond": recipe.svd_rcond,
        }
    manifest = {
        "schema_version": 2,
        "task": "forward_pa_residual_analysis",
        "model_class": selection["model_class"],
        "dataset": dataset,
        "dataset_resolution": {
            "runner_config_value": config["dataset"],
            "runner_config_resolved_path": dataset,
            "legacy_selection_manifest_value": selection.get("dataset"),
            "legacy_path_used_for_io": False,
            "identity": (
                "runner-config repo-relative dataset with exact selection-"
                "manifest train/validation/spec hashes"
            ),
        },
        "selection_manifest": selection_path,
        "selection_manifest_sha256": initial_selection_sha256,
        "selection_config": selection_config_path,
        "selection_config_sha256": initial_selection_config_sha256,
        "selection_config_resolution": {
            "runner_config_value": config["selection_config"],
            "legacy_selection_manifest_value": selection.get("config"),
            "legacy_path_used_for_io": False,
        },
        "selected_model": model_path,
        "selected_model_sha256": initial_model_sha256,
        "selected_model_resolution": {
            "policy": model_resolution,
            "runner_config_value": config.get("selected_model"),
            "legacy_selection_manifest_value": legacy_model_value,
            "legacy_absolute_path_used_for_io": False,
        },
        "dataset_files_sha256": frozen_dataset_hashes,
        "accessed_splits": ["train", "validation"],
        "test_split_accessed": False,
        "test_file_hashes_recorded": False,
        "discovery_split": (
            "leave-one-explicit-frame-out training residual; coefficient-only "
            "refits conditional on topology/solver selected on validation"
        ),
        "oof_scope": {
            "coefficient_fit_only": True,
            "topology_validation_selected": True,
            "solver_validation_selected": True,
            "preprocessing_nested_within_oof": False,
            "conditional_preprocessing": (
                "integer/fractional alignment diagnostics and residual bins "
                "are frozen once from the complete training split"
            ),
        },
        "validation_role": (
            "descriptive confirmation only; this validation split already "
            "selected the architecture, solver recipe, and regularization"
        ),
        "future_branch_selection_rule": (
            "use train inner blocked CV; do not choose a branch from this "
            "validation report and then reuse the same validation as confirmation"
        ),
        "segment_policy": (
            "nperseg-derived explicit model-reset frames; not claimed to be "
            "independent physical captures"
        ),
        "common_warmup_samples_per_frame": common_warmup,
        "common_future_cooldown_samples_per_frame": common_cooldown,
        "selection_manifest_common_boundary": {
            "warmup_samples_per_frame": selection_common_warmup,
            "future_cooldown_samples_per_frame": selection_common_cooldown,
        },
        "analysis_common_boundary": {
            "warmup_samples_per_frame": common_warmup,
            "future_cooldown_samples_per_frame": common_cooldown,
            "relationship_to_selection_manifest": (
                "exact_match"
                if (
                    common_warmup == selection_common_warmup
                    and common_cooldown == selection_common_cooldown
                )
                else "conservative_mp_extension"
            ),
            "rationale": (
                "GMP analysis uses the exact common boundary frozen by model "
                "selection; legacy MP analysis may conservatively extend it "
                "to the matched cross-model comparison support"
            ),
        },
        "selected_model_required_boundary": {
            "warmup_samples_per_frame": recipe.causal_warmup_samples,
            "cooldown_samples_per_frame": recipe.lookahead_samples,
        },
        "selected_architecture": selected_architecture,
        "selected_recipe": recipe.to_dict(),
        "operation_count_verification": operation_verification,
        "full_training_refit": {
            "fit_seconds": full_fit_seconds,
            "fit_diagnostics": full_fit_diagnostics,
            "frozen_npz_reproduction": reproduction,
            "validation_predictor": (
                "integrity-verified frozen selected NPZ; independent full-"
                "training refit is used only as a reproduction check"
            ),
        },
        "oof_fold_count": len(fold_reports),
        "oof_total_fit_seconds": total_oof_fit_seconds,
        "oof_folds": fold_reports,
        "train_oof_summary": _summarize(oof_report),
        "validation_summary": _summarize(validation_report),
        "reset_boundary_diagnostics": {
            "train_oof": _reset_boundary_summary(
                oof_prediction,
                train_output,
                train_valid,
                train_model_valid,
            ),
            "validation": _reset_boundary_summary(
                validation_prediction,
                validation_output,
                validation_valid,
                validation_model_valid,
            ),
        },
        "state_conditioned_model_gate": (
            "locked: fewer than two declared independent captures cannot "
            "establish slow thermal/bias state"
            if spec.independent_capture_count < 2
            else
            "diagnostics available; branch selection still requires nested "
            "training-only validation"
        ),
        "train_oof_report": oof_report_path,
        "train_oof_report_sha256": file_sha256(oof_report_path),
        "validation_report": validation_report_path,
        "validation_report_sha256": file_sha256(validation_report_path),
        "predictions": prediction_path,
        "predictions_sha256": file_sha256(prediction_path),
        "config": source_config,
        "config_sha256": initial_config_sha256,
        "source_sha256": initial_source_hashes,
        "selection_source_integrity": selection_source_integrity,
        "input_integrity": {
            "all_hashes_frozen_before_waveform_access": True,
            "all_inputs_reverified_after_waveform_load": True,
            "all_inputs_reverified_before_artifact_publication": True,
            "selection_source_dependencies_reverified": True,
        },
        "publication": {
            "immutable_bundle": True,
            "completion_manifest_published_last": True,
            "atomic_single_writer_lock": {
                "path": lock_path,
                "owner_pid": os.getpid(),
                "owner_payload_sha256": initial_lock_sha256,
                "failure_policy": (
                    "lock remains as explicit incomplete-run marker"
                ),
            },
            "atomic_artifact_protocol": (
                "same-directory temp followed by os.replace"
            ),
        },
    }

    verify_all_inputs("pre-completion-manifest")
    _verify_owned_lock(
        lock_path,
        lock_payload,
        scope="before completion manifest",
    )
    _verify_frozen_hashes(
        {
            "train_oof_report": oof_report_path,
            "validation_report": validation_report_path,
            "predictions": prediction_path,
        },
        {
            "train_oof_report": manifest["train_oof_report_sha256"],
            "validation_report": manifest["validation_report_sha256"],
            "predictions": manifest["predictions_sha256"],
        },
        scope="pre-completion artifact",
    )
    _atomic_write_json(manifest_path, manifest_temp, manifest)
    _verify_owned_lock(
        lock_path,
        lock_payload,
        scope="after completion manifest",
    )
    lock_path.unlink()
    return manifest


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze integrity-gated MP/GMP PA residuals on train OOF and "
            "validation only; never open test."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = analyze_from_config(args.config)
    train_nmse = manifest["train_oof_summary"]["global_metrics"][
        "pooled_complex_nmse_db"
    ]
    validation_nmse = manifest["validation_summary"]["global_metrics"][
        "pooled_complex_nmse_db"
    ]
    print(
        "Residual analysis:",
        f"train OOF NMSE={train_nmse:.6f} dB",
        f"validation NMSE={validation_nmse:.6f} dB",
        "state branch gate=locked",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
