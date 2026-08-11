"""Validation-only spline-memory DPD selection for ``BlackBoxData``.

The selector deliberately has no release/test input.  Its only data entry
point is the compact ``selection_view.json`` prepared by
``prepare_blackbox_data.py``.  It also requires a complete, hash-bound PA
selection bundle and refuses a standalone PA ``.npz`` file.

Calibration and deployment directions are kept separate:

* train-only ILA calibration: ``u_train = y_train / g_train -> x_train``;
* validation selection: ``x_val -> DPD -> frozen PA -> compare g_train*x_val``.

Measured validation output is used only to quantify frozen-evaluator fidelity
and the post-selection evaluator-headroom diagnostic.  It is never a DPD
input.  The headroom gate is evaluated after model selection and therefore
cannot influence which DPD wins.

The capture contains no sample rate or RF-band definitions.  Consequently the
only spectral artifact produced here is a normalized-frequency Welch PSD.  No
ACLR, harmonic attenuation, Hz axis, or Huawei pass/fail claim is inferred.
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
from typing import Any, Iterable

import numpy as np

from baseline.alignment import complex_ls_gain, overlap_for_delay
from baseline.complex_spline_dpd import make_knots
from baseline.complexity import OperationCount
from baseline.gmp_pa import GeneralizedMemoryPolynomialPA
from baseline.metrics import welch_numpy
from baseline.pa_models import MemoryPolynomialPA
from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    SplineMemoryState,
    spline_memory_design_matrix,
)
from baseline.train_spline import file_sha256, write_json
from experiments.select_blackbox_pa import (
    _load_normalized_pairs,
    _reject_symlink_components,
    _verify_selection_view,
    load_frozen_blackbox_pa_selection,
)


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = {
    "schema_version",
    "selection_dir",
    "expected_source_sha256",
    "expected_selection_view_sha256",
    "pa_bundle_dir",
    "expected_pa_completion_sha256",
    "output_dir",
    "dataset_label",
    "knot_strategy",
    "knot_counts",
    "ridge_values",
    "branch_topologies",
    "selection_tolerance_db",
    "evaluator_headroom_gate_db",
    "maximum_fit_count",
    "psd_nperseg",
    "psd_noverlap",
}
TOPOLOGY_KEYS = {"name", "branches"}
BRANCH_KEYS = {"signal_delay", "envelope_delay"}
BOUND_OUTPUTS = {
    "selected_dpd.npz",
    "validation_trials.json",
    "pareto_frontier.json",
    "normalized_psd.npz",
    "selection_manifest.json",
}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _resolve_project_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _strict_integer(value: Any, *, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _strict_float(
    value: Any,
    *,
    name: str,
    minimum: float,
    strictly_greater: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    invalid = result <= minimum if strictly_greater else result < minimum
    if invalid:
        relation = ">" if strictly_greater else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _sha256_digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer_values(value: Any, *, name: str, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON list")
    result = tuple(
        _strict_integer(item, name=f"{name} entry", minimum=minimum)
        for item in value
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _ridge_values(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("ridge_values must be a non-empty JSON list")
    result = tuple(
        _strict_float(item, name="ridge_values entry", minimum=0.0)
        for item in value
    )
    if len(set(result)) != len(result):
        raise ValueError("ridge_values entries must be unique")
    if 0.0 not in result:
        raise ValueError(
            "ridge_values must include 0 so raw-design rank is measured "
            "without ridge augmentation"
        )
    return result


def _branch_topologies(
    value: Any,
) -> tuple[tuple[str, tuple[SplineMemoryBranch, ...]], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("branch_topologies must be a non-empty JSON list")
    result: list[tuple[str, tuple[SplineMemoryBranch, ...]]] = []
    identities: set[tuple[SplineMemoryBranch, ...]] = set()
    names: set[str] = set()
    for raw_topology in value:
        if not isinstance(raw_topology, dict):
            raise ValueError("every branch topology must be an object")
        unknown = set(raw_topology) - TOPOLOGY_KEYS
        if unknown:
            raise ValueError(f"branch topology has unknown keys: {sorted(unknown)}")
        if set(raw_topology) != TOPOLOGY_KEYS:
            raise ValueError("branch topology must contain name and branches")
        name = raw_topology["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("branch topology name must be non-empty")
        name = name.strip()
        if name in names:
            raise ValueError("branch topology names must be unique")
        names.add(name)
        raw_branches = raw_topology["branches"]
        if not isinstance(raw_branches, list) or not raw_branches:
            raise ValueError(f"{name}.branches must be a non-empty list")
        branches: list[SplineMemoryBranch] = []
        for raw_branch in raw_branches:
            if not isinstance(raw_branch, dict):
                raise ValueError(f"every {name} branch must be an object")
            if set(raw_branch) != BRANCH_KEYS:
                raise ValueError(
                    f"every {name} branch must contain exactly "
                    "signal_delay and envelope_delay"
                )
            branches.append(
                SplineMemoryBranch(
                    _strict_integer(
                        raw_branch["signal_delay"],
                        name=f"{name}.signal_delay",
                        minimum=0,
                    ),
                    _strict_integer(
                        raw_branch["envelope_delay"],
                        name=f"{name}.envelope_delay",
                        minimum=0,
                    ),
                )
            )
        branch_tuple = tuple(branches)
        if len(set(branch_tuple)) != len(branch_tuple):
            raise ValueError(f"{name} contains duplicate branches")
        if branch_tuple in identities:
            raise ValueError("duplicate branch topology")
        identities.add(branch_tuple)
        result.append((name, branch_tuple))
    return tuple(result)


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_json_object(path, label="config")
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("config schema_version must equal 1")
    missing = CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"config has unknown keys: {sorted(unknown)}")
    if not isinstance(config["dataset_label"], str) or not config[
        "dataset_label"
    ].strip():
        raise ValueError("dataset_label must be a non-empty string")
    if config["knot_strategy"] not in {
        "uniform_amplitude",
        "uniform_power",
        "quantile",
        "compression_aware",
    }:
        raise ValueError("unsupported knot_strategy")
    for key in (
        "expected_source_sha256",
        "expected_selection_view_sha256",
        "expected_pa_completion_sha256",
    ):
        _sha256_digest(config[key], name=key)
    return config


def enumerate_candidate_recipes(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the deterministic topology-by-knot-by-ridge ledger."""

    knots = _integer_values(config["knot_counts"], name="knot_counts", minimum=2)
    ridges = _ridge_values(config["ridge_values"])
    topologies = _branch_topologies(config["branch_topologies"])
    recipes: list[dict[str, Any]] = []
    for topology_name, branches in topologies:
        # Run unregularized fit first so every ridge trial is explicitly bound
        # to the same raw-design rank diagnostic.
        ridge_order = (0.0,) + tuple(item for item in ridges if item != 0.0)
        for knot_count in knots:
            empty = SparseSplineMemoryDPD(
                knots=np.linspace(0.0, 1.0, knot_count),
                branches=branches,
                coefficients=np.zeros(
                    (len(branches), knot_count),
                    dtype=np.complex128,
                ),
                knot_strategy="counting_only",
            )
            for ridge in ridge_order:
                recipes.append(
                    {
                        "topology_name": topology_name,
                        "branches": branches,
                        "knot_count": knot_count,
                        "ridge": ridge,
                        "feature_count": len(branches) * knot_count,
                        "maximum_delay": empty.maximum_delay,
                        "operation_count": empty.operation_count(),
                    }
                )
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


def _pa_warmup_samples(
    model: MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
) -> int:
    if isinstance(model, MemoryPolynomialPA):
        return int(model.causal_warmup_samples)
    if isinstance(model, GeneralizedMemoryPolynomialPA):
        return int(model.config.causal_warmup_samples)
    raise TypeError("unsupported frozen PA model")


def _error_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    warmup_samples: int,
) -> dict[str, Any]:
    estimate = np.asarray(estimate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if estimate.ndim != 1 or estimate.shape != reference.shape:
        raise ValueError("metric inputs must be equal-length vectors")
    if warmup_samples < 0 or warmup_samples >= estimate.size:
        raise ValueError("warmup must leave at least one scored sample")
    scored_estimate = estimate[warmup_samples:]
    scored_reference = reference[warmup_samples:]
    if not np.all(np.isfinite(scored_estimate)) or not np.all(
        np.isfinite(scored_reference)
    ):
        raise ValueError("metric inputs contain non-finite values")
    error_power = float(np.mean(np.abs(scored_estimate - scored_reference) ** 2))
    reference_power = float(np.mean(np.abs(scored_reference) ** 2))
    if reference_power <= 0.0:
        raise ValueError("metric reference must have positive power")
    relative = error_power / reference_power
    perfect = error_power == 0.0
    nmse_db = None if perfect else float(10.0 * np.log10(relative))
    return {
        "complex_nmse_pooled_db": nmse_db,
        "perfect_reconstruction": perfect,
        "mse": error_power,
        "reference_power": reference_power,
        "relative_error_power": relative,
        "warmup_samples_at_record_start": int(warmup_samples),
        "scored_sample_count": int(scored_estimate.size),
        "discarded_sample_count": int(warmup_samples),
    }


def _signal_summary(signal: np.ndarray) -> dict[str, float]:
    values = np.asarray(signal, dtype=np.complex128)
    power = float(np.mean(np.abs(values) ** 2))
    peak = float(np.max(np.abs(values)))
    papr = None if power == 0.0 else float(10.0 * np.log10(peak * peak / power))
    return {
        "rms_amplitude": float(np.sqrt(power)),
        "maximum_amplitude": peak,
        "papr_db": papr,
    }


def _trial_score(trial: dict[str, Any]) -> float:
    score = trial.get("selection_score_db")
    if score is None:
        metrics = trial.get("validation_correct_direction", {})
        if metrics.get("perfect_reconstruction") is True:
            return -np.inf
        return np.inf
    result = float(score)
    return result if np.isfinite(result) else np.inf


def _cost_key(trial: dict[str, Any]) -> tuple[Any, ...]:
    cost = trial["operation_count_per_complex_sample"]
    return (
        int(cost["real_multiplications"]),
        int(cost["real_additions"]),
        int(cost["real_divisions"]),
        int(cost["nonlinear_operations"]),
        int(cost["comparisons"]),
        int(cost["lookups"]),
        int(cost["real_memory_reads"]),
        int(cost["real_memory_writes"]),
        int(cost["state_real_values"]),
        int(cost["stored_real_coefficients"]),
        int(cost["stored_real_constants"]),
    )


def choose_validation_winner(
    trials: Iterable[dict[str, Any]],
    *,
    tolerance_db: float,
    validity_field: str = "hard_valid",
) -> dict[str, Any]:
    """Choose cheapest hard-valid trial within ``tolerance_db`` of best NMSE."""

    tolerance = _strict_float(
        tolerance_db,
        name="selection_tolerance_db",
        minimum=0.0,
    )
    valid = [
        trial
        for trial in trials
        if trial.get(validity_field) is True and _trial_score(trial) < np.inf
    ]
    if not valid:
        raise ValueError("no hard-valid DPD candidate remains")
    best_score = min(_trial_score(trial) for trial in valid)
    near_best = [
        trial
        for trial in valid
        if _trial_score(trial) <= best_score + tolerance + 1e-12
    ]
    return min(
        near_best,
        key=lambda trial: (
            _cost_key(trial),
            _trial_score(trial),
            str(trial["topology_name"]),
            int(trial["knot_count"]),
            float(trial["ridge"]),
        ),
    )


def select_dpd_candidate(
    trials: Iterable[dict[str, Any]],
    *,
    tolerance_db: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a deployable improvement or a clearly diagnostic fallback."""

    trial_list = list(trials)
    deployable = [trial for trial in trial_list if trial.get("hard_valid") is True]
    if deployable:
        selected = choose_validation_winner(
            deployable,
            tolerance_db=tolerance_db,
        )
        return selected, {
            "deployment_recommended": True,
            "selection_role": "deployment_candidate_on_frozen_surrogate",
            "no_dpd_is_recommended_when_false": False,
        }
    diagnostic = choose_validation_winner(
        trial_list,
        tolerance_db=tolerance_db,
        validity_field="eligible_for_diagnostic_selection",
    )
    return diagnostic, {
        "deployment_recommended": False,
        "selection_role": "diagnostic_best_dpd_but_not_a_deployment_winner",
        "no_dpd_is_recommended_when_false": True,
        "reason": "all structurally valid DPD candidates fail improves_no_dpd",
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_cost = _cost_key(left)
    right_cost = _cost_key(right)
    left_vector = (
        _trial_score(left),
        *left_cost,
        float(left["validation_drive"]["maximum_amplitude"]),
    )
    right_vector = (
        _trial_score(right),
        *right_cost,
        float(right["validation_drive"]["maximum_amplitude"]),
    )
    return all(a <= b for a, b in zip(left_vector, right_vector, strict=True)) and any(
        a < b for a, b in zip(left_vector, right_vector, strict=True)
    )


def pareto_frontier(trials: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return hard-valid non-dominated validation trials."""

    valid = [
        trial
        for trial in trials
        if trial.get("hard_valid") is True and _trial_score(trial) < np.inf
    ]
    frontier = [
        trial
        for trial in valid
        if not any(
            other is not trial and _dominates(other, trial)
            for other in valid
        )
    ]
    return sorted(
        frontier,
        key=lambda trial: (
            _trial_score(trial),
            _cost_key(trial),
            str(trial["topology_name"]),
            int(trial["knot_count"]),
            float(trial["ridge"]),
        ),
    )


@dataclasses.dataclass(frozen=True)
class GroupedSplineFactorization:
    """One steady-state SVD reused for every ridge in a topology/K group."""

    knots: np.ndarray
    branches: tuple[SplineMemoryBranch, ...]
    singular_values: np.ndarray
    right_singular_vectors_h: np.ndarray
    projected_target: np.ndarray
    causal_warmup_samples: int
    steady_sample_count: int
    feature_count: int
    data_design_rank: int
    singular_value_cutoff: float
    data_design_condition_number: float | None
    factorization_seconds: float


def factorize_spline_group(
    calibration_input: np.ndarray,
    target: np.ndarray,
    *,
    knots: np.ndarray,
    branches: Iterable[SplineMemoryBranch],
) -> GroupedSplineFactorization:
    """Factorize the normalized causal-steady design exactly once.

    For ``w=max(m_b,d_b)`` this constructs only the regression objective

    ``||D c-q||^2 + ridge*||c||^2``

    with ``D=Phi[w:]/sqrt(N)`` and ``q=target[w:]/sqrt(N)``.  Therefore the
    zero-padded record-start rows never affect coefficients or rank.
    """

    started = time.perf_counter()
    values = np.asarray(calibration_input, dtype=np.complex128)
    desired = np.asarray(target, dtype=np.complex128)
    if values.ndim != 1 or values.shape != desired.shape or values.size == 0:
        raise ValueError("calibration_input and target must be equal non-empty vectors")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(desired)):
        raise ValueError("calibration_input and target must be finite")
    branch_tuple = tuple(branches)
    if not branch_tuple:
        raise ValueError("at least one branch is required")
    warmup = max(
        max(branch.signal_delay, branch.envelope_delay)
        for branch in branch_tuple
    )
    if warmup >= values.size:
        raise ValueError("causal warmup consumes the calibration record")
    design = spline_memory_design_matrix(values, knots, branch_tuple)[warmup:]
    steady_target = desired[warmup:]
    steady_count = int(steady_target.size)
    normalization = np.sqrt(float(steady_count))
    design /= normalization
    normalized_target = steady_target / normalization
    left, singular_values, right_h = np.linalg.svd(
        design,
        full_matrices=False,
    )
    projected_target = left.conj().T @ normalized_target
    del left, design
    maximum_singular = float(singular_values[0])
    cutoff = float(
        np.finfo(np.float64).eps
        * max(steady_count, right_h.shape[1])
        * maximum_singular
    )
    rank = int(np.count_nonzero(singular_values > cutoff))
    minimum_singular = float(singular_values[-1])
    condition = (
        None
        if minimum_singular <= 0.0
        else float(maximum_singular / minimum_singular)
    )
    return GroupedSplineFactorization(
        knots=np.asarray(knots, dtype=np.float64).copy(),
        branches=branch_tuple,
        singular_values=singular_values,
        right_singular_vectors_h=right_h,
        projected_target=projected_target,
        causal_warmup_samples=warmup,
        steady_sample_count=steady_count,
        feature_count=int(right_h.shape[1]),
        data_design_rank=rank,
        singular_value_cutoff=cutoff,
        data_design_condition_number=condition,
        factorization_seconds=time.perf_counter() - started,
    )


def solve_grouped_spline(
    factorization: GroupedSplineFactorization,
    *,
    ridge: float,
    knot_strategy: str,
) -> tuple[SparseSplineMemoryDPD, float]:
    """Solve one ridge value from a cached direct SVD factorization."""

    started = time.perf_counter()
    ridge = _strict_float(ridge, name="ridge", minimum=0.0)
    singular = factorization.singular_values
    if ridge == 0.0:
        factors = np.zeros_like(singular)
        retained = singular > factorization.singular_value_cutoff
        factors[retained] = 1.0 / singular[retained]
    else:
        factors = singular / (singular * singular + ridge)
    flat = factorization.right_singular_vectors_h.conj().T @ (
        factors * factorization.projected_target
    )
    coefficients = flat.reshape(
        len(factorization.branches),
        factorization.knots.size,
    )
    model = SparseSplineMemoryDPD(
        knots=factorization.knots,
        branches=factorization.branches,
        coefficients=coefficients,
        knot_strategy=knot_strategy,
    )
    return model, time.perf_counter() - started


def _improvement_gate(
    candidate_metrics: dict[str, Any],
    no_dpd_metrics: dict[str, Any],
) -> dict[str, Any]:
    candidate_error = float(candidate_metrics["mse"])
    no_dpd_error = float(no_dpd_metrics["mse"])
    passed = candidate_error < no_dpd_error
    positive_infinite = candidate_error == 0.0 and no_dpd_error > 0.0
    undefined_zero_reference = no_dpd_error == 0.0
    if positive_infinite or undefined_zero_reference:
        improvement_db = None
    else:
        improvement_db = float(10.0 * np.log10(no_dpd_error / candidate_error))
    return {
        "pass": passed,
        "candidate_error_power": candidate_error,
        "no_dpd_error_power": no_dpd_error,
        "improvement_db_10log10_no_dpd_over_candidate": improvement_db,
        "positive_infinite_improvement_due_to_zero_candidate_error": (
            positive_infinite
        ),
        "undefined_because_no_dpd_error_is_zero": undefined_zero_reference,
        "policy": "strictly lower correct-direction validation error power",
    }


def _evaluator_headroom_diagnostic(
    selected_metrics: dict[str, Any],
    evaluator_metrics: dict[str, Any],
    *,
    required_margin_db: float,
) -> dict[str, Any]:
    """Return same-validation evaluator headroom; never a confirmation gate."""

    required = _strict_float(
        required_margin_db,
        name="evaluator_headroom_gate_db",
        minimum=0.0,
    )
    selected_error = float(selected_metrics["mse"])
    evaluator_error = float(evaluator_metrics["mse"])
    positive_infinite = evaluator_error == 0.0 and selected_error > 0.0
    negative_infinite = selected_error == 0.0 and evaluator_error > 0.0
    both_zero = selected_error == 0.0 and evaluator_error == 0.0
    if positive_infinite or negative_infinite or both_zero:
        margin_db = None
    else:
        margin_db = float(10.0 * np.log10(selected_error / evaluator_error))
    passed = bool(
        positive_infinite
        or (margin_db is not None and margin_db >= required)
    )
    return {
        "selected_cascade_error_power": selected_error,
        "evaluator_model_error_power": evaluator_error,
        "margin_db_10log10_dpd_error_over_pa_model_error": margin_db,
        "positive_infinite_margin_due_to_zero_evaluator_error": (
            positive_infinite
        ),
        "negative_infinite_margin_due_to_zero_dpd_error": negative_infinite,
        "undefined_because_both_errors_are_zero": both_zero,
        "required_margin_db": required,
        "pass": passed,
    }


def _evaluate_factorized_trial(
    recipe: dict[str, Any],
    *,
    factorization: GroupedSplineFactorization,
    calibration_input: np.ndarray,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    pa_model: MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
    gain: complex,
    common_warmup_samples: int,
    pa_train_input_peak: float,
    knot_strategy: str,
    no_dpd_metrics: dict[str, Any],
) -> tuple[dict[str, Any], SparseSplineMemoryDPD]:
    model, solve_seconds = solve_grouped_spline(
        factorization,
        ridge=float(recipe["ridge"]),
        knot_strategy=knot_strategy,
    )
    training_prediction = np.asarray(
        model.predict(calibration_input),
        dtype=np.complex128,
    )
    training_finite = bool(np.all(np.isfinite(training_prediction)))
    training_metrics = (
        _error_metrics(
            training_prediction,
            train_x,
            warmup_samples=factorization.causal_warmup_samples,
        )
        if training_finite
        else {
            "complex_nmse_pooled_db": None,
            "perfect_reconstruction": False,
            "failure": "non-finite train reconstruction",
        }
    )
    drive = np.asarray(model.predict(validation_x), dtype=np.complex128)
    cascade = np.asarray(pa_model.predict(drive), dtype=np.complex128)
    ideal = gain * validation_x
    finite = bool(np.all(np.isfinite(drive)) and np.all(np.isfinite(cascade)))
    maximum_validation_radius = float(np.max(np.abs(validation_x)))
    knot_maximum = float(model.knots[-1])
    maximum_drive = float(np.max(np.abs(drive))) if finite else float("inf")
    support_tolerance = 64.0 * np.finfo(float).eps
    validation_in_knots = bool(
        maximum_validation_radius
        <= knot_maximum * (1.0 + support_tolerance) + support_tolerance
    )
    drive_in_pa_support = bool(
        maximum_drive
        <= pa_train_input_peak * (1.0 + support_tolerance) + support_tolerance
    )
    full_rank = factorization.data_design_rank == factorization.feature_count
    requested_knots_preserved = model.knot_count == int(recipe["knot_count"])
    structural_reasons: list[str] = []
    if not finite:
        structural_reasons.append("non_finite_drive_or_cascade")
    if not training_finite:
        structural_reasons.append("non_finite_train_reconstruction")
    if not full_rank:
        structural_reasons.append("steady_data_design_rank_deficient")
    if not validation_in_knots:
        structural_reasons.append("validation_desired_amplitude_outside_train_knots")
    if not drive_in_pa_support:
        structural_reasons.append(
            "predistorted_drive_outside_train_pa_input_support"
        )
    if not requested_knots_preserved:
        structural_reasons.append("duplicate_train_quantiles_reduced_knot_count")
    metrics = (
        _error_metrics(
            cascade,
            ideal,
            warmup_samples=common_warmup_samples,
        )
        if finite
        else {
            "complex_nmse_pooled_db": None,
            "perfect_reconstruction": False,
            "mse": None,
            "failure": "non-finite validation cascade",
        }
    )
    improvement = (
        _improvement_gate(metrics, no_dpd_metrics)
        if finite
        else {
            "pass": False,
            "candidate_error_power": None,
            "no_dpd_error_power": float(no_dpd_metrics["mse"]),
            "improvement_db_10log10_no_dpd_over_candidate": None,
            "policy": "strictly lower correct-direction validation error power",
            "failure": "non-finite validation cascade",
        }
    )
    reasons = list(structural_reasons)
    if not improvement["pass"]:
        reasons.append("does_not_improve_no_dpd")
    operation_count = model.operation_count().to_dict()
    return (
        {
            "candidate_kind": "spline_memory_dpd",
            "topology_name": recipe["topology_name"],
            "branches": [
                {
                    "signal_delay": branch.signal_delay,
                    "envelope_delay": branch.envelope_delay,
                }
                for branch in model.branches
            ],
            "requested_knot_count": int(recipe["knot_count"]),
            "knot_count": model.knot_count,
            "ridge": float(recipe["ridge"]),
            "fit_direction": (
                "train-slice ILA: measured y/g -> known PA input x"
            ),
            "fit_solver": {
                "name": "cached_direct_svd_normalized_steady_design",
                "objective": (
                    "mean(|Phi_steady*c-target_steady|^2) + "
                    "ridge*sum(|c|^2)"
                ),
                "factorization_reused_across_ridges": True,
                "factorization_seconds_shared": (
                    factorization.factorization_seconds
                ),
                "ridge_solve_seconds": solve_seconds,
                "causal_zero_padding_rows_used_for_fit_or_rank": False,
                "causal_warmup_samples_excluded": (
                    factorization.causal_warmup_samples
                ),
                "steady_scored_sample_count": factorization.steady_sample_count,
                "raw_steady_design_rank": factorization.data_design_rank,
                "raw_steady_design_feature_count": factorization.feature_count,
                "raw_steady_design_full_column_rank": full_rank,
                "singular_value_cutoff": factorization.singular_value_cutoff,
                "raw_steady_design_condition_number": (
                    factorization.data_design_condition_number
                ),
            },
            "train_ila_reconstruction_steady": training_metrics,
            "validation_direction": (
                "desired x -> DPD -> frozen PA -> compare with train-slice g*x"
            ),
            "validation_correct_direction": metrics,
            "selection_score_db": metrics.get("complex_nmse_pooled_db"),
            "improves_no_dpd_gate": improvement,
            "validation_drive": _signal_summary(drive),
            "support_checks": {
                "maximum_validation_desired_amplitude": maximum_validation_radius,
                "maximum_train_ila_input_knot": knot_maximum,
                "validation_desired_within_train_knots": validation_in_knots,
                "maximum_validation_predistorted_amplitude": maximum_drive,
                "maximum_train_pa_input_amplitude": pa_train_input_peak,
                "predistorted_drive_within_train_pa_input_support": (
                    drive_in_pa_support
                ),
                "requested_knot_count_preserved": requested_knots_preserved,
                "extrapolation_permitted": False,
            },
            "operation_count_class": "analytical_optimized_datapath",
            "operation_count_is_measured_numpy_timing": False,
            "operation_count_per_complex_sample": operation_count,
            "structurally_valid": not structural_reasons,
            "structural_invalid_reasons": structural_reasons,
            "eligible_for_diagnostic_selection": not structural_reasons,
            "hard_valid": not reasons,
            "hard_invalid_reasons": reasons,
        },
        model,
    )


def _normalized_psd(
    signals: dict[str, np.ndarray],
    *,
    nperseg: int,
    noverlap: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    common_reference_peak: float | None = None
    for name, signal in signals.items():
        frequency, spectrum = welch_numpy(
            signal,
            fs=1.0,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="spectrum",
            detrend="constant",
        )
        frequency = np.fft.fftshift(frequency)
        spectrum = np.fft.fftshift(np.asarray(spectrum, dtype=np.float64))
        if "normalized_frequency_cycles_per_sample" not in result:
            result["normalized_frequency_cycles_per_sample"] = frequency
        elif not np.array_equal(
            result["normalized_frequency_cycles_per_sample"], frequency
        ):
            raise RuntimeError("Welch frequency grids disagree")
        result[f"{name}_power"] = spectrum
        if name == "ideal":
            common_reference_peak = float(np.max(spectrum))
    if common_reference_peak is None or common_reference_peak <= 0.0:
        raise ValueError("ideal PSD must have positive peak power")
    tiny = np.finfo(np.float64).tiny
    for name in signals:
        result[f"{name}_db_relative_to_ideal_psd_peak"] = 10.0 * np.log10(
            np.maximum(result[f"{name}_power"], tiny) / common_reference_peak
        )
    return result


def _verify_pa_data_binding(
    pa_manifest: dict[str, Any],
    verification: dict[str, Any],
    *,
    normalization_scale: float,
) -> None:
    provenance = pa_manifest.get("data_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("PA manifest lacks data provenance")
    if provenance.get("selection_view_sha256") != verification["view_sha256"]:
        raise ValueError("PA bundle and current selection_view hash disagree")
    if provenance.get("verified_selection_files_sha256") != verification[
        "verified_file_hashes"
    ]:
        raise ValueError("PA bundle and current selection-file hashes disagree")
    if provenance.get("split_contract") != verification["split_contract"]:
        raise ValueError("PA bundle and current split contract disagree")
    expected_scale = float(verification["training_input_peak"])
    if not np.isclose(normalization_scale, expected_scale, rtol=1e-12, atol=0.0):
        raise ValueError("PA bundle and selection normalization scale disagree")


def select_from_config(config_path: str | Path) -> dict[str, Any]:
    """Fit the declared train-only ILA grid and select on validation cascade."""

    selection_started = time.perf_counter()
    raw_config = Path(config_path)
    source_config = (
        raw_config if raw_config.is_absolute() else PROJECT_ROOT / raw_config
    ).resolve()
    config = _load_config(source_config)
    raw_selection_dir = Path(str(config["selection_dir"]))
    configured_selection_dir = (
        raw_selection_dir
        if raw_selection_dir.is_absolute()
        else PROJECT_ROOT / raw_selection_dir
    )
    _reject_symlink_components(
        configured_selection_dir,
        name="selection_dir",
    )
    raw_pa_bundle_dir = Path(str(config["pa_bundle_dir"]))
    configured_pa_bundle_dir = (
        raw_pa_bundle_dir
        if raw_pa_bundle_dir.is_absolute()
        else PROJECT_ROOT / raw_pa_bundle_dir
    )
    _reject_symlink_components(
        configured_pa_bundle_dir,
        name="pa_bundle_dir",
    )
    selection_dir = _resolve_project_path(config["selection_dir"], name="selection_dir")
    pa_bundle_dir = _resolve_project_path(config["pa_bundle_dir"], name="pa_bundle_dir")
    output_dir = _resolve_project_path(config["output_dir"], name="output_dir")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )

    # Pin the complete PA evaluator before loading its model or fitting a DPD.
    pa_completion_path = pa_bundle_dir / "completion_manifest.json"
    if (
        pa_completion_path.is_symlink()
        or pa_completion_path.resolve().parent != pa_bundle_dir
    ):
        raise ValueError("PA completion manifest must be a contained regular file")
    expected_pa_completion_sha256 = str(
        config["expected_pa_completion_sha256"]
    )
    actual_pa_completion_sha256 = file_sha256(pa_completion_path)
    if actual_pa_completion_sha256 != expected_pa_completion_sha256:
        raise ValueError(
            "PA completion SHA-256 mismatch: refusing unpinned evaluator"
        )

    # This strict verifier rejects symlinks/escapes before reading the view or
    # its five explicitly named selection files.
    verification = _verify_selection_view(selection_dir)
    expected_selection_view_sha256 = str(
        config["expected_selection_view_sha256"]
    )
    actual_selection_view_sha256 = str(verification["view_sha256"])
    if actual_selection_view_sha256 != expected_selection_view_sha256:
        raise ValueError(
            "selection_view SHA-256 mismatch: refusing unpinned data view"
        )
    expected_source_sha256 = str(config["expected_source_sha256"])
    actual_source_sha256 = verification["view"].get("source_sha256")
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "source SHA-256 mismatch: refusing unpinned source capture"
        )
    frozen_pa = load_frozen_blackbox_pa_selection(pa_bundle_dir)
    _verify_pa_data_binding(
        frozen_pa.manifest,
        verification,
        normalization_scale=frozen_pa.normalization_scale,
    )
    recipes = enumerate_candidate_recipes(config)
    train_x, train_y, validation_x, validation_y, scaling = _load_normalized_pairs(
        selection_dir,
        scale=frozen_pa.normalization_scale,
        expected_counts=verification["split_contract"],
    )
    train_x, train_y = overlap_for_delay(
        train_x,
        train_y,
        frozen_pa.integer_delay_samples,
    )
    validation_x, validation_y = overlap_for_delay(
        validation_x,
        validation_y,
        frozen_pa.integer_delay_samples,
    )
    if max(recipe["maximum_delay"] for recipe in recipes) >= min(
        train_x.size,
        validation_x.size,
    ):
        raise ValueError("DPD branch delay consumes an aligned record")

    gain = complex_ls_gain(train_x, train_y)
    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("train-only complex gain is invalid")
    pa_warmup = _pa_warmup_samples(frozen_pa.model)
    maximum_dpd_warmup = max(recipe["maximum_delay"] for recipe in recipes)
    common_warmup = pa_warmup + maximum_dpd_warmup
    if common_warmup >= validation_x.size:
        raise ValueError("common DPD+PA warmup consumes validation")

    calibration_input = train_y / gain
    pa_train_input_peak = float(np.max(np.abs(train_x)))
    knot_strategy = str(config["knot_strategy"])
    ideal = gain * validation_x
    # Compute the zero-cost reference before any DPD factorization or
    # hyperparameter selection.  Candidate validity is defined relative to
    # this frozen reference and cannot move after seeing candidate results.
    no_dpd_output = np.asarray(
        frozen_pa.model.predict(validation_x),
        dtype=np.complex128,
    )
    no_dpd_metrics = _error_metrics(
        no_dpd_output,
        ideal,
        warmup_samples=common_warmup,
    )
    measured_validation_no_dpd_capture = _error_metrics(
        validation_y,
        ideal,
        warmup_samples=common_warmup,
    )
    no_dpd_trial: dict[str, Any] = {
        "candidate_kind": "no_dpd_reference",
        "topology_name": "no_dpd",
        "branches": [],
        "requested_knot_count": 0,
        "knot_count": 0,
        "ridge": 0.0,
        "validation_direction": (
            "desired x -> frozen PA -> compare with train-slice g*x"
        ),
        "validation_correct_direction": no_dpd_metrics,
        "selection_score_db": no_dpd_metrics["complex_nmse_pooled_db"],
        "validation_drive": _signal_summary(validation_x),
        "operation_count_class": "analytical_optimized_datapath",
        "operation_count_is_measured_numpy_timing": False,
        "operation_count_per_complex_sample": OperationCount().to_dict(),
        "structurally_valid": True,
        "eligible_for_diagnostic_selection": False,
        "hard_valid": True,
        "hard_invalid_reasons": [],
        "deployment_role": "reference_and_safe_fallback",
    }

    trials: list[dict[str, Any]] = []
    models: dict[tuple[str, int, float], SparseSplineMemoryDPD] = {}
    factorization_records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for recipe in recipes:
        grouped.setdefault(
            (str(recipe["topology_name"]), int(recipe["knot_count"])),
            [],
        ).append(recipe)

    for group_recipes in grouped.values():
        first = group_recipes[0]
        knots = make_knots(
            # Full input history is legitimate: a steady scored row n>=w can
            # reference calibration_input[n-d] with n-d<w.  Only regression
            # and target rows 0:w are excluded below.
            calibration_input,
            int(first["knot_count"]),
            knot_strategy,
        )
        factorization = factorize_spline_group(
            calibration_input,
            train_x,
            knots=knots,
            branches=first["branches"],
        )
        factorization_records.append(
            {
                "topology_name": first["topology_name"],
                "knot_count": int(first["knot_count"]),
                "seconds": factorization.factorization_seconds,
                "steady_sample_count": factorization.steady_sample_count,
                "feature_count": factorization.feature_count,
                "rank": factorization.data_design_rank,
            }
        )
        for recipe in group_recipes:
            trial, model = _evaluate_factorized_trial(
                recipe,
                factorization=factorization,
                calibration_input=calibration_input,
                train_x=train_x,
                validation_x=validation_x,
                pa_model=frozen_pa.model,
                gain=gain,
                common_warmup_samples=common_warmup,
                pa_train_input_peak=pa_train_input_peak,
                knot_strategy=knot_strategy,
                no_dpd_metrics=no_dpd_metrics,
            )
            trials.append(trial)
            models[
                (
                    str(recipe["topology_name"]),
                    int(recipe["knot_count"]),
                    float(recipe["ridge"]),
                )
            ] = model

    tolerance_db = _strict_float(
        config["selection_tolerance_db"],
        name="selection_tolerance_db",
        minimum=0.0,
    )
    selected_trial, selection_policy = select_dpd_candidate(
        trials,
        tolerance_db=tolerance_db,
    )
    selected_key = (
        str(selected_trial["topology_name"]),
        int(selected_trial["requested_knot_count"]),
        float(selected_trial["ridge"]),
    )
    selected_model = models[selected_key]
    frontier = pareto_frontier([no_dpd_trial, *trials])

    selected_drive = np.asarray(
        selected_model.predict(validation_x),
        dtype=np.complex128,
    )
    selected_output = np.asarray(
        frozen_pa.model.predict(selected_drive),
        dtype=np.complex128,
    )
    pa_validation_prediction = no_dpd_output
    selected_metrics = _error_metrics(
        selected_output,
        ideal,
        warmup_samples=common_warmup,
    )
    evaluator_fidelity = _error_metrics(
        pa_validation_prediction,
        validation_y,
        warmup_samples=common_warmup,
    )
    headroom = _evaluator_headroom_diagnostic(
        selected_metrics,
        evaluator_fidelity,
        required_margin_db=float(config["evaluator_headroom_gate_db"]),
    )

    psd_nperseg = _strict_integer(
        config["psd_nperseg"],
        name="psd_nperseg",
        minimum=2,
    )
    psd_noverlap = _strict_integer(
        config["psd_noverlap"],
        name="psd_noverlap",
        minimum=0,
    )
    if psd_noverlap >= psd_nperseg:
        raise ValueError("psd_noverlap must be less than psd_nperseg")
    if psd_nperseg > validation_x.size - common_warmup:
        raise ValueError("PSD segment is longer than scored validation")
    psd = _normalized_psd(
        {
            "ideal": ideal[common_warmup:],
            "no_dpd_pa_output": no_dpd_output[common_warmup:],
            "selected_dpd_pa_output": selected_output[common_warmup:],
            "selected_predistorted_drive": selected_drive[common_warmup:],
        },
        nperseg=psd_nperseg,
        noverlap=psd_noverlap,
    )
    selection_wall_seconds_before_publication = (
        time.perf_counter() - selection_started
    )

    pa_selection_manifest_path = pa_bundle_dir / "selection_manifest.json"
    pa_model_path = pa_bundle_dir / "selected_pa.npz"
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "task": "blackbox_validation_only_spline_memory_dpd_selection",
        "calibration_direction": "train-only measured y/g -> known x",
        "selection_direction": (
            "validation desired x -> DPD -> frozen PA -> compare train-only g*x"
        ),
        "measured_validation_y_usage": (
            "frozen PA fidelity and post-selection headroom diagnostic only; "
            "never DPD input"
        ),
        "common_warmup_samples_at_record_start": common_warmup,
        "no_dpd_reference": no_dpd_trial,
        "trials": trials,
    }
    frontier_document = {
        "schema_version": SCHEMA_VERSION,
        "task": "blackbox_validation_only_spline_memory_dpd_pareto_frontier",
        "dominance_dimensions": [
            "validation correct-direction complex NMSE",
            "real MUL",
            "real ADD",
            "real DIV",
            "nonlinear operations",
            "comparisons",
            "LUT accesses",
            "memory reads",
            "memory writes",
            "state values",
            "stored coefficients",
            "stored constants",
            "maximum predistorted amplitude",
        ],
        "frontier": frontier,
        "no_dpd_reference_included": True,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        staged_model = stage / "selected_dpd.npz"
        staged_trials = stage / "validation_trials.json"
        staged_frontier = stage / "pareto_frontier.json"
        staged_psd = stage / "normalized_psd.npz"
        staged_manifest = stage / "selection_manifest.json"
        staged_completion = stage / "completion_manifest.json"
        selected_model.save(staged_model)
        write_json(staged_trials, ledger)
        write_json(staged_frontier, frontier_document)
        np.savez(staged_psd, **psd)

        final_model = output_dir / "selected_dpd.npz"
        final_trials = output_dir / "validation_trials.json"
        final_frontier = output_dir / "pareto_frontier.json"
        final_psd = output_dir / "normalized_psd.npz"
        final_manifest = output_dir / "selection_manifest.json"
        source = Path(__file__).resolve()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "task": "blackbox_validation_only_spline_memory_dpd_selection",
            "scope": "train_validation_surrogate_only",
            "dataset_label": str(config["dataset_label"]),
            "semantics_status": verification["view"].get("semantics"),
            "claim_boundary": {
                "physical_pa_measurement": False,
                "surrogate_only": True,
                "x_as_desired_signal_is_provisional": True,
                "opendpd_superiority_proven": False,
                "huawei_compliance_proven": False,
            },
            "data_provenance": {
                "selection_directory": _manifest_path(selection_dir),
                "selection_view": _manifest_path(verification["view_path"]),
                "expected_source_sha256": expected_source_sha256,
                "actual_source_sha256": actual_source_sha256,
                "expected_actual_source_hash_match": True,
                "source_sha256": actual_source_sha256,
                "expected_selection_view_sha256": (
                    expected_selection_view_sha256
                ),
                "actual_selection_view_sha256": actual_selection_view_sha256,
                "expected_actual_selection_view_hash_match": True,
                "selection_view_sha256": actual_selection_view_sha256,
                "verified_selection_files_sha256": verification[
                    "verified_file_hashes"
                ],
                "split_contract": verification["split_contract"],
                "split_relationship": (
                    "adjacent chronological slices of one capture; not "
                    "independent captures"
                ),
                "sealed_or_test_data_available_to_selector": False,
            },
            "frozen_pa_evaluator": {
                "bundle_directory": _manifest_path(pa_bundle_dir),
                "expected_completion_manifest_sha256": (
                    expected_pa_completion_sha256
                ),
                "actual_completion_manifest_sha256": (
                    actual_pa_completion_sha256
                ),
                "expected_actual_completion_hash_match": True,
                "selection_manifest_sha256": file_sha256(pa_selection_manifest_path),
                "model_sha256": file_sha256(pa_model_path),
                "model_family": frozen_pa.manifest["selected_model"]["model_family"],
                "normalization_scale": frozen_pa.normalization_scale,
                "integer_delay_samples": frozen_pa.integer_delay_samples,
                "independent_from_dpd_candidate_fitting": True,
                "validation_independence": (
                    "non-independent: PA architecture and DPD are both selected "
                    "using the same validation slice"
                ),
            },
            "normalization": {
                **scaling,
                "policy": "divide x and y by one common train-input peak",
                "validation_refit": False,
            },
            "alignment": {
                "integer_delay_samples": frozen_pa.integer_delay_samples,
                "source": "frozen train-only PA bundle",
                "validation_refit": False,
                "fractional_delay_applied": False,
            },
            "gain": {
                "real": float(gain.real),
                "imag": float(gain.imag),
                "formula": "sum(conj(x_train)*y_train)/sum(abs(x_train)^2)",
                "fit_split": "aligned train only",
                "validation_refit": False,
            },
            "directions": {
                "ila_calibration": "train-only y/g -> x",
                "validation_deployment": (
                    "desired x -> DPD -> frozen PA -> compare g*x"
                ),
                "measured_validation_y": (
                    "used only for frozen PA fidelity/headroom after selection"
                ),
                "measured_validation_y_used_as_dpd_input": False,
            },
            "sequence_contract": {
                "train": (
                    "earlier chronological slice of the same source capture"
                ),
                "validation": (
                    "immediately adjacent later slice of the same source capture"
                ),
                "independent_captures": False,
                "state_reset": "once at each record start",
                "pa_warmup_samples": pa_warmup,
                "maximum_candidate_dpd_warmup_samples": maximum_dpd_warmup,
                "common_cascade_warmup_samples": common_warmup,
            },
            "candidate_grid": {
                "fit_count": len(recipes),
                "maximum_fit_count": int(config["maximum_fit_count"]),
                "topology_count": len(_branch_topologies(config["branch_topologies"])),
                "knot_counts": list(_integer_values(
                    config["knot_counts"], name="knot_counts", minimum=2
                )),
                "ridge_values": list(_ridge_values(config["ridge_values"])),
                "knot_strategy": knot_strategy,
                "selection_split": "validation",
                "selection_metric": "correct-direction pooled complex NMSE",
                "selection_tolerance_db": tolerance_db,
                "tie_policy": (
                    "within tolerance of best NMSE choose lexicographically "
                    "cheapest complete operation/storage vector"
                ),
                "test_used_for_architecture_or_hyperparameter_selection": False,
                "operation_count_class": "analytical_optimized_datapath",
                "operation_count_is_measured_numpy_timing": False,
            },
            "selection": {
                "selected_trial": selected_trial,
                **selection_policy,
                "selected_after_all_hard_validity_gates": bool(
                    selection_policy["deployment_recommended"]
                ),
                "no_dpd_reference": no_dpd_trial,
                "evaluator_headroom_used_for_ranking": False,
            },
            "selected_model": {
                "path": _manifest_path(final_model),
                "sha256": file_sha256(staged_model),
                "model_type": "phase_equivariant_sparse_spline_memory_dpd",
                "normalization_scale_required_for_source_units": True,
                "model_npz_is_not_safe_as_a_source_unit_api": True,
                "required_loader": (
                    "experiments.select_blackbox_dpd."
                    "load_frozen_blackbox_dpd_selection"
                ),
                "safe_source_unit_api": ["predict", "predict_chunk"],
                "source_unit_safety_wrapper": {
                    "checks_desired_against_knot_support": True,
                    "checks_each_output_or_chunk_against_train_pa_input_support": (
                        True
                    ),
                    "overhead_included_in_analytical_optimized_datapath_count": (
                        False
                    ),
                    "overhead": (
                        "runtime magnitude, maximum, and comparison safety "
                        "checks; deployment implementation-dependent"
                    ),
                },
                "artifact_role": selection_policy["selection_role"],
                "deployment_recommended": selection_policy[
                    "deployment_recommended"
                ],
            },
            "validation_summary": {
                "surrogate_no_dpd_reference_used_for_dpd_ranking": (
                    no_dpd_metrics
                ),
                "measured_validation_no_dpd_capture": {
                    **measured_validation_no_dpd_capture,
                    "diagnostic_only": True,
                    "used_for_dpd_ranking": False,
                    "meaning": (
                        "recorded y validation slice versus train-slice g*x; "
                        "this is an actual captured no-DPD baseline only if the "
                        "provisional x/y semantics are confirmed"
                    ),
                },
                "selected_dpd": selected_metrics,
                "selected_drive": _signal_summary(selected_drive),
                "no_dpd_drive": _signal_summary(validation_x),
            },
            "evaluator_headroom_gate": {
                "computed_after_selection": True,
                "used_for_ranking": False,
                "diagnostic_only": True,
                "independent_confirmation": False,
                "independence_limitation": (
                    "PA architecture and DPD were selected on the same "
                    "validation slice; train/validation are adjacent slices "
                    "of one capture"
                ),
                "evaluator_fidelity_against_measured_validation_y": (
                    evaluator_fidelity
                ),
                **headroom,
                "requirement_source": (
                    "internal diagnostic guard, not a Huawei requirement"
                ),
                "pass_does_not_confirm_physical_or_independent_performance": True,
            },
            "calibration_timing": {
                "measurement_scope": "local Python/NumPy wall-clock",
                "selection_wall_seconds_before_publication": float(
                    selection_wall_seconds_before_publication
                ),
                "selection_wall_scope": (
                    "select_from_config entry through data loading, grouped "
                    "factorization, all ridge solves, validation cascade "
                    "evaluation, selection, diagnostics, and PSD computation; "
                    "excludes artifact publication and inference timing"
                ),
                "factorization_count": len(factorization_records),
                "expected_factorization_count": len(grouped),
                "factorizations": factorization_records,
                "factorization_seconds_total": float(
                    sum(item["seconds"] for item in factorization_records)
                ),
                "ridge_solve_seconds_total": float(
                    sum(
                        trial["fit_solver"]["ridge_solve_seconds"]
                        for trial in trials
                    )
                ),
                "inference_throughput_measured": False,
                "inference_throughput_artifact": "pending separate benchmark",
            },
            "spectral_artifact": {
                "path": _manifest_path(final_psd),
                "sha256": file_sha256(staged_psd),
                "frequency_axis": "normalized cycles/sample, fs=1.0",
                "welch": {
                    "nperseg": psd_nperseg,
                    "noverlap": psd_noverlap,
                    "window": "periodic Hann",
                    "scaling": "spectrum",
                    "detrend": "constant",
                    "fft_shifted": True,
                },
                "aclr_or_harmonic_metrics_computed": False,
                "reason": "sample rate and RF acceptance regions are unknown",
            },
            "artifacts": {
                "validation_trials": _manifest_path(final_trials),
                "validation_trials_sha256": file_sha256(staged_trials),
                "pareto_frontier": _manifest_path(final_frontier),
                "pareto_frontier_sha256": file_sha256(staged_frontier),
                "selection_manifest": _manifest_path(final_manifest),
            },
            "config": {
                "path": _manifest_path(source_config),
                "sha256": file_sha256(source_config),
            },
            "code_sha256": {
                "experiments/select_blackbox_dpd.py": file_sha256(source),
                "experiments/select_blackbox_pa.py": file_sha256(
                    PROJECT_ROOT / "experiments" / "select_blackbox_pa.py"
                ),
                "baseline/spline_memory_dpd.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "spline_memory_dpd.py"
                ),
                "baseline/complex_spline_dpd.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "complex_spline_dpd.py"
                ),
                "baseline/alignment.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "alignment.py"
                ),
                "baseline/metrics.py": file_sha256(
                    PROJECT_ROOT / "baseline" / "metrics.py"
                ),
            },
            "determinism": {
                "stochastic_fitting": False,
                "seed": None,
                "coefficient_dtype": "complex128",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        }
        write_json(staged_manifest, manifest)
        bindings = {
            "selected_dpd.npz": file_sha256(staged_model),
            "validation_trials.json": file_sha256(staged_trials),
            "pareto_frontier.json": file_sha256(staged_frontier),
            "normalized_psd.npz": file_sha256(staged_psd),
            "selection_manifest.json": file_sha256(staged_manifest),
        }
        completion = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "blackbox_dpd_selection_completion",
            "status": "complete",
            "bound_files_sha256": bindings,
            "publication_contract": (
                "the bundle is valid only when every bound hash verifies"
            ),
        }
        write_json(staged_completion, completion)
        load_frozen_blackbox_dpd_selection(stage)
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to replace concurrently created output: {output_dir}"
            )
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


@dataclasses.dataclass(frozen=True)
class FrozenBlackBoxDPDSelection:
    model: SparseSplineMemoryDPD
    normalization_scale: float
    manifest: dict[str, Any]

    def _maximum_normalized_pa_input(self) -> float:
        value = float(
            self.manifest.get("selection", {})
            .get("selected_trial", {})
            .get("support_checks", {})
            .get("maximum_train_pa_input_amplitude", float("nan"))
        )
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("frozen DPD manifest lacks valid PA input support")
        return value

    def _normalized_source_input(self, signal: np.ndarray) -> np.ndarray:
        source = np.asarray(signal)
        if source.ndim != 1 or source.size == 0:
            raise ValueError("source-unit DPD input must be a non-empty vector")
        values = np.asarray(source, dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise ValueError("source-unit DPD input contains non-finite values")
        normalized = values / self.normalization_scale
        tolerance = 64.0 * np.finfo(float).eps
        if float(np.max(np.abs(normalized))) > (
            float(self.model.knots[-1]) * (1.0 + tolerance) + tolerance
        ):
            raise ValueError(
                "source-unit DPD input exceeds frozen train-knot support"
            )
        return normalized

    def _validate_normalized_drive(self, drive: np.ndarray) -> None:
        values = np.asarray(drive, dtype=np.complex128)
        if not np.all(np.isfinite(values)):
            raise ValueError("frozen DPD produced non-finite drive samples")
        tolerance = 64.0 * np.finfo(float).eps
        maximum_support = self._maximum_normalized_pa_input()
        if float(np.max(np.abs(values))) > (
            maximum_support * (1.0 + tolerance) + tolerance
        ):
            raise ValueError(
                "frozen DPD output exceeds frozen train PA-input support"
            )

    def initial_state(self) -> SplineMemoryState:
        return self.model.initial_state()

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Predistort source-unit samples with explicit normalize/de-normalize."""

        source = np.asarray(signal)
        normalized = self._normalized_source_input(source)
        normalized_output = self.model.predict(normalized)
        self._validate_normalized_drive(normalized_output)
        output = normalized_output * self.normalization_scale
        dtype = np.complex64 if source.dtype == np.complex64 else np.complex128
        return np.asarray(output, dtype=dtype)

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: SplineMemoryState,
    ) -> tuple[np.ndarray, SplineMemoryState]:
        """Streaming source-unit inference with explicit carried DPD state."""

        source = np.asarray(signal)
        normalized = self._normalized_source_input(source)
        normalized_output, next_state = self.model.predict_chunk(normalized, state)
        self._validate_normalized_drive(normalized_output)
        output = normalized_output * self.normalization_scale
        dtype = np.complex64 if source.dtype == np.complex64 else np.complex128
        return np.asarray(output, dtype=dtype), next_state


def load_frozen_blackbox_dpd_selection(
    output_dir: str | Path,
) -> FrozenBlackBoxDPDSelection:
    """Verify the completed DPD bundle before loading model and scale."""

    bundle = _resolve_project_path(output_dir, name="output_dir")
    completion = _load_json_object(
        bundle / "completion_manifest.json",
        label="completion_manifest",
    )
    if (
        int(completion.get("schema_version", -1)) != SCHEMA_VERSION
        or completion.get("artifact_type") != "blackbox_dpd_selection_completion"
        or completion.get("status") != "complete"
    ):
        raise ValueError("invalid or incomplete BlackBox DPD selection bundle")
    bindings = completion.get("bound_files_sha256")
    if not isinstance(bindings, dict) or set(bindings) != BOUND_OUTPUTS:
        raise ValueError("DPD completion manifest has incomplete file bindings")
    for name in sorted(BOUND_OUTPUTS):
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
    spectral = manifest.get("spectral_artifact")
    if not all(isinstance(item, dict) for item in (model_record, artifacts, spectral)):
        raise ValueError("DPD selection manifest lacks artifact bindings")
    if model_record.get("sha256") != bindings["selected_dpd.npz"]:
        raise ValueError("DPD model hash disagrees with completion manifest")
    if artifacts.get("validation_trials_sha256") != bindings[
        "validation_trials.json"
    ]:
        raise ValueError("DPD trial hash disagrees with completion manifest")
    if artifacts.get("pareto_frontier_sha256") != bindings[
        "pareto_frontier.json"
    ]:
        raise ValueError("DPD Pareto hash disagrees with completion manifest")
    if spectral.get("sha256") != bindings["normalized_psd.npz"]:
        raise ValueError("DPD PSD hash disagrees with completion manifest")
    if Path(str(model_record.get("path"))).name != "selected_dpd.npz":
        raise ValueError("DPD model path is inconsistent")
    scale = float(
        manifest.get("frozen_pa_evaluator", {}).get(
            "normalization_scale",
            float("nan"),
        )
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("DPD bundle normalization scale is invalid")
    frozen = FrozenBlackBoxDPDSelection(
        SparseSplineMemoryDPD.load(bundle / "selected_dpd.npz"),
        scale,
        manifest,
    )
    # Reject a malformed safety contract at load time, before any source-unit
    # inference can be attempted through the frozen wrapper.
    frozen._maximum_normalized_pa_input()
    return frozen


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a spline-memory DPD on BlackBox validation through a "
            "hash-bound frozen PA evaluator."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = select_from_config(args.config)
    selected = manifest["selection"]["selected_trial"]
    gate = manifest["evaluator_headroom_gate"]
    print(
        "Selected BlackBox DPD:",
        f"topology={selected['topology_name']}",
        f"K={selected['knot_count']}",
        f"ridge={selected['ridge']}",
        f"validation_nmse_db={selected['selection_score_db']}",
        f"evaluator_headroom_pass={gate['pass']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
