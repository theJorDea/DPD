"""Preregistered staged selection for a standalone spline-Hammerstein PA.

Recipe enumeration and ranking are intentionally independent of waveform I/O.
The experiment runner added below this layer must first prove that its exact
candidate set and maximum fit count agree with the frozen JSON protocol.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from baseline.complexity import OperationCount, spline_hammerstein_pa_cost
from baseline.metrics import (
    nmse_opendpd_db,
    nmse_pooled_db,
    time_domain_rms_evm_db,
)
from baseline.spline_hammerstein_pa import (
    SplineHammersteinPA,
    fit_spline_hammerstein_pa,
    sph_coordinate_values,
)


SCHEMA_VERSION = 1
SUPPORTED_TASK = "forward_pa_model_spline_hammerstein_selection"
S0_RETENTION_WINDOW_DB = 0.05
S0_MAX_RETAINED_TOPOLOGIES = 2
VARIANT_COORDINATES = {
    "amplitude_uniform": "amplitude",
    "amplitude_uniform_power_placement": "amplitude",
    "amplitude_quantile": "amplitude",
    "amplitude_compression_aware_p2": "amplitude",
    "power_uniform": "power",
}


@dataclasses.dataclass(frozen=True)
class SPHOOFProtocol:
    segment_length: int
    common_warmup_samples: int
    common_cooldown_samples: int = 0
    maximum_alternations: int = 20
    minimum_alternations: int = 2
    convergence_tolerance: float = 1e-7
    objective_increase_tolerance: float = 1e-10
    real_multiplication_limit_exclusive: int = 1000

    def __post_init__(self) -> None:
        for name, minimum in (
            ("segment_length", 1),
            ("common_warmup_samples", 0),
            ("common_cooldown_samples", 0),
            ("maximum_alternations", 1),
            ("minimum_alternations", 1),
            ("real_multiplication_limit_exclusive", 1),
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.minimum_alternations > self.maximum_alternations:
            raise ValueError("minimum alternations exceed maximum alternations")
        if (
            self.common_warmup_samples + self.common_cooldown_samples
            >= self.segment_length
        ):
            raise ValueError("common boundary exclusion consumes a full frame")
        for name in (
            "convergence_tolerance",
            "objective_increase_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclasses.dataclass(frozen=True)
class SPHRecipe:
    variant: str
    knot_count: int
    fir_length: int
    control_ridge: float
    smoothness: float
    fir_ridge: float

    def __post_init__(self) -> None:
        if self.variant not in VARIANT_COORDINATES:
            raise ValueError(f"unsupported SPH variant: {self.variant}")
        for name in ("knot_count", "fir_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < (2 if name == "knot_count" else 1):
                raise ValueError(f"invalid {name}: {value}")
        for name in ("control_ridge", "smoothness", "fir_ridge"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def coordinate(self) -> str:
        return VARIANT_COORDINATES[self.variant]

    @property
    def operation_count(self) -> OperationCount:
        return spline_hammerstein_pa_cost(
            self.knot_count,
            self.fir_length,
            coordinate=self.coordinate,
        )

    @property
    def name(self) -> str:
        return (
            f"{self.variant}_K{self.knot_count}_L{self.fir_length}"
            f"_cr{self.control_ridge:.0e}_sm{self.smoothness:.0e}"
            f"_fr{self.fir_ridge:.0e}"
        )

    @property
    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "coordinate": self.coordinate,
            "knot_count": self.knot_count,
            "fir_length": self.fir_length,
            "control_ridge": self.control_ridge,
            "smoothness": self.smoothness,
            "fir_ridge": self.fir_ridge,
            "name": self.name,
        }


def load_sph_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SPH config must contain one JSON object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("SPH config schema_version mismatch")
    if config.get("task") != SUPPORTED_TASK:
        raise ValueError("unexpected SPH selection task")
    if config.get("status") != (
        "preregistered_before_model_implementation_and_candidate_fit"
    ):
        raise ValueError("SPH config does not carry the preregistered status")
    scope = config.get("scope")
    if not isinstance(scope, dict) or scope.get(
        "test_split_access_permitted"
    ) is not False:
        raise ValueError("SPH selection config must prohibit test access")
    required = {
        "dataset",
        "output_dir",
        "fit",
        "coordinate_and_knot_variants",
        "staged_search",
        "search_budget",
        "operation_count_convention",
        "selection",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"SPH config is missing keys: {sorted(missing)}")
    return config


def _unique_ints(
    values: Any,
    *,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        for value in values
    ):
        raise ValueError(f"{name} contains an invalid integer")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _unique_floats(values: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} values must be finite and non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _stage(config: dict[str, Any], name: str) -> dict[str, Any]:
    stages = config["staged_search"]
    if not isinstance(stages, list):
        raise ValueError("staged_search must be a list")
    matches = [stage for stage in stages if stage.get("stage") == name]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f"staged_search must contain exactly one {name}")
    return matches[0]


def _default_regularization(
    config: dict[str, Any],
) -> tuple[float, float, float]:
    fit = config["fit"]
    if not isinstance(fit, dict):
        raise ValueError("fit must be an object")
    values = tuple(
        float(fit[key])
        for key in (
            "default_control_ridge",
            "default_smoothness",
            "default_fir_ridge",
        )
    )
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("default regularization is invalid")
    return values  # type: ignore[return-value]


def enumerate_s0_recipes(config: dict[str, Any]) -> tuple[SPHRecipe, ...]:
    stage = _stage(config, "S0_coordinate_and_memory")
    knot_counts = _unique_ints(
        stage.get("knot_counts"),
        name="S0 knot_counts",
        minimum=2,
    )
    lengths = _unique_ints(
        stage.get("fir_lengths"),
        name="S0 fir_lengths",
        minimum=1,
    )
    variants = stage.get("variant_names")
    if not isinstance(variants, list) or not variants:
        raise ValueError("S0 variant_names must be a non-empty list")
    if len(set(variants)) != len(variants):
        raise ValueError("S0 variant_names must be unique")
    if any(variant not in VARIANT_COORDINATES for variant in variants):
        raise ValueError("S0 contains an unsupported variant")
    declared_variants = {
        item.get("name")
        for item in config["coordinate_and_knot_variants"]
        if isinstance(item, dict)
    }
    if set(variants) != declared_variants:
        raise ValueError("S0 variants disagree with coordinate declarations")
    control_ridge, smoothness, fir_ridge = _default_regularization(config)
    return tuple(
        SPHRecipe(
            variant=str(variant),
            knot_count=knot_count,
            fir_length=fir_length,
            control_ridge=control_ridge,
            smoothness=smoothness,
            fir_ridge=fir_ridge,
        )
        for knot_count in knot_counts
        for variant in variants
        for fir_length in lengths
    )


def enumerate_s1_recipes(
    config: dict[str, Any],
    retained_topologies: Iterable[tuple[str, int]],
) -> tuple[SPHRecipe, ...]:
    topologies = tuple(retained_topologies)
    if not 1 <= len(topologies) <= S0_MAX_RETAINED_TOPOLOGIES:
        raise ValueError("S1 requires one or two retained S0 topologies")
    if len(set(topologies)) != len(topologies):
        raise ValueError("S1 retained topologies must be unique")
    stage = _stage(config, "S1_knot_count")
    knot_counts = _unique_ints(
        stage.get("knot_counts"),
        name="S1 knot_counts",
        minimum=2,
    )
    control_ridge, smoothness, fir_ridge = _default_regularization(config)
    return tuple(
        SPHRecipe(
            variant=variant,
            knot_count=knot_count,
            fir_length=fir_length,
            control_ridge=control_ridge,
            smoothness=smoothness,
            fir_ridge=fir_ridge,
        )
        for variant, fir_length in topologies
        for knot_count in knot_counts
    )


def enumerate_s2_recipes(
    config: dict[str, Any],
    topology: tuple[str, int, int],
) -> tuple[SPHRecipe, ...]:
    variant, fir_length, knot_count = topology
    stage = _stage(config, "S2_control_and_smoothness")
    control_ridges = _unique_floats(
        stage.get("control_ridges"),
        name="S2 control_ridges",
    )
    smoothnesses = _unique_floats(
        stage.get("smoothnesses"),
        name="S2 smoothnesses",
    )
    fir_ridge = float(stage.get("fir_ridge"))
    if not np.isfinite(fir_ridge) or fir_ridge < 0.0:
        raise ValueError("S2 fir_ridge is invalid")
    return tuple(
        SPHRecipe(
            variant=variant,
            knot_count=knot_count,
            fir_length=fir_length,
            control_ridge=control_ridge,
            smoothness=smoothness,
            fir_ridge=fir_ridge,
        )
        for control_ridge in control_ridges
        for smoothness in smoothnesses
    )


def enumerate_s3_recipes(
    config: dict[str, Any],
    recipe: SPHRecipe,
) -> tuple[SPHRecipe, ...]:
    stage = _stage(config, "S3_fir_ridge")
    fir_ridges = _unique_floats(
        stage.get("fir_ridges"),
        name="S3 fir_ridges",
    )
    return tuple(
        SPHRecipe(
            variant=recipe.variant,
            knot_count=recipe.knot_count,
            fir_length=recipe.fir_length,
            control_ridge=recipe.control_ridge,
            smoothness=recipe.smoothness,
            fir_ridge=fir_ridge,
        )
        for fir_ridge in fir_ridges
    )


def _finite_score(value: Any, *, name: str) -> float:
    score = float(value)
    if not np.isfinite(score) and score != -np.inf:
        raise ValueError(f"{name} must be finite or negative infinity")
    return score


def _trial_recipe(trial: dict[str, Any]) -> SPHRecipe:
    recipe = trial.get("recipe")
    if not isinstance(recipe, SPHRecipe):
        raise TypeError("trial.recipe must be an SPHRecipe")
    return recipe


def select_ranked_trial(
    trials: Iterable[dict[str, Any]],
    *,
    tolerance_db: float,
) -> dict[str, Any]:
    """Apply primary-window, secondary-quality, then hardware tie-breaks."""

    if not np.isfinite(tolerance_db) or tolerance_db < 0.0:
        raise ValueError("ranking tolerance must be finite and non-negative")
    rows = [trial for trial in trials if trial.get("hard_valid") is True]
    if not rows:
        raise ValueError("no hard-valid SPH trial is available")
    identities = [_trial_recipe(row).canonical_sha256 for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate hard-valid SPH trial recipe")
    best_primary = min(
        _finite_score(row["full_record_nmse_db"], name="full NMSE")
        for row in rows
    )
    near = [
        row
        for row in rows
        if _finite_score(row["full_record_nmse_db"], name="full NMSE")
        <= best_primary + tolerance_db
    ]
    near.sort(
        key=lambda row: (
            _finite_score(
                row["common_interior_nmse_db"],
                name="common NMSE",
            ),
            _trial_recipe(row).operation_count.real_multiplications,
            _trial_recipe(row).operation_count.nonlinear_operations,
            _trial_recipe(row).operation_count.stored_real_coefficients,
            _trial_recipe(row).fir_length,
            _trial_recipe(row).knot_count,
            _trial_recipe(row).name,
        )
    )
    return near[0]


def retain_s0_topologies(
    trials: Iterable[dict[str, Any]],
) -> tuple[tuple[str, int], ...]:
    """Keep at most two hard-valid topology points inside the 0.05 dB window."""

    rows = [trial for trial in trials if trial.get("hard_valid") is True]
    if not rows:
        raise ValueError("S0 produced no hard-valid topology")
    best = min(
        _finite_score(row["full_record_nmse_db"], name="S0 full NMSE")
        for row in rows
    )
    retained = [
        row
        for row in rows
        if _finite_score(row["full_record_nmse_db"], name="S0 full NMSE")
        <= best + S0_RETENTION_WINDOW_DB
    ]
    retained.sort(
        key=lambda row: (
            _finite_score(row["full_record_nmse_db"], name="S0 full NMSE"),
            _finite_score(
                row["common_interior_nmse_db"],
                name="S0 common NMSE",
            ),
            _trial_recipe(row).operation_count.nonlinear_operations,
            _trial_recipe(row).fir_length,
            _trial_recipe(row).name,
        )
    )
    return tuple(
        (recipe.variant, recipe.fir_length)
        for recipe in (
            _trial_recipe(row)
            for row in retained[:S0_MAX_RETAINED_TOPOLOGIES]
        )
    )


def validate_search_budget(config: dict[str, Any]) -> dict[str, int]:
    """Prove the preregistered worst-case recipe and OOF-fit count."""

    s0 = enumerate_s0_recipes(config)
    s0_count = len(s0)
    placeholder_topologies = (
        ("amplitude_uniform", 1),
        ("power_uniform", 8),
    )
    s1 = enumerate_s1_recipes(config, placeholder_topologies)
    s1_count = len(s1)
    s2 = enumerate_s2_recipes(
        config,
        (s1[0].variant, s1[0].fir_length, s1[0].knot_count),
    )
    s2_count = len(s2)
    s3_count = len(enumerate_s3_recipes(config, s2[0]))
    budget = config["search_budget"]
    declared = {
        "S0": int(budget["maximum_S0_candidate_recipes"]),
        "S1": int(budget["maximum_S1_candidate_recipes"]),
        "S2": int(budget["maximum_S2_candidate_recipes"]),
        "S3": int(budget["maximum_S3_candidate_recipes"]),
    }
    actual = {"S0": s0_count, "S1": s1_count, "S2": s2_count, "S3": s3_count}
    if actual != declared:
        raise ValueError(
            f"enumerated SPH search {actual} disagrees with config {declared}"
        )
    raw_folds = budget["oof_fold_count"]
    if not isinstance(raw_folds, int) or isinstance(raw_folds, bool):
        raise ValueError("oof_fold_count must be an integer")
    folds = int(raw_folds)
    if folds < 2:
        raise ValueError("oof_fold_count must be at least two")
    fit_calls = folds * sum(actual.values())
    if fit_calls != int(budget["maximum_oof_fit_calls_without_cache"]):
        raise ValueError("SPH maximum OOF fit count disagrees with config")
    operation = config["operation_count_convention"]
    if operation.get("complex_multiply") != (
        "4 real multiplications + 2 real additions"
    ):
        raise ValueError("SPH runner requires the frozen 4M+2A convention")
    limit = operation.get("real_multiplication_limit_exclusive")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("real multiplication limit must be a positive integer")
    if any(recipe.operation_count.real_multiplications >= limit for recipe in s0):
        raise ValueError("an S0 recipe violates the strict multiplication limit")
    frozen_points = operation.get("frozen_length_points")
    if not isinstance(frozen_points, list) or not frozen_points:
        raise ValueError("frozen_length_points must be a non-empty list")
    for point in frozen_points:
        if not isinstance(point, dict):
            raise ValueError("every frozen length point must be an object")
        length = int(point["L"])
        expected = spline_hammerstein_pa_cost(24, length)
        observed = (
            int(point["real_multiplications"]),
            int(point["real_additions"]),
            int(point["state_real_values"]),
        )
        calculated = (
            expected.real_multiplications,
            expected.real_additions,
            expected.state_real_values,
        )
        if observed != calculated:
            raise ValueError("frozen SPH operation point disagrees with code")
    return {**actual, "folds": folds, "maximum_oof_fit_calls": fit_calls}


def _complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _frame_ids(sample_count: int, segment_length: int) -> np.ndarray:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return np.arange(sample_count, dtype=np.int64) // int(segment_length)


def _common_mask(
    sample_count: int,
    protocol: SPHOOFProtocol,
) -> np.ndarray:
    mask = np.zeros(sample_count, dtype=bool)
    for start in range(0, sample_count, protocol.segment_length):
        stop = min(start + protocol.segment_length, sample_count)
        left = min(start + protocol.common_warmup_samples, stop)
        right = max(start, stop - protocol.common_cooldown_samples)
        if left < right:
            mask[left:right] = True
    if not np.any(mask):
        raise ValueError("common boundary mask contains no scored samples")
    return mask


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _streaming_checks(
    model: SplineHammersteinPA,
    signal: np.ndarray,
) -> dict[str, Any]:
    samples = _complex_vector(signal, name="streaming_signal")
    expected = model.predict(samples)
    boundaries = sorted(
        {
            0,
            1,
            samples.size // 3,
            min(samples.size // 3 + 7, samples.size),
            max(samples.size - 2, 0),
            samples.size,
        }
    )
    state = model.initial_state()
    chunks: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop == start:
            continue
        output, state = model.predict_chunk(samples[start:stop], state)
        chunks.append(output)
    streamed = np.concatenate(chunks)
    reset = model.predict_segments(samples, samples.size)
    streaming_error = float(np.max(np.abs(streamed - expected), initial=0.0))
    reset_error = float(np.max(np.abs(reset - expected), initial=0.0))
    return {
        "streaming_chunk_equivalence_passed": bool(
            np.array_equal(streamed, expected)
        ),
        "reset_at_frame_equivalence_passed": bool(
            np.array_equal(reset, expected)
        ),
        "maximum_streaming_error": streaming_error,
        "maximum_reset_error": reset_error,
        "final_state_sha256": _array_sha256(state.history),
    }


def _metric_summary(
    prediction: np.ndarray,
    reference: np.ndarray,
    protocol: SPHOOFProtocol,
) -> dict[str, Any]:
    estimate = _complex_vector(prediction, name="prediction")
    target = _complex_vector(reference, name="reference")
    if estimate.shape != target.shape:
        raise ValueError("prediction and reference must have equal length")
    mask = _common_mask(estimate.size, protocol)
    frame_ids = _frame_ids(estimate.size, protocol.segment_length)
    per_frame = []
    complete_estimates: list[np.ndarray] = []
    complete_targets: list[np.ndarray] = []
    for frame_id in np.unique(frame_ids):
        indices = np.flatnonzero(frame_ids == frame_id)
        per_frame.append(
            {
                "frame_id": int(frame_id),
                "sample_count": int(indices.size),
                "full_record_nmse_db": nmse_pooled_db(
                    estimate[indices],
                    target[indices],
                ),
            }
        )
        if indices.size == protocol.segment_length:
            complete_estimates.append(estimate[indices])
            complete_targets.append(target[indices])
    opendpd = (
        nmse_opendpd_db(
            np.stack(complete_estimates),
            np.stack(complete_targets),
        )
        if complete_estimates
        else None
    )
    return {
        "full_record_nmse_db": nmse_pooled_db(estimate, target),
        "common_interior_nmse_db": nmse_pooled_db(
            estimate[mask],
            target[mask],
        ),
        "full_record_time_domain_rms_evm_db": time_domain_rms_evm_db(
            estimate,
            target,
        ),
        "common_interior_time_domain_rms_evm_db": time_domain_rms_evm_db(
            estimate[mask],
            target[mask],
        ),
        "opendpd_compatible_nmse_db": opendpd,
        "opendpd_complete_frame_count": len(complete_estimates),
        "scored_sample_count_full": int(estimate.size),
        "scored_sample_count_common": int(np.count_nonzero(mask)),
        "per_frame": per_frame,
    }


def evaluate_oof_recipe(
    recipe: SPHRecipe,
    train_input: np.ndarray,
    train_output: np.ndarray,
    *,
    protocol: SPHOOFProtocol,
    reference_gmp_oof_prediction: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit one recipe leave-one-explicit-frame-out on the train split only."""

    samples = _complex_vector(train_input, name="train_input")
    target = _complex_vector(train_output, name="train_output")
    if samples.shape != target.shape:
        raise ValueError("train_input and train_output must have equal length")
    reference_gmp = None
    if reference_gmp_oof_prediction is not None:
        reference_gmp = _complex_vector(
            reference_gmp_oof_prediction,
            name="reference_gmp_oof_prediction",
        )
        if reference_gmp.shape != target.shape:
            raise ValueError("reference GMP OOF prediction has the wrong length")
    frame_ids = _frame_ids(samples.size, protocol.segment_length)
    unique_frames = tuple(int(value) for value in np.unique(frame_ids))
    if len(unique_frames) < 2:
        raise ValueError("OOF evaluation requires at least two explicit frames")
    if recipe.fir_length > protocol.segment_length:
        raise ValueError("recipe FIR length exceeds the explicit frame length")
    cost = recipe.operation_count
    if cost.real_multiplications >= protocol.real_multiplication_limit_exclusive:
        raise ValueError("recipe violates the strict real multiplication limit")

    prediction = np.empty(samples.size, dtype=np.complex128)
    fold_reports: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    for held_frame in unique_frames:
        fit_indices = np.flatnonzero(frame_ids != held_frame)
        held_indices = np.flatnonzero(frame_ids == held_frame)
        started = time.perf_counter()
        model, diagnostics = fit_spline_hammerstein_pa(
            samples[fit_indices],
            target[fit_indices],
            knot_count=recipe.knot_count,
            knot_variant=recipe.variant,  # type: ignore[arg-type]
            fir_length=recipe.fir_length,
            segment_length=protocol.segment_length,
            control_ridge=recipe.control_ridge,
            smoothness=recipe.smoothness,
            fir_ridge=recipe.fir_ridge,
            maximum_alternations=protocol.maximum_alternations,
            minimum_alternations=protocol.minimum_alternations,
            convergence_tolerance=protocol.convergence_tolerance,
            objective_increase_tolerance=(
                protocol.objective_increase_tolerance
            ),
            coefficient_dtype=np.complex128,
        )
        fit_seconds = time.perf_counter() - started
        total_fit_seconds += fit_seconds
        held_prediction = model.predict(samples[held_indices])
        prediction[held_indices] = held_prediction
        held_size = int(held_indices.size)
        held_warmup = min(
            protocol.common_warmup_samples,
            max(held_size - 1, 0),
        )
        held_cooldown = min(
            protocol.common_cooldown_samples,
            max(held_size - held_warmup - 1, 0),
        )
        held_protocol = SPHOOFProtocol(
            segment_length=held_size,
            common_warmup_samples=held_warmup,
            common_cooldown_samples=held_cooldown,
            maximum_alternations=protocol.maximum_alternations,
            minimum_alternations=protocol.minimum_alternations,
            convergence_tolerance=protocol.convergence_tolerance,
            objective_increase_tolerance=protocol.objective_increase_tolerance,
            real_multiplication_limit_exclusive=(
                protocol.real_multiplication_limit_exclusive
            ),
        )
        held_metrics = _metric_summary(
            held_prediction,
            target[held_indices],
            held_protocol,
        )
        coordinate = sph_coordinate_values(
            samples[held_indices],
            model.coordinate,
        )
        above = coordinate > model.knots[-1]
        gmp_fold_nmse = None
        gmp_fold_common_nmse = None
        fold_gain = None
        fold_common_gain = None
        if reference_gmp is not None:
            gmp_held_metrics = _metric_summary(
                reference_gmp[held_indices],
                target[held_indices],
                held_protocol,
            )
            gmp_fold_nmse = float(gmp_held_metrics["full_record_nmse_db"])
            gmp_fold_common_nmse = float(
                gmp_held_metrics["common_interior_nmse_db"]
            )
            fold_gain = gmp_fold_nmse - float(
                held_metrics["full_record_nmse_db"]
            )
            fold_common_gain = gmp_fold_common_nmse - float(
                held_metrics["common_interior_nmse_db"]
            )
        stream_checks = _streaming_checks(model, samples[held_indices])
        fold_reports.append(
            {
                "held_frame_id": held_frame,
                "fit_frame_ids": [
                    frame for frame in unique_frames if frame != held_frame
                ],
                "fit_sample_count": int(fit_indices.size),
                "held_sample_count": int(held_indices.size),
                "fit_seconds": fit_seconds,
                "fit_diagnostics": dataclasses.asdict(diagnostics),
                "held_metrics": held_metrics,
                "reference_gmp_full_record_nmse_db": gmp_fold_nmse,
                "reference_gmp_common_interior_nmse_db": gmp_fold_common_nmse,
                "gain_over_gmp_full_record_db": fold_gain,
                "gain_over_gmp_common_interior_db": fold_common_gain,
                "streaming_checks": stream_checks,
                "knot_sha256": _array_sha256(model.knots),
                "control_points_sha256": _array_sha256(model.control_points),
                "fir_tail_sha256": _array_sha256(model.fir_tail),
                "prediction_sha256": _array_sha256(held_prediction),
                "input_support": {
                    "fraction_above_fit_maximum": float(np.mean(above)),
                    "count_above_fit_maximum": int(np.count_nonzero(above)),
                    "maximum_held_coordinate": float(np.max(coordinate)),
                    "maximum_fit_coordinate": float(model.knots[-1]),
                },
            }
        )

    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("SPH OOF prediction contains non-finite samples")
    metrics = _metric_summary(prediction, target, protocol)
    stream_pass = all(
        fold["streaming_checks"]["streaming_chunk_equivalence_passed"]
        and fold["streaming_checks"]["reset_at_frame_equivalence_passed"]
        for fold in fold_reports
    )
    monotonic_pass = all(
        fold["fit_diagnostics"]["all_updates_monotonic"]
        for fold in fold_reports
    )
    rank_pass = all(
        fold["fit_diagnostics"]["all_data_designs_full_column_rank"]
        for fold in fold_reports
    )
    finite_diagnostic_fields = (
        "zero_model_objective",
        "memoryless_initial_objective",
        "optimization_final_objective",
        "serialized_model_objective",
        "maximum_calibration_coordinate",
        "control_point_l2_norm",
        "fir_tail_l2_norm",
        "target_power",
        "training_mse",
        "training_relative_error_power",
        "fit_wall_time_seconds",
    )
    finite_update_fields = (
        "data_design_condition_number",
        "data_minimum_singular_value",
        "data_maximum_singular_value",
        "augmented_condition_number",
        "augmented_minimum_singular_value",
        "augmented_maximum_singular_value",
        "objective_before",
        "objective_after",
        "relative_objective_decrease",
        "coefficient_l2_norm",
    )
    numerics_pass = all(
        all(
            np.isfinite(float(fold["fit_diagnostics"][name]))
            for name in finite_diagnostic_fields
        )
        and all(
            np.isfinite(float(update[name]))
            for update in fold["fit_diagnostics"]["updates"]
            for name in finite_update_fields
        )
        for fold in fold_reports
    )
    serialized_relative_objective_deltas = [
        abs(
            float(fold["fit_diagnostics"]["serialized_model_objective"])
            - float(fold["fit_diagnostics"]["optimization_final_objective"])
        )
        / max(
            abs(
                float(
                    fold["fit_diagnostics"]["optimization_final_objective"]
                )
            ),
            np.finfo(np.float64).eps
            * float(fold["fit_diagnostics"]["target_power"]),
        )
        for fold in fold_reports
    ]
    recipe_pass = all(
        fold["fit_diagnostics"]["knot_count"] == recipe.knot_count
        and fold["fit_diagnostics"]["fir_length"] == recipe.fir_length
        and fold["fit_diagnostics"]["coordinate"] == recipe.coordinate
        and fold["fit_diagnostics"]["h0_contract"]
        == "1+0j fixed and not stored"
        for fold in fold_reports
    )
    hard_checks = {
        "all_predictions_finite": True,
        "all_updates_monotonic": monotonic_pass,
        "all_data_designs_full_column_rank": rank_pass,
        "all_required_numerics_finite": numerics_pass,
        "all_streaming_and_reset_checks_exact": stream_pass,
        "all_fold_recipes_match": recipe_pass,
        "real_multiplications_strictly_below_limit": (
            cost.real_multiplications
            < protocol.real_multiplication_limit_exclusive
        ),
    }
    fold_gains = [
        float(fold["gain_over_gmp_full_record_db"])
        for fold in fold_reports
        if fold["gain_over_gmp_full_record_db"] is not None
    ]
    fold_common_gains = [
        float(fold["gain_over_gmp_common_interior_db"])
        for fold in fold_reports
        if fold["gain_over_gmp_common_interior_db"] is not None
    ]
    gmp_metrics = (
        _metric_summary(reference_gmp, target, protocol)
        if reference_gmp is not None
        else None
    )
    return {
        "recipe": recipe,
        "recipe_sha256": recipe.canonical_sha256,
        "operation_count": cost.to_dict(),
        "full_record_nmse_db": metrics["full_record_nmse_db"],
        "common_interior_nmse_db": metrics["common_interior_nmse_db"],
        "metrics": metrics,
        "reference_gmp_metrics": gmp_metrics,
        "gain_over_gmp_full_record_db": (
            None
            if gmp_metrics is None
            else float(gmp_metrics["full_record_nmse_db"])
            - float(metrics["full_record_nmse_db"])
        ),
        "gain_over_gmp_common_interior_db": (
            None
            if gmp_metrics is None
            else float(gmp_metrics["common_interior_nmse_db"])
            - float(metrics["common_interior_nmse_db"])
        ),
        "minimum_fold_gain_over_gmp_full_record_db": (
            min(fold_gains) if fold_gains else None
        ),
        "minimum_fold_gain_over_gmp_common_interior_db": (
            min(fold_common_gains) if fold_common_gains else None
        ),
        "hard_valid": all(hard_checks.values()),
        "hard_validity_checks": hard_checks,
        "numerical_schedule_diagnostics": {
            "maximum_serialized_vs_matrix_relative_objective_delta": max(
                serialized_relative_objective_deltas,
                default=0.0,
            ),
            "role": (
                "reported arithmetic-order diagnostic; the frozen monotonic "
                "tolerance applies to ALS block updates"
            ),
        },
        "fit_seconds": total_fit_seconds,
        "fold_count": len(fold_reports),
        "fold_reports": fold_reports,
        "oof_prediction": prediction,
        "oof_prediction_sha256": _array_sha256(prediction),
        "accessed_split": "train_only",
        "test_split_accessed": False,
    }
