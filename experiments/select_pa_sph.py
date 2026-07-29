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
from typing import Any, Iterable

import numpy as np

from baseline.complexity import OperationCount, spline_hammerstein_pa_cost


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
