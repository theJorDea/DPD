import copy
from pathlib import Path
import unittest

from experiments.select_pa_sph import (
    SPHRecipe,
    enumerate_s0_recipes,
    enumerate_s1_recipes,
    enumerate_s2_recipes,
    enumerate_s3_recipes,
    load_sph_config,
    retain_s0_topologies,
    select_ranked_trial,
    validate_search_budget,
)


class SPHRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path("experiments/configs/pa_sph_apa200.json")
        cls.config = load_sph_config(cls.path)

    def test_preregistered_stage_sizes_and_fit_budget_are_exact(self) -> None:
        summary = validate_search_budget(self.config)
        self.assertEqual(
            summary,
            {
                "S0": 20,
                "S1": 14,
                "S2": 25,
                "S3": 5,
                "folds": 3,
                "maximum_oof_fit_calls": 192,
            },
        )
        s0 = enumerate_s0_recipes(self.config)
        self.assertEqual(len({recipe.canonical_sha256 for recipe in s0}), 20)
        self.assertEqual({recipe.knot_count for recipe in s0}, {24})
        self.assertEqual({recipe.fir_length for recipe in s0}, {1, 2, 4, 8})
        self.assertEqual(
            {recipe.operation_count.real_multiplications for recipe in s0},
            {9, 13, 21, 37},
        )

    def test_later_stage_enumeration_changes_only_declared_axes(self) -> None:
        s1 = enumerate_s1_recipes(
            self.config,
            (("amplitude_uniform", 2), ("power_uniform", 4)),
        )
        self.assertEqual(len(s1), 14)
        self.assertEqual(
            {recipe.knot_count for recipe in s1},
            {8, 12, 16, 24, 32, 48, 64},
        )
        self.assertEqual(
            {(recipe.variant, recipe.fir_length) for recipe in s1},
            {("amplitude_uniform", 2), ("power_uniform", 4)},
        )
        topology = (s1[0].variant, s1[0].fir_length, s1[0].knot_count)
        s2 = enumerate_s2_recipes(self.config, topology)
        self.assertEqual(len(s2), 25)
        self.assertEqual(len({recipe.control_ridge for recipe in s2}), 5)
        self.assertEqual(len({recipe.smoothness for recipe in s2}), 5)
        s3 = enumerate_s3_recipes(self.config, s2[0])
        self.assertEqual(len(s3), 5)
        self.assertEqual(len({recipe.fir_ridge for recipe in s3}), 5)

    def test_config_budget_tamper_is_detected(self) -> None:
        config = copy.deepcopy(self.config)
        config["search_budget"]["maximum_S2_candidate_recipes"] = 24
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_search_budget(config)

        config = copy.deepcopy(self.config)
        config["operation_count_convention"]["frozen_length_points"][0][
            "real_multiplications"
        ] = 8
        with self.assertRaisesRegex(ValueError, "operation point"):
            validate_search_budget(config)


class SPHRankingTests(unittest.TestCase):
    @staticmethod
    def _trial(
        recipe: SPHRecipe,
        full: float,
        common: float,
        *,
        valid: bool = True,
    ) -> dict[str, object]:
        return {
            "recipe": recipe,
            "full_record_nmse_db": full,
            "common_interior_nmse_db": common,
            "hard_valid": valid,
        }

    def test_primary_window_uses_common_metric_before_complexity(self) -> None:
        cheap = SPHRecipe("power_uniform", 8, 1, 1e-8, 1e-6, 1e-8)
        accurate_common = SPHRecipe(
            "amplitude_uniform",
            64,
            8,
            1e-8,
            1e-6,
            1e-8,
        )
        rows = [
            self._trial(cheap, -30.010, -30.1),
            self._trial(accurate_common, -30.000, -31.0),
        ]
        selected = select_ranked_trial(rows, tolerance_db=0.02)
        self.assertIs(selected["recipe"], accurate_common)

    def test_outside_primary_window_cannot_win_on_secondary(self) -> None:
        primary = SPHRecipe("power_uniform", 8, 1, 1e-8, 1e-6, 1e-8)
        outside = SPHRecipe(
            "amplitude_uniform",
            64,
            8,
            1e-8,
            1e-6,
            1e-8,
        )
        rows = [
            self._trial(primary, -30.10, -30.0),
            self._trial(outside, -30.079, -40.0),
        ]
        selected = select_ranked_trial(rows, tolerance_db=0.02)
        self.assertIs(selected["recipe"], primary)

    def test_hard_invalid_trial_is_never_selected_or_retained(self) -> None:
        valid = SPHRecipe("power_uniform", 24, 1, 1e-8, 1e-6, 1e-8)
        invalid = SPHRecipe(
            "amplitude_uniform",
            24,
            8,
            1e-8,
            1e-6,
            1e-8,
        )
        rows = [
            self._trial(valid, -30.0, -30.0),
            self._trial(invalid, -50.0, -50.0, valid=False),
        ]
        self.assertIs(
            select_ranked_trial(rows, tolerance_db=0.02)["recipe"],
            valid,
        )
        self.assertEqual(retain_s0_topologies(rows), (("power_uniform", 1),))

    def test_s0_retains_at_most_two_inside_frozen_window(self) -> None:
        recipes = [
            SPHRecipe(name, 24, length, 1e-8, 1e-6, 1e-8)
            for name, length in (
                ("amplitude_uniform", 1),
                ("power_uniform", 2),
                ("amplitude_quantile", 4),
                ("amplitude_compression_aware_p2", 8),
            )
        ]
        rows = [
            self._trial(recipes[0], -30.00, -30.0),
            self._trial(recipes[1], -29.98, -30.1),
            self._trial(recipes[2], -29.97, -30.2),
            self._trial(recipes[3], -29.94, -40.0),
        ]
        self.assertEqual(
            retain_s0_topologies(rows),
            (("amplitude_uniform", 1), ("power_uniform", 2)),
        )


if __name__ == "__main__":
    unittest.main()
