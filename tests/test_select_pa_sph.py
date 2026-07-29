import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baseline.spline_hammerstein_pa import SplineHammersteinPA
from experiments.select_pa_sph import (
    SPHOOFProtocol,
    SPHRecipe,
    enumerate_s0_recipes,
    enumerate_s1_recipes,
    enumerate_s2_recipes,
    enumerate_s3_recipes,
    evaluate_oof_recipe,
    load_sph_config,
    load_verified_gmp_oof_prediction,
    retain_s0_topologies,
    run_staged_oof_search,
    select_ranked_trial,
    validate_search_budget,
    verify_sph_preregistered_inputs,
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


class SPHIntegrityTests(unittest.TestCase):
    def test_production_evidence_verifies_before_waveform_load(self) -> None:
        verified = verify_sph_preregistered_inputs(
            "experiments/configs/pa_sph_apa200.json"
        )
        self.assertTrue(verified["verified_before_waveform_load"])
        self.assertFalse(verified["test_split_accessed"])
        self.assertFalse(verified["test_file_hashes_recorded"])
        self.assertEqual(
            set(verified["dataset_hashes"]),
            {
                "spec.json",
                "train_input.csv",
                "train_output.csv",
                "val_input.csv",
                "val_output.csv",
            },
        )
        prediction = load_verified_gmp_oof_prediction(verified)
        self.assertEqual(prediction.shape, (58980,))
        self.assertEqual(prediction.dtype, np.dtype(np.complex128))
        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_tampered_dataset_or_evidence_hash_fails_verification(self) -> None:
        source = Path("experiments/configs/pa_sph_apa200.json")
        config = json.loads(source.read_text(encoding="utf-8"))
        with self.subTest("dataset"):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "tampered_dataset.json"
                changed = copy.deepcopy(config)
                changed["dataset_contract"]["required_files_sha256"][
                    "train_input.csv"
                ] = "0" * 64
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "dataset hashes differ"):
                    verify_sph_preregistered_inputs(path)
        with self.subTest("evidence"):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "tampered_evidence.json"
                changed = copy.deepcopy(config)
                changed["evidence"]["design_document"]["sha256"] = "0" * 64
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verify_sph_preregistered_inputs(path)

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


class SPHOOFEvaluationTests(unittest.TestCase):
    @staticmethod
    def _synthetic_records() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1811)
        frames = []
        for length in (128, 128, 91):
            radius = np.linspace(0.02, 1.0, length)
            phase = rng.uniform(-np.pi, np.pi, size=length)
            frames.append(radius * np.exp(1j * phase))
        signal = np.concatenate(frames)
        truth = SplineHammersteinPA(
            knots=np.linspace(0.0, 1.0, 5),
            control_points=np.asarray(
                [1.4 + 0.08j, 1.25, 1.05 - 0.08j, 0.85 - 0.15j, 0.68 - 0.2j]
            ),
            fir_tail=np.asarray([0.06 + 0.02j, -0.02 + 0.01j]),
        )
        measured = truth.predict_segments(signal, 128)
        measured += 1e-7 * (
            rng.normal(size=signal.size) + 1j * rng.normal(size=signal.size)
        )
        reference_gmp = (1.05 - 0.03j) * signal
        return signal, measured, reference_gmp

    def test_train_only_oof_handles_partial_frame_and_exact_streaming(self) -> None:
        signal, measured, reference_gmp = self._synthetic_records()
        recipe = SPHRecipe(
            "amplitude_uniform",
            5,
            3,
            0.0,
            0.0,
            0.0,
        )
        protocol = SPHOOFProtocol(
            segment_length=128,
            common_warmup_samples=4,
            maximum_alternations=20,
            minimum_alternations=2,
            convergence_tolerance=1e-10,
        )
        result = evaluate_oof_recipe(
            recipe,
            signal,
            measured,
            protocol=protocol,
            reference_gmp_oof_prediction=reference_gmp,
        )
        self.assertTrue(result["hard_valid"])
        self.assertTrue(
            result["hard_validity_checks"]["all_required_numerics_finite"]
        )
        self.assertLess(
            result["numerical_schedule_diagnostics"][
                "maximum_serialized_vs_matrix_relative_objective_delta"
            ],
            1e-9,
        )
        self.assertFalse(result["test_split_accessed"])
        self.assertEqual(result["accessed_split"], "train_only")
        self.assertEqual(result["fold_count"], 3)
        self.assertEqual(
            [fold["held_sample_count"] for fold in result["fold_reports"]],
            [128, 128, 91],
        )
        self.assertEqual(result["metrics"]["opendpd_complete_frame_count"], 2)
        self.assertEqual(result["metrics"]["scored_sample_count_common"], 335)
        self.assertEqual(result["operation_count"]["real_multiplications"], 17)
        self.assertLess(result["full_record_nmse_db"], -90.0)
        self.assertGreater(result["gain_over_gmp_full_record_db"], 40.0)
        self.assertTrue(
            all(
                fold["streaming_checks"][
                    "streaming_chunk_equivalence_passed"
                ]
                and fold["streaming_checks"][
                    "reset_at_frame_equivalence_passed"
                ]
                for fold in result["fold_reports"]
            )
        )

    def test_held_target_cannot_leak_into_its_oof_prediction(self) -> None:
        signal, measured, _ = self._synthetic_records()
        recipe = SPHRecipe(
            "amplitude_uniform",
            5,
            3,
            0.0,
            0.0,
            0.0,
        )
        protocol = SPHOOFProtocol(
            segment_length=128,
            common_warmup_samples=4,
            maximum_alternations=8,
            minimum_alternations=2,
            convergence_tolerance=1e-9,
        )
        original = evaluate_oof_recipe(
            recipe,
            signal,
            measured,
            protocol=protocol,
        )
        changed = measured.copy()
        changed[:128] += (0.12 - 0.04j) * signal[:128]
        perturbed = evaluate_oof_recipe(
            recipe,
            signal,
            changed,
            protocol=protocol,
        )
        np.testing.assert_array_equal(
            original["oof_prediction"][:128],
            perturbed["oof_prediction"][:128],
        )
        self.assertGreater(
            np.max(
                np.abs(
                    original["oof_prediction"][128:]
                    - perturbed["oof_prediction"][128:]
                )
            ),
            1e-4,
        )

    def test_oof_protocol_rejects_consumed_frame_or_nonexclusive_budget(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "consumes"):
            SPHOOFProtocol(
                segment_length=8,
                common_warmup_samples=4,
                common_cooldown_samples=4,
            )
        signal, measured, _ = self._synthetic_records()
        recipe = SPHRecipe(
            "amplitude_uniform",
            5,
            3,
            0.0,
            0.0,
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "strict"):
            evaluate_oof_recipe(
                recipe,
                signal,
                measured,
                protocol=SPHOOFProtocol(
                    segment_length=128,
                    common_warmup_samples=4,
                    real_multiplication_limit_exclusive=17,
                ),
            )


class SPHStagedSearchTests(unittest.TestCase):
    def test_stages_use_train_only_cache_and_frozen_ranking(self) -> None:
        config = load_sph_config(
            "experiments/configs/pa_sph_apa200.json"
        )
        signal = np.ones(6, dtype=np.complex128)
        target = np.ones(6, dtype=np.complex128)
        reference = np.ones(6, dtype=np.complex128)
        calls: list[str] = []

        def fake_evaluate(
            recipe: SPHRecipe,
            train_input: np.ndarray,
            train_output: np.ndarray,
            *,
            protocol: SPHOOFProtocol,
            reference_gmp_oof_prediction: np.ndarray,
        ) -> dict[str, object]:
            del train_input, train_output, protocol, reference_gmp_oof_prediction
            calls.append(recipe.canonical_sha256)
            score = -30.0
            score += 0.0 if recipe.variant == "amplitude_uniform" else 1.0
            score += 0.0 if recipe.fir_length == 2 else 0.5
            score += abs(recipe.knot_count - 24) * 0.01
            score += 0.0 if recipe.control_ridge == 1e-8 else 0.03
            score += 0.0 if recipe.smoothness == 1e-6 else 0.03
            score += 0.0 if recipe.fir_ridge == 1e-8 else 0.03
            return {
                "recipe": recipe,
                "recipe_sha256": recipe.canonical_sha256,
                "operation_count": recipe.operation_count.to_dict(),
                "full_record_nmse_db": score,
                "common_interior_nmse_db": score - 0.1,
                "metrics": {"full_record_nmse_db": score},
                "reference_gmp_metrics": {
                    "full_record_nmse_db": score + 1.0,
                    "common_interior_nmse_db": score + 0.9,
                },
                "gain_over_gmp_full_record_db": 1.0,
                "gain_over_gmp_common_interior_db": 1.0,
                "minimum_fold_gain_over_gmp_full_record_db": 0.5,
                "minimum_fold_gain_over_gmp_common_interior_db": 0.5,
                "hard_valid": True,
                "hard_validity_checks": {"synthetic": True},
                "fit_seconds": 0.01,
                "fold_count": 3,
                "fold_reports": [],
                "oof_prediction": np.zeros(6, dtype=np.complex128),
                "oof_prediction_sha256": "synthetic",
                "accessed_split": "train_only",
                "test_split_accessed": False,
            }

        with patch(
            "experiments.select_pa_sph.evaluate_oof_recipe",
            side_effect=fake_evaluate,
        ):
            result = run_staged_oof_search(
                config,
                signal,
                target,
                protocol=SPHOOFProtocol(
                    segment_length=2,
                    common_warmup_samples=0,
                ),
                reference_gmp_oof_prediction=reference,
            )

        final = result["final_recipe"]
        self.assertEqual(final.variant, "amplitude_uniform")
        self.assertEqual(final.fir_length, 2)
        self.assertEqual(final.knot_count, 24)
        self.assertEqual(final.control_ridge, 1e-8)
        self.assertEqual(final.smoothness, 1e-6)
        self.assertEqual(final.fir_ridge, 1e-8)
        self.assertEqual(result["stage_recipe_associations"], 57)
        self.assertEqual(result["unique_recipe_evaluations"], 54)
        self.assertEqual(result["cache_hits"], 3)
        self.assertEqual(result["completed_unique_oof_fit_calls"], 162)
        self.assertEqual(
            result["evaluated_recipe_oof_fit_call_upper_bound"],
            162,
        )
        self.assertEqual(
            result[
                "stage_association_oof_fit_call_upper_bound_without_cache"
            ],
            171,
        )
        self.assertEqual(len(calls), 54)
        self.assertEqual(result["accessed_splits"], ["train"])
        self.assertFalse(result["validation_loaded"])
        self.assertFalse(result["test_split_accessed"])
        self.assertTrue(
            result["decision"]["evaluator_replacement_eligible"]
        )
        self.assertFalse(result["decision"]["gate_a_to_b_opened"])
        self.assertFalse(result["decision"]["old_apa_test_permitted"])


if __name__ == "__main__":
    unittest.main()
