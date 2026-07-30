import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.select_pa_sparse_spline_memory import (
    SparseRecipe,
    common_mask,
    evaluate_recipe_oof,
    frame_segments,
    load_config,
    rank_valid_records,
    retain_topologies,
    run_staged_search,
    validate_search_budget,
)
from baseline.sparse_spline_memory_pa import (
    SparseSplineMemoryPA,
    SparseSplineMemoryPABranch,
)


class SparseSplineMemorySelectionTests(unittest.TestCase):
    @staticmethod
    def _signal(length: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        radius = rng.uniform(0.05, 0.95, length)
        phase = rng.uniform(-np.pi, np.pi, length)
        return radius * np.exp(1j * phase)

    def test_frame_partition_and_common_mask(self) -> None:
        signal = self._signal(17, 1)
        segments = frame_segments(signal, (5, 7, 5))
        self.assertEqual(tuple(segment.size for segment in segments), (5, 7, 5))
        mask = common_mask((5, 7, 5), 2)
        self.assertEqual(int(np.count_nonzero(mask)), 11)
        self.assertFalse(mask[0])
        self.assertTrue(mask[2])
        self.assertFalse(mask[5])

    def test_recipe_hash_is_canonical(self) -> None:
        first = SparseRecipe("a", ((0, 0), (2, 0)), 8, 1e-8)
        second = SparseRecipe("a", ((0, 0), (2, 0)), 8, 1e-8)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.name, SparseRecipe("a", ((0, 0),), 8, 1e-8).name)

    def test_existing_implementation_preregistration_status_is_explicit(self) -> None:
        source = Path(
            "experiments/configs/pa_sparse_spline_memory_apa200.json"
        )
        config = json.loads(source.read_text(encoding="utf-8"))
        config["status"] = (
            "preregistered_before_candidate_fit_using_frozen_existing_implementation"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_config(path)["status"], config["status"])

    def test_oof_fit_does_not_cross_frame_boundaries(self) -> None:
        branches = ((0, 0), (2, 0))
        knots = np.linspace(0.0, 1.0, 8)
        coefficients = np.asarray(
            [
                np.linspace(0.8 + 0.1j, 1.1 - 0.1j, 8),
                np.linspace(0.02 - 0.02j, 0.04 + 0.01j, 8),
            ]
        )
        # Build the synthetic reference through the public model API without
        # making the selector depend on any test split.
        synthetic = SparseSplineMemoryPA(
            knots,
            tuple(SparseSplineMemoryPABranch(*pair) for pair in branches),
            coefficients,
        )
        # Give every fold the same calibrated amplitude endpoint.  This keeps
        # the synthetic recovery test focused on frame isolation rather than
        # on the model's explicitly documented endpoint extrapolation rule.
        prepared = []
        for seed in (2, 3, 4):
            segment = self._signal(80, seed)
            radius = np.abs(segment)
            segment = segment * np.minimum(1.0, 0.8 / radius)
            segment[0] = 0.8 * np.exp(1j * np.angle(segment[0]))
            prepared.append(segment)
        inputs = tuple(prepared)
        outputs = tuple(synthetic.predict(segment) for segment in inputs)
        # Use a finite, intentionally weaker reference so the metric helper
        # does not hit the ``-inf`` exact-reconstruction edge case.
        reference = np.concatenate(outputs) + (1e-3 + 2e-3j)
        record = evaluate_recipe_oof(
            SparseRecipe("synthetic", branches, 8, 0.0),
            inputs,
            outputs,
            reference,
            frame_lengths=(80, 80, 80),
            common_warmup=4,
            gates={
                "maximum_augmented_condition_number": 1e12,
                "maximum_absolute_coefficient": 1e6,
                "maximum_support_exceedance_fraction": 1.0,
                "real_multiplications_strictly_below": 1000,
            },
        )
        self.assertTrue(record["hard_valid"])
        self.assertLess(record["full_record_nmse_db"], -200.0)
        self.assertGreater(record["minimum_fold_gain_over_gmp_full_db"], 100.0)

    def test_ranking_uses_primary_then_secondary(self) -> None:
        def row(name: str, full: float, common: float, digest: str) -> dict:
            return {
                "recipe": {"name": name, "branches": [], "knot_count": 8},
                "recipe_sha256": digest,
                "full_record_nmse_db": full,
                "common_interior_nmse_db": common,
                "hard_valid": True,
                "operation_count": {"real_multiplications": 20},
            }

        ranked = rank_valid_records(
            [row("a", -30.0, -31.0, "a"), row("b", -29.99, -32.0, "b")],
            tie_tolerance_db=0.02,
        )
        self.assertEqual(ranked[0]["recipe"]["name"], "b")

    def test_retain_topologies_has_window_and_limit(self) -> None:
        rows = []
        for index, score in enumerate((-30.0, -29.9, -29.7, -28.0)):
            rows.append(
                {
                    "recipe": {"name": str(index), "branches": [], "knot_count": 8},
                    "recipe_sha256": str(index),
                    "full_record_nmse_db": score,
                    "common_interior_nmse_db": score,
                    "hard_valid": True,
                    "operation_count": {"real_multiplications": 20},
                }
            )
        retained = retain_topologies(rows, maximum=3, window_db=0.25)
        self.assertEqual([row["recipe"]["name"] for row in retained], ["0", "1"])


class SparseSplineMemoryStagedSearchTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {
            "dataset_contract": {
                "frame_lengths": [64, 64, 64],
                "common_warmup_samples_per_frame": 2,
            },
            "branch_families": {
                "memoryless": [[0, 0]],
                "short_signal_memory": [[0, 0], [1, 0]],
            },
            "search": {
                "stage_s0_topology_screen": {
                    "knot_count": 4,
                    "ridge": 1e-8,
                    "retain_topologies": 1,
                    "retention_window_db": 0.25,
                },
                "stage_s1_knot_count": {
                    "knot_counts": [4, 6],
                    "ridge": 1e-8,
                },
                "stage_s2_ridge": {"ridges": [0.0, 1e-8]},
                "ranking": {"tie_tolerance_db": 0.02},
            },
            "gates": {
                "maximum_augmented_condition_number": 1e12,
                "maximum_absolute_coefficient": 1e6,
                "maximum_support_exceedance_fraction": 0.0,
                "real_multiplications_strictly_below": 1000,
                "cheap_pareto_max_full_loss_vs_mp_db": 1000.0,
                "cheap_pareto_max_common_loss_vs_mp_db": 1000.0,
                "evaluator_min_full_gain_over_gmp_db": -1000.0,
                "evaluator_min_common_gain_over_gmp_db": -1000.0,
                "evaluator_minimum_fold_gain_over_gmp_db": -1000.0,
                "incremental_min_full_gain_db": 0.25,
                "incremental_min_common_gain_db": 0.25,
                "incremental_minimum_fold_gain_db": 0.1,
            },
            "reference_models": {
                "matched_mp_oof": {
                    "full_record_nmse_db": -10.0,
                    "common_interior_nmse_db": -10.0,
                },
                "matched_gmp_oof": {
                    "full_record_nmse_db": -10.0,
                    "common_interior_nmse_db": -10.0,
                },
                "incremental_control_oof": {
                    "full_record_nmse_db": -10.0,
                    "common_interior_nmse_db": -10.0,
                    "fold_records": [
                        {
                            "held_frame_id": held_frame,
                            "held_metrics": {
                                "full_record_nmse_db": -10.0,
                                "common_interior_nmse_db": -10.0,
                            },
                        }
                        for held_frame in range(3)
                    ],
                },
            },
            "search_budget": {
                "oof_fold_count": 3,
                "maximum_s0_recipes": 2,
                "maximum_s1_recipes": 2,
                "maximum_s2_recipes": 2,
                "maximum_unique_recipes": 6,
                "maximum_oof_fit_calls_without_cache": 18,
            },
        }

    def test_production_search_budget_is_exact(self) -> None:
        config = json.loads(
            Path("experiments/configs/pa_sparse_spline_memory_apa200.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            validate_search_budget(config),
            {"S0": 7, "S1": 12, "S2": 5, "folds": 3},
        )

    def test_stages_use_train_oof_and_recipe_cache(self) -> None:
        rng = np.random.default_rng(401)
        inputs = []
        for _ in range(3):
            radius = np.linspace(0.02, 1.0, 64)
            phase = rng.uniform(-np.pi, np.pi, 64)
            inputs.append(radius * np.exp(1j * phase))
        input_segments = tuple(inputs)
        truth = SparseSplineMemoryPA(
            knots=np.linspace(0.0, 1.0, 4),
            branches=(SparseSplineMemoryPABranch(0, 0),),
            coefficients=np.asarray(
                [[1.2 + 0.1j, 1.1, 0.9 - 0.1j, 0.7 - 0.2j]]
            ),
        )
        output_segments = tuple(truth.predict(signal) for signal in input_segments)
        search = run_staged_search(
            self._config(),
            input_segments,
            output_segments,
            np.concatenate(input_segments),
        )
        self.assertEqual(search["stage_recipe_associations"], 6)
        self.assertGreaterEqual(search["cache_hits"], 2)
        self.assertLessEqual(search["unique_recipe_evaluations"], 6)
        self.assertLessEqual(search["completed_oof_fit_calls"], 18)
        self.assertTrue(search["final_trial"]["hard_valid"])
        self.assertTrue(
            search["decision"]["incremental_hypothesis_gate_passed"]
        )
        self.assertFalse(search["decision"]["gate_a_to_b_opened"])


if __name__ == "__main__":
    unittest.main()
