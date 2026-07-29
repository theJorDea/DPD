import unittest

import numpy as np

from experiments.select_pa_sparse_spline_memory import (
    SparseRecipe,
    common_mask,
    evaluate_recipe_oof,
    frame_segments,
    rank_valid_records,
    retain_topologies,
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


if __name__ == "__main__":
    unittest.main()
