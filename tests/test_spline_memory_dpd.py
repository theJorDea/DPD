import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    fit_ila_sparse_spline_memory_dpd,
    fit_sparse_spline_memory_dpd,
    spline_memory_design_matrix,
)


class SplineMemoryDPDTests(unittest.TestCase):
    def setUp(self) -> None:
        self.knots = np.asarray([0.0, 0.25, 0.55, 0.8, 1.0])
        self.branches = (
            SplineMemoryBranch(0, 0),
            SplineMemoryBranch(1, 1),
        )
        self.coefficients = np.asarray(
            [
                [
                    1.00 + 0.10j,
                    1.10 + 0.05j,
                    1.20 - 0.10j,
                    1.30 - 0.20j,
                    1.40 - 0.10j,
                ],
                [
                    0.05 - 0.03j,
                    0.04 - 0.02j,
                    0.03 + 0.01j,
                    0.02 + 0.02j,
                    0.01 + 0.03j,
                ],
            ]
        )
        self.model = SparseSplineMemoryDPD(
            self.knots,
            self.branches,
            self.coefficients,
        )

    @staticmethod
    def _random_signal(count: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        radius = rng.uniform(0.03, 0.97, count)
        phase = rng.uniform(-np.pi, np.pi, count)
        return radius * np.exp(1j * phase)

    def test_design_has_local_support_per_branch(self) -> None:
        signal = self._random_signal(100, 8)
        design = spline_memory_design_matrix(
            signal,
            self.knots,
            self.branches,
        )
        self.assertEqual(design.shape, (100, 10))
        for branch_index, branch in enumerate(self.branches):
            block = design[
                :,
                branch_index * self.knots.size:
                (branch_index + 1) * self.knots.size,
            ]
            self.assertTrue(
                np.all(np.count_nonzero(np.abs(block) > 1e-15, axis=1) <= 2)
            )
            for sample_index in range(branch.signal_delay, signal.size):
                lagged = signal[sample_index - branch.signal_delay]
                np.testing.assert_allclose(
                    np.sum(block[sample_index]) / lagged,
                    1.0,
                    rtol=1e-13,
                    atol=1e-13,
                )

    def test_joint_complex_fit_recovers_all_branch_control_points(self) -> None:
        signal = self._random_signal(4000, 91)
        target = self.model.predict(signal)
        fitted, diagnostics = fit_sparse_spline_memory_dpd(
            signal,
            target,
            branches=self.branches,
            knots=self.knots,
            ridge=0.0,
        )
        self.assertEqual(diagnostics.solver, "augmented_complex_lstsq")
        self.assertEqual(diagnostics.solver_rank, diagnostics.feature_count)
        self.assertLess(diagnostics.training_relative_error_power_full, 1e-24)
        np.testing.assert_allclose(
            fitted.coefficients,
            self.coefficients,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_phase_equivariance(self) -> None:
        signal = self._random_signal(300, 17)
        rotation = np.exp(1j * 0.734)
        np.testing.assert_allclose(
            self.model.predict(signal * rotation),
            self.model.predict(signal) * rotation,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_continuous_chunks_match_full_record(self) -> None:
        signal = self._random_signal(79, 29)
        expected = self.model.predict(signal)
        state = self.model.initial_state()
        chunks: list[np.ndarray] = []
        start = 0
        for length in (3, 11, 2, 23, 40):
            stop = start + length
            chunk, state = self.model.predict_chunk(
                signal[start:stop],
                state,
            )
            chunks.append(chunk)
            start = stop
        self.assertEqual(start, signal.size)
        np.testing.assert_allclose(
            np.concatenate(chunks),
            expected,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_array_equal(
            state.history,
            signal[-self.model.maximum_delay:],
        )

    def test_segment_reset_matches_independent_records(self) -> None:
        signal = self._random_signal(23, 37)
        segmented = self.model.predict_segments(signal, segment_length=8)
        expected = np.concatenate(
            (
                self.model.predict(signal[:8]),
                self.model.predict(signal[8:16]),
                self.model.predict(signal[16:]),
            )
        )
        np.testing.assert_allclose(segmented, expected)
        continuous = self.model.predict(signal)
        self.assertGreater(abs(segmented[8] - continuous[8]), 1e-6)

    def test_ila_uses_normalized_measured_output_as_fit_input(self) -> None:
        normalized_observation = self._random_signal(3500, 51)
        known_pa_input = self.model.predict(normalized_observation)
        gain = 2.1 - 0.3j
        measured_pa_output = gain * normalized_observation
        fitted, diagnostics = fit_ila_sparse_spline_memory_dpd(
            known_pa_input,
            measured_pa_output,
            gain,
            branches=self.branches,
            knots=self.knots,
            ridge=0.0,
        )
        self.assertEqual(diagnostics.solver_rank, diagnostics.feature_count)
        np.testing.assert_allclose(
            fitted.predict(normalized_observation),
            known_pa_input,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_storage_and_operation_hooks(self) -> None:
        model = SparseSplineMemoryDPD(
            knots=np.linspace(0.0, 1.0, 8),
            branches=(
                SplineMemoryBranch(0, 0),
                SplineMemoryBranch(1, 0),
                SplineMemoryBranch(2, 1),
            ),
            coefficients=np.ones((3, 8), dtype=np.complex128),
        )
        self.assertEqual(model.stored_complex_coefficients, 24)
        self.assertEqual(model.stored_real_coefficients, 48)
        cost = model.operation_count(convention="4m2a", indexing="binary")
        self.assertEqual(cost.real_multiplications, 24)
        self.assertEqual(cost.real_additions, 26)
        self.assertEqual(cost.nonlinear_operations, 2)
        self.assertEqual(cost.comparisons, 6)
        self.assertEqual(cost.stored_real_coefficients, 48)
        self.assertEqual(cost.stored_real_constants, 15)
        self.assertEqual(cost.state_real_values, 4)

        memoryless_equivalent = SparseSplineMemoryDPD(
            knots=np.linspace(0.0, 1.0, 8),
            branches=(SplineMemoryBranch(0, 0),),
            coefficients=np.ones((1, 8), dtype=np.complex128),
        ).operation_count()
        self.assertEqual(memoryless_equivalent.real_multiplications, 9)
        self.assertEqual(memoryless_equivalent.real_additions, 8)
        self.assertEqual(memoryless_equivalent.real_memory_reads, 6)

    def test_npz_round_trip(self) -> None:
        signal = self._random_signal(50, 101)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spline_memory.npz"
            self.model.save(path)
            loaded = SparseSplineMemoryDPD.load(path)
            self.assertEqual(loaded.branches, self.model.branches)
            np.testing.assert_array_equal(loaded.knots, self.model.knots)
            np.testing.assert_array_equal(
                loaded.coefficients,
                self.model.coefficients,
            )
            np.testing.assert_allclose(
                loaded.predict(signal),
                self.model.predict(signal),
            )


if __name__ == "__main__":
    unittest.main()
