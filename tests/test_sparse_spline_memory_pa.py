import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.sparse_spline_memory_pa import (
    SparseSplineMemoryPA,
    SparseSplineMemoryPABranch,
    fit_sparse_spline_memory_pa_segments,
    sparse_spline_memory_pa_design_matrix,
)


class SparseSplineMemoryPATests(unittest.TestCase):
    @staticmethod
    def _signal(length: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        radius = rng.uniform(0.03, 0.97, length)
        phase = rng.uniform(-np.pi, np.pi, length)
        return radius * np.exp(1j * phase)

    def setUp(self) -> None:
        self.knots = np.linspace(0.0, 1.0, 8)
        self.branches = (
            SparseSplineMemoryPABranch(0, 0),
            SparseSplineMemoryPABranch(2, 0),
            SparseSplineMemoryPABranch(3, 2),
        )
        coefficients = np.asarray(
            [
                np.linspace(0.8 + 0.1j, 1.2 - 0.1j, 8),
                np.linspace(0.03 - 0.04j, 0.01 + 0.02j, 8),
                np.linspace(-0.02 + 0.03j, 0.04 + 0.01j, 8),
            ],
            dtype=np.complex128,
        )
        self.model = SparseSplineMemoryPA(
            knots=self.knots,
            branches=self.branches,
            coefficients=coefficients,
        )

    def test_design_has_two_active_points_per_branch(self) -> None:
        signal = self._signal(300, 1)
        design = sparse_spline_memory_pa_design_matrix(
            (signal,),
            self.knots,
            self.branches,
        )
        self.assertEqual(design.shape, (300, 24))
        for branch_index in range(len(self.branches)):
            block = design[:, branch_index * 8 : (branch_index + 1) * 8]
            self.assertTrue(
                np.all(np.count_nonzero(np.abs(block) > 1e-14, axis=1) <= 2)
            )

    def test_phase_equivariance(self) -> None:
        signal = self._signal(300, 2)
        rotation = np.exp(1j * 0.731)
        np.testing.assert_allclose(
            self.model.predict(signal * rotation),
            self.model.predict(signal) * rotation,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_segmented_fit_recovers_synthetic_forward_model(self) -> None:
        segments = (
            self._signal(900, 3),
            self._signal(870, 4),
            self._signal(840, 5),
        )
        targets = tuple(self.model.predict(segment) for segment in segments)
        fitted, diagnostics = fit_sparse_spline_memory_pa_segments(
            segments,
            targets,
            branches=self.branches,
            knots=self.knots,
            ridge=0.0,
        )
        self.assertEqual(diagnostics.data_design_rank, diagnostics.feature_count)
        self.assertGreater(diagnostics.minimum_nonzero_feature_samples, 0)
        self.assertEqual(
            diagnostics.augmented_solver_rank,
            diagnostics.feature_count,
        )
        self.assertLess(diagnostics.training_relative_error_power_full, 1e-24)
        np.testing.assert_allclose(
            fitted.coefficients,
            self.model.coefficients,
            rtol=2e-11,
            atol=2e-11,
        )

    def test_segment_boundaries_reset_state(self) -> None:
        first = self._signal(40, 6)
        second = self._signal(40, 7)
        reset = np.concatenate((self.model.predict(first), self.model.predict(second)))
        segmented = self.model.predict_segments(
            np.concatenate((first, second)),
            segment_length=40,
        )
        np.testing.assert_allclose(segmented, reset, rtol=1e-13, atol=1e-13)
        continuous = self.model.predict(np.concatenate((first, second)))
        self.assertGreater(abs(continuous[40] - reset[40]), 1e-8)

    def test_streaming_chunks_match_full_record(self) -> None:
        signal = self._signal(127, 8)
        expected = self.model.predict(signal)
        state = self.model.initial_state()
        chunks = []
        start = 0
        for length in (3, 17, 2, 31, 74):
            stop = start + length
            chunk, state = self.model.predict_chunk(signal[start:stop], state)
            chunks.append(chunk)
            start = stop
        np.testing.assert_allclose(
            np.concatenate(chunks),
            expected,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_array_equal(state.history, signal[-self.model.maximum_delay :])

    def test_causality(self) -> None:
        signal = self._signal(100, 9)
        changed = signal.copy()
        changed[80:] *= np.exp(1j * 1.1)
        np.testing.assert_array_equal(
            self.model.predict(signal)[:80],
            self.model.predict(changed)[:80],
        )

    def test_operation_count(self) -> None:
        cost = self.model.operation_count(convention="4m2a", indexing="binary")
        self.assertEqual(cost.real_multiplications, 24)
        self.assertEqual(cost.real_additions, 26)
        self.assertEqual(cost.nonlinear_operations, 2)
        self.assertEqual(cost.comparisons, 6)
        self.assertEqual(cost.stored_real_coefficients, 48)
        self.assertEqual(cost.stored_real_constants, 15)
        self.assertEqual(cost.state_real_values, 6)

    def test_save_load_preserves_pa_direction(self) -> None:
        signal = self._signal(73, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            self.model.save(path)
            loaded = SparseSplineMemoryPA.load(path)
        self.assertEqual(loaded.metadata["direction"], "measured_pa_input_to_predicted_pa_output")
        np.testing.assert_array_equal(loaded.knots, self.model.knots)
        self.assertEqual(loaded.branches, self.model.branches)
        np.testing.assert_allclose(loaded.predict(signal), self.model.predict(signal))


if __name__ == "__main__":
    unittest.main()
