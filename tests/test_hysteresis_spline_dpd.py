import unittest

import numpy as np

from baseline.hysteresis_spline_dpd import (
    ResidualHysteresisSplineMemoryDPD,
    branch_hysteresis_gate,
    fit_residual_hysteresis_spline_memory_dpd,
    hysteresis_design_matrices,
)
from baseline.spline_memory_dpd import SplineMemoryBranch


class HysteresisSplineDPDTests(unittest.TestCase):
    def setUp(self):
        n = 400
        t = np.arange(n, dtype=float)
        radius = 0.35 + 0.25 * np.sin(2.0 * np.pi * t / 73.0)
        self.signal = radius * np.exp(1j * (0.03 * t + 0.2 * np.sin(t / 31.0)))
        self.branches = (
            SplineMemoryBranch(0, 0),
            SplineMemoryBranch(1, 1),
            SplineMemoryBranch(2, 2),
        )
        self.knots = np.linspace(0.0, 0.7, 8)

    def test_gate_is_branch_local_and_ternary(self):
        gates = branch_hysteresis_gate(self.signal, self.branches, deadband=1e-5)
        self.assertEqual(gates.shape, (self.signal.size, 3))
        self.assertTrue(np.all(np.isin(gates, (-1, 0, 1))))
        # A delayed branch must have a delayed direction sequence, not the
        # common current-envelope direction.
        self.assertGreater(np.count_nonzero(gates[:, 0] != gates[:, 1]), 0)

    def test_residual_dictionary_is_gated_baseline_dictionary(self):
        base, residual, gates = hysteresis_design_matrices(
            self.signal, self.knots, self.branches, deadband=0.0
        )
        k = self.knots.size
        for branch_index in range(len(self.branches)):
            start = branch_index * k
            stop = start + k
            np.testing.assert_allclose(
                residual[:, start:stop],
                base[:, start:stop] * gates[:, branch_index, None],
            )

    def test_fit_and_phase_equivariance(self):
        baseline = np.array(
            [[0.8 + 0.05j] * self.knots.size,
             [0.12 - 0.03j] * self.knots.size,
             [-0.04 + 0.01j] * self.knots.size],
            dtype=np.complex128,
        )
        residual = np.array(
            [[0.02 + 0.01j] * self.knots.size,
             [0.01 - 0.02j] * self.knots.size,
             [-0.01 + 0.01j] * self.knots.size],
            dtype=np.complex128,
        )
        probe = ResidualHysteresisSplineMemoryDPD(
            self.knots, self.branches, baseline, residual, deadband=0.002
        )
        target = probe(self.signal)
        fitted, diagnostics = fit_residual_hysteresis_spline_memory_dpd(
            self.signal,
            target,
            branches=self.branches,
            knots=self.knots,
            knot_count=self.knots.size,
            deadband=0.002,
            ridge_baseline=1e-12,
            ridge_residual=1e-12,
        )
        self.assertTrue(np.isfinite(diagnostics.augmented_condition_number))
        np.testing.assert_allclose(fitted(self.signal)[5:], target[5:], atol=2e-5)
        phi = 0.713
        rotated = np.exp(1j * phi) * self.signal
        np.testing.assert_allclose(
            fitted(rotated), np.exp(1j * phi) * fitted(self.signal), atol=1e-12
        )

    def test_operation_count_keeps_runtime_bank_selection_explicit(self):
        zeros = np.zeros((3, self.knots.size), dtype=np.complex128)
        model = ResidualHysteresisSplineMemoryDPD(
            self.knots, self.branches, zeros, zeros, deadband=0.01
        )
        precomputed = model.operation_count(precomputed_banks=True)
        online = model.operation_count(precomputed_banks=False)
        self.assertEqual(precomputed.real_multiplications, 27)
        self.assertEqual(precomputed.real_additions, 31)
        self.assertEqual(precomputed.lookups, 6)
        self.assertGreater(online.real_multiplications, precomputed.real_multiplications)
        self.assertEqual(model.stored_complex_coefficients, 48)
        self.assertEqual(model.maximum_delay, 3)


if __name__ == "__main__":
    unittest.main()
