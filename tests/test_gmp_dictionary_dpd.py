"""Unit tests for the composite spline + GMP-class DPD."""

from __future__ import annotations

import unittest

import numpy as np

from baseline.direct_learning import nmse_db
from baseline.gmp_dictionary_dpd import (
    CompositeSplineGmpDPD,
    GmpDictionaryGrid,
    GmpMember,
    composite_design_matrix,
    fit_gmp_residual_members,
    gmp_dictionary_columns,
    gmp_member_operation_count,
    orthogonal_matching_pursuit,
)
from baseline.sparse_spline_memory_pa import SparseSplineMemoryPA
from baseline.spline_memory_dpd import (
    fit_sparse_spline_memory_dpd,
)


def _deterministic_signal(
    sample_count: int,
    *,
    seed: int,
    peak: float = 0.9,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(sample_count + 64)
    imag = rng.standard_normal(sample_count + 64)
    kernel = np.ones(33) / 33.0
    real = np.convolve(real, kernel, mode="same")
    imag = np.convolve(imag, kernel, mode="same")
    signal = (real + 1j * imag)[64:]
    signal = signal / np.max(np.abs(signal)) * peak
    return np.asarray(signal, dtype=np.complex128)


def _synthetic_pa(maximum: float) -> SparseSplineMemoryPA:
    knots = np.linspace(0.0, maximum, 9)
    coefficients = np.array(
        [
            [1.0 + 0.05j, 0.9 + 0.1j, 0.7 + 0.2j, 0.5 + 0.3j,
             0.4 + 0.25j, 0.5 + 0.15j, 0.7 + 0.1j, 0.9 + 0.05j, 1.0 + 0.0j],
            [0.05 + 0.02j, 0.08 + 0.03j, 0.1 + 0.05j, 0.08 + 0.04j,
             0.06 + 0.03j, 0.05 + 0.02j, 0.04 + 0.02j, 0.03 + 0.01j,
             0.02 + 0.01j],
            [0.02 + 0.01j, 0.03 + 0.01j, 0.04 + 0.02j, 0.03 + 0.02j,
             0.02 + 0.01j, 0.02 + 0.01j, 0.01 + 0.01j, 0.01 + 0.0j,
             0.01 + 0.0j],
        ],
        dtype=np.complex128,
    )
    return SparseSplineMemoryPA(
        knots=knots,
        branches=((0, 0), (1, 1), (2, 2)),
        coefficients=coefficients,
    )


def _least_squares_gain(input_signal: np.ndarray, output_signal: np.ndarray) -> complex:
    return complex(
        np.vdot(input_signal, output_signal)
        / np.vdot(input_signal, input_signal)
    )


class DictionaryTests(unittest.TestCase):
    def test_design_columns_match_direct_definition(self) -> None:
        signal = _deterministic_signal(512, seed=5)
        members = (GmpMember(0, 0, 1), GmpMember(2, 1, 3))
        design = gmp_dictionary_columns(signal, members)
        expected_0 = signal * (np.abs(signal) ** 2)
        expected_1 = np.zeros_like(signal)
        expected_1[2:] = signal[:-2] * (np.abs(signal)[1:-1] ** 6)
        np.testing.assert_allclose(design[:, 0], expected_0, rtol=1e-12)
        np.testing.assert_allclose(design[:, 1], expected_1, rtol=1e-12)

    def test_grid_size(self) -> None:
        grid = GmpDictionaryGrid(
            maximum_signal_delay=8,
            maximum_envelope_delay=8,
            maximum_exponent=8,
        )
        self.assertEqual(len(grid.members), 512)


class OmpTests(unittest.TestCase):
    def test_recovers_sparse_support(self) -> None:
        signal = _deterministic_signal(4096, seed=7)
        members = (GmpMember(0, 0, 1), GmpMember(1, 0, 2), GmpMember(3, 2, 1))
        design = gmp_dictionary_columns(signal, members)
        true_coefficients = np.array([0.4 + 0.1j, -0.2 + 0.3j, 0.1 - 0.05j])
        target = design @ true_coefficients + 0.001 * (
            _deterministic_signal(4096, seed=8, peak=1.0)
        )
        selected, coefficients, _ = orthogonal_matching_pursuit(
            design,
            target,
            maximum_members=3,
            ridge=0.0,
        )
        self.assertEqual(len(selected), 3)
        for index in range(3):
            self.assertAlmostEqual(
                coefficients[index].real,
                true_coefficients[index].real,
                places=2,
            )
            self.assertAlmostEqual(
                coefficients[index].imag,
                true_coefficients[index].imag,
                places=2,
            )

    def test_rejects_empty_design(self) -> None:
        signal = _deterministic_signal(64, seed=9)
        design = np.zeros((64, 2), dtype=np.complex128)
        with self.assertRaises(ValueError):
            orthogonal_matching_pursuit(
                design,
                signal,
                maximum_members=2,
                ridge=0.0,
            )


class OperationCountTests(unittest.TestCase):
    def test_member_count_within_budget(self) -> None:
        members = tuple(GmpMember(m, d, k) for m in (0, 1) for d in (0, 2)
                        for k in (1, 2, 4))
        count = gmp_member_operation_count(
            members,
            shared_envelope_delays={0},
        )
        self.assertEqual(count.real_multiplications, 2 + (3 + 3) + 12 * 4)
        self.assertLess(count.real_multiplications, 1000)

    def test_empty_members_cost_nothing(self) -> None:
        count = gmp_member_operation_count(())
        self.assertEqual(count.real_multiplications, 0)


class CompositeTests(unittest.TestCase):
    def setUp(self) -> None:
        train = _deterministic_signal(16_384, seed=13)
        self.train = train
        peak = float(np.max(np.abs(train)))
        self.pa = _synthetic_pa(peak * 1.05)
        self.gain = _least_squares_gain(train, self.pa.predict(train))
        self.warmup = 4
        self.spline_dpd, _ = fit_sparse_spline_memory_dpd(
            self.pa.predict(train[:8192]) / self.gain,
            train[:8192],
            branches=((0, 0), (1, 1), (2, 2)),
            knot_count=6,
            ridge=1e-8,
        )
        self.grid = GmpDictionaryGrid(
            maximum_signal_delay=6,
            maximum_envelope_delay=6,
            maximum_exponent=4,
        )

    def _composite(self) -> CompositeSplineGmpDPD:
        diagnostics = fit_gmp_residual_members(
            self.train[:8192],
            spline_model=self.spline_dpd,
            pa=self.pa,
            gain=self.gain,
            grid=self.grid,
            member_budgets=(12,),
            ridge_values=(1e-8,),
            warmup=self.warmup,
        )
        candidate = diagnostics["candidates"][0]
        composite = CompositeSplineGmpDPD(
            spline=self.spline_dpd,
            members=tuple(
                GmpMember(
                    row["signal_delay"],
                    row["envelope_delay"],
                    row["exponent"],
                )
                for row in candidate["selected_members"]
            ),
            member_coefficients=candidate["member_coefficients"],
        )
        return composite

    def test_composite_improves_cascade(self) -> None:
        composite = self._composite()
        check = self.train[8192:12_288]
        baseline_nmse = nmse_db(
            self.pa.predict(self.spline_dpd.predict(check)),
            self.gain * check,
            self.warmup,
        )
        composite_nmse = nmse_db(
            self.pa.predict(composite.predict(check)),
            self.gain * check,
            self.warmup,
        )
        self.assertLess(composite_nmse, baseline_nmse)

    def test_phase_equivariance(self) -> None:
        composite = self._composite()
        signal = self.train[:1024]
        rotation = np.exp(1j * np.pi / 2)
        direct = composite.predict(signal * rotation)
        rotated = rotation * composite.predict(signal)
        np.testing.assert_allclose(direct, rotated, rtol=1e-9, atol=1e-9)

    def test_chunk_equivalence(self) -> None:
        composite = self._composite()
        signal = self.train[:4096]
        full = composite.predict(signal)
        state = composite.initial_state()
        pieces = []
        for start in range(0, signal.size, 512):
            chunk_output, state = composite.predict_chunk(
                signal[start : start + 512],
                state,
            )
            pieces.append(chunk_output)
        streamed = np.concatenate(pieces)
        np.testing.assert_allclose(full, streamed, rtol=1e-10, atol=1e-10)

    def test_causality(self) -> None:
        composite = self._composite()
        signal = self.train[:2048]
        reference = composite.predict(signal)
        perturbed = signal.copy()
        perturbed[1024:] += 0.5 * (1.0 + 1j)
        changed = composite.predict(perturbed)
        np.testing.assert_allclose(
            reference[:1024],
            changed[:1024],
            rtol=0.0,
            atol=1e-12,
        )

    def test_save_load_roundtrip(self) -> None:
        import tempfile
        from pathlib import Path

        composite = self._composite()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "composite.npz"
            composite.save(path)
            loaded = CompositeSplineGmpDPD.load(path)
        np.testing.assert_allclose(
            loaded.predict(self.train[:512]),
            composite.predict(self.train[:512]),
            rtol=0.0,
            atol=0.0,
        )

    def test_operation_count_reflects_members(self) -> None:
        composite = self._composite()
        empty = CompositeSplineGmpDPD(
            spline=self.spline_dpd,
            members=(),
            member_coefficients=np.empty(0, dtype=np.complex128),
        )
        self.assertGreater(
            composite.operation_count().real_multiplications,
            empty.operation_count().real_multiplications,
        )

    def test_composite_design_matrix_width(self) -> None:
        composite = self._composite()
        design = composite_design_matrix(composite, self.train[:256])
        expected = (
            self.spline_dpd.branch_count * self.spline_dpd.knot_count
            + len(composite.members)
        )
        self.assertEqual(design.shape, (256, expected))


if __name__ == "__main__":
    unittest.main()
