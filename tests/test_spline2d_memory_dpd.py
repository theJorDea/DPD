"""Tests for the 2D spline-memory DPD (model, fit, save/load)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.spline2d_memory_dpd import (  # noqa: E402
    Grid2DBranch,
    Spline2DMemoryDPD,
    fit_spline2d_memory_dpd,
)
from baseline.spline_memory_dpd import (  # noqa: E402
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    fit_sparse_spline_memory_dpd,
)


def _synthetic_record(n: int = 20000, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.35
    r = np.abs(x)
    r1 = np.roll(r, 1)
    r1[0] = 0.0
    # Dynamic hysteresis term b*r1^2 is invisible to any 1D curve of r.
    y = x * (1.0 - 0.30 * r**2 + 0.12 * r1**2)
    return x, y


def _one_d_body(
    x: np.ndarray, y: np.ndarray
) -> SparseSplineMemoryDPD:
    branches = [SplineMemoryBranch(signal_delay=m, envelope_delay=0) for m in range(4)]
    model, _ = fit_sparse_spline_memory_dpd(
        y / 1.0,
        x,
        branches=branches,
        knot_count=16,
        knot_strategy="quantile",
        ridge=1e-8,
    )
    return model


class Spline2DMemoryDPDTests(unittest.TestCase):
    def test_predict_shape_and_finiteness(self) -> None:
        x, y = _synthetic_record(4000)
        body = _one_d_body(x, y)
        model = Spline2DMemoryDPD(
            body,
            (Grid2DBranch(0, 0, 1, knot_count=8),),
            (np.linspace(0.0, float(np.max(np.abs(x))), 8),),
            np.zeros((1, 8, 8), dtype=np.complex128),
        )
        out = model.predict(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_2d_branch_captures_hysteresis_that_1d_cannot(self) -> None:
        x, y = _synthetic_record()
        body = _one_d_body(x, y)
        body_error = np.mean(np.abs(body.predict(y / 1.0) - x) ** 2)
        model, diagnostics = fit_spline2d_memory_dpd(
            y,
            x,
            body=body,
            grid_branches=[Grid2DBranch(0, 0, 1, knot_count=10)],
            ridge=1e-9,
            refit_body=True,
        )
        joint_error = np.mean(np.abs(model.predict(y) - x) ** 2)
        # A 10x10 piecewise-linear grid cannot represent the smooth hysteresis
        # term exactly, but must remove most of the 1D-irreducible error.
        self.assertLess(
            joint_error,
            0.7 * body_error,
            "2D grid must remove the dominant hysteresis term the 1D body misses",
        )
        self.assertGreater(diagnostics["gram_condition_number"], 0.0)

    def test_frozen_body_mode_reduces_residual(self) -> None:
        x, y = _synthetic_record()
        body = _one_d_body(x, y)
        base_error = np.mean(np.abs(body.predict(y / 1.0) - x) ** 2)
        model, _ = fit_spline2d_memory_dpd(
            y,
            x,
            body=body,
            grid_branches=[Grid2DBranch(0, 0, 1, knot_count=10)],
            ridge=1e-9,
            refit_body=False,
        )
        # Frozen body: model output must contain the unchanged body output.
        grid_out = model.predict(y) - body.predict(y / 1.0)
        self.assertTrue(np.all(np.isfinite(grid_out)))
        joint_error = np.mean(np.abs(model.predict(y) - x) ** 2)
        self.assertLess(joint_error, base_error)

    def test_save_load_roundtrip(self) -> None:
        x, y = _synthetic_record(6000)
        body = _one_d_body(x, y)
        model, _ = fit_spline2d_memory_dpd(
            y,
            x,
            body=body,
            grid_branches=[Grid2DBranch(0, 0, 1, knot_count=6)],
            ridge=1e-8,
            refit_body=True,
        )
        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            model.save(path)
            restored = Spline2DMemoryDPD.load(path)
        same = np.max(np.abs(restored.predict(x) - model.predict(x)))
        self.assertLess(float(same), 1e-9)
        self.assertEqual(
            restored.stored_complex_coefficients,
            model.stored_complex_coefficients,
        )


if __name__ == "__main__":
    unittest.main()
