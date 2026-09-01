"""Unit tests for bounded iterative direct learning helpers."""

from __future__ import annotations

import unittest

import numpy as np

from baseline.direct_learning import (
    DirectLearningConfig,
    bounded_direct_update,
    ilc_waveform_refinement,
    iterative_direct_schedule,
    model_with_delta,
    nmse_db,
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
    """Causal, finite complex baseband signal with a controlled peak."""

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
    """Known causal phase-equivariant spline-memory PA."""

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


class NmseTests(unittest.TestCase):
    def test_known_scaling(self) -> None:
        reference = np.ones(4096, dtype=np.complex128)
        deterministic_noise = np.exp(
            1j * 2.0 * np.pi * np.arange(4096) * 7.0 / 4096.0
        )
        estimate = reference + 0.1 * deterministic_noise
        self.assertAlmostEqual(nmse_db(estimate, reference, 0), -20.0, places=6)

    def test_rejects_bad_warmup(self) -> None:
        with self.assertRaises(ValueError):
            nmse_db(np.ones(8, dtype=complex), np.ones(8, dtype=complex), 8)


class BoundedDirectUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        train = _deterministic_signal(24_576, seed=11)
        self.train = train
        self.peak = float(np.max(np.abs(train)))
        self.pa = _synthetic_pa(self.peak * 1.05)
        raw_output = self.pa.predict(train)
        self.gain = _least_squares_gain(train, raw_output)
        self.maximum_pa_input = self.peak * 1.05
        self.warmup = 4
        # ILA baseline postdistorter on an early train block only.
        block = train[:8192]
        baseline, _ = fit_sparse_spline_memory_dpd(
            self.pa.predict(block) / self.gain,
            block,
            branches=((0, 0), (1, 1), (2, 2)),
            knot_count=6,
            ridge=1e-8,
        )
        self.baseline = baseline
        advisor = train[8192:16_384]
        self.baseline_advisor_nmse = nmse_db(
            self.pa.predict(baseline.predict(advisor)),
            self.gain * advisor,
            self.warmup,
        )

    def test_update_does_not_worsen_and_learns(self) -> None:
        result = bounded_direct_update(
            model=self.baseline,
            pa=self.pa,
            gain=self.gain,
            fit_x=self.train[:4096],
            advisor_x=self.train[16_384:20_480],
            warmup=self.warmup,
            maximum_pa_input=self.maximum_pa_input,
        )
        self.assertTrue(np.all(np.isfinite(result.delta)))
        self.assertLessEqual(
            result.advisor_nmse_db,
            self.baseline_advisor_nmse + 1e-6,
        )
        self.assertGreater(result.candidate_count, 0)
        self.assertTrue(result.support_valid)
        self.assertGreaterEqual(result.advisor_improvement_db, 0.0)

    def test_updated_model_improves_cascade(self) -> None:
        result = bounded_direct_update(
            model=self.baseline,
            pa=self.pa,
            gain=self.gain,
            fit_x=self.train[:4096],
            advisor_x=self.train[16_384:20_480],
            warmup=self.warmup,
            maximum_pa_input=self.maximum_pa_input,
        )
        updated = model_with_delta(
            self.baseline,
            result.delta,
            result.selected_step,
        )
        check = self.train[20_480:24_576]
        baseline_nmse = nmse_db(
            self.pa.predict(self.baseline.predict(check)),
            self.gain * check,
            self.warmup,
        )
        updated_nmse = nmse_db(
            self.pa.predict(updated.predict(check)),
            self.gain * check,
            self.warmup,
        )
        self.assertLessEqual(updated_nmse, baseline_nmse + 1e-9)

    def test_causality_is_preserved(self) -> None:
        result = bounded_direct_update(
            model=self.baseline,
            pa=self.pa,
            gain=self.gain,
            fit_x=self.train[:4096],
            advisor_x=self.train[16_384:20_480],
            warmup=self.warmup,
            maximum_pa_input=self.maximum_pa_input,
        )
        updated = model_with_delta(
            self.baseline,
            result.delta,
            result.selected_step,
        )
        signal = self.train[:2048]
        reference = updated.predict(signal)
        perturbed = signal.copy()
        perturbed[512:] += 0.5 * (1.0 + 1j)
        changed = updated.predict(perturbed)
        # Delay-zero branches make output[n] depend on x[n] itself, so the
        # outputs must coincide exactly up to (not including) the boundary.
        np.testing.assert_allclose(
            reference[:512],
            changed[:512],
            rtol=0.0,
            atol=1e-12,
        )

    def test_phase_equivariance_after_update(self) -> None:
        result = bounded_direct_update(
            model=self.baseline,
            pa=self.pa,
            gain=self.gain,
            fit_x=self.train[:4096],
            advisor_x=self.train[16_384:20_480],
            warmup=self.warmup,
            maximum_pa_input=self.maximum_pa_input,
        )
        updated = model_with_delta(
            self.baseline,
            result.delta,
            result.selected_step,
        )
        signal = self.train[:1024]
        rotation = np.exp(1j * np.pi / 2)
        direct = updated.predict(signal * rotation)
        rotated = rotation * updated.predict(signal)
        np.testing.assert_allclose(direct, rotated, rtol=1e-9, atol=1e-9)

    def test_rejects_invalid_gain(self) -> None:
        with self.assertRaises(ValueError):
            bounded_direct_update(
                model=self.baseline,
                pa=self.pa,
                gain=0.0,
                fit_x=self.train[:4096],
                advisor_x=self.train[16_384:20_480],
                warmup=self.warmup,
                maximum_pa_input=self.maximum_pa_input,
            )


class IterativeScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        train = _deterministic_signal(32_768, seed=23)
        self.train = train
        peak = float(np.max(np.abs(train)))
        self.pa = _synthetic_pa(peak * 1.05)
        raw_output = self.pa.predict(train[:8192])
        self.gain = _least_squares_gain(train[:8192], raw_output)
        self.maximum_pa_input = peak * 1.05
        self.warmup = 4
        baseline, _ = fit_sparse_spline_memory_dpd(
            self.pa.predict(train[:8192]) / self.gain,
            train[:8192],
            branches=((0, 0), (1, 1), (2, 2)),
            knot_count=6,
            ridge=1e-8,
        )
        self.baseline = baseline

    def test_schedule_runs_and_records(self) -> None:
        fit_slices = [slice(8192, 12_288), slice(16_384, 20_480)]
        advisor_slices = [slice(12_288, 16_384), slice(20_480, 24_576)]
        final, records = iterative_direct_schedule(
            self.baseline,
            pa=self.pa,
            gain=self.gain,
            train_x=self.train,
            fit_slices=fit_slices,
            advisor_slices=advisor_slices,
            warmup=self.warmup,
            maximum_pa_input=self.maximum_pa_input,
            minimum_improvement_db=-1.0,
        )
        self.assertEqual(len(records), 2)
        self.assertIsNone(records[0]["advisor_gain_over_previous_db"])
        self.assertIsNotNone(records[1]["advisor_gain_over_previous_db"])
        self.assertEqual(final.branch_count, self.baseline.branch_count)
        self.assertEqual(final.knot_count, self.baseline.knot_count)

    def test_validation_is_diagnostic_only(self) -> None:
        fit_slices = [slice(8192, 12_288)]
        advisor_slices = [slice(12_288, 16_384)]
        final, records = iterative_direct_schedule(
            self.baseline,
            pa=self.pa,
            gain=self.gain,
            train_x=self.train,
            fit_slices=fit_slices,
            advisor_slices=advisor_slices,
            warmup=self.warmup,
            maximum_pa_input=self.maximum_pa_input,
            validation_x=self.train[:4096],
        )
        diagnostic = records[-1]
        self.assertTrue(diagnostic["validation_diagnostic_only"])
        self.assertFalse(diagnostic["validation_used_for_selection"])
        self.assertEqual(len(records), 2)
        self.assertEqual(final.branch_count, self.baseline.branch_count)

    def test_joint_objective_runs_and_improves_both(self) -> None:
        fit_slices = [slice(8192, 12_288), slice(16_384, 20_480)]
        advisor_slices = [slice(12_288, 16_384), slice(20_480, 24_576)]
        joint, joint_records = iterative_direct_schedule(
            self.baseline,
            pa=self.pa,
            gain=self.gain,
            train_x=self.train,
            fit_slices=fit_slices,
            advisor_slices=advisor_slices,
            warmup=self.warmup,
            maximum_pa_input=self.maximum_pa_input,
            secondary=(self.pa, self.gain),
            joint_objective=True,
            minimum_improvement_db=-1.0,
        )
        self.assertEqual(len(joint_records), 2)
        self.assertEqual(joint.branch_count, self.baseline.branch_count)
        for x_slice in (slice(24_576, 28_672), slice(28_672, 32_768)):
            x = self.train[x_slice]
            base_drive = self.baseline.predict(x)
            joint_drive = joint.predict(x)
            base_output = self.pa.predict(base_drive)
            joint_output = self.pa.predict(joint_drive)
            base_nmse = nmse_db(base_output, self.gain * x, self.warmup)
            joint_nmse = nmse_db(joint_output, self.gain * x, self.warmup)
            self.assertLessEqual(joint_nmse, base_nmse + 1e-9)

    def test_rejects_overlapping_slices(self) -> None:
        with self.assertRaises(ValueError):
            iterative_direct_schedule(
                self.baseline,
                pa=self.pa,
                gain=self.gain,
                train_x=self.train,
                fit_slices=[slice(8192, 12_288)],
                advisor_slices=[slice(12_000, 16_384)],
                warmup=self.warmup,
                maximum_pa_input=self.maximum_pa_input,
            )

    def test_rejects_out_of_range_slices(self) -> None:
        with self.assertRaises(ValueError):
            iterative_direct_schedule(
                self.baseline,
                pa=self.pa,
                gain=self.gain,
                train_x=self.train,
                fit_slices=[slice(30_000, 40_000)],
                advisor_slices=[slice(8192, 12_288)],
                warmup=self.warmup,
                maximum_pa_input=self.maximum_pa_input,
            )


class IlcWaveformTests(unittest.TestCase):
    def test_ilc_improves_cascade_waveform(self) -> None:
        train = _deterministic_signal(16_384, seed=31)
        peak = float(np.max(np.abs(train)))
        pa = _synthetic_pa(peak * 1.05)
        raw_output = pa.predict(train)
        gain = _least_squares_gain(train, raw_output)
        maximum_pa_input = peak * 1.05
        warmup = 4
        drive, records = ilc_waveform_refinement(
            train,
            pa=pa,
            gain=gain,
            warmup=warmup,
            maximum_pa_input=maximum_pa_input,
            beta=1.0,
            maximum_iterations=6,
        )
        self.assertEqual(records[0]["iteration"], 0)
        self.assertLess(
            records[-1]["cascade_nmse_db"],
            records[0]["cascade_nmse_db"] - 0.01,
        )
        self.assertLessEqual(
            float(np.max(np.abs(drive))),
            maximum_pa_input + 1e-12,
        )
        # A causal DPD refit on the ILC pairs must not be worse than the
        # plain ILA fit through the same evaluator.
        ila, _ = fit_sparse_spline_memory_dpd(
            pa.predict(train) / gain,
            train,
            branches=((0, 0), (1, 1), (2, 2)),
            knot_count=6,
            ridge=1e-8,
        )
        ilc_fitted, _ = fit_sparse_spline_memory_dpd(
            train,
            drive,
            branches=((0, 0), (1, 1), (2, 2)),
            knot_count=6,
            ridge=1e-8,
        )
        check = train[8192:12_288]
        ila_nmse = nmse_db(
            pa.predict(ila.predict(check)),
            gain * check,
            warmup,
        )
        ilc_nmse = nmse_db(
            pa.predict(ilc_fitted.predict(check)),
            gain * check,
            warmup,
        )
        self.assertLessEqual(ilc_nmse, ila_nmse + 1e-6)

    def test_rejects_bad_beta(self) -> None:
        train = _deterministic_signal(2048, seed=41)
        peak = float(np.max(np.abs(train)))
        pa = _synthetic_pa(peak * 1.05)
        with self.assertRaises(ValueError):
            ilc_waveform_refinement(
                train,
                pa=pa,
                gain=1.0,
                warmup=4,
                maximum_pa_input=peak,
                beta=0.0,
            )


class ConfigTests(unittest.TestCase):
    def test_rejects_invalid_grid(self) -> None:
        with self.assertRaises(ValueError):
            DirectLearningConfig(ridge_values=())
        with self.assertRaises(ValueError):
            DirectLearningConfig(step_values=(-1.0,))
        with self.assertRaises(ValueError):
            DirectLearningConfig(epsilon=0.0)


if __name__ == "__main__":
    unittest.main()
