import unittest
from unittest.mock import patch

import numpy as np

from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
)
from experiments.benchmark_dpd_timing import (
    REFERENCE_REAL_MULS,
    _measure_pair,
    benchmark_model,
    reference_1000_mul_kernel,
)


class DpdTimingTests(unittest.TestCase):
    def _model(self) -> SparseSplineMemoryDPD:
        coefficients = np.asarray(
            [
                [1.0 + 0.1j, 0.9 + 0.2j, 0.8 - 0.1j, 0.7 + 0.05j],
                [0.2 - 0.1j, 0.1 + 0.05j, -0.1 + 0.2j, 0.05 - 0.2j],
                [0.03 + 0.02j, -0.02 + 0.01j, 0.01 - 0.03j, 0.02j],
            ],
            dtype=np.complex128,
        )
        return SparseSplineMemoryDPD(
            knots=np.linspace(0.0, 1.0, 4),
            branches=(
                SplineMemoryBranch(0, 0),
                SplineMemoryBranch(1, 0),
                SplineMemoryBranch(2, 1),
            ),
            coefficients=coefficients,
        )

    def test_reference_has_explicit_1000_mul_contract(self) -> None:
        signal = np.asarray([1.0 + 2.0j, 0.5 - 1.0j])
        output = reference_1000_mul_kernel(signal)
        self.assertEqual(output.shape, signal.shape)
        self.assertEqual(REFERENCE_REAL_MULS, 1000)
        coefficients = np.linspace(0.25, 1.25, 1000)
        expected = (
            signal.real * np.sum(coefficients[:500])
            + 1j * signal.imag * np.sum(coefficients[500:])
        )
        np.testing.assert_allclose(output, expected, rtol=1e-14, atol=1e-14)

    def test_chunk_equivalence_and_time_ratio_are_recorded(self) -> None:
        signal = np.exp(1j * np.linspace(0.0, 2.0, 32))
        result = benchmark_model(
            self._model(),
            signal.astype(np.complex64),
            chunk_sizes=(1, 7, 32),
            warmup=0,
            repeats=1,
        )
        self.assertFalse(result["claims_scope"]["pa_included"])
        self.assertFalse(result["claims_scope"]["customer_gate_evaluable"])
        self.assertEqual(
            result["reference_contract"][
                "nominal_real_multiplications_per_sample"
            ],
            1000,
        )
        self.assertEqual(len(result["dpd"]["chunk_results"]), 3)
        self.assertTrue(
            result["dpd"]["analytical_vector_is_not_timed_python_trace"]
        )
        for row in result["dpd"]["chunk_results"]:
            self.assertTrue(row["streaming_equivalent"])
            self.assertGreaterEqual(
                row["host_python_time_ratio_to_scalar_reference"]["median"],
                0.0,
            )
            self.assertEqual(len(row["execution_order"]), 1)

    def test_reference_budget_and_chunk_inputs_are_strict(self) -> None:
        signal = np.ones(8, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "exactly 1000"):
            benchmark_model(
                self._model(),
                signal,
                chunk_sizes=(1,),
                reference_real_muls=999,
                warmup=0,
                repeats=1,
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            benchmark_model(
                self._model(),
                signal,
                chunk_sizes=(),
                warmup=0,
                repeats=1,
            )
        with self.assertRaisesRegex(ValueError, "timed sample count"):
            benchmark_model(
                self._model(),
                signal,
                chunk_sizes=(9,),
                warmup=0,
                repeats=1,
            )
        with self.assertRaisesRegex(ValueError, "positive integers"):
            benchmark_model(
                self._model(),
                signal,
                chunk_sizes=(True,),
                warmup=0,
                repeats=1,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            benchmark_model(
                self._model(),
                signal,
                chunk_sizes=(1, 1),
                warmup=0,
                repeats=1,
            )

    def test_timing_prefix_is_explicit_and_bounded(self) -> None:
        signal = np.exp(1j * np.linspace(0.0, 2.0, 32))
        result = benchmark_model(
            self._model(),
            signal,
            chunk_sizes=(4,),
            warmup=0,
            repeats=1,
            sample_limit=8,
        )
        self.assertEqual(result["signal"]["sample_count"], 8)
        self.assertEqual(result["signal"]["source_sample_count"], 32)
        self.assertEqual(
            result["signal"]["timing_prefix"],
            "first samples from desired_input",
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            benchmark_model(
                self._model(),
                signal,
                sample_limit=33,
                warmup=0,
                repeats=1,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            benchmark_model(
                self._model(),
                signal,
                sample_limit=True,
                warmup=0,
                repeats=1,
            )

    def test_paired_measurement_alternates_execution_order(self) -> None:
        calls: list[str] = []

        def dpd(values: np.ndarray) -> np.ndarray:
            calls.append("dpd")
            return values

        def reference(values: np.ndarray) -> np.ndarray:
            calls.append("reference")
            return values

        clock = [0, 10, 10, 30, 30, 60, 60, 75, 75, 87, 87, 111]
        with patch(
            "experiments.benchmark_dpd_timing.time.perf_counter_ns",
            side_effect=clock,
        ):
            result = _measure_pair(
                dpd,
                reference,
                np.ones(2, dtype=np.complex128),
                warmup=0,
                repeats=3,
            )
        self.assertEqual(
            calls,
            ["dpd", "reference", "reference", "dpd", "dpd", "reference"],
        )
        self.assertEqual(
            result["execution_order"],
            [
                "dpd_then_reference",
                "reference_then_dpd",
                "dpd_then_reference",
            ],
        )
        np.testing.assert_allclose(
            result["host_python_time_ratio_to_scalar_reference"][
                "per_repeat"
            ],
            [0.5, 0.5, 0.5],
        )


if __name__ == "__main__":
    unittest.main()
