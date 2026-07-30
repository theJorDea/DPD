import unittest

import numpy as np

from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
)
from experiments.benchmark_dpd_timing import (
    REFERENCE_REAL_MULS,
    benchmark_model,
    reference_1000_mul_kernel,
)


class DpdTimingTests(unittest.TestCase):
    def _model(self) -> SparseSplineMemoryDPD:
        return SparseSplineMemoryDPD(
            knots=np.linspace(0.0, 1.0, 4),
            branches=(
                SplineMemoryBranch(0, 0),
                SplineMemoryBranch(1, 0),
            ),
            coefficients=np.ones((2, 4), dtype=np.complex128),
        )

    def test_reference_has_explicit_1000_mul_contract(self) -> None:
        signal = np.asarray([1.0 + 2.0j, 0.5 - 1.0j])
        output = reference_1000_mul_kernel(signal)
        self.assertEqual(output.shape, signal.shape)
        self.assertEqual(REFERENCE_REAL_MULS, 1000)

    def test_chunk_equivalence_and_time_ratio_are_recorded(self) -> None:
        signal = np.exp(1j * np.linspace(0.0, 2.0, 32))
        result = benchmark_model(
            self._model(),
            signal.astype(np.complex64),
            chunk_sizes=(1, 8, 32),
            warmup=0,
            repeats=1,
        )
        self.assertFalse(result["claims_scope"]["pa_included"])
        self.assertEqual(
            result["reference"]["nominal_real_multiplications_per_sample"],
            1000,
        )
        self.assertEqual(len(result["dpd"]["chunk_results"]), 3)
        for row in result["dpd"]["chunk_results"]:
            self.assertTrue(row["streaming_equivalent"])
            self.assertGreaterEqual(row["time_ratio_to_reference"], 0.0)

    def test_reference_budget_and_chunk_inputs_are_strict(self) -> None:
        signal = np.ones(8, dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "exactly 1000"):
            benchmark_model(
                self._model(),
                signal,
                reference_real_muls=999,
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


if __name__ == "__main__":
    unittest.main()
