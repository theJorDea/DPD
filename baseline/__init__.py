"""Auditable low-complexity DPD baselines.

The package deliberately keeps the reference implementations NumPy-only.  It
is intended for calibration/evaluation and for producing coefficients that can
later be mapped to a streaming fixed-point implementation.
"""

from .complex_spline_dpd import (
    ComplexLinearSplineDPD,
    SplineFitDiagnostics,
    fit_complex_linear_spline,
    fit_ila_postdistorter,
)
from .pa_models import (
    MemoryPolynomialFitDiagnostics,
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
    memory_polynomial_design_matrix,
    memory_polynomial_segmented_design_matrix,
    segmented_steady_state_mask,
)
from .spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    SplineMemoryFitDiagnostics,
    SplineMemoryState,
    fit_ila_sparse_spline_memory_dpd,
    fit_sparse_spline_memory_dpd,
    spline_memory_design_matrix,
)
from .complexity import memory_polynomial_inference_cost

__all__ = [
    "ComplexLinearSplineDPD",
    "SplineFitDiagnostics",
    "fit_complex_linear_spline",
    "fit_ila_postdistorter",
    "MemoryPolynomialPA",
    "MemoryPolynomialFitDiagnostics",
    "fit_memory_polynomial_pa",
    "memory_polynomial_design_matrix",
    "memory_polynomial_segmented_design_matrix",
    "segmented_steady_state_mask",
    "SparseSplineMemoryDPD",
    "SplineMemoryBranch",
    "SplineMemoryFitDiagnostics",
    "SplineMemoryState",
    "fit_sparse_spline_memory_dpd",
    "fit_ila_sparse_spline_memory_dpd",
    "spline_memory_design_matrix",
    "memory_polynomial_inference_cost",
]
