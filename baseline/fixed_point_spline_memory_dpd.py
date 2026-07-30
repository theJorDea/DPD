"""Bit-accurate fixed-point reference for sparse spline-memory DPD.

The deployed predistorter is

    z[n] = sum_b x[n-m_b] C_b(|x[n-d_b]|),

where ``x`` is the desired transmit signal and ``z`` is the predistorted PA
drive.  The integer arithmetic is identical to the already-audited sparse
spline-memory forward kernel, so this module wraps that kernel instead of
duplicating square-root, interpolation, saturation and state logic.

The wrapper preserves DPD semantics and type safety:

* it accepts only :class:`SparseSplineMemoryDPD`, never a PA model;
* ``self.model`` remains the frozen DPD coefficient set;
* streaming state must contain exactly ``maximum_delay`` complex samples;
* metadata states the desired-input to predistorted-drive direction.

This is a numerical reference, not an RTL/HLS latency claim.
"""

from __future__ import annotations

import numpy as np

from .complexity import ComplexMultiplyConvention, OperationCount
from .fixed_point_pa import (
    FixedPointPAConfig,
    FixedPointPAResult,
    FixedPointPAState,
    FixedPointPAStats,
)
from .fixed_point_sparse_spline_pa import FixedPointSparseSplineMemoryPA
from .sparse_spline_memory_pa import SparseSplineMemoryPA
from .spline_memory_dpd import SparseSplineMemoryDPD


FixedPointDPDConfig = FixedPointPAConfig
FixedPointDPDState = FixedPointPAState
FixedPointDPDStats = FixedPointPAStats
FixedPointDPDResult = FixedPointPAResult


class FixedPointSparseSplineMemoryDPD:
    """Causal integer reference for a frozen sparse spline-memory DPD."""

    def __init__(
        self,
        model: SparseSplineMemoryDPD,
        config: FixedPointDPDConfig,
    ) -> None:
        if not isinstance(model, SparseSplineMemoryDPD):
            raise TypeError("model must be SparseSplineMemoryDPD")
        if not isinstance(config, FixedPointPAConfig):
            raise TypeError("config must be FixedPointDPDConfig")
        self.model = model
        self.config = config
        arithmetic_proxy = SparseSplineMemoryPA(
            knots=model.knots,
            branches=model.branches,
            coefficients=model.coefficients,
            knot_strategy=model.knot_strategy,
        )
        self._kernel = FixedPointSparseSplineMemoryPA(
            arithmetic_proxy,
            config,
        )

    @property
    def history_length(self) -> int:
        return self._kernel.history_length

    @property
    def knot_codes(self) -> np.ndarray:
        return self._kernel.knot_codes

    @property
    def coefficient_saturation_count(self) -> int:
        return self._kernel.coefficient_saturation_count

    @property
    def knot_code_collision_count(self) -> int:
        return self._kernel.knot_code_collision_count

    @property
    def maximum_knot_code_shift(self) -> int:
        return self._kernel.maximum_knot_code_shift

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "fixed_point_sparse_spline_memory_dpd",
            "direction": "desired_input_to_predistorted_drive",
            "equation": "sum_b x[n-m_b] * C_b(abs(x[n-d_b]))",
            "floating_model_metadata": self.model.metadata,
            "fixed_point_config": self.config.to_dict(),
            "integer_arithmetic_reuse": (
                "sparse spline-memory kernel only; no PA direction semantics"
            ),
            "hardware_latency_or_resources": False,
        }

    def initial_state(self) -> FixedPointDPDState:
        return self._kernel.initial_state()

    def _validate_state(self, state: FixedPointDPDState | None) -> None:
        if state is None:
            return
        if not isinstance(state, FixedPointPAState):
            raise TypeError("state must be FixedPointDPDState")
        if state.size != self.history_length:
            raise ValueError(
                "DPD streaming state must contain exactly maximum_delay "
                "complex samples"
            )

    def predict_chunk(
        self,
        signal: np.ndarray,
        state: FixedPointDPDState | None = None,
    ) -> FixedPointDPDResult:
        """Predistort one chunk and return the state for the next chunk."""

        self._validate_state(state)
        return self._kernel.predict_chunk(signal, state)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Predistort one independent record with zero initial history."""

        return self.predict_chunk(signal).output

    __call__ = predict

    def predict_segments(
        self,
        signal: np.ndarray,
        segment_length: int,
    ) -> np.ndarray:
        """Predistort independent segments, resetting state at each boundary."""

        return self._kernel.predict_segments(signal, segment_length)

    def operation_count(
        self,
        *,
        convention: ComplexMultiplyConvention = "4m2a",
        indexing: str = "binary",
    ) -> OperationCount:
        """Return the explicit integer DPD schedule per complex sample."""

        return self._kernel.operation_count(
            convention=convention,
            indexing=indexing,
        )


__all__ = [
    "FixedPointDPDConfig",
    "FixedPointDPDResult",
    "FixedPointDPDState",
    "FixedPointDPDStats",
    "FixedPointSparseSplineMemoryDPD",
]
