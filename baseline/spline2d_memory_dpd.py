"""Two-dimensional spline-memory DPD: ``x[n-m] * C2(|x[n-d0]|, |x[n-d1]|)``.

The 2D branch adds a bilinear control grid over the current and a delayed
envelope, catching dynamic AM/AM hysteresis that one-dimensional curves
from separate delays cannot represent.  Coefficients stay linear in the
least-squares fit: on each sample a bilinear patch activates four
(k, l) control points, so the design columns are
``x[n-m] * phi_k(|x[n-d0]|) * psi_l(|x[n-d1]|)``.

Model layout: an optional frozen 1D ``SparseSplineMemoryDPD`` body plus
one or more 2D grids.  Everything is fitted jointly by augmented ridge
least squares through normal equations (Gram system, complex128).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from baseline.complex_spline_dpd import local_spline_coordinates
from baseline.spline_memory_dpd import SparseSplineMemoryDPD, SplineMemoryBranch


@dataclass(frozen=True)
class Grid2DBranch:
    """One bilinear grid branch: signal delay m, envelope delays d0/d1."""

    signal_delay: int
    envelope_delay_0: int
    envelope_delay_1: int
    knot_count: int = 12

    def __post_init__(self) -> None:
        for name in ("signal_delay", "envelope_delay_0", "envelope_delay_1"):
            value = getattr(self, name)
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.knot_count) < 4:
            raise ValueError("knot_count must be at least 4")


class Spline2DMemoryDPD:
    """1D shared-knot spline body plus bilinear 2D grid branches."""

    def __init__(
        self,
        body: SparseSplineMemoryDPD | None,
        grid_branches: tuple[Grid2DBranch, ...],
        grid_knots: tuple[np.ndarray, ...],
        grid_values: np.ndarray,
    ) -> None:
        if not grid_branches:
            raise ValueError("at least one 2D grid branch is required")
        if body is not None and not isinstance(body, SparseSplineMemoryDPD):
            raise TypeError("body must be a SparseSplineMemoryDPD or None")
        branches = tuple(grid_branches)
        knots = tuple(np.asarray(k, dtype=np.float64) for k in grid_knots)
        if len(knots) != len(branches):
            raise ValueError("one knot array per grid branch is required")
        values = np.asarray(grid_values)
        if values.ndim != 3 or values.shape[0] != len(branches):
            raise ValueError(
                "grid_values must have shape (branches, K0, K1)"
            )
        for index, (branch, knot_pair) in enumerate(zip(branches, knots)):
            if knot_pair.size != branch.knot_count:
                raise ValueError(
                    f"grid branch {index} expects {branch.knot_count} knots"
                )
            if values.shape[1:] != (branch.knot_count, branch.knot_count):
                raise ValueError(
                    "grid_values inner shape must match knot_count per branch"
                )
        if not np.all(np.isfinite(values)):
            raise ValueError("grid values contain NaN or infinite values")
        self.body = body
        self.grid_branches = branches
        self.grid_knots = knots
        self.grid_values = values.astype(np.complex128, copy=False)

    @property
    def maximum_delay(self) -> int:
        body_delay = self.body.maximum_delay if self.body is not None else 0
        grid_delay = max(
            max(
                branch.signal_delay,
                branch.envelope_delay_0,
                branch.envelope_delay_1,
            )
            for branch in self.grid_branches
        )
        return max(body_delay, grid_delay)

    @property
    def stored_complex_coefficients(self) -> int:
        body_count = (
            self.body.stored_complex_coefficients if self.body is not None else 0
        )
        return body_count + int(self.grid_values.size)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        samples = np.asarray(signal, dtype=np.complex128).reshape(-1)
        if self.body is not None:
            output = self.body.predict(samples).astype(np.complex128)
        else:
            output = np.zeros(samples.size, dtype=np.complex128)
        n = samples.size
        magnitude = np.abs(samples)
        for index, branch in enumerate(self.grid_branches):
            knots = self.grid_knots[index]
            left0, weight0 = local_spline_coordinates(magnitude, knots)
            shifted = np.zeros(n, dtype=np.float64)
            if branch.envelope_delay_1 < n:
                shifted[branch.envelope_delay_1:] = magnitude[
                    : n - branch.envelope_delay_1
                ]
            left1, weight1 = local_spline_coordinates(shifted, knots)
            signal_shift = np.zeros(n, dtype=np.complex128)
            if branch.signal_delay < n:
                signal_shift[branch.signal_delay:] = samples[
                    : n - branch.signal_delay
                ]
            values = self.grid_values[index]
            k0 = np.stack(
                (left0, np.minimum(left0 + 1, knots.size - 1)), axis=1
            )
            k1 = np.stack(
                (left1, np.minimum(left1 + 1, knots.size - 1)), axis=1
            )
            w0 = np.stack((1.0 - weight0, weight0), axis=1)
            w1 = np.stack((1.0 - weight1, weight1), axis=1)
            patch = np.zeros(n, dtype=np.complex128)
            for di in (0, 1):
                for dj in (0, 1):
                    patch += (
                        w0[:, di]
                        * w1[:, dj]
                        * values[k0[:, di], k1[:, dj]]
                    )
            output += signal_shift * patch
        return output

    def save(self, path: str | Path) -> None:
        payload: dict[str, object] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "model_type": np.asarray("spline2d_memory_dpd"),
            "grid_signal_delay": np.asarray(
                [b.signal_delay for b in self.grid_branches], dtype=np.int64
            ),
            "grid_envelope_delay_0": np.asarray(
                [b.envelope_delay_0 for b in self.grid_branches], dtype=np.int64
            ),
            "grid_envelope_delay_1": np.asarray(
                [b.envelope_delay_1 for b in self.grid_branches], dtype=np.int64
            ),
            "grid_knot_counts": np.asarray(
                [b.knot_count for b in self.grid_branches], dtype=np.int64
            ),
            "grid_knots": np.concatenate(
                [k for k in self.grid_knots]
            ),
            "grid_values": self.grid_values.reshape(-1),
        }
        if self.body is not None:
            body_path = Path(path).with_suffix(".body.npz")
            self.body.save(body_path)
            payload["body_model_path"] = np.asarray(body_path.name)
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Spline2DMemoryDPD":
        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported spline2d schema")
            if str(data["model_type"]) != "spline2d_memory_dpd":
                raise ValueError("unexpected spline2d model type")
            delays0 = data["grid_signal_delay"].tolist()
            delays1 = data["grid_envelope_delay_0"].tolist()
            delays2 = data["grid_envelope_delay_1"].tolist()
            counts = data["grid_knot_counts"].tolist()
            knots_flat = data["grid_knots"]
            values_flat = data["grid_values"]
        branches = tuple(
            Grid2DBranch(
                signal_delay=int(delays0[i]),
                envelope_delay_0=int(delays1[i]),
                envelope_delay_1=int(delays2[i]),
                knot_count=int(counts[i]),
            )
            for i in range(len(counts))
        )
        knots: list[np.ndarray] = []
        offset = 0
        for count in counts:
            knots.append(knots_flat[offset : offset + count])
            offset += count
        values = tuple(
            values_flat[
                i * int(counts[i]) * int(counts[i]) :
                (i + 1) * int(counts[i]) * int(counts[i])
            ].reshape(int(counts[i]), int(counts[i]))
            for i in range(len(counts))
        )
        body = None
        body_path = Path(path).with_suffix(".body.npz")
        if body_path.exists():
            body = SparseSplineMemoryDPD.load(body_path)
        return cls(body, branches, knots, values)


def _grid_design_columns(
    samples: np.ndarray,
    branch: Grid2DBranch,
    knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense bilinear design columns and index bookkeeping."""

    n = samples.size
    magnitude = np.abs(samples)
    phi = _spline_basis(magnitude, knots)
    shifted_env = np.zeros(n, dtype=np.float64)
    if branch.envelope_delay_1 < n:
        shifted_env[branch.envelope_delay_1 :] = magnitude[
            : n - branch.envelope_delay_1
        ]
    psi = _spline_basis(shifted_env, knots)
    signal_shift = np.zeros(n, dtype=np.complex128)
    if branch.signal_delay < n:
        signal_shift[branch.signal_delay :] = samples[: n - branch.signal_delay]
    k = knots.size
    columns = np.empty((n, k * k), dtype=np.complex128)
    offset = 0
    for ki in range(k):
        for kj in range(k):
            columns[:, offset] = signal_shift * phi[:, ki] * psi[:, kj]
            offset += 1
    return columns, phi, psi


def _spline_basis(radius: np.ndarray, knots: np.ndarray) -> np.ndarray:
    left, weight = local_spline_coordinates(radius, knots)
    basis = np.zeros((radius.size, knots.size), dtype=np.float64)
    rows = np.arange(radius.size)
    basis[rows, left] = 1.0 - weight
    basis[rows, left + 1] += weight
    return basis


def fit_spline2d_memory_dpd(
    calibration_input: np.ndarray,
    target: np.ndarray,
    *,
    body: SparseSplineMemoryDPD | None = None,
    grid_branches: Sequence[Grid2DBranch],
    knot_strategy: str = "quantile",
    ridge: float = 1e-8,
    refit_body: bool = True,
) -> tuple[Spline2DMemoryDPD, dict[str, object]]:
    """Jointly fit the 1D body coefficients and 2D grid values by ridge LS.

    ``calibration_input`` is the ILA observation (typically ``y/g``) and
    ``target`` the desired predistorted sequence (typically ``x``).  When
    ``refit_body`` is false the supplied body coefficients stay frozen and
    only the 2D grids are fitted on the residual.
    """

    samples = np.asarray(calibration_input, dtype=np.complex128).reshape(-1)
    desired = np.asarray(target, dtype=np.complex128).reshape(-1)
    if samples.shape != desired.shape:
        raise ValueError("calibration_input and target must have equal length")
    if not grid_branches:
        raise ValueError("at least one 2D grid branch is required")
    if body is None:
        raise ValueError("a 1D body is required (branches define the knots)")

    branch_list = list(grid_branches)
    knots: list[np.ndarray] = []
    for branch in branch_list:
        if branch.envelope_delay_1 == 0:
            radius = np.abs(samples)
        else:
            shifted = np.zeros(samples.size, dtype=np.float64)
            shifted[branch.envelope_delay_1 :] = np.abs(samples)[
                : samples.size - branch.envelope_delay_1
            ]
            radius = shifted
        positive = radius[radius > 0.0]
        if positive.size == 0:
            raise ValueError("calibration record has zero amplitude")
        unit = np.linspace(0.0, 1.0, branch.knot_count)
        knot = np.quantile(positive, unit)
        knot[0] = 0.0
        knot[-1] = float(np.max(np.abs(samples)))
        knots.append(np.unique(knot))

    pieces: list[np.ndarray] = []
    for branch, knot in zip(branch_list, knots):
        columns, _, _ = _grid_design_columns(samples, branch, knot)
        pieces.append(columns)
    grid_columns = np.concatenate(pieces, axis=1)

    n = samples.size
    normalization = float(np.sqrt(n))
    if refit_body:
        body_basis = _body_design_columns(samples, body)
        design = np.concatenate((body_basis, grid_columns), axis=1)
        body_free = True
    else:
        base_prediction = body.predict(samples).astype(np.complex128)
        design = grid_columns
        desired = desired - base_prediction
        body_free = False

    design_n = design / normalization
    target_full = desired
    target_n = target_full / normalization
    gram = design_n.conj().T @ design_n
    rhs = design_n.conj().T @ target_n
    feature_count = design.shape[1]
    gram += ridge * np.eye(feature_count, dtype=np.complex128)
    condition = float(np.linalg.cond(gram))
    flat = np.linalg.solve(gram, rhs)

    if body_free:
        body_coefficients = flat[: body.coefficients.size].reshape(
            body.coefficients.shape
        )
        fitted_body = SparseSplineMemoryDPD(
            knots=body.knots,
            branches=body.branches,
            coefficients=body_coefficients,
            knot_strategy=body.knot_strategy,
        )
        grid_flat = flat[body.coefficients.size :]
    else:
        fitted_body = body
        grid_flat = flat

    per_branch: list[np.ndarray] = []
    offset = 0
    for index, branch in enumerate(branch_list):
        knot_count = knots[index].size
        count = knot_count * knot_count
        piece = grid_flat[offset : offset + count].reshape(knot_count, knot_count)
        padded = np.pad(
            piece,
            (
                (0, branch.knot_count - piece.shape[0]),
                (0, branch.knot_count - piece.shape[1]),
            ),
        )
        per_branch.append(padded)
        offset += count
    grid_values = np.stack(per_branch, axis=0)
    model = Spline2DMemoryDPD(fitted_body, tuple(branch_list), knots, grid_values)
    prediction = model.predict(samples).astype(np.complex128)
    warm = model.maximum_delay
    error = prediction[warm:] - target_full[warm:]
    reference = target_full[warm:]
    training_nmse_db = float(
        10 * np.log10(np.mean(np.abs(error) ** 2) / np.mean(np.abs(reference) ** 2))
    )
    diagnostics: dict[str, object] = {
        "sample_count": int(n),
        "grid_branches": [
            {
                "signal_delay": branch.signal_delay,
                "envelope_delay_0": branch.envelope_delay_0,
                "envelope_delay_1": branch.envelope_delay_1,
                "effective_knot_count": int(knots[index].size),
            }
            for index, branch in enumerate(branch_list)
        ],
        "feature_count": int(feature_count),
        "ridge": float(ridge),
        "gram_condition_number": condition,
        "body_refitted": bool(body_free),
        "causal_warmup_samples": int(warm),
        "training_nmse_db_after_warmup": training_nmse_db,
    }
    return model, diagnostics


def _body_design_columns(
    samples: np.ndarray, body: SparseSplineMemoryDPD
) -> np.ndarray:
    """Design columns of the shared-knot 1D body (linear in knot values)."""

    n = samples.size
    columns = np.empty(
        (n, body.branch_count * body.knot_count), dtype=np.complex128
    )
    for branch_index, branch in enumerate(body.branches):
        signal_shift = np.zeros(n, dtype=np.complex128)
        if branch.signal_delay < n:
            signal_shift[branch.signal_delay :] = samples[: n - branch.signal_delay]
        envelope_shift = np.zeros(n, dtype=np.float64)
        if branch.envelope_delay < n:
            envelope_shift[branch.envelope_delay :] = np.abs(samples)[
                : n - branch.envelope_delay
            ]
        left, weight = local_spline_coordinates(envelope_shift, body.knots)
        for knot_index in range(body.knot_count):
            column = np.zeros(n, dtype=np.complex128)
            mask_lo = left == knot_index
            mask_hi = left + 1 == knot_index
            column[mask_lo] = signal_shift[mask_lo] * (1.0 - weight[mask_lo])
            column[mask_hi] += signal_shift[mask_hi] * weight[mask_hi]
            columns[:, branch_index * body.knot_count + knot_index] = column
    return columns
