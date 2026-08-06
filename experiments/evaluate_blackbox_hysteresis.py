"""Evaluate the branch-specific residual hysteresis DPD on BlackBox capture.

This is a deliberately separate experiment.  It leaves the published
``Spline_memory_DPD`` baseline untouched and uses the same train/validation
split, train-only normalization/gain, and frozen GMP PA evaluator as the
detailed notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from baseline.complex_spline_dpd import make_knots
from baseline.gmp_pa import GMPConfig, fit_gmp_pa
from baseline.hysteresis_spline_dpd import (
    ResidualHysteresisSplineMemoryDPD,
    branch_hysteresis_gate,
    fit_residual_hysteresis_spline_memory_dpd,
    hysteresis_design_matrices,
)
from baseline.metrics import nmse_pooled_db
from baseline.spline_memory_dpd import (
    SplineMemoryBranch,
    fit_ila_sparse_spline_memory_dpd,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_DIR = PROJECT_ROOT / "data/private/blackbox_v3/selection"
OUTPUT_PATH = PROJECT_ROOT / "experiments/results/blackbox_hysteresis_v1.json"
BRANCHES = (
    SplineMemoryBranch(0, 0),
    SplineMemoryBranch(1, 1),
    SplineMemoryBranch(2, 2),
)
KNOT_COUNT = 24
RIDGE_BASELINE = 1e-4
RIDGE_RESIDUAL_VALUES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
WARMUP = 3


def read_iq(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.asarray(raw[:, 0] + 1j * raw[:, 1], dtype=np.complex128)


def pooled_metrics(estimate: np.ndarray, reference: np.ndarray, warmup: int = WARMUP) -> dict[str, float]:
    error = estimate[warmup:] - reference[warmup:]
    ref = reference[warmup:]
    mse = float(np.mean(np.abs(error) ** 2))
    ref_power = float(np.mean(np.abs(ref) ** 2))
    return {
        "nmse_db": float(10.0 * np.log10(mse / ref_power)),
        "mse": mse,
        "peak": float(np.max(np.abs(estimate))),
        "papr_db": float(10.0 * np.log10(np.max(np.abs(estimate) ** 2) / np.mean(np.abs(estimate) ** 2))),
    }


def segment_nmse(estimate: np.ndarray, reference: np.ndarray, segments: int = 8) -> list[float]:
    values = []
    for segment in np.array_split(np.arange(reference.size), segments):
        if segment.size <= WARMUP:
            continue
        values.append(float(nmse_pooled_db(estimate[segment][WARMUP:], reference[segment][WARMUP:])))
    return values


def delayed_radius(signal: np.ndarray, delay: int) -> np.ndarray:
    """Return causal ``|signal[n-delay]|`` with zero record-start padding."""
    result = np.zeros(signal.size, dtype=float)
    if delay == 0:
        result[:] = np.abs(signal)
    elif delay < signal.size:
        result[delay:] = np.abs(signal[:-delay])
    return result


def residual_direction_analysis(
    error: np.ndarray,
    signal: np.ndarray,
    branches: tuple[SplineMemoryBranch, ...],
    bins: int = 8,
) -> dict[str, object]:
    """Compare baseline residual means for rising and falling envelopes."""
    result = []
    for branch in branches:
        radius = delayed_radius(signal, branch.envelope_delay)
        previous = delayed_radius(signal, branch.envelope_delay + 1)
        delta = radius - previous
        edges = np.quantile(radius[WARMUP:], np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(edges)
        rows = []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (radius >= left) & (radius <= right)
            rising = mask & (delta > 0.0)
            falling = mask & (delta < 0.0)
            if np.count_nonzero(rising) < 10 or np.count_nonzero(falling) < 10:
                continue
            mean_rising = np.mean(error[rising])
            mean_falling = np.mean(error[falling])
            rows.append({
                "radius_left": float(left),
                "radius_right": float(right),
                "rising_count": int(np.count_nonzero(rising)),
                "falling_count": int(np.count_nonzero(falling)),
                "mean_error_rising_real": float(mean_rising.real),
                "mean_error_rising_imag": float(mean_rising.imag),
                "mean_error_falling_real": float(mean_falling.real),
                "mean_error_falling_imag": float(mean_falling.imag),
                "mean_difference_abs": float(abs(mean_rising - mean_falling)),
            })
        result.append({
            "signal_delay": branch.signal_delay,
            "envelope_delay": branch.envelope_delay,
            "bins": rows,
            "maximum_mean_difference_abs": max((row["mean_difference_abs"] for row in rows), default=0.0),
        })
    return {"branches": result}


def fit_one_branch_residual(
    calibration_input: np.ndarray,
    target: np.ndarray,
    *,
    knots: np.ndarray,
    branches: tuple[SplineMemoryBranch, ...],
    active_branch: int,
    deadband: float,
    ridge_baseline: float,
    ridge_residual: float,
) -> ResidualHysteresisSplineMemoryDPD:
    """Fit baseline plus one selected hysteresis branch only."""
    base, residual, _ = hysteresis_design_matrices(
        calibration_input, knots, branches, deadband
    )
    warmup = max(max(b.signal_delay, b.envelope_delay + 1) for b in branches)
    base = base[warmup:]
    residual = residual[warmup:]
    target = target[warmup:]
    knot_count = knots.size
    start = active_branch * knot_count
    stop = start + knot_count
    residual_active = residual[:, start:stop]
    feature_count = base.shape[1]
    joint = np.column_stack((base, residual_active)) / np.sqrt(float(target.size))
    regularizer = np.zeros((feature_count + knot_count, feature_count + knot_count), dtype=np.complex128)
    regularizer[:feature_count, :feature_count] = np.sqrt(ridge_baseline) * np.eye(feature_count)
    regularizer[feature_count:, feature_count:] = np.sqrt(ridge_residual) * np.eye(knot_count)
    augmented = np.vstack((joint, regularizer))
    augmented_target = np.concatenate((target / np.sqrt(float(target.size)), np.zeros(feature_count + knot_count, dtype=np.complex128)))
    solved = np.linalg.lstsq(augmented, augmented_target, rcond=None)[0]
    residual_coefficients = np.zeros((len(branches), knot_count), dtype=np.complex128)
    residual_coefficients[active_branch] = solved[feature_count:]
    return ResidualHysteresisSplineMemoryDPD(
        knots=knots,
        branches=branches,
        baseline_coefficients=solved[:feature_count].reshape(len(branches), knot_count),
        residual_coefficients=residual_coefficients,
        deadband=deadband,
    )


def main() -> None:
    train_x_raw = read_iq(SELECTION_DIR / "train_input.csv")
    train_y_raw = read_iq(SELECTION_DIR / "train_output.csv")
    val_x_raw = read_iq(SELECTION_DIR / "val_input.csv")
    val_y_raw = read_iq(SELECTION_DIR / "val_output.csv")
    scale = float(np.max(np.abs(train_x_raw)))
    train_x, train_y = train_x_raw / scale, train_y_raw / scale
    val_x, val_y = val_x_raw / scale, val_y_raw / scale
    gain = np.vdot(train_x, train_y) / np.vdot(train_x, train_x)

    pa_config = GMPConfig(
        ka=9, la=9, kb=3, lb=7, mb=3, kc=3, lc=7, mc=3,
        leading_policy="causal_leading",
    )
    started = time.perf_counter()
    pa_model, pa_fit = fit_gmp_pa(
        train_x, train_y, config=pa_config, ridge=1e-10,
        segment_length=train_x.size,
    )
    pa_fit_seconds = time.perf_counter() - started
    pa_val_hat = pa_model.predict(val_x)
    ideal_val = gain * val_x
    no_dpd_output = pa_val_hat
    no_dpd = pooled_metrics(no_dpd_output, ideal_val)

    ila_input = train_y / gain
    knots = make_knots(ila_input, KNOT_COUNT, "quantile")
    baseline, baseline_diag = fit_ila_sparse_spline_memory_dpd(
        train_x, train_y, gain,
        branches=BRANCHES,
        knots=knots,
        ridge=RIDGE_BASELINE,
    )
    baseline_drive = baseline.predict(val_x)
    baseline_output = pa_model.predict(baseline_drive)
    baseline_metrics = pooled_metrics(baseline_output, ideal_val)
    residual_analysis = residual_direction_analysis(
        baseline_output - ideal_val, val_x, BRANCHES
    )

    delta_values = []
    for branch in BRANCHES:
        current = np.abs(np.concatenate((np.zeros(branch.envelope_delay), val_x))[:val_x.size])
        previous = np.abs(np.concatenate((np.zeros(branch.envelope_delay + 1), val_x))[:val_x.size])
        delta_values.append(current - previous)
    delta_scale = float(np.median(np.abs(np.concatenate(delta_values))))
    alphas = (0.0, 0.1, 0.25, 0.5, 1.0)
    hysteresis_rows = []
    for alpha in alphas:
        for ridge_residual in RIDGE_RESIDUAL_VALUES:
            deadband = alpha * delta_scale
            started = time.perf_counter()
            model, diagnostics = fit_residual_hysteresis_spline_memory_dpd(
                ila_input, train_x,
                branches=BRANCHES,
                knots=knots,
                knot_count=KNOT_COUNT,
                deadband=deadband,
                ridge_baseline=RIDGE_BASELINE,
                ridge_residual=ridge_residual,
            )
            fit_seconds = time.perf_counter() - started
            drive = model.predict(val_x)
            output = pa_model.predict(drive)
            metrics = pooled_metrics(output, ideal_val)
            hysteresis_rows.append({
                "alpha": alpha,
                "ridge_residual": ridge_residual,
                "deadband": deadband,
                "metrics": metrics,
                "improvement_vs_no_dpd_db": float(10.0 * np.log10(no_dpd["mse"] / metrics["mse"])),
                "improvement_vs_baseline_db": float(10.0 * np.log10(baseline_metrics["mse"] / metrics["mse"])),
                "fit_seconds": fit_seconds,
                "up_fraction": diagnostics.up_fraction,
                "steady_fraction": diagnostics.steady_fraction,
                "down_fraction": diagnostics.down_fraction,
                "condition_number": diagnostics.augmented_condition_number,
                "residual_to_baseline_norm": diagnostics.residual_to_baseline_norm,
                "max_drive": float(np.max(np.abs(drive))),
                "segment_nmse_db": segment_nmse(output, ideal_val),
                "operation_count_precomputed_banks": model.operation_count(precomputed_banks=True).to_dict(),
                "operation_count_online_residual": model.operation_count(precomputed_banks=False).to_dict(),
            })

    sparse_rows = []
    for active_branch in range(len(BRANCHES)):
        for ridge_residual in (1e-2, 1e-1, 1.0):
            model = fit_one_branch_residual(
                ila_input, train_x, knots=knots, branches=BRANCHES,
                active_branch=active_branch, deadband=0.5 * delta_scale,
                ridge_baseline=RIDGE_BASELINE, ridge_residual=ridge_residual,
            )
            drive = model.predict(val_x)
            output = pa_model.predict(drive)
            metrics = pooled_metrics(output, ideal_val)
            sparse_rows.append({
                "active_branch": active_branch,
                "ridge_residual": ridge_residual,
                "metrics": metrics,
                "improvement_vs_baseline_db": float(10.0 * np.log10(baseline_metrics["mse"] / metrics["mse"])),
                "residual_to_baseline_norm": float(np.linalg.norm(model.residual_coefficients) / max(np.linalg.norm(model.baseline_coefficients), 1e-30)),
                "segment_nmse_db": segment_nmse(output, ideal_val),
            })

    best = min(hysteresis_rows, key=lambda row: row["metrics"]["nmse_db"])
    report = {
        "schema_version": 1,
        "dataset": "BlackBox capture; train 92000, validation 23000",
        "pa_evaluator": {
            "model": "causal GMP",
            "coefficient_count": pa_config.coefficient_count,
            "fit_seconds": pa_fit_seconds,
            "validation_nmse_db": float(nmse_pooled_db(pa_val_hat[WARMUP:], val_y[WARMUP:])),
        },
        "baseline": {
            "branches": [[b.signal_delay, b.envelope_delay] for b in BRANCHES],
            "knot_count": KNOT_COUNT,
            "metrics": baseline_metrics,
            "fit_diagnostics": baseline_diag.__dict__,
        },
        "no_dpd": no_dpd,
        "delta_scale_median": delta_scale,
        "baseline_residual_direction_analysis": residual_analysis,
        "hysteresis": hysteresis_rows,
        "sparse_one_branch_hysteresis": sparse_rows,
        "best_by_validation_nmse": best,
        "status": "surrogate-only; validation used for alpha selection; no physical PA measurement",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({
        "no_dpd_nmse_db": no_dpd["nmse_db"],
        "baseline_nmse_db": baseline_metrics["nmse_db"],
        "best_hysteresis_alpha": best["alpha"],
        "best_hysteresis_nmse_db": best["metrics"]["nmse_db"],
        "best_gain_vs_baseline_db": best["improvement_vs_baseline_db"],
        "best_sparse_gain_vs_baseline_db": max(row["improvement_vs_baseline_db"] for row in sparse_rows),
        "best_residual_ratio": best["residual_to_baseline_norm"],
        "output": str(OUTPUT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
