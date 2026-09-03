"""Round-8 probes: spline-forward third judge, tail-knot DPD, peak-weighted judge.

1. Spline-forward: fit the DPD's own spline class FORWARD (x -> y). Its
   fidelity is a direct measurement of the DPD-class capacity ceiling;
   as an evaluator it extrapolates linearly instead of exploding.
2. Tail knots: hybrid knot vector (16 quantile + 8 uniform in the top
   amplitude decile) for the ILA DPD, quantile-only baseline for
   comparison, same protocol.
3. Peak-weighted judge: refit judge A's GMP with weights
   w = 1 + alpha * 1[|x| > q95] and compare binwise fidelity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.gmp_pa import (  # noqa: E402
    GeneralizedMemoryPolynomialPA,
    gmp_terms,
)
from baseline.spline_memory_dpd import (  # noqa: E402
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    fit_sparse_spline_memory_dpd,
    spline_memory_design_matrix,
)
from baseline.train_spline import load_split_pair  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402

WARM = 64


def hybrid_knots(x: np.ndarray, quantile_count: int = 16, tail_count: int = 8) -> np.ndarray:
    amp = np.abs(x)
    base = np.quantile(amp, np.linspace(0.0, 1.0, quantile_count + 1))
    base = base[:-1]  # keep 16 quantile knots below the tail
    top_start = np.quantile(amp, 0.90)
    # Tail knots live INSIDE the calibration range; the last knot must
    # sit exactly on the max (stretching it past max ruins peak fits).
    tail = np.linspace(top_start, float(amp.max()), tail_count + 1)[1:]
    knots = np.unique(np.concatenate([base, tail]))
    knots[0] = 0.0
    return knots


def quantile_knots(x: np.ndarray, count: int = 24) -> np.ndarray:
    amp = np.abs(x)
    knots = np.quantile(amp, np.linspace(0.0, 1.0, count))
    knots[0] = 0.0
    knots[-1] = float(amp.max())
    return np.unique(knots)


def ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    norm = np.linalg.norm(design, axis=0)
    norm[norm == 0] = 1.0
    dn = design / norm
    n = dn.shape[0]
    gram = dn.conj().T @ dn / n + ridge * np.eye(dn.shape[1])
    rhs = dn.conj().T @ target / n
    return np.linalg.solve(gram, rhs) / norm


def gmp_columns(signal: np.ndarray, config) -> np.ndarray:
    n = signal.size
    max_exp = max(config.ka - 1, config.kb, config.kc)
    powers = {1: np.abs(signal), 2: np.abs(signal) ** 2}
    for exponent in range(3, max_exp + 1):
        powers[exponent] = powers[exponent - 2] * powers[2]

    def shift(v: np.ndarray, d: int) -> np.ndarray:
        out = np.zeros_like(v)
        if d == 0:
            out[:] = v
        elif d > 0:
            out[d:] = v[: n - d]
        else:
            out[: n + d] = v[-d:]
        return out

    columns = np.empty((n, len(gmp_terms(config))), dtype=np.complex128)
    for index, term in enumerate(gmp_terms(config)):
        if term.exponent == 0:
            columns[:, index] = shift(signal, term.signal_delay)
        else:
            columns[:, index] = shift(signal, term.signal_delay) * shift(
                powers[term.exponent], term.envelope_delay
            )
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", required=True)
    parser.add_argument("--fit-block", required=True, help="lo,hi")
    parser.add_argument("--selection-block", required=True, help="lo,hi")
    parser.add_argument("--branches", required=True, help="m,d;m,d;...")
    parser.add_argument("--knot-count", type=int, default=24)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pa_a = GeneralizedMemoryPolynomialPA.load(Path(args.pa).resolve())
    pa_b = GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())
    train_x, train_y = load_split_pair(Path(args.dataset).resolve(), "train")
    val_x, val_y = load_split_pair(Path(args.dataset).resolve(), "val")
    train_x = np.asarray(train_x).reshape(-1)
    train_y = np.asarray(train_y).reshape(-1)
    val_x = np.asarray(val_x).reshape(-1)
    val_y = np.asarray(val_y).reshape(-1)
    fit_lo, fit_hi = (int(v) for v in args.fit_block.split(","))
    sel_lo, sel_hi = (int(v) for v in args.selection_block.split(","))
    fit_x, fit_y = train_x[fit_lo:fit_hi], train_y[fit_lo:fit_hi]
    sel_x = train_x[sel_lo:sel_hi]

    warm = max(pa_a.config.causal_warmup_samples, WARM)
    gain = complex(
        np.vdot(fit_x, pa_a.predict(fit_x)) / np.vdot(fit_x, fit_x)
    )
    branches = [
        SplineMemoryBranch(int(pair.split(",")[0]), int(pair.split(",")[1]))
        for pair in args.branches.split(";")
    ]

    report: dict[str, object] = {"label": args.label, "gain": [gain.real, gain.imag]}

    # --- 1. Spline-forward third judge ---
    forward, fwd_diag = fit_sparse_spline_memory_dpd(
        fit_x,
        fit_y,
        branches=branches,
        knot_count=args.knot_count,
        knot_strategy="quantile",
        ridge=1e-8,
    )
    forward_fidelity_val = float(nmse_db(forward.predict(val_x), val_y, warm))
    forward_fidelity_train = float(nmse_db(forward.predict(fit_x), fit_y, warm))
    report["spline_forward"] = {
        "val_nmse_db": forward_fidelity_val,
        "train_nmse_db": forward_fidelity_train,
    }
    print(
        f"spline-forward: train {forward_fidelity_train:.2f} | val {forward_fidelity_val:.2f}",
        flush=True,
    )

    # --- 2. Tail-knot DPD vs quantile-knot DPD (ILA, same branches) ---
    knot_variants = {
        "quantile24": quantile_knots(fit_y / gain, args.knot_count),
        "hybrid16_8tail": hybrid_knots(np.abs(fit_y / gain)),
    }
    report["knot_variants"] = {}
    for name, knots in knot_variants.items():
        for ridge in (1e-8, 1e-7):
            model, fit_diag = fit_sparse_spline_memory_dpd(
                fit_y / gain,
                fit_x,
                branches=branches,
                knots=knots,
                ridge=ridge,
            )
            drive = model.predict(sel_x)
            n_a = float(nmse_db(pa_a.predict(drive), gain * sel_x, warm))
            n_b = float(nmse_db(pa_b.predict(drive), gain * sel_x, warm))
            report["knot_variants"][f"{name}_ridge{ridge:g}"] = {
                "selection_a": n_a,
                "selection_b": n_b,
                "worst_case": max(n_a, n_b),
                "train_nmse_db": fit_diag.training_nmse_db_after_warmup,
            }
            print(
                f"{name} ridge {ridge:g}: sel A {n_a:.3f} | B {n_b:.3f} | "
                f"WC {max(n_a, n_b):.3f}",
                flush=True,
            )
            del model

    # --- 3. Peak-weighted judge refit (fidelity metrics against measured y) ---
    q95 = float(np.quantile(np.abs(fit_x), 0.95))
    columns = gmp_columns(fit_x, pa_a.config)
    target = fit_y
    val_columns = gmp_columns(val_x, pa_a.config)
    base_val = float(nmse_db(pa_a.predict(val_x), val_y, warm))
    report["peak_weighted_judge"] = {"unweighted_val_nmse_db": base_val}
    weight_mask = np.abs(fit_x) > q95
    bin_mask = np.abs(val_x) > float(np.quantile(np.abs(val_x), 0.95))
    for alpha in (0.0, 1.0, 3.0, 10.0):
        weights = 1.0 + alpha * weight_mask
        sqrt_w = np.sqrt(weights)
        norm = np.linalg.norm(columns * sqrt_w[:, None], axis=0)
        norm[norm == 0] = 1.0
        dw = columns * sqrt_w[:, None] / norm
        gram = dw.conj().T @ dw + 1e-7 * np.eye(dw.shape[1])
        rhs = dw.conj().T @ (sqrt_w * target)
        coefficients = np.linalg.solve(gram, rhs) / norm
        val_pred = val_columns @ coefficients
        overall = float(nmse_db(val_pred, val_y, warm))
        top = float(
            10
            * np.log10(
                np.mean(np.abs((val_pred - val_y)[bin_mask]) ** 2)
                / np.mean(np.abs((val_y)[bin_mask]) ** 2)
            )
        )
        base_top = float(
            10
            * np.log10(
                np.mean(
                    np.abs((pa_a.predict(val_x) - val_y)[bin_mask]) ** 2
                )
                / np.mean(np.abs((val_y)[bin_mask]) ** 2)
            )
        )
        report["peak_weighted_judge"][f"alpha_{alpha:g}"] = {
            "val_nmse_db": overall,
            "top_bin_nmse_db": top,
            "baseline_top_bin_nmse_db": base_top,
        }
        print(
            f"peak-weighted alpha {alpha:g}: val {overall:.3f} | "
            f"top bin {top:.3f} (base {base_top:.3f})",
            flush=True,
        )

    Path(args.output).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
