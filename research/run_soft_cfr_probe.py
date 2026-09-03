"""Soft CFR probe: clip the DPD drive peaks and re-score the cascade.

DPA amplitude-bin attribution says 91% of judge divergence sits in the
top amplitude bins; if judge A mis-models peaks, a soft roll-off of the
drive can trade tiny in-band EVM for a large reduction of that error
zone.  Grid over cap levels, worst-case A/B scoring, report only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA  # noqa: E402
from baseline.spline_memory_dpd import SparseSplineMemoryDPD  # noqa: E402
from baseline.train_spline import load_split_pair  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402


def soft_cfr(u: np.ndarray, cap: float, width: float = 0.02) -> np.ndarray:
    """Cosine roll-off above cap: |u|<=cap(1-width) unchanged, >=cap hard."""
    mag = np.abs(u)
    phase = np.exp(1j * np.angle(u))
    upper = cap
    lower = cap * (1.0 - width)
    out = mag.copy()
    zone = (mag > lower) & (mag < upper)
    out[zone] = lower + (upper - lower) * np.sin(
        0.5 * np.pi * (mag[zone] - lower) / (upper - lower)
    )
    out[mag >= upper] = upper
    return out * phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", required=True)
    parser.add_argument("--dpd", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pa_a = GeneralizedMemoryPolynomialPA.load(Path(args.pa).resolve())
    pa_b = GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())
    dpd = SparseSplineMemoryDPD.load(Path(args.dpd).resolve())
    train_x, train_y = load_split_pair(Path(args.dataset).resolve(), "train")
    val_x, val_y = load_split_pair(Path(args.dataset).resolve(), "val")
    train_x = np.asarray(train_x).reshape(-1)
    val_x = np.asarray(val_x).reshape(-1)
    val_y = np.asarray(val_y).reshape(-1)

    warm = max(pa_a.config.causal_warmup_samples, 16)
    gain = complex(
        np.vdot(train_x, pa_a.predict(train_x)) / np.vdot(train_x, train_x)
    )
    u = dpd.predict(val_x)
    cap_ref = float(np.max(np.abs(u)))

    # Cascade target is the ideal PA output for the INPUT signal.
    ideal = gain * val_x
    report: dict[str, object] = {"label": args.label, "cap_ref": cap_ref}
    for cap_frac in (1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85):
        driven = (
            u
            if cap_frac >= 1.0
            else soft_cfr(u, cap_ref * cap_frac)
        )
        n_a = float(nmse_db(pa_a.predict(driven), ideal, warm))
        n_b = float(nmse_db(pa_b.predict(driven), ideal, warm))
        report[f"cap_{cap_frac:.2f}"] = {
            "nmse_a": n_a,
            "nmse_b": n_b,
            "worst_case": max(n_a, n_b),
        }
        print(
            f"cap {cap_frac:.2f}: A {n_a:.3f} | B {n_b:.3f} | WC {max(n_a, n_b):.3f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
