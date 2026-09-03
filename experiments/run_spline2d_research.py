"""Spline2D DPD research runner: augment the ILA spline body with bilinear
2D grids, evaluate the cascade through both frozen GMP evaluators with the
campaign block protocol (fit/advisor/selection disjoint), rank by
worst-case selection NMSE, and save the winner package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA  # noqa: E402
from baseline.spline2d_memory_dpd import (  # noqa: E402
    Grid2DBranch,
    Spline2DMemoryDPD,
    fit_spline2d_memory_dpd,
)
from baseline.spline_memory_dpd import SparseSplineMemoryDPD  # noqa: E402
from baseline.train_spline import load_split_pair  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402


def cascade_nmse(pa, drive, desired, gain, warmup) -> float:
    return float(nmse_db(pa.predict(drive), gain * desired, warmup))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    dataset = Path(cfg["dataset"]).resolve()
    pa_a = GeneralizedMemoryPolynomialPA.load(
        (ROOT / cfg["primary_pa_npz"]).resolve()
    )
    pa_b = GeneralizedMemoryPolynomialPA.load(
        (ROOT / cfg["secondary_pa_npz"]).resolve()
    )
    base_spline = SparseSplineMemoryDPD.load(
        (ROOT / cfg["base_spline_npz"]).resolve()
    )

    train_x, train_y = load_split_pair(dataset, "train")
    train_x = np.asarray(train_x).reshape(-1)
    train_y = np.asarray(train_y).reshape(-1)

    fit_lo, fit_hi = cfg["fit_block"]
    adv_lo, adv_hi = cfg["advisor_block"]
    sel_lo, sel_hi = cfg["selection_block"]
    fit_x, fit_y = train_x[fit_lo:fit_hi], train_y[fit_lo:fit_hi]
    adv_x = train_x[adv_lo:adv_hi]
    sel_x, sel_y = train_x[sel_lo:sel_hi], train_y[sel_lo:sel_hi]

    gain = complex(
        np.vdot(fit_x, pa_a.predict(fit_x)) / np.vdot(fit_x, fit_x)
    )
    warm = max(
        base_spline.maximum_delay,
        pa_a.config.causal_warmup_samples,
        pa_b.config.causal_warmup_samples,
        16,
    )

    def score(model) -> dict[str, float]:
        drive_adv = model.predict(adv_x)
        drive_sel = model.predict(sel_x)
        a_adv = cascade_nmse(pa_a, drive_adv, adv_x, gain, warm)
        a_sel = cascade_nmse(pa_a, drive_sel, sel_x, gain, warm)
        b_adv = cascade_nmse(pa_b, drive_adv, adv_x, gain, warm)
        b_sel = cascade_nmse(pa_b, drive_sel, sel_x, gain, warm)
        return {
            "advisor_primary": a_adv,
            "advisor_secondary": b_adv,
            "advisor_worst_case": max(a_adv, b_adv),
            "selection_primary": a_sel,
            "selection_secondary": b_sel,
            "selection_worst_case": max(a_sel, b_sel),
        }

    out_dir = (ROOT / cfg["output_dir"]).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline: the frozen ILA body alone under the same protocol.
    results: dict[str, object] = {
        "baseline_ila": score(base_spline),
        "gain": {"real": gain.real, "imag": gain.imag},
        "warmup": warm,
        "protocol": {
            "fit_block": [fit_lo, fit_hi],
            "advisor_block": [adv_lo, adv_hi],
            "selection_block": [sel_lo, sel_hi],
        },
    }
    print(
        f"baseline_ila: selection worst-case "
        f"{results['baseline_ila']['selection_worst_case']:.3f}",
        flush=True,
    )

    trials = []
    for variant in cfg["variants"]:
        grids = [
            Grid2DBranch(
                signal_delay=int(g[0]),
                envelope_delay_0=int(g[1]),
                envelope_delay_1=int(g[2]),
                knot_count=int(g[3]),
            )
            for g in variant["grids"]
        ]
        model, diagnostics = fit_spline2d_memory_dpd(
            fit_y / gain,
            fit_x,
            body=base_spline,
            grid_branches=grids,
            ridge=float(cfg.get("ridge", 1e-7)),
            refit_body=variant["mode"] == "joint",
        )
        scores = score(model)
        trials.append(
            {
                "name": variant["name"],
                "mode": variant["mode"],
                "diagnostics": diagnostics,
                "scores": scores,
            }
        )
        print(
            f"{variant['name']}: advisor WC {scores['advisor_worst_case']:.3f} | "
            f"selection WC {scores['selection_worst_case']:.3f}",
            flush=True,
        )

    winner = min(trials, key=lambda t: t["scores"]["selection_worst_case"])
    if winner["scores"]["selection_worst_case"] < results["baseline_ila"][
        "selection_worst_case"
    ]:
        best = min(
            trials,
            key=lambda t: t["scores"]["selection_worst_case"],
        )
        grids = [
            Grid2DBranch(
                signal_delay=int(g[0]),
                envelope_delay_0=int(g[1]),
                envelope_delay_1=int(g[2]),
                knot_count=int(g[3]),
            )
            for g in next(v for v in cfg["variants"] if v["name"] == best["name"])[
                "grids"
            ]
        ]
        model, _ = fit_spline2d_memory_dpd(
            fit_y / gain,
            fit_x,
            body=base_spline,
            grid_branches=grids,
            ridge=float(cfg.get("ridge", 1e-7)),
            refit_body=best["mode"] == "joint",
        )
        model.save(out_dir / "spline2d_dpd.npz")
        results["winner"] = best["name"]
    else:
        results["winner"] = "baseline_ila"
    results["trials"] = trials

    (out_dir / "spline2d_research_report.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"WINNER: {results['winner']}", flush=True)


if __name__ == "__main__":
    main()
