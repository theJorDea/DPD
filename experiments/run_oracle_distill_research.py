"""Distill the oracle drive into a spline DPD and score the cascade.

The oracle experiment showed that on DPA a drive optimized through judge
A with a soft support penalty is simultaneously good for both judges
(A -34.8 / B -37.0 on the validation block) - 3.8 dB above the current
final cascade.  This runner distills that drive into deployable spline
DPD models by plain least squares on (fit_x -> u*) and scores the
cascade through both frozen judges on disjoint advisor/selection blocks.
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
from baseline.spline_memory_dpd import (  # noqa: E402
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    fit_sparse_spline_memory_dpd,
)
from baseline.train_spline import load_split_pair  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402


def cascade_score(model, pa_a, pa_b, x_block, gain, warm) -> dict[str, float]:
    drive = model.predict(x_block)
    n_a = float(nmse_db(pa_a.predict(drive), gain * x_block, warm))
    n_b = float(nmse_db(pa_b.predict(drive), gain * x_block, warm))
    return {"a": n_a, "b": n_b, "worst_case": max(n_a, n_b)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", required=True)
    parser.add_argument("--oracle-drive", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--fit-subblock", required=True, help="lo,hi indices of the oracle block inside its source array")
    parser.add_argument("--advisor-subblock", required=True, help="lo,hi inside train")
    parser.add_argument("--selection-subblock", required=True, help="lo,hi inside train")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pa_a = GeneralizedMemoryPolynomialPA.load(Path(args.pa).resolve())
    pa_b = GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())
    dataset = Path(args.dataset).resolve()
    train_x, _ = load_split_pair(dataset, "train")
    train_x = np.asarray(train_x).reshape(-1)
    with np.load(args.oracle_drive) as data:
        drive = data["oracle_drive"]
        block = str(data["block"])

    warm = max(pa_a.config.causal_warmup_samples, 16)
    gain = complex(
        np.vdot(train_x, pa_a.predict(train_x)) / np.vdot(train_x, train_x)
    )

    # Drive length must match the requested block (train or val).
    val_x, _ = load_split_pair(dataset, "val")
    val_x = np.asarray(val_x).reshape(-1)
    if block == "fit":
        source_x = train_x
    else:
        source_x = val_x
    fit_lo, fit_hi = (int(v) for v in args.fit_subblock.split(","))
    if source_x[fit_lo:fit_hi].size != drive.size:
        raise ValueError(
            f"oracle drive {drive.size} != source_x[{fit_lo}:{fit_hi}] length "
            f"{source_x[fit_lo:fit_hi].size}"
        )

    adv_lo, adv_hi = (int(v) for v in args.advisor_subblock.split(","))
    sel_lo, sel_hi = (int(v) for v in args.selection_subblock.split(","))
    fit_x = source_x[fit_lo:fit_hi]
    advisor_x = train_x[adv_lo:adv_hi]
    selection_x = train_x[sel_lo:sel_hi]

    families = {
        "signal12_env2x24": (
            [SplineMemoryBranch(m, 0) for m in range(12)]
            + [SplineMemoryBranch(m, 2) for m in range(1, 3)],
            24,
        ),
        "signal10_env1_env2x24": (
            [SplineMemoryBranch(m, 0) for m in range(10)]
            + [SplineMemoryBranch(m, 1) for m in range(1, 3)]
            + [SplineMemoryBranch(m, 2) for m in range(1, 3)],
            24,
        ),
        "signal14_env1x24": (
            [SplineMemoryBranch(m, 0) for m in range(14)]
            + [SplineMemoryBranch(m, 1) for m in range(1, 3)],
            24,
        ),
    }

    report: dict[str, object] = {
        "block": block,
        "fit_samples": int(fit_x.size),
        "blocks": {
            "fit": [fit_lo, fit_hi],
            "advisor": [adv_lo, adv_hi],
            "selection": [sel_lo, sel_hi],
        },
        "gain": [gain.real, gain.imag],
    }
    trials = []
    for name, (branches, knots) in families.items():
        model, diagnostics = fit_sparse_spline_memory_dpd(
            fit_x,
            drive,
            branches=branches,
            knot_count=knots,
            knot_strategy="quantile",
            ridge=1e-9,
        )
        coefficients = int(model.coefficients.size)
        mul = int(
            sum(
                2 if branch.envelope_delay else 1
                for branch in model.branches
            )
        )
        advisor = cascade_score(model, pa_a, pa_b, advisor_x, gain, warm)
        selection = cascade_score(model, pa_a, pa_b, selection_x, gain, warm)
        drive_fidelity = float(
            nmse_db(model.predict(fit_x), drive, warm)
        )
        trials.append(
            {
                "name": name,
                "coefficients": coefficients,
                "mul_per_sample": mul,
                "drive_fidelity_nmse_db": drive_fidelity,
                "advisor": advisor,
                "selection": selection,
            }
        )
        print(
            f"{name}: coef {coefficients} | drive fit {drive_fidelity:.2f} | "
            f"advisor WC {advisor['worst_case']:.3f} | "
            f"selection WC {selection['worst_case']:.3f}",
            flush=True,
        )

    winner = min(trials, key=lambda t: t["selection"]["worst_case"])
    best_model, _ = fit_sparse_spline_memory_dpd(
        fit_x,
        drive,
        branches=families[winner["name"]][0],
        knot_count=families[winner["name"]][1],
        knot_strategy="quantile",
        ridge=1e-9,
    )
    best_model.save(out_dir / "oracle_distilled_dpd.npz")
    report["trials"] = trials
    report["winner"] = winner["name"]
    (out_dir / "oracle_distill_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("WINNER", winner["name"], flush=True)


if __name__ == "__main__":
    main()
