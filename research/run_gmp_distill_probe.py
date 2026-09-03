"""Diagnostic: distill the oracle drive into a LARGE GMP dictionary.

The spline distillation topped out at drive fidelity -29.6 dB -> cascade
-31.2.  This script asks whether the limit is the spline class or the
parameteric GMP class as a whole (outside the MUL budget, diagnostic
only): ridge LS on (x -> u*) over a dense phase-equivariant member grid,
then cascade scoring on the disjoint selection block.
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
from baseline.gmp_dictionary_dpd import (  # noqa: E402
    GmpMember,
    gmp_dictionary_columns,
)
from baseline.train_spline import load_split_pair  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", required=True)
    parser.add_argument("--oracle-drive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-m", type=int, default=12)
    parser.add_argument("--max-d", type=int, default=12)
    parser.add_argument("--max-k", type=int, default=8)
    args = parser.parse_args()

    pa_a = GeneralizedMemoryPolynomialPA.load(Path(args.pa).resolve())
    pa_b = GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())
    train_x, _ = load_split_pair(Path(args.dataset).resolve(), "train")
    train_x = np.asarray(train_x).reshape(-1)
    with np.load(args.oracle_drive) as data:
        drive = data["oracle_drive"]
        gain = complex(data["gain"][0], data["gain"][1])
    fit_x = train_x[: drive.size]
    warm = max(pa_a.config.causal_warmup_samples, 16)

    members = tuple(
        GmpMember(m, d, k)
        for k in range(1, args.max_k + 1)
        for d in range(args.max_d)
        for m in range(args.max_m)
    )
    columns = gmp_dictionary_columns(fit_x, members)
    print(f"design: {columns.shape}", flush=True)

    # Conditioning fix: unit-norm columns, ridge sweep on the normalized
    # Gram, SVD-based solve in float64.
    col_norms = np.linalg.norm(columns, axis=0)
    col_norms[col_norms == 0] = 1.0
    design_n = columns / col_norms
    gram_n = design_n.conj().T @ design_n / design_n.shape[0]
    report: dict[str, object] = {
        "members": len(members),
        "gram_condition_number": float(np.linalg.cond(gram_n)),
    }
    for ridge in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
        gram_r = gram_n + ridge * np.eye(gram_n.shape[0], dtype=np.complex128)
        rhs = design_n.conj().T @ drive / design_n.shape[0]
        coefficients_unit = np.linalg.solve(gram_r, rhs)
        coefficients = coefficients_unit / col_norms
        prediction = columns @ coefficients
        fidelity = float(nmse_db(prediction, drive, warm))
        sel_x = train_x[16384:23040]
        sel_pred = gmp_dictionary_columns(sel_x, members) @ coefficients
        s_a = float(nmse_db(pa_a.predict(sel_pred), gain * sel_x, warm))
        s_b = float(nmse_db(pa_b.predict(sel_pred), gain * sel_x, warm))
        report[f"ridge_{ridge:g}"] = {
            "drive_fidelity_nmse_db": fidelity,
            "selection_cascade_a_db": s_a,
            "selection_cascade_b_db": s_b,
            "selection_worst_case_db": max(s_a, s_b),
        }
        print(
            f"ridge {ridge:g}: fidelity {fidelity:.2f} | sel A {s_a:.3f} "
            f"B {s_b:.3f} WC {max(s_a, s_b):.3f}",
            flush=True,
        )
    Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
