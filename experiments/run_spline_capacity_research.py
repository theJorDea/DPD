"""Spline DPD capacity research: larger families/knots, cross-evaluator gate.

Fits ILA spline-memory DPD candidates on measured train data only (no
surrogate in the fit), with fit/advisor/selection blocks disjoint, and
ranks every candidate by the worst-case cascade NMSE across the two
frozen GMP evaluators.  Validation is a read-only diagnostic.  This is
the last untried local lever: increasing DPD capacity instead of adding
GMP members on top of the frozen spline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.alignment import align_and_estimate_gain
from baseline.direct_learning import nmse_db
from baseline.gmp_pa import GeneralizedMemoryPolynomialPA, gmp_terms
from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    fit_ila_sparse_spline_memory_dpd,
)
from baseline.train_spline import (
    align_split_pair,
    gain_from_training,
    load_split_pair,
    write_json,
)


def _parse_branches(spec: list[list[int]]) -> tuple[tuple[SplineMemoryBranch, ...], ...]:
    return tuple(
        tuple(SplineMemoryBranch(int(delay), int(envelope)) for delay, envelope in family)
        for family in spec
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("task") != "spline_capacity_research":
        raise ValueError("unexpected task")

    dataset = PROJECT_ROOT / config["dataset"]
    train_input, train_output = load_split_pair(dataset, "train")
    val_input, val_output = load_split_pair(dataset, "val")

    _, _, delay, _ = align_and_estimate_gain(
        train_input, train_output, max_abs_delay=32
    )
    train_input, train_output = align_split_pair(
        train_input, train_output, delay=delay
    )
    val_input, _ = align_split_pair(val_input, val_output, delay=delay)
    gain, _ = gain_from_training(train_input, train_output, strategy="complex_ls")

    pa = GeneralizedMemoryPolynomialPA.load(
        PROJECT_ROOT / config["primary_pa_npz"]
    )
    pa_b = GeneralizedMemoryPolynomialPA.load(
        PROJECT_ROOT / config["secondary_pa_npz"]
    )
    gain_eval = complex(
        np.vdot(train_input, pa.predict(train_input))
        / np.vdot(train_input, train_input)
    )
    gain_eval_b = complex(
        np.vdot(train_input, pa_b.predict(train_input))
        / np.vdot(train_input, train_input)
    )
    warmup = max(
        int(max(term.signal_delay for term in gmp_terms(pa.config))) + 1,
        int(max(term.signal_delay for term in gmp_terms(pa_b.config))) + 1,
    )

    fit_block = slice(*[int(v) for v in config["fit_block"]])
    advisor_block = slice(*[int(v) for v in config["advisor_block"]])
    selection_block = slice(*[int(v) for v in config["selection_block"]])
    advisor_desired = train_input[advisor_block]
    selection_desired = train_input[selection_block]

    def cascade_scores(model: SparseSplineMemoryDPD, desired: np.ndarray):
        drive = model.predict(desired)
        out_a = np.asarray(pa.predict(drive), dtype=np.complex128)
        out_b = np.asarray(pa_b.predict(drive), dtype=np.complex128)
        return (
            nmse_db(out_a, gain_eval * desired, warmup),
            nmse_db(out_b, gain_eval_b * desired, warmup),
        )

    families = _parse_branches(config["branch_families"])
    strategy_spec = config.get("knot_strategy", "quantile")
    strategies = (
        [str(strategy_spec)]
        if isinstance(strategy_spec, str)
        else [str(value) for value in strategy_spec]
    )
    trials: list[dict[str, Any]] = []
    for family_index, branches in enumerate(families):
        family_name = "_".join(f"{b.signal_delay}{b.envelope_delay}" for b in branches)
        for knot_count in config["knot_counts"]:
            for ridge in config["ridges"]:
                for knot_strategy in strategies:
                    fit_started = time.perf_counter()
                    model, diagnostics = fit_ila_sparse_spline_memory_dpd(
                        train_input[fit_block],
                        train_output[fit_block],
                        gain,
                        branches=branches,
                        knot_count=int(knot_count),
                        knot_strategy=knot_strategy,
                        ridge=float(ridge),
                    )
                    advisor_a, advisor_b = cascade_scores(model, advisor_desired)
                    valid = bool(
                        diagnostics.solver_rank == diagnostics.feature_count
                        and np.isfinite(advisor_a)
                        and np.isfinite(advisor_b)
                    )
                    trials.append(
                        {
                            "family": family_name,
                            "branches": [
                                [branch.signal_delay, branch.envelope_delay]
                                for branch in branches
                            ],
                            "knot_count": int(knot_count),
                            "ridge": float(ridge),
                            "knot_strategy": knot_strategy,
                            "advisor_nmse_db": [advisor_a, advisor_b],
                            "worst_case_advisor_nmse_db": max(advisor_a, advisor_b),
                            "valid_for_selection": valid,
                            "fit_seconds": time.perf_counter() - fit_started,
                            "feature_count": int(diagnostics.feature_count),
                            "solver_rank": int(diagnostics.solver_rank),
                        }
                    )
                    print(
                        f"{family_name} K={knot_count} ridge={ridge:g} "
                        f"{knot_strategy}: worst-case advisor {max(advisor_a, advisor_b):.4f} dB"
                    )

    valid_trials = [trial for trial in trials if trial["valid_for_selection"]]
    if not valid_trials:
        raise RuntimeError("no valid candidates")
    winner = min(valid_trials, key=lambda trial: trial["worst_case_advisor_nmse_db"])

    # Refit the winner on the full fit block and score the selection block.
    winner_branches = tuple(
        SplineMemoryBranch(delay, envelope) for delay, envelope in winner["branches"]
    )
    winner_model, winner_diagnostics = fit_ila_sparse_spline_memory_dpd(
        train_input[fit_block],
        train_output[fit_block],
        gain,
        branches=winner_branches,
        knot_count=int(winner["knot_count"]),
        knot_strategy=winner.get("knot_strategy", "quantile"),
        ridge=float(winner["ridge"]),
    )
    selection_a, selection_b = cascade_scores(winner_model, selection_desired)
    val_scores = cascade_scores(winner_model, val_input)

    output_dir = PROJECT_ROOT / config["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)
    model_path = output_dir / "capacity_research_dpd.npz"
    winner_model.save(model_path)
    report = {
        "schema_version": 1,
        "task": "spline_capacity_research",
        "winner": winner,
        "winner_selection_nmse_db": [selection_a, selection_b],
        "winner_worst_case_selection_nmse_db": max(selection_a, selection_b),
        "winner_validation_nmse_db": list(val_scores),
        "baseline_reference_nmse_db": config.get("baseline_reference_nmse_db"),
        "trial_count": len(trials),
        "trials": trials,
        "blocks": {
            "fit": config["fit_block"],
            "advisor": config["advisor_block"],
            "selection": config["selection_block"],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "selection_policy": "train-only worst-case across evaluators A and B",
    }
    write_json(output_dir / "capacity_research_report.json", report)
    print(
        "WINNER:",
        winner["family"],
        "K=",
        winner["knot_count"],
        "ridge=",
        winner["ridge"],
        f"selection worst-case {max(selection_a, selection_b):.4f} dB",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
