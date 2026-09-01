"""Research runner: composite spline + sparse GMP-class DPD.

Stage 3 of the minus-50 plan.  Starting from the frozen spline-memory DPD
and a frozen GMP PA evaluator, this runner

1. selects phase-equivariant GMP-class residual members by greedy
   orthogonal matching pursuit on a train fit block (linearized inverse
   of the cascade residual);
2. fits member coefficients with complex ridge least squares for every
   (member budget, ridge) candidate and scores the full cascade NMSE on a
   disjoint train advisor block (support-bounded);
3. polishes the winning composite with bounded Gauss-Newton direct
   updates through the evaluator on further disjoint train slices;
4. performs the final family selection (baseline / composite /
   composite+GN) on a held-out train block disjoint from everything
   above; the validation split is evaluated once per candidate as a
   read-only diagnostic; the test split is never opened.

Deployed-cost gate: every candidate must stay within the 1000
real-multiplications-per-sample budget under the repository accounting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from baseline.direct_learning import (
    DirectLearningConfig,
    iterative_direct_schedule_core,
    nmse_db,
    signal_summary,
)
from baseline.gmp_dictionary_dpd import (
    CompositeSplineGmpDPD,
    GmpDictionaryGrid,
    GmpMember,
    composite_design_matrix,
    fit_gmp_residual_members,
)
from baseline.gmp_pa import GeneralizedMemoryPolynomialPA, gmp_terms
from baseline.spline_memory_dpd import SparseSplineMemoryDPD
from baseline.train_spline import (
    file_sha256,
    load_split_pair,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("config must be one JSON object with schema_version 1")
    if value.get("task") != "composite_spline_gmp_dpd_research":
        raise ValueError("unexpected config task")
    required = {
        "dataset",
        "baseline_dpd_npz",
        "pa_model_npz",
        "output_dir",
        "grid",
        "member_budgets",
        "ridge_values",
        "fit_block",
        "advisor_block",
        "gauss_newton_schedule",
        "selection_block",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    return value


def _pa_warmup(pa: GeneralizedMemoryPolynomialPA) -> int:
    return int(max(term.signal_delay for term in gmp_terms(pa.config)) + 1)


def _cascade_nmse(
    pa: GeneralizedMemoryPolynomialPA,
    drive: np.ndarray,
    desired: np.ndarray,
    gain: complex,
    warmup: int,
) -> float:
    output = np.asarray(pa.predict(drive), dtype=np.complex128)
    return nmse_db(output, gain * desired, warmup)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = _read_config(args.config)

    dataset = PROJECT_ROOT / config["dataset"]
    train_input, _ = load_split_pair(dataset, "train")
    validation_input, _ = load_split_pair(dataset, "val")

    baseline_dpd = SparseSplineMemoryDPD.load(
        PROJECT_ROOT / config["baseline_dpd_npz"]
    )
    pa = GeneralizedMemoryPolynomialPA.load(
        PROJECT_ROOT / config["pa_model_npz"]
    )
    warmup = max(
        int(config.get("minimum_warmup", 0)),
        baseline_dpd.maximum_delay,
        _pa_warmup(pa),
    )
    gain = complex(
        np.vdot(train_input, pa.predict(train_input))
        / np.vdot(train_input, train_input)
    )

    fit_block = slice(*[int(v) for v in config["fit_block"]])
    advisor_block = slice(*[int(v) for v in config["advisor_block"]])
    selection_block = slice(*[int(v) for v in config["selection_block"]])
    for name, span in (
        ("fit_block", fit_block),
        ("advisor_block", advisor_block),
        ("selection_block", selection_block),
    ):
        if span.start < 0 or span.stop > train_input.size:
            raise ValueError(f"{name} outside train split")
    if (
        max(fit_block.start, advisor_block.start)
        < min(fit_block.stop, advisor_block.stop)
    ):
        raise ValueError("fit and advisor blocks must be disjoint")
    if (
        max(fit_block.start, selection_block.start)
        < min(fit_block.stop, selection_block.stop)
        or max(advisor_block.start, selection_block.start)
        < min(advisor_block.stop, selection_block.stop)
    ):
        raise ValueError("selection block must be disjoint from fit/advisor")

    baseline_advisor_drive = baseline_dpd.predict(train_input[advisor_block])
    # The frozen ILA baseline already drives 1.19 against a 1.0 identification
    # support, and a DOMP residual correction concentrates exactly at the
    # peaks where the PA compresses, so a structural candidate inherently
    # demands peak headroom.  The guard rejects runaway growth, not the
    # extrapolation that predistortion requires; the winner's peak is
    # reported as explicit extrapolation exposure.
    allowed_growth = float(config.get("maximum_drive_growth", 0.15))
    maximum_drive = float(np.max(np.abs(baseline_advisor_drive))) * (
        1.0 + allowed_growth
    )
    identification_input_peak = float(np.max(np.abs(train_input)))
    mul_budget = int(config.get("real_multiplication_budget", 1000))

    grid = GmpDictionaryGrid(
        maximum_signal_delay=int(config["grid"]["maximum_signal_delay"]),
        maximum_envelope_delay=int(config["grid"]["maximum_envelope_delay"]),
        maximum_exponent=int(config["grid"]["maximum_exponent"]),
    )
    budgets = tuple(int(v) for v in config["member_budgets"])
    ridges = tuple(float(v) for v in config["ridge_values"])

    # ---- Stage 3a: DOMP member selection and ridge fits on the fit block.
    selection_diagnostics = fit_gmp_residual_members(
        train_input[fit_block],
        spline_model=baseline_dpd,
        pa=pa,
        gain=gain,
        grid=grid,
        member_budgets=budgets,
        ridge_values=ridges,
        warmup=warmup,
    )
    advisor_desired = train_input[advisor_block]
    candidates: list[dict[str, Any]] = []
    for candidate in selection_diagnostics["candidates"]:
        if "selected_members" not in candidate:
            continue
        composite = CompositeSplineGmpDPD(
            spline=baseline_dpd,
            members=tuple(
                GmpMember(
                    row["signal_delay"],
                    row["envelope_delay"],
                    row["exponent"],
                )
                for row in candidate["selected_members"]
            ),
            member_coefficients=candidate["member_coefficients"],
        )
        operation_count = composite.operation_count()
        drive = composite.predict(advisor_desired)
        peak = float(np.max(np.abs(drive)))
        support_valid = bool(
            peak <= maximum_drive * (1.0 + 64 * np.finfo(float).eps)
        )
        within_budget = bool(
            int(operation_count.real_multiplications) <= mul_budget
        )
        score = (
            _cascade_nmse(pa, drive, advisor_desired, gain, warmup)
            if support_valid and within_budget
            else float("inf")
        )
        if support_valid and within_budget:
            rejection_reason = None
        elif not support_valid:
            rejection_reason = "drive_peak_exceeds_bound"
        else:
            rejection_reason = "real_multiplication_budget"
        candidates.append(
            {
                "member_budget": int(candidate["member_budget"]),
                "ridge": float(candidate["ridge"]),
                "member_count": len(composite.members),
                "advisor_cascade_nmse_db": (
                    score if np.isfinite(score) else None
                ),
                "rejection_reason": rejection_reason,
                "drive_peak": peak,
                "support_valid": support_valid,
                "real_multiplications": int(
                    operation_count.real_multiplications
                ),
                "within_mul_budget": within_budget,
                "_composite": composite,
            }
        )
    feasible = [
        row
        for row in candidates
        if row["support_valid"] and row["within_mul_budget"]
    ]
    if not feasible:
        raise ValueError(
            "no composite candidate is support-valid and within budget"
        )
    best_member_fit = min(feasible, key=lambda row: row["advisor_cascade_nmse_db"])
    composite = best_member_fit["_composite"]

    # ---- Stage 3b: bounded Gauss-Newton polish of the composite.
    gn_schedule = config["gauss_newton_schedule"]
    fit_slices = [slice(int(row["fit"][0]), int(row["fit"][1])) for row in gn_schedule]
    advisor_slices = [
        slice(int(row["advisor"][0]), int(row["advisor"][1]))
        for row in gn_schedule
    ]
    spline_branch_count = baseline_dpd.coefficients.shape[0]
    spline_knot_count = baseline_dpd.coefficients.shape[1]
    member_count = len(composite.members)

    def composite_design_fn(signal: np.ndarray) -> np.ndarray:
        return composite_design_matrix(composite, signal)

    initial_flat = np.concatenate(
        (
            composite.spline.coefficients.reshape(-1),
            composite.member_coefficients,
        )
    )
    dl_config = DirectLearningConfig(
        ridge_values=tuple(
            float(v) for v in config.get("gn_ridge_values", (1e-8, 1e-6, 1e-4, 1e-2))
        ),
        step_values=tuple(
            float(v) for v in config.get("gn_step_values", (0.0625, 0.25, 1.0))
        ),
    )
    final_flat, gn_records = iterative_direct_schedule_core(
        initial_flat_coefficients=initial_flat,
        pa=pa,
        gain=gain,
        train_x=train_input,
        fit_slices=fit_slices,
        advisor_slices=advisor_slices,
        warmup=warmup,
        maximum_pa_input=maximum_drive,
        design_fn=composite_design_fn,
        config=dl_config,
    )
    polished_spline = SparseSplineMemoryDPD(
        knots=baseline_dpd.knots,
        branches=baseline_dpd.branches,
        coefficients=final_flat[: spline_branch_count * spline_knot_count].reshape(
            spline_branch_count,
            spline_knot_count,
        ),
        knot_strategy=baseline_dpd.knot_strategy,
    )
    polished = CompositeSplineGmpDPD(
        spline=polished_spline,
        members=composite.members,
        member_coefficients=final_flat[spline_branch_count * spline_knot_count :],
    )

    # ---- Train-only family selection on the held-out train block.
    selection_desired = train_input[selection_block]
    families = {
        "baseline_spline": baseline_dpd,
        "composite_domp": composite,
        "composite_domp_gauss_newton": polished,
    }
    family_scores: dict[str, float] = {}
    for name, model in families.items():
        drive = model.predict(selection_desired)
        family_scores[name] = _cascade_nmse(
            pa, drive, selection_desired, gain, warmup
        )
    winner_name = min(family_scores, key=family_scores.get)
    winner = families[winner_name]

    validation_diagnostics: dict[str, Any] = {}
    for name, model in families.items():
        drive = model.predict(validation_input)
        validation_diagnostics[name] = {
            "validation_nmse_db": _cascade_nmse(
                pa, drive, validation_input, gain, warmup
            ),
            "drive_summary": signal_summary(drive, warmup),
        }

    output_dir = PROJECT_ROOT / config["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)
    if winner is not baseline_dpd:
        winner.save(output_dir / "composite_dpd.npz")

    member_report = [
        {
            "signal_delay": member.signal_delay,
            "envelope_delay": member.envelope_delay,
            "exponent": member.exponent,
        }
        for member in composite.members
    ]
    report = {
        "schema_version": 1,
        "task": "composite_spline_gmp_dpd_research",
        "claims_scope": {
            "surrogate_only": True,
            "physical_pa_result": False,
            "test_split_accessed": False,
            "validation_used_for_selection": False,
            "validation_evaluated_as_diagnostic": True,
            "selection_rule": (
                "train-only blocks; selection block disjoint from fit, "
                "advisor, and Gauss-Newton slices"
            ),
        },
        "inputs": {
            "dataset": config["dataset"],
            "baseline_dpd_npz": config["baseline_dpd_npz"],
            "baseline_dpd_sha256": file_sha256(
                PROJECT_ROOT / config["baseline_dpd_npz"]
            ),
            "pa_model_npz": config["pa_model_npz"],
            "pa_model_sha256": file_sha256(
                PROJECT_ROOT / config["pa_model_npz"]
            ),
        },
        "protocol": {
            "warmup_samples": warmup,
            "gain_policy": "train_least_squares_frozen",
            "gain": gain,
            "identification_input_peak": identification_input_peak,
            "baseline_drive_peak": float(np.max(np.abs(baseline_advisor_drive))),
            "maximum_drive_bound": maximum_drive,
            "maximum_drive_growth_allowed": allowed_growth,
            "real_multiplication_budget": mul_budget,
        },
        "grid": {
            "maximum_signal_delay": grid.maximum_signal_delay,
            "maximum_envelope_delay": grid.maximum_envelope_delay,
            "maximum_exponent": grid.maximum_exponent,
            "candidate_member_count": len(grid.members),
        },
        "member_selection": {
            "cascade_nmse_before_members_db": (
                selection_diagnostics["cascade_nmse_before_members_db"]
            ),
            "candidate_count": len(candidates),
            "chosen_member_budget": int(best_member_fit["member_budget"]),
            "chosen_ridge": float(best_member_fit["ridge"]),
            "members": member_report,
            "member_count": member_count,
            "real_multiplications": best_member_fit["real_multiplications"],
            "advisor_cascade_nmse_db": best_member_fit[
                "advisor_cascade_nmse_db"
            ],
            "all_candidates": [
                {key: value for key, value in row.items() if key != "_composite"}
                for row in candidates
            ],
        },
        "gauss_newton_polish": {
            "schedule": gn_schedule,
            "records": gn_records,
            "final_flat_coefficient_count": int(final_flat.size),
        },
        "train_only_selection_nmse_db": family_scores,
        "selected_family": winner_name,
        "validation_diagnostics": validation_diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_limits": [
            "cascade NMSE is measured through the same frozen GMP evaluator",
            "a gain is surrogate-only until verified on an independent evaluator",
            "validation numbers are diagnostics; selection used train blocks only",
            (
                "the composite drive exceeds the PA identification support "
                "(peak extrapolation); physical verification is required "
                "before any hardware claim"
            ),
        ],
    }
    write_json(output_dir / "composite_report.json", report)
    print("selected family:", winner_name)
    print("train-only selection NMSE (dB):", json.dumps(family_scores, indent=2))
    print(
        "validation diagnostics (dB):",
        json.dumps(
            {
                name: round(row["validation_nmse_db"], 6)
                for name, row in validation_diagnostics.items()
            },
            indent=2,
        ),
    )
    print(
        "deployed real multiplications:",
        best_member_fit["real_multiplications"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
