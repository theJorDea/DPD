"""Cross-evaluator composite DPD runner (surrogate-exploitation guard).

Variant of ``run_composite_dpd`` hardened against the failure observed on
2026-08-30: members selected by OMP against one evaluator's residual plus
Gauss-Newton polishing through the same evaluator improved the cascade on
that evaluator but *degraded* it on an independent second evaluator.

Protocol here:

* DOMP member selection and coefficient fits still happen on train fit
  blocks against the primary evaluator's linearized residual;
* every (budget, ridge) candidate must improve the frozen baseline on
  BOTH evaluators on the disjoint train advisor block (cross gate) and is
  ranked by the worst-case NMSE across the two evaluators;
* Gauss-Newton polishing ranks ridge/step candidates by worst-case
  advisor NMSE across both evaluators;
* final family selection uses a held-out train block and again requires
  improvement over the baseline on both evaluators;
* the test split is never opened; validation is a read-only diagnostic.
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
    if value.get("task") != "composite_spline_gmp_dpd_cross_research":
        raise ValueError("unexpected config task")
    required = {
        "dataset",
        "baseline_dpd_npz",
        "pa_model_npz",
        "secondary_pa_model_npz",
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
    pa_b = GeneralizedMemoryPolynomialPA.load(
        PROJECT_ROOT / config["secondary_pa_model_npz"]
    )
    warmup = max(
        int(config.get("minimum_warmup", 0)),
        baseline_dpd.maximum_delay,
        _pa_warmup(pa),
        _pa_warmup(pa_b),
    )
    gain = complex(
        np.vdot(train_input, pa.predict(train_input))
        / np.vdot(train_input, train_input)
    )
    gain_b = complex(
        np.vdot(train_input, pa_b.predict(train_input))
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
    allowed_growth = float(config.get("maximum_drive_growth", 0.15))
    maximum_drive = float(np.max(np.abs(baseline_advisor_drive))) * (
        1.0 + allowed_growth
    )
    mul_budget = int(config.get("real_multiplication_budget", 1000))

    grid = GmpDictionaryGrid(
        maximum_signal_delay=int(config["grid"]["maximum_signal_delay"]),
        maximum_envelope_delay=int(config["grid"]["maximum_envelope_delay"]),
        maximum_exponent=int(config["grid"]["maximum_exponent"]),
    )
    budgets = tuple(int(v) for v in config["member_budgets"])
    ridges = tuple(float(v) for v in config["ridge_values"])

    consensus = bool(config.get("consensus_target", False))
    selection_diagnostics = fit_gmp_residual_members(
        train_input[fit_block],
        spline_model=baseline_dpd,
        pa=pa,
        gain=gain,
        grid=grid,
        member_budgets=budgets,
        ridge_values=ridges,
        warmup=warmup,
        consensus_secondary=(pa_b, gain_b) if consensus else None,
    )

    advisor_desired = train_input[advisor_block]
    baseline_nmse_a = _cascade_nmse(
        pa,
        baseline_advisor_drive,
        advisor_desired,
        gain,
        warmup,
    )
    baseline_nmse_b = _cascade_nmse(
        pa_b,
        baseline_advisor_drive,
        advisor_desired,
        gain_b,
        warmup,
    )
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
        score_a = _cascade_nmse(pa, drive, advisor_desired, gain, warmup)
        score_b = _cascade_nmse(pa_b, drive, advisor_desired, gain_b, warmup)
        improves_both = bool(
            score_a < baseline_nmse_a and score_b < baseline_nmse_b
        )
        feasible = bool(
            support_valid and within_budget and improves_both
        )
        if feasible:
            worst_case = max(score_a, score_b)
            rejection_reason = None
        elif not support_valid:
            worst_case = None
            rejection_reason = "drive_peak_exceeds_bound"
        elif not within_budget:
            worst_case = None
            rejection_reason = "real_multiplication_budget"
        else:
            worst_case = None
            rejection_reason = "cross_evaluator_gate"
        candidates.append(
            {
                "member_budget": int(candidate["member_budget"]),
                "ridge": float(candidate["ridge"]),
                "member_count": len(composite.members),
                "advisor_nmse_primary_db": score_a,
                "advisor_nmse_secondary_db": score_b,
                "worst_case_nmse_db": worst_case,
                "improves_both_evaluators": improves_both,
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
    feasible = [row for row in candidates if row["worst_case_nmse_db"] is not None]
    if not feasible:
        report = {
            "schema_version": 1,
            "task": "composite_spline_gmp_dpd_cross_research",
            "decision": "HOLD",
            "hold_reason": (
                "no DOMP candidate improves the baseline on both evaluators"
            ),
            "baseline_advisor_nmse_db": {
                "primary": baseline_nmse_a,
                "secondary": baseline_nmse_b,
            },
            "candidate_count": len(candidates),
        }
        output_dir = PROJECT_ROOT / config["output_dir"]
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {output_dir}"
            )
        output_dir.mkdir(parents=True)
        write_json(output_dir / "composite_cross_report.json", report)
        print("CROSS-EVALUATOR HOLD: no candidate improves both evaluators")
        print("baseline advisor NMSE A:", baseline_nmse_a)
        print("baseline advisor NMSE B:", baseline_nmse_b)
        return 0

    best_member_fit = min(
        feasible, key=lambda row: row["worst_case_nmse_db"]
    )
    composite = best_member_fit["_composite"]

    # Gauss-Newton polish with worst-case ranking across both evaluators.
    gn_schedule = config["gauss_newton_schedule"]
    fit_slices = [
        slice(int(row["fit"][0]), int(row["fit"][1])) for row in gn_schedule
    ]
    advisor_slices = [
        slice(int(row["advisor"][0]), int(row["advisor"][1]))
        for row in gn_schedule
    ]
    spline_shape = baseline_dpd.coefficients.shape

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
            float(v)
            for v in config.get("gn_ridge_values", (1e-8, 1e-6, 1e-4, 1e-2))
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
        secondary=(pa_b, gain_b),
    )
    polished_spline = SparseSplineMemoryDPD(
        knots=baseline_dpd.knots,
        branches=baseline_dpd.branches,
        coefficients=final_flat[: spline_shape[0] * spline_shape[1]].reshape(
            spline_shape
        ),
        knot_strategy=baseline_dpd.knot_strategy,
    )
    polished = CompositeSplineGmpDPD(
        spline=polished_spline,
        members=composite.members,
        member_coefficients=final_flat[spline_shape[0] * spline_shape[1] :],
    )

    # Final family selection on the held-out train block, cross gate kept.
    selection_desired = train_input[selection_block]
    families = {
        "baseline_spline": baseline_dpd,
        "composite_domp_cross": composite,
        "composite_domp_gauss_newton_cross": polished,
    }
    family_scores: dict[str, dict[str, float]] = {}
    for name, model in families.items():
        drive = model.predict(selection_desired)
        family_scores[name] = {
            "primary": _cascade_nmse(
                pa, drive, selection_desired, gain, warmup
            ),
            "secondary": _cascade_nmse(
                pa_b, drive, selection_desired, gain_b, warmup
            ),
        }
    eligible = {
        name: scores
        for name, scores in family_scores.items()
        if name == "baseline_spline"
        or (
            scores["primary"] < family_scores["baseline_spline"]["primary"]
            and scores["secondary"]
            < family_scores["baseline_spline"]["secondary"]
        )
    }
    if len(eligible) == 1:
        winner_name = "baseline_spline"
        decision = "HOLD"
        hold_reason = (
            "no composite family improves the baseline on both evaluators "
            "on the held-out train block; baseline retained"
        )
    else:
        winner_name = min(
            eligible,
            key=lambda name: max(
                family_scores[name]["primary"],
                family_scores[name]["secondary"],
            ),
        )
        decision = "PASS"
        hold_reason = None
    winner = families[winner_name]

    validation_diagnostics: dict[str, Any] = {}
    for name, model in families.items():
        drive = model.predict(validation_input)
        validation_diagnostics[name] = {
            "primary_validation_nmse_db": _cascade_nmse(
                pa, drive, validation_input, gain, warmup
            ),
            "secondary_validation_nmse_db": _cascade_nmse(
                pa_b, drive, validation_input, gain_b, warmup
            ),
            "drive_summary": signal_summary(drive, warmup),
        }

    output_dir = PROJECT_ROOT / config["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)
    if winner is not baseline_dpd:
        winner.save(output_dir / "composite_cross_dpd.npz")

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
        "task": "composite_spline_gmp_dpd_cross_research",
        "decision": decision,
        "hold_reason": hold_reason,
        "selected_family": winner_name,
        "claims_scope": {
            "surrogate_only": True,
            "physical_pa_result": False,
            "test_split_accessed": False,
            "validation_used_for_selection": False,
            "validation_evaluated_as_diagnostic": True,
            "selection_rule": (
                "train-only blocks; cross gate requires improvement on both "
                "frozen evaluators; worst-case ranking"
            ),
        },
        "inputs": {
            "dataset": config["dataset"],
            "primary_pa_model_npz": config["pa_model_npz"],
            "primary_pa_model_sha256": file_sha256(
                PROJECT_ROOT / config["pa_model_npz"]
            ),
            "secondary_pa_model_npz": config["secondary_pa_model_npz"],
            "secondary_pa_model_sha256": file_sha256(
                PROJECT_ROOT / config["secondary_pa_model_npz"]
            ),
        },
        "protocol": {
            "warmup_samples": warmup,
            "gain_primary": gain,
            "gain_secondary": gain_b,
            "consensus_target": bool(config.get("consensus_target", False)),
            "maximum_drive_bound": maximum_drive,
            "real_multiplication_budget": mul_budget,
        },
        "member_selection": {
            "chosen_member_budget": int(best_member_fit["member_budget"]),
            "chosen_ridge": float(best_member_fit["ridge"]),
            "member_count": len(member_report),
            "members": member_report,
            "real_multiplications": int(
                best_member_fit["real_multiplications"]
            ),
            "candidate_count": len(candidates),
            "all_candidates": [
                {key: value for key, value in row.items() if key != "_composite"}
                for row in candidates
            ],
        },
        "gauss_newton_polish": {
            "schedule": gn_schedule,
            "records": gn_records,
        },
        "train_only_selection_nmse_db": family_scores,
        "validation_diagnostics": validation_diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_limits": [
            "cascade NMSE is measured through frozen GMP surrogate evaluators",
            "the cross gate is a surrogate-level guard, not physical proof",
            "validation numbers are diagnostics; selection used train blocks only",
        ],
    }
    write_json(output_dir / "composite_cross_report.json", report)
    print("decision:", decision, "| selected family:", winner_name)
    print(
        "train-only selection NMSE (primary/secondary):",
        json.dumps(family_scores, indent=2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
