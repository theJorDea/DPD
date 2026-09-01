"""Research runner: bounded iterative direct learning for the spline DPD.

This is a non-sealed research tool in the spirit of
``evaluate_blackbox_dpd_hypotheses``.  It improves a frozen spline-memory
DPD by direct updates through a frozen GMP PA evaluator:

    desired x (train slice) -> DPD -> frozen GMP PA -> g*x

Discipline:

* coefficient updates, ILC refits and model-family selection all use
  *train* slices only (fit and advisor segments are disjoint);
* the validation split is evaluated exactly once per candidate at the end
  and is recorded as a read-only diagnostic;
* the test split is never opened by this runner;
* two improvement mechanisms are compared on a held-out train block:
  (a) damped Gauss-Newton direct updates, (b) ILC waveform refinement with
  a causal least-squares refit on the resulting (x, u) pairs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from baseline.direct_learning import (
    DirectLearningConfig,
    bounded_direct_update,
    iterative_direct_schedule,
    model_with_delta,
    nmse_db,
    signal_summary,
)
from baseline.gmp_pa import GeneralizedMemoryPolynomialPA, gmp_terms
from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    fit_sparse_spline_memory_dpd,
)
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
    if value.get("task") != "direct_learning_spline_memory_dpd_research":
        raise ValueError("unexpected config task")
    required = {
        "dataset",
        "baseline_dpd_npz",
        "pa_model_npz",
        "output_dir",
        "gauss_newton_schedule",
        "ilc",
        "selection_block",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    return value


def _pa_warmup(pa: GeneralizedMemoryPolynomialPA) -> int:
    maximum_delay = max(
        term.signal_delay for term in gmp_terms(pa.config)
    )
    return int(maximum_delay + 1)


def _slice_pairs(config_value: list[dict[str, Any]]) -> tuple[list[slice], list[slice]]:
    fit_slices = [slice(int(row["fit"][0]), int(row["fit"][1])) for row in config_value]
    advisor_slices = [
        slice(int(row["advisor"][0]), int(row["advisor"][1]))
        for row in config_value
    ]
    return fit_slices, advisor_slices


def ilc_config_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("ilc", {}).get("enabled", True))


def _train_only_nmse(
    pa: GeneralizedMemoryPolynomialPA,
    model: SparseSplineMemoryDPD,
    desired: np.ndarray,
    gain: complex,
    warmup: int,
) -> float:
    drive = model.predict(desired)
    output = pa.predict(drive)
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

    # Frozen train-only gain (complex least squares), matching the
    # repository identification convention.
    gain = complex(
        np.vdot(train_input, pa.predict(train_input))
        / np.vdot(train_input, train_input)
    )

    # The evaluator was identified on train inputs whose peak is 1.0, while
    # the frozen baseline DPD already drives peaks above that (known
    # "unseen drive amplitude" exposure, disclosed in the risk register).
    # The update guard therefore bounds *growth* relative to the baseline
    # drive instead of pretending the support starts at the input peak.
    identification_input_peak = float(np.max(np.abs(train_input)))
    fit_slices, advisor_slices = _slice_pairs(config["gauss_newton_schedule"])
    baseline_drive_peaks = []
    for advisor_span in advisor_slices:
        baseline_drive_peaks.append(
            float(
                np.max(
                    np.abs(
                        baseline_dpd.predict(train_input[advisor_span])
                    )
                )
            )
        )
    if ilc_config_enabled(config):
        ilc_peak = float(
            np.max(
                np.abs(
                    baseline_dpd.predict(
                        train_input[
                            int(config["ilc"]["block"][0]) : int(
                                config["ilc"]["block"][1]
                            )
                        ]
                    )
                )
            )
        )
        baseline_drive_peaks.append(ilc_peak)
    allowed_growth = float(config.get("maximum_drive_growth", 0.02))
    maximum_pa_input = max(baseline_drive_peaks) * (1.0 + allowed_growth)

    dl_config = DirectLearningConfig(
        ridge_values=tuple(config.get("ridge_values", (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3))),
        step_values=tuple(config.get("step_values", (0.0625, 0.125, 0.25, 0.5, 1.0))),
    )

    baseline_selection_nmse = _train_only_nmse(
        pa,
        baseline_dpd,
        train_input[config["selection_block"][0] : config["selection_block"][1]],
        gain,
        warmup,
    )

    # (a) Bounded Gauss-Newton direct updates on disjoint train slices.
    secondary = None
    secondary_gain = None
    if config.get("secondary_pa_model_npz"):
        pa_b = GeneralizedMemoryPolynomialPA.load(
            PROJECT_ROOT / config["secondary_pa_model_npz"]
        )
        secondary_gain = complex(
            np.vdot(train_input, pa_b.predict(train_input))
            / np.vdot(train_input, train_input)
        )
        secondary = (pa_b, secondary_gain)
    gn_model, gn_records = iterative_direct_schedule(
        baseline_dpd,
        pa=pa,
        gain=gain,
        train_x=train_input,
        fit_slices=fit_slices,
        advisor_slices=advisor_slices,
        warmup=warmup,
        maximum_pa_input=maximum_pa_input,
        config=dl_config,
        secondary=secondary,
        joint_objective=bool(config.get("gauss_newton_joint", False)),
    )

    # (b) ILC waveform refinement plus a causal refit on the same knots.
    ilc_config = config["ilc"]
    ilc_block = slice(int(ilc_config["block"][0]), int(ilc_config["block"][1]))
    ilc_desired = train_input[ilc_block]
    ilc_records: list[dict[str, Any]] = []
    ilc_model: SparseSplineMemoryDPD | None = None
    if ilc_config.get("enabled", True):
        from baseline.direct_learning import ilc_waveform_refinement

        drive, ilc_records = ilc_waveform_refinement(
            ilc_desired,
            pa=pa,
            gain=gain,
            warmup=warmup,
            maximum_pa_input=maximum_pa_input,
            beta=float(ilc_config.get("beta", 1.0)),
            maximum_iterations=int(ilc_config.get("maximum_iterations", 6)),
        )
        ridge_grid = tuple(ilc_config.get("refit_ridges", (1e-8, 1e-6, 1e-4)))
        best_ridge = None
        best_score = float("inf")
        for ridge in ridge_grid:
            candidate, _ = fit_sparse_spline_memory_dpd(
                ilc_desired,
                drive,
                knots=baseline_dpd.knots,
                branches=baseline_dpd.branches,
                ridge=float(ridge),
            )
            score = _train_only_nmse(
                pa,
                candidate,
                train_input[config["selection_block"][0] : config["selection_block"][1]],
                gain,
                warmup,
            )
            ilc_records.append(
                {
                    "stage": "refit",
                    "ridge": float(ridge),
                    "selection_block_nmse_db": score,
                }
            )
            if score < best_score:
                best_score = score
                best_ridge = ridge
                ilc_model = candidate
        if ilc_model is None:
            raise ValueError("ILC refit produced no candidate")

    # Train-only family selection on a block disjoint from every fit and
    # advisor slice used above.
    selection_block = slice(
        int(config["selection_block"][0]),
        int(config["selection_block"][1]),
    )
    selection_signal = train_input[selection_block]
    candidates = {
        "baseline_ila": baseline_dpd,
        "gauss_newton_direct": gn_model,
    }
    if ilc_model is not None:
        candidates["ilc_refit"] = ilc_model
    selection_scores: dict[str, dict[str, float | None]] = {}
    for name, model in candidates.items():
        drive = model.predict(selection_signal)
        primary = nmse_db(
            pa.predict(drive), gain * selection_signal, warmup
        )
        secondary_score: float | None = None
        if secondary is not None:
            secondary_score = nmse_db(
                pa_b.predict(drive),
                secondary_gain * selection_signal,
                warmup,
            )
        worst = primary if secondary_score is None else max(
            primary, secondary_score
        )
        selection_scores[name] = {
            "primary": primary,
            "secondary": secondary_score,
            "worst_case": worst,
        }
    winner_name = min(
        selection_scores,
        key=lambda name: selection_scores[name]["worst_case"],
    )
    winner = candidates[winner_name]

    # Validation split: read-only diagnostics, evaluated once per model.
    validation_diagnostics = {}
    for name, model in candidates.items():
        drive = model.predict(validation_input)
        output = pa.predict(drive)
        entry = {
            "validation_nmse_db": nmse_db(output, gain * validation_input, warmup),
            "drive_summary": signal_summary(drive, warmup),
        }
        if secondary is not None:
            entry["secondary_validation_nmse_db"] = nmse_db(
                pa_b.predict(drive),
                secondary_gain * validation_input,
                warmup,
            )
        validation_diagnostics[name] = entry

    output_dir = PROJECT_ROOT / config["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)
    winner.save(output_dir / "direct_improved_model.npz")

    operation_count_changed = (
        winner.branch_count != baseline_dpd.branch_count
        or winner.knot_count != baseline_dpd.knot_count
    )
    report = {
        "schema_version": 1,
        "task": "direct_learning_spline_memory_dpd_research",
        "claims_scope": {
            "surrogate_only": True,
            "physical_pa_result": False,
            "test_split_accessed": False,
            "validation_used_for_selection": False,
            "validation_evaluated_as_diagnostic": True,
            "selection_rule": "train_only_block_disjoint_from_all_fit_and_advisor_slices",
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
            "baseline_drive_peak": max(baseline_drive_peaks),
            "maximum_drive_growth_allowed": allowed_growth,
            "maximum_pa_input_bound": maximum_pa_input,
            "selection_block": list(config["selection_block"]),
        },
        "gauss_newton": {
            "schedule": config["gauss_newton_schedule"],
            "records": gn_records,
        },
        "ilc": {
            "config": config["ilc"],
            "records": ilc_records,
        },
        "train_only_selection_nmse_db": selection_scores,
        "selected_family": winner_name,
        "selection_gain_over_baseline_db": (
            baseline_selection_nmse
            - selection_scores[winner_name]["worst_case"]
        ),
        "validation_diagnostics": validation_diagnostics,
        "operation_count_changed": operation_count_changed,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_limits": [
            "cascade NMSE is measured through the same frozen GMP evaluator",
            "a gain is surrogate-only until verified on an independent evaluator",
            "validation numbers are diagnostics; selection used train blocks only",
        ],
    }
    write_json(output_dir / "direct_report.json", report)
    print("selected family:", winner_name)
    print("train-only selection NMSE (dB):", json.dumps(selection_scores, indent=2))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
