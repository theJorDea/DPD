"""Fast validation-only experiments for BlackBox DPD improvement hypotheses.

This is deliberately a scratch research runner, not a frozen release tool.
It never reads the sealed/test split and never uses measured validation output
as a DPD input.  The common deployment evaluation is always

    desired x -> candidate DPD -> frozen PA evaluator -> compare with g*x.

Stages are run separately so intermediate negative results can stop later work.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from baseline.alignment import overlap_for_delay
from baseline.gmp_pa import GMPConfig, fit_gmp_pa
from baseline.fixed_point_spline_memory_dpd import (
    FixedPointSparseSplineMemoryDPD,
)
from baseline.spline_memory_dpd import (
    fit_sparse_spline_memory_dpd,
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    spline_memory_design_matrix,
)
from experiments.select_blackbox_dpd import (
    load_frozen_blackbox_dpd_selection,
)
from experiments.select_blackbox_pa import (
    _load_normalized_pairs,
    _verify_selection_view,
    load_frozen_blackbox_pa_selection,
)
from experiments.evaluate_fixed_point_dpd import _make_fixed_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_DIR = PROJECT_ROOT / "data/private/blackbox_v3/selection"
PA_BUNDLE = PROJECT_ROOT / "experiments/results/blackbox_pa_v2_selection"
DPD_BUNDLE = PROJECT_ROOT / "experiments/results/blackbox_dpd_v1_selection"
OUTPUT_DIR = PROJECT_ROOT / "experiments/results/blackbox_dpd_hypotheses_quick"
COMMON_WARMUP = 11
SEGMENT_COUNT = 8


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value)
        + "\n",
        encoding="utf-8",
    )


def _delay(values: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.complex128)
    if delay == 0:
        result[:] = values
    elif delay < values.size:
        result[delay:] = values[:-delay]
    return result


def nmse_db(estimate: np.ndarray, reference: np.ndarray, warmup: int) -> float:
    error = estimate[warmup:] - reference[warmup:]
    error_power = float(np.mean(np.abs(error) ** 2))
    reference_power = float(np.mean(np.abs(reference[warmup:]) ** 2))
    if reference_power <= 0.0:
        raise ValueError("NMSE reference must have positive power")
    if error_power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(error_power / reference_power))


def signal_summary(signal: np.ndarray, warmup: int = 0) -> dict[str, float]:
    values = np.asarray(signal)[warmup:]
    power = float(np.mean(np.abs(values) ** 2))
    peak = float(np.max(np.abs(values)))
    return {
        "rms": float(np.sqrt(power)),
        "peak": peak,
        "papr_db": float(10.0 * np.log10(peak * peak / power)),
    }


def segment_improvements(
    no_dpd: np.ndarray,
    candidate: np.ndarray,
    ideal: np.ndarray,
) -> list[float]:
    indices = np.arange(COMMON_WARMUP, ideal.size)
    result = []
    for segment in np.array_split(indices, SEGMENT_COUNT):
        baseline_score = nmse_db(no_dpd[segment], ideal[segment], 0)
        candidate_score = nmse_db(candidate[segment], ideal[segment], 0)
        result.append(float(baseline_score - candidate_score))
    return result


@dataclass(frozen=True)
class Context:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    gain: complex
    pa: object
    baseline: SparseSplineMemoryDPD
    maximum_pa_input: float


def load_context() -> Context:
    verification = _verify_selection_view(SELECTION_DIR.resolve())
    frozen_pa = load_frozen_blackbox_pa_selection(PA_BUNDLE)
    frozen_dpd = load_frozen_blackbox_dpd_selection(DPD_BUNDLE)
    train_x, train_y, validation_x, validation_y, _ = _load_normalized_pairs(
        SELECTION_DIR.resolve(),
        scale=frozen_pa.normalization_scale,
        expected_counts=verification["split_contract"],
    )
    train_x, train_y = overlap_for_delay(
        train_x, train_y, frozen_pa.integer_delay_samples
    )
    validation_x, validation_y = overlap_for_delay(
        validation_x, validation_y, frozen_pa.integer_delay_samples
    )
    gain_record = frozen_dpd.manifest["gain"]
    gain = complex(float(gain_record["real"]), float(gain_record["imag"]))
    maximum_pa_input = float(
        frozen_dpd.manifest["selection"]["selected_trial"]["support_checks"]
        ["maximum_train_pa_input_amplitude"]
    )
    return Context(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        gain=gain,
        pa=frozen_pa.model,
        baseline=frozen_dpd.model,
        maximum_pa_input=maximum_pa_input,
    )


def evaluate_model(
    context: Context,
    model: SparseSplineMemoryDPD,
    desired: np.ndarray,
) -> dict:
    drive = np.asarray(model.predict(desired), dtype=np.complex128)
    output = np.asarray(context.pa.predict(drive), dtype=np.complex128)
    no_dpd = np.asarray(context.pa.predict(desired), dtype=np.complex128)
    ideal = context.gain * desired
    score = nmse_db(output, ideal, COMMON_WARMUP)
    baseline_score = nmse_db(no_dpd, ideal, COMMON_WARMUP)
    return {
        "nmse_db": score,
        "improvement_over_no_dpd_db": baseline_score - score,
        "no_dpd_nmse_db": baseline_score,
        "drive": drive,
        "output": output,
        "no_dpd_output": no_dpd,
        "ideal": ideal,
        "drive_summary": signal_summary(drive, COMMON_WARMUP),
        "support_valid": bool(
            np.max(np.abs(drive))
            <= context.maximum_pa_input * (1.0 + 64.0 * np.finfo(float).eps)
        ),
        "segment_improvements_db": segment_improvements(
            no_dpd, output, ideal
        ),
    }


def _compact_evaluation(evaluation: dict) -> dict:
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"drive", "output", "no_dpd_output", "ideal"}
    }


def _complex_correlation(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.sqrt(
        np.vdot(left, left).real * np.vdot(right, right).real
    )
    return float(abs(np.vdot(left, right)) / max(denominator, 1e-30))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / max(denominator, 1e-30))


def _fit_residual_branch(
    design: np.ndarray,
    target: np.ndarray,
    *,
    warmup: int,
    ridge: float,
) -> tuple[np.ndarray, float]:
    design = design[warmup:]
    target = target[warmup:]
    normalization = np.sqrt(float(design.shape[0]))
    fit_design = design / normalization
    fit_target = target / normalization
    regularizer = np.sqrt(ridge) * np.eye(
        design.shape[1], dtype=np.complex128
    )
    coefficients = np.linalg.lstsq(
        np.vstack((fit_design, regularizer)),
        np.concatenate((fit_target, np.zeros(design.shape[1], complex))),
        rcond=None,
    )[0]
    residual_after = target - design @ coefficients
    improvement = float(
        10.0
        * np.log10(
            np.mean(np.abs(target) ** 2)
            / np.mean(np.abs(residual_after) ** 2)
        )
    )
    return coefficients, improvement


def run_residual_analysis(context: Context) -> dict:
    started = time.perf_counter()
    baseline_validation = evaluate_model(
        context, context.baseline, context.validation_x
    )
    cascade_residual = (
        baseline_validation["output"] - baseline_validation["ideal"]
    )
    scored_residual = cascade_residual[COMMON_WARMUP:]
    scored_x = context.validation_x[COMMON_WARMUP:]

    lagged_correlations = []
    for delay in range(0, 17):
        feature = _delay(context.validation_x, delay)[COMMON_WARMUP:]
        lagged_correlations.append(
            {
                "delay": delay,
                "complex_correlation": _complex_correlation(
                    feature, scored_residual
                ),
            }
        )

    amplitude = np.abs(scored_x)
    amplitude_diagnostics = {
        "corr_error_power_vs_amplitude": _pearson(
            np.abs(scored_residual) ** 2, amplitude
        ),
        "corr_error_power_vs_power": _pearson(
            np.abs(scored_residual) ** 2, amplitude**2
        ),
    }
    slow_state = []
    input_power = np.abs(context.validation_x) ** 2
    for beta in (0.9, 0.99, 0.999, 0.9999):
        state = np.empty(input_power.size, dtype=float)
        state[0] = input_power[0]
        for index in range(1, input_power.size):
            state[index] = beta * state[index - 1] + (1.0 - beta) * input_power[index]
        slow_state.append(
            {
                "beta": beta,
                "corr_error_power": _pearson(
                    np.abs(scored_residual) ** 2,
                    state[COMMON_WARMUP:],
                ),
            }
        )

    # Rank one-branch spline additions on train-only inverse residual.
    ila_input = context.train_y / context.gain
    inverse_prediction = context.baseline.predict(ila_input)
    inverse_residual = context.train_x - inverse_prediction
    active = set(context.baseline.branches)
    branch_scores = []
    for signal_delay in range(0, 7):
        for envelope_delay in range(0, 7):
            branch = SplineMemoryBranch(signal_delay, envelope_delay)
            if branch in active:
                continue
            design = spline_memory_design_matrix(
                ila_input, context.baseline.knots, (branch,)
            )
            _, improvement = _fit_residual_branch(
                design,
                inverse_residual,
                warmup=max(context.baseline.maximum_delay, signal_delay, envelope_delay),
                ridge=1e-4,
            )
            branch_scores.append(
                {
                    "signal_delay": signal_delay,
                    "envelope_delay": envelope_delay,
                    "train_inverse_residual_improvement_db": improvement,
                }
            )
    branch_scores.sort(
        key=lambda row: row["train_inverse_residual_improvement_db"],
        reverse=True,
    )
    result = {
        "stage": "residual_analysis",
        "scope": "train_ranking_and_validation_diagnostics_only",
        "baseline_validation": _compact_evaluation(baseline_validation),
        "lagged_complex_correlations": lagged_correlations,
        "amplitude_diagnostics": amplitude_diagnostics,
        "slow_state_diagnostics": slow_state,
        "top_cross_memory_branch_scores": branch_scores[:12],
        "candidate_pair_count": len(branch_scores),
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_limits": [
            "raw residual correlation is not a causal PA-sensitivity score",
            "branch ranking uses train inverse residual only",
            "validation measured y is never a DPD input",
        ],
    }
    _write_json(OUTPUT_DIR / "residual_analysis.json", result)
    return result


def _real_jacobian_forward_difference(
    context: Context,
    design: np.ndarray,
    coefficients: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    base_drive = design @ flat
    base_output = np.asarray(context.pa.predict(base_drive), dtype=np.complex128)
    parameter_count = flat.size
    jacobian = np.empty(
        (2 * (design.shape[0] - COMMON_WARMUP), 2 * parameter_count),
        dtype=float,
    )
    for parameter in range(parameter_count):
        direction = design[:, parameter]
        real_output = context.pa.predict(base_drive + epsilon * direction)
        imag_output = context.pa.predict(base_drive + 1j * epsilon * direction)
        real_derivative = (
            np.asarray(real_output, complex) - base_output
        ) / epsilon
        imag_derivative = (
            np.asarray(imag_output, complex) - base_output
        ) / epsilon
        real_derivative = real_derivative[COMMON_WARMUP:]
        imag_derivative = imag_derivative[COMMON_WARMUP:]
        jacobian[:, parameter] = np.concatenate(
            (real_derivative.real, real_derivative.imag)
        )
        jacobian[:, parameter_count + parameter] = np.concatenate(
            (imag_derivative.real, imag_derivative.imag)
        )
    return jacobian, base_drive, base_output


def _model_with_delta(
    baseline: SparseSplineMemoryDPD,
    delta: np.ndarray,
    step: float,
) -> SparseSplineMemoryDPD:
    parameter_count = baseline.coefficients.size
    complex_delta = (
        delta[:parameter_count] + 1j * delta[parameter_count:]
    ).reshape(baseline.coefficients.shape)
    return SparseSplineMemoryDPD(
        knots=baseline.knots,
        branches=baseline.branches,
        coefficients=baseline.coefficients + step * complex_delta,
        knot_strategy=baseline.knot_strategy,
    )


def run_direct_refinement(context: Context) -> dict:
    started = time.perf_counter()
    fit_slice = slice(16_384, 20_480)  # 4096 contiguous train samples
    advisor_slice = slice(32_768, 40_960)  # disjoint 8192 train samples
    fit_x = context.train_x[fit_slice]
    advisor_x = context.train_x[advisor_slice]
    design = spline_memory_design_matrix(
        fit_x, context.baseline.knots, context.baseline.branches
    )
    epsilon = 1e-4
    jacobian, _, base_output = _real_jacobian_forward_difference(
        context,
        design,
        context.baseline.coefficients,
        epsilon=epsilon,
    )
    fit_target = context.gain * fit_x
    complex_residual = fit_target[COMMON_WARMUP:] - base_output[COMMON_WARMUP:]
    target = np.concatenate((complex_residual.real, complex_residual.imag))
    normalization = np.sqrt(float(target.size))
    normalized_jacobian = jacobian / normalization
    normalized_target = target / normalization

    candidates = []
    ridge_values = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
    step_values = (0.0625, 0.125, 0.25, 0.5, 1.0)
    advisor_ideal = context.gain * advisor_x
    advisor_no_dpd = context.pa.predict(advisor_x)
    advisor_no_dpd_nmse = nmse_db(
        advisor_no_dpd, advisor_ideal, COMMON_WARMUP
    )
    for ridge in ridge_values:
        regularizer = np.sqrt(ridge) * np.eye(
            normalized_jacobian.shape[1], dtype=float
        )
        delta = np.linalg.lstsq(
            np.vstack((normalized_jacobian, regularizer)),
            np.concatenate(
                ((normalized_target, np.zeros(regularizer.shape[0], dtype=float)))
            ),
            rcond=None,
        )[0]
        for step in step_values:
            candidate_model = _model_with_delta(context.baseline, delta, step)
            drive = candidate_model.predict(advisor_x)
            output = context.pa.predict(drive)
            score = nmse_db(output, advisor_ideal, COMMON_WARMUP)
            support_valid = bool(
                np.max(np.abs(drive)) <= context.maximum_pa_input
                * (1.0 + 64.0 * np.finfo(float).eps)
            )
            candidates.append(
                {
                    "ridge": ridge,
                    "step": step,
                    "advisor_train_nmse_db": score,
                    "advisor_train_improvement_over_no_dpd_db": (
                        advisor_no_dpd_nmse - score
                    ),
                    "support_valid": support_valid,
                    "maximum_drive": float(np.max(np.abs(drive))),
                    "delta_relative_norm": float(
                        step
                        * np.linalg.norm(delta)
                        / max(
                            np.linalg.norm(
                                np.concatenate(
                                    ((context.baseline.coefficients.real.ravel(),
                                      context.baseline.coefficients.imag.ravel()))
                                )
                            ),
                            1e-30,
                        )
                    ),
                    "delta": delta,
                }
            )
    valid = [row for row in candidates if row["support_valid"]]
    selected = min(valid, key=lambda row: row["advisor_train_nmse_db"])
    refined_model = _model_with_delta(
        context.baseline, selected["delta"], selected["step"]
    )
    baseline_validation = evaluate_model(
        context, context.baseline, context.validation_x
    )
    refined_validation = evaluate_model(
        context, refined_model, context.validation_x
    )
    coefficient_change = refined_model.coefficients - context.baseline.coefficients
    refined_model.save(OUTPUT_DIR / "direct_refined_model.npz")
    public_candidates = [
        {key: value for key, value in row.items() if key != "delta"}
        for row in candidates
    ]
    selected_public = {
        key: value for key, value in selected.items() if key != "delta"
    }
    result = {
        "stage": "direct_refinement",
        "calibration_split": {
            "jacobian": [fit_slice.start, fit_slice.stop],
            "advisor": [advisor_slice.start, advisor_slice.stop],
            "validation_used_for_coefficient_or_step_selection": False,
        },
        "finite_difference_epsilon": epsilon,
        "jacobian_shape": list(jacobian.shape),
        "candidate_count": len(candidates),
        "selected_on_disjoint_train_advisor": selected_public,
        "baseline_validation": _compact_evaluation(baseline_validation),
        "refined_validation": _compact_evaluation(refined_validation),
        "validation_gain_over_baseline_dpd_db": (
            baseline_validation["nmse_db"] - refined_validation["nmse_db"]
        ),
        "coefficient_change_relative_norm": float(
            np.linalg.norm(coefficient_change)
            / max(np.linalg.norm(context.baseline.coefficients), 1e-30)
        ),
        "operation_count_changed": False,
        "operation_count": context.baseline.operation_count().to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "all_advisor_candidates": public_candidates,
        "claim_limit": (
            "direct refinement optimizes through the same frozen PA evaluator; "
            "a gain is surrogate-only until independently verified"
        ),
    }
    _write_json(OUTPUT_DIR / "direct_refinement.json", result)
    return result


def _one_direct_robustness_case(
    context: Context,
    *,
    fit_start: int,
    advisor_start: int,
    epsilon: float,
) -> dict:
    fit_x = context.train_x[fit_start:fit_start + 4096]
    advisor_x = context.train_x[advisor_start:advisor_start + 8192]
    design = spline_memory_design_matrix(
        fit_x, context.baseline.knots, context.baseline.branches
    )
    jacobian, _, base_output = _real_jacobian_forward_difference(
        context,
        design,
        context.baseline.coefficients,
        epsilon=epsilon,
    )
    fit_target = context.gain * fit_x
    residual = fit_target[COMMON_WARMUP:] - base_output[COMMON_WARMUP:]
    target = np.concatenate((residual.real, residual.imag))
    normalization = np.sqrt(float(target.size))
    normalized_jacobian = jacobian / normalization
    normalized_target = target / normalization
    ridge = 1e-8
    regularizer = np.sqrt(ridge) * np.eye(
        normalized_jacobian.shape[1], dtype=float
    )
    delta = np.linalg.lstsq(
        np.vstack((normalized_jacobian, regularizer)),
        np.concatenate(
            ((normalized_target, np.zeros(regularizer.shape[0], dtype=float)))
        ),
        rcond=None,
    )[0]
    advisor_ideal = context.gain * advisor_x
    advisor_candidates = []
    for step in (0.5, 0.75, 1.0):
        model = _model_with_delta(context.baseline, delta, step)
        drive = model.predict(advisor_x)
        output = context.pa.predict(drive)
        advisor_candidates.append(
            {
                "step": step,
                "nmse_db": nmse_db(output, advisor_ideal, COMMON_WARMUP),
                "support_valid": bool(
                    np.max(np.abs(drive)) <= context.maximum_pa_input
                    * (1.0 + 64.0 * np.finfo(float).eps)
                ),
            }
        )
    selected = min(
        (row for row in advisor_candidates if row["support_valid"]),
        key=lambda row: row["nmse_db"],
    )
    model = _model_with_delta(context.baseline, delta, selected["step"])
    validation = evaluate_model(context, model, context.validation_x)
    baseline_validation = evaluate_model(
        context, context.baseline, context.validation_x
    )
    parameter_count = context.baseline.coefficients.size
    complex_delta = (
        delta[:parameter_count] + 1j * delta[parameter_count:]
    ).reshape(context.baseline.coefficients.shape)
    return {
        "fit_range": [fit_start, fit_start + 4096],
        "advisor_range": [advisor_start, advisor_start + 8192],
        "epsilon": epsilon,
        "selected_step_on_train_advisor": selected,
        "advisor_candidates": advisor_candidates,
        "validation": _compact_evaluation(validation),
        "validation_gain_over_baseline_dpd_db": (
            baseline_validation["nmse_db"] - validation["nmse_db"]
        ),
        "coefficient_change_relative_norm": float(
            selected["step"] * np.linalg.norm(complex_delta)
            / max(np.linalg.norm(context.baseline.coefficients), 1e-30)
        ),
    }


def run_direct_robustness(context: Context) -> dict:
    started = time.perf_counter()
    specifications = (
        (0, 8192, 3e-5),
        (8192, 24_576, 1e-4),
        (49_152, 65_536, 3e-4),
    )
    cases = [
        _one_direct_robustness_case(
            context,
            fit_start=fit_start,
            advisor_start=advisor_start,
            epsilon=epsilon,
        )
        for fit_start, advisor_start, epsilon in specifications
    ]
    gains = np.asarray(
        [case["validation_gain_over_baseline_dpd_db"] for case in cases]
    )
    result = {
        "stage": "direct_refinement_robustness",
        "case_count": len(cases),
        "cases": cases,
        "validation_gain_summary_db": {
            "minimum": float(np.min(gains)),
            "median": float(np.median(gains)),
            "maximum": float(np.max(gains)),
        },
        "all_cases_improve_every_validation_segment": bool(
            all(
                min(case["validation"]["segment_improvements_db"])
                > min(
                    evaluate_model(
                        context, context.baseline, context.validation_x
                    )["segment_improvements_db"]
                )
                for case in cases
            )
        ),
        "operation_count_changed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "robustness across train blocks and finite-difference steps does "
            "not remove same-surrogate optimization bias"
        ),
    }
    _write_json(OUTPUT_DIR / "direct_refinement_robustness.json", result)
    return result


def run_cross_memory(context: Context) -> dict:
    started = time.perf_counter()
    residual_path = OUTPUT_DIR / "residual_analysis.json"
    if not residual_path.is_file():
        raise FileNotFoundError(
            "run --stage residual before cross-memory selection"
        )
    residual = json.loads(residual_path.read_text(encoding="utf-8"))
    ranked = residual["top_cross_memory_branch_scores"][:6]
    ila_input = context.train_y / context.gain
    baseline_branches = tuple(context.baseline.branches)
    trials = []

    def fit_trial(branch_record: dict, ridge: float) -> tuple[dict, SparseSplineMemoryDPD]:
        added = SplineMemoryBranch(
            int(branch_record["signal_delay"]),
            int(branch_record["envelope_delay"]),
        )
        model, diagnostics = fit_sparse_spline_memory_dpd(
            ila_input,
            context.train_x,
            branches=baseline_branches + (added,),
            knots=context.baseline.knots,
            ridge=ridge,
        )
        evaluation = evaluate_model(context, model, context.validation_x)
        record = {
            "added_branch": asdict(added),
            "ridge": ridge,
            "train_inverse_residual_rank_score_db": branch_record[
                "train_inverse_residual_improvement_db"
            ],
            "validation": _compact_evaluation(evaluation),
            "fit_diagnostics": asdict(diagnostics),
            "operation_count": model.operation_count().to_dict(),
        }
        return record, model

    models: list[SparseSplineMemoryDPD] = []
    # Fast screen: one frozen ridge for six train-ranked branches.
    for branch_record in ranked:
        record, model = fit_trial(branch_record, 1e-4)
        trials.append(record)
        models.append(model)

    screened_best_index = min(
        range(len(trials)),
        key=lambda index: trials[index]["validation"]["nmse_db"],
    )
    screened_best_branch = trials[screened_best_index]["added_branch"]
    source_record = next(
        row for row in ranked
        if int(row["signal_delay"]) == int(screened_best_branch["signal_delay"])
        and int(row["envelope_delay"]) == int(screened_best_branch["envelope_delay"])
    )
    # Only the winning topology receives two extra ridge checks.
    for ridge in (1e-5, 1e-3):
        record, model = fit_trial(source_record, ridge)
        trials.append(record)
        models.append(model)

    selected_index = min(
        range(len(trials)),
        key=lambda index: trials[index]["validation"]["nmse_db"],
    )
    selected = trials[selected_index]
    selected_model = models[selected_index]
    selected_model.save(OUTPUT_DIR / "cross_memory_model.npz")
    baseline_validation = evaluate_model(
        context, context.baseline, context.validation_x
    )
    direct_path = OUTPUT_DIR / "direct_refined_model.npz"
    direct_validation = None
    if direct_path.is_file():
        direct_validation = _compact_evaluation(
            evaluate_model(
                context,
                SparseSplineMemoryDPD.load(direct_path),
                context.validation_x,
            )
        )
    result = {
        "stage": "one_cross_memory_spline_branch",
        "selection_split": "validation",
        "candidate_source": "top six train-only inverse-residual branch scores",
        "trial_count": len(trials),
        "baseline_validation": _compact_evaluation(baseline_validation),
        "direct_refined_validation": direct_validation,
        "trials": trials,
        "selected": selected,
        "gain_over_baseline_dpd_db": (
            baseline_validation["nmse_db"]
            - selected["validation"]["nmse_db"]
        ),
        "gain_over_direct_refined_db": (
            None
            if direct_validation is None
            else direct_validation["nmse_db"]
            - selected["validation"]["nmse_db"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "branch and ridge are selected on validation; sealed test and "
            "independent PA evidence remain unavailable"
        ),
    }
    _write_json(OUTPUT_DIR / "cross_memory.json", result)
    return result


def _gmp_residual_feature(
    signal: np.ndarray,
    *,
    signal_delay: int,
    envelope_delay: int,
    exponent: int,
) -> np.ndarray:
    signal_lag = _delay(signal, signal_delay)
    envelope = np.abs(_delay(signal, envelope_delay))
    return signal_lag * envelope**exponent


def _predict_direct_plus_gmp(
    base_model: SparseSplineMemoryDPD,
    signal: np.ndarray,
    *,
    signal_delay: int,
    envelope_delay: int,
    exponent: int,
    coefficient: complex,
) -> np.ndarray:
    return base_model.predict(signal) + coefficient * _gmp_residual_feature(
        signal,
        signal_delay=signal_delay,
        envelope_delay=envelope_delay,
        exponent=exponent,
    )


def _evaluate_direct_plus_gmp(
    context: Context,
    base_model: SparseSplineMemoryDPD,
    desired: np.ndarray,
    recipe: dict,
    coefficient: complex,
) -> dict:
    drive = _predict_direct_plus_gmp(
        base_model, desired, coefficient=coefficient, **recipe
    )
    output = context.pa.predict(drive)
    no_dpd = context.pa.predict(desired)
    ideal = context.gain * desired
    return {
        "nmse_db": nmse_db(output, ideal, COMMON_WARMUP),
        "no_dpd_nmse_db": nmse_db(no_dpd, ideal, COMMON_WARMUP),
        "drive_summary": signal_summary(drive, COMMON_WARMUP),
        "support_valid": bool(
            np.max(np.abs(drive)) <= context.maximum_pa_input
            * (1.0 + 64.0 * np.finfo(float).eps)
        ),
        "segment_improvements_db": segment_improvements(
            no_dpd, output, ideal
        ),
    }


def _gmp_residual_incremental_cost(
    base_model: SparseSplineMemoryDPD,
    recipe: dict,
) -> dict:
    exponent = int(recipe["exponent"])
    envelope_delay = int(recipe["envelope_delay"])
    maximum_delay = max(
        int(recipe["signal_delay"]), envelope_delay
    )
    existing_envelopes = {
        branch.envelope_delay for branch in base_model.branches
    }
    new_envelope = envelope_delay not in existing_envelopes
    return {
        "real_multiplications": 6 + max(exponent - 1, 0)
        + (3 if new_envelope else 0),
        "real_additions": 4 + (2 if new_envelope else 0),
        "nonlinear_operations": 1 if new_envelope else 0,
        "stored_real_coefficients": 2,
        "incremental_state_real_values": 2
        * max(maximum_delay - base_model.maximum_delay, 0),
        "notes": (
            "conservative explicit x_lag*|x_env|^p, complex coefficient, "
            "and accumulation schedule"
        ),
    }


def run_gmp_residual(context: Context) -> dict:
    started = time.perf_counter()
    direct_path = OUTPUT_DIR / "direct_refined_model.npz"
    if not direct_path.is_file():
        raise FileNotFoundError("run --stage direct before GMP residual")
    base_model = SparseSplineMemoryDPD.load(direct_path)
    fit_x = context.train_x[49_152:53_248]
    advisor_x = context.train_x[65_536:73_728]
    epsilon = 1e-4
    base_fit_drive = base_model.predict(fit_x)
    base_fit_output = context.pa.predict(base_fit_drive)
    fit_target = context.gain * fit_x
    fit_residual = fit_target[COMMON_WARMUP:] - base_fit_output[COMMON_WARMUP:]
    real_target = np.concatenate((fit_residual.real, fit_residual.imag))
    recipes = [
        {
            "signal_delay": signal_delay,
            "envelope_delay": envelope_delay,
            "exponent": exponent,
        }
        for signal_delay in range(0, 7)
        for envelope_delay in range(0, 7)
        for exponent in (1, 2, 3)
    ]
    ranked = []
    for recipe in recipes:
        feature = _gmp_residual_feature(fit_x, **recipe)
        real_output = context.pa.predict(base_fit_drive + epsilon * feature)
        imag_output = context.pa.predict(base_fit_drive + 1j * epsilon * feature)
        derivative_real = (
            np.asarray(real_output) - base_fit_output
        )[COMMON_WARMUP:] / epsilon
        derivative_imag = (
            np.asarray(imag_output) - base_fit_output
        )[COMMON_WARMUP:] / epsilon
        jacobian = np.column_stack(
            (
                np.concatenate((derivative_real.real, derivative_real.imag)),
                np.concatenate((derivative_imag.real, derivative_imag.imag)),
            )
        )
        normalization = np.sqrt(float(real_target.size))
        ridge = 1e-8
        regularizer = np.sqrt(ridge) * np.eye(2)
        delta = np.linalg.lstsq(
            np.vstack((jacobian / normalization, regularizer)),
            np.concatenate((real_target / normalization, np.zeros(2))),
            rcond=None,
        )[0]
        predicted_residual = real_target - jacobian @ delta
        predicted_gain = float(
            10.0
            * np.log10(
                np.mean(real_target**2)
                / np.mean(predicted_residual**2)
            )
        )
        ranked.append(
            {
                **recipe,
                "fit_linearized_gain_db": predicted_gain,
                "coefficient": complex(delta[0], delta[1]),
            }
        )
    ranked.sort(key=lambda row: row["fit_linearized_gain_db"], reverse=True)

    advisor_trials = []
    for candidate in ranked[:12]:
        recipe = {
            key: int(candidate[key])
            for key in ("signal_delay", "envelope_delay", "exponent")
        }
        for step in (0.5, 0.75, 1.0):
            coefficient = step * candidate["coefficient"]
            evaluation = _evaluate_direct_plus_gmp(
                context, base_model, advisor_x, recipe, coefficient
            )
            advisor_trials.append(
                {
                    **recipe,
                    "step": step,
                    "coefficient": coefficient,
                    "fit_linearized_gain_db": candidate[
                        "fit_linearized_gain_db"
                    ],
                    "advisor_train_nmse_db": evaluation["nmse_db"],
                    "support_valid": evaluation["support_valid"],
                }
            )
    valid = [row for row in advisor_trials if row["support_valid"]]
    selected = min(valid, key=lambda row: row["advisor_train_nmse_db"])
    selected_recipe = {
        key: int(selected[key])
        for key in ("signal_delay", "envelope_delay", "exponent")
    }
    validation = _evaluate_direct_plus_gmp(
        context,
        base_model,
        context.validation_x,
        selected_recipe,
        selected["coefficient"],
    )
    direct_validation = evaluate_model(
        context, base_model, context.validation_x
    )
    np.savez(
        OUTPUT_DIR / "direct_plus_gmp_residual.npz",
        signal_delay=np.asarray(selected_recipe["signal_delay"]),
        envelope_delay=np.asarray(selected_recipe["envelope_delay"]),
        exponent=np.asarray(selected_recipe["exponent"]),
        coefficient=np.asarray(selected["coefficient"]),
    )
    result = {
        "stage": "one_sparse_gmp_residual_branch",
        "base_model": "direct_refined_spline_memory",
        "selection": "top 12 fit-block linearized scores, step on disjoint train advisor",
        "validation_used_for_branch_or_step_selection": False,
        "recipe_count": len(recipes),
        "advisor_trial_count": len(advisor_trials),
        "selected": selected,
        "direct_validation": _compact_evaluation(direct_validation),
        "residual_validation": validation,
        "validation_gain_over_direct_refined_db": (
            direct_validation["nmse_db"] - validation["nmse_db"]
        ),
        "incremental_operation_count": _gmp_residual_incremental_cost(
            base_model, selected_recipe
        ),
        "top_fit_linearized_recipes": ranked[:12],
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "candidate is optimized through the same frozen PA evaluator; "
            "validation is only a final surrogate check"
        ),
    }
    _write_json(OUTPUT_DIR / "gmp_residual.json", result)
    return result


def _predict_direct_plus_fir(
    base_model: SparseSplineMemoryDPD,
    signal: np.ndarray,
    tail_coefficients: np.ndarray,
) -> np.ndarray:
    base_drive = np.asarray(base_model.predict(signal), dtype=np.complex128)
    output = base_drive.copy()
    for delay, coefficient in enumerate(tail_coefficients, start=1):
        output += coefficient * _delay(base_drive, delay)
    return output


def _evaluate_direct_plus_fir(
    context: Context,
    base_model: SparseSplineMemoryDPD,
    desired: np.ndarray,
    tail_coefficients: np.ndarray,
) -> dict:
    drive = _predict_direct_plus_fir(base_model, desired, tail_coefficients)
    output = context.pa.predict(drive)
    no_dpd = context.pa.predict(desired)
    ideal = context.gain * desired
    return {
        "nmse_db": nmse_db(output, ideal, COMMON_WARMUP),
        "no_dpd_nmse_db": nmse_db(no_dpd, ideal, COMMON_WARMUP),
        "drive_summary": signal_summary(drive, COMMON_WARMUP),
        "support_valid": bool(
            np.max(np.abs(drive)) <= context.maximum_pa_input
            * (1.0 + 64.0 * np.finfo(float).eps)
        ),
        "segment_improvements_db": segment_improvements(
            no_dpd, output, ideal
        ),
    }


def run_short_fir(context: Context) -> dict:
    started = time.perf_counter()
    direct_path = OUTPUT_DIR / "direct_refined_model.npz"
    if not direct_path.is_file():
        raise FileNotFoundError("run --stage direct before short FIR")
    base_model = SparseSplineMemoryDPD.load(direct_path)
    fit_x = context.train_x[49_152:53_248]
    advisor_x = context.train_x[65_536:73_728]
    base_fit_drive = base_model.predict(fit_x)
    base_fit_output = context.pa.predict(base_fit_drive)
    target = context.gain * fit_x
    residual = target[COMMON_WARMUP:] - base_fit_output[COMMON_WARMUP:]
    real_target = np.concatenate((residual.real, residual.imag))
    epsilon = 1e-4
    maximum_tail = 5
    real_columns = []
    imag_columns = []
    for delay in range(1, maximum_tail + 1):
        feature = _delay(base_fit_drive, delay)
        real_output = context.pa.predict(base_fit_drive + epsilon * feature)
        imag_output = context.pa.predict(base_fit_drive + 1j * epsilon * feature)
        real_derivative = (
            np.asarray(real_output) - base_fit_output
        )[COMMON_WARMUP:] / epsilon
        imag_derivative = (
            np.asarray(imag_output) - base_fit_output
        )[COMMON_WARMUP:] / epsilon
        real_columns.append(
            np.concatenate((real_derivative.real, real_derivative.imag))
        )
        imag_columns.append(
            np.concatenate((imag_derivative.real, imag_derivative.imag))
        )
    full_jacobian = np.column_stack(real_columns + imag_columns)
    normalization = np.sqrt(float(real_target.size))
    advisor_trials = []
    for tail_count in range(1, maximum_tail + 1):
        indices = list(range(tail_count)) + list(
            range(maximum_tail, maximum_tail + tail_count)
        )
        jacobian = full_jacobian[:, indices]
        for ridge in (1e-8, 1e-6, 1e-4):
            regularizer = np.sqrt(ridge) * np.eye(2 * tail_count)
            delta = np.linalg.lstsq(
                np.vstack((jacobian / normalization, regularizer)),
                np.concatenate(
                    ((real_target / normalization,
                      np.zeros(2 * tail_count, dtype=float)))
                ),
                rcond=None,
            )[0]
            coefficients = delta[:tail_count] + 1j * delta[tail_count:]
            for step in (0.5, 0.75, 1.0):
                stepped = step * coefficients
                evaluation = _evaluate_direct_plus_fir(
                    context, base_model, advisor_x, stepped
                )
                advisor_trials.append(
                    {
                        "fir_length": tail_count + 1,
                        "tail_count": tail_count,
                        "ridge": ridge,
                        "step": step,
                        "tail_coefficients": stepped,
                        "advisor_train_nmse_db": evaluation["nmse_db"],
                        "support_valid": evaluation["support_valid"],
                    }
                )
    valid = [row for row in advisor_trials if row["support_valid"]]
    selected = min(valid, key=lambda row: row["advisor_train_nmse_db"])
    validation = _evaluate_direct_plus_fir(
        context,
        base_model,
        context.validation_x,
        selected["tail_coefficients"],
    )
    direct_validation = evaluate_model(
        context, base_model, context.validation_x
    )
    tail_count = int(selected["tail_count"])
    np.savez(
        OUTPUT_DIR / "direct_plus_short_fir.npz",
        tail_coefficients=np.asarray(selected["tail_coefficients"]),
        fir_length=np.asarray(selected["fir_length"]),
    )
    selected_public = dict(selected)
    selected_public["tail_coefficients"] = [
        complex(value) for value in selected["tail_coefficients"]
    ]
    result = {
        "stage": "direct_refined_spline_plus_short_fir",
        "fir_form": "z = base_spline(x) + sum_{l=1}^{L-1} h_l*base_spline(x)[n-l]",
        "selection": "length/ridge/step selected on disjoint train advisor",
        "validation_used_for_selection": False,
        "advisor_trial_count": len(advisor_trials),
        "selected": selected_public,
        "direct_validation": _compact_evaluation(direct_validation),
        "fir_validation": validation,
        "validation_gain_over_direct_refined_db": (
            direct_validation["nmse_db"] - validation["nmse_db"]
        ),
        "incremental_operation_count": {
            "real_multiplications": 4 * tail_count,
            "real_additions": 4 * tail_count,
            "stored_real_coefficients": 2 * tail_count,
            "state_real_values": 2 * tail_count,
            "notes": "proper-complex FIR tail; h0 fixed to one",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "FIR is optimized through the same frozen PA evaluator and has "
            "not been evaluated on sealed test or physical PA"
        ),
    }
    _write_json(OUTPUT_DIR / "short_fir.json", result)
    return result


def _bounded_direct_update(
    context: Context,
    model: SparseSplineMemoryDPD,
    *,
    fit_start: int,
    advisor_start: int,
    epsilon: float = 1e-4,
) -> tuple[SparseSplineMemoryDPD, dict]:
    fit_x = context.train_x[fit_start:fit_start + 4096]
    advisor_x = context.train_x[advisor_start:advisor_start + 8192]
    design = spline_memory_design_matrix(
        fit_x, model.knots, model.branches
    )
    jacobian, _, base_output = _real_jacobian_forward_difference(
        context, design, model.coefficients, epsilon=epsilon
    )
    residual = (
        context.gain * fit_x[COMMON_WARMUP:]
        - base_output[COMMON_WARMUP:]
    )
    target = np.concatenate((residual.real, residual.imag))
    normalization = np.sqrt(float(target.size))
    ridge = 1e-8
    regularizer = np.sqrt(ridge) * np.eye(jacobian.shape[1])
    delta = np.linalg.lstsq(
        np.vstack((jacobian / normalization, regularizer)),
        np.concatenate((target / normalization, np.zeros(jacobian.shape[1]))),
        rcond=None,
    )[0]
    advisor_ideal = context.gain * advisor_x
    trials = []
    for step in (0.0, 0.25, 0.5, 0.75, 1.0):
        candidate = _model_with_delta(model, delta, step)
        drive = candidate.predict(advisor_x)
        output = context.pa.predict(drive)
        trials.append(
            {
                "step": step,
                "advisor_train_nmse_db": nmse_db(
                    output, advisor_ideal, COMMON_WARMUP
                ),
                "support_valid": bool(
                    np.max(np.abs(drive)) <= context.maximum_pa_input
                    * (1.0 + 64.0 * np.finfo(float).eps)
                ),
            }
        )
    selected = min(
        (row for row in trials if row["support_valid"]),
        key=lambda row: row["advisor_train_nmse_db"],
    )
    updated = _model_with_delta(model, delta, selected["step"])
    return updated, {
        "fit_range": [fit_start, fit_start + 4096],
        "advisor_range": [advisor_start, advisor_start + 8192],
        "epsilon": epsilon,
        "ridge": ridge,
        "selected": selected,
        "trials": trials,
    }


def _run_direct_schedule(
    context: Context,
    initial: SparseSplineMemoryDPD,
    schedule: tuple[tuple[int, int], ...],
) -> tuple[SparseSplineMemoryDPD, list[dict]]:
    model = initial
    records = []
    for iteration, (fit_start, advisor_start) in enumerate(schedule, start=1):
        model, update = _bounded_direct_update(
            context,
            model,
            fit_start=fit_start,
            advisor_start=advisor_start,
        )
        validation = evaluate_model(context, model, context.validation_x)
        records.append(
            {
                "iteration": iteration,
                **update,
                "validation_diagnostic_not_used_for_stopping": (
                    _compact_evaluation(validation)
                ),
            }
        )
    return model, records


def run_iterative_direct(context: Context) -> dict:
    started = time.perf_counter()
    first_update = SparseSplineMemoryDPD.load(
        OUTPUT_DIR / "direct_refined_model.npz"
    )
    schedules = {
        "predeclared_A": (
            (49_152, 65_536),
            (0, 8_192),
            (8_192, 24_576),
        ),
        "order_robustness_B": (
            (0, 8_192),
            (49_152, 65_536),
            (8_192, 24_576),
        ),
    }
    schedule_results = {}
    models = {}
    for name, schedule in schedules.items():
        model, records = _run_direct_schedule(
            context, first_update, schedule
        )
        validation = evaluate_model(context, model, context.validation_x)
        schedule_results[name] = {
            "schedule": [list(pair) for pair in schedule],
            "updates": records,
            "final_validation": _compact_evaluation(validation),
        }
        models[name] = model
    # Schedule A is frozen before looking at validation; B is robustness only.
    selected_name = "predeclared_A"
    selected_model = models[selected_name]
    selected_model.save(OUTPUT_DIR / "iterative_direct_model.npz")
    first_validation = evaluate_model(
        context, first_update, context.validation_x
    )
    final_validation = evaluate_model(
        context, selected_model, context.validation_x
    )
    result = {
        "stage": "iterative_bounded_direct_refinement",
        "initial_model": "one direct update selected on train-only advisor",
        "fixed_additional_iteration_count": 3,
        "validation_used_for_update_or_stopping": False,
        "selected_schedule": selected_name,
        "schedules": schedule_results,
        "one_update_validation": _compact_evaluation(first_validation),
        "final_validation": _compact_evaluation(final_validation),
        "validation_gain_over_one_update_db": (
            first_validation["nmse_db"] - final_validation["nmse_db"]
        ),
        "schedule_final_nmse_difference_db": abs(
            schedule_results["predeclared_A"]["final_validation"]["nmse_db"]
            - schedule_results["order_robustness_B"]["final_validation"]["nmse_db"]
        ),
        "operation_count_changed": False,
        "operation_count": context.baseline.operation_count().to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "all updates use the same frozen PA evaluator; schedule "
            "robustness does not establish physical-PA generalization"
        ),
    }
    _write_json(OUTPUT_DIR / "iterative_direct.json", result)
    return result


def run_knot_strategies(context: Context) -> dict:
    started = time.perf_counter()
    ila_input = context.train_y / context.gain
    recipes = [
        {"strategy": "uniform_amplitude", "compression_power": 2.0},
        {"strategy": "uniform_power", "compression_power": 2.0},
        {"strategy": "quantile", "compression_power": 2.0},
        {"strategy": "compression_aware", "compression_power": 1.5},
        {"strategy": "compression_aware", "compression_power": 2.0},
        {"strategy": "compression_aware", "compression_power": 3.0},
        {"strategy": "compression_aware", "compression_power": 4.0},
    ]
    trials = []
    models = []
    for recipe in recipes:
        model, diagnostics = fit_sparse_spline_memory_dpd(
            ila_input,
            context.train_x,
            branches=context.baseline.branches,
            knot_count=24,
            knot_strategy=recipe["strategy"],
            compression_power=recipe["compression_power"],
            ridge=1e-4,
        )
        ila_validation = evaluate_model(
            context, model, context.validation_x
        )
        direct_model, direct_update = _bounded_direct_update(
            context,
            model,
            fit_start=16_384,
            advisor_start=32_768,
        )
        direct_validation = evaluate_model(
            context, direct_model, context.validation_x
        )
        trials.append(
            {
                **recipe,
                "effective_knot_count": model.knot_count,
                "fit_diagnostics": asdict(diagnostics),
                "ila_validation": _compact_evaluation(ila_validation),
                "direct_update": direct_update,
                "direct_validation": _compact_evaluation(direct_validation),
                "operation_count": direct_model.operation_count().to_dict(),
            }
        )
        models.append(direct_model)
    selected_index = min(
        range(len(trials)),
        key=lambda index: trials[index]["direct_validation"]["nmse_db"],
    )
    selected = trials[selected_index]
    selected_model = models[selected_index]
    selected_model.save(OUTPUT_DIR / "knot_strategy_model.npz")
    iterative = json.loads(
        (OUTPUT_DIR / "iterative_direct.json").read_text(encoding="utf-8")
    )
    result = {
        "stage": "knot_placement_ablation",
        "fixed_topology": [asdict(branch) for branch in context.baseline.branches],
        "fixed_knot_count": 24,
        "fixed_ila_ridge": 1e-4,
        "direct_update_fit_and_advisor_are_train_only": True,
        "strategy_selected_on_validation": True,
        "trials": trials,
        "selected": selected,
        "gain_over_quantile_one_update_db": (
            next(
                row for row in trials
                if row["strategy"] == "quantile"
            )["direct_validation"]["nmse_db"]
            - selected["direct_validation"]["nmse_db"]
        ),
        "gain_over_iterative_quantile_db": (
            iterative["final_validation"]["nmse_db"]
            - selected["direct_validation"]["nmse_db"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "knot strategy is selected on the same validation capture; "
            "no sealed-test claim is permitted"
        ),
    }
    _write_json(OUTPUT_DIR / "knot_strategies.json", result)
    return result


def _candidate_drive(
    desired: np.ndarray,
    model: SparseSplineMemoryDPD,
    *,
    fir_tail: np.ndarray | None = None,
) -> np.ndarray:
    """Return a candidate DPD drive, optionally with a causal FIR tail."""

    drive = np.asarray(model.predict(desired), dtype=np.complex128)
    if fir_tail is None:
        return drive
    result = drive.copy()
    for delay, coefficient in enumerate(fir_tail, start=1):
        result += coefficient * _delay(drive, delay)
    return result


def run_alternative_evaluator(context: Context) -> dict:
    """Check DPD ranking on a separately fitted, lower-fidelity GMP model.

    This is deliberately labelled a weak robustness check.  The evaluator has
    a different GMP topology and is fitted from scratch, but it uses the same
    measured train/validation capture as the primary evaluator.
    """

    started = time.perf_counter()
    alternative_config = GMPConfig(
        ka=7,
        la=9,
        kb=2,
        lb=9,
        mb=2,
        kc=2,
        lc=9,
        mc=2,
        leading_policy="causal_leading",
    )
    alternative_pa, diagnostics = fit_gmp_pa(
        context.train_x,
        context.train_y,
        config=alternative_config,
        ridge=1e-10,
        segment_length=int(context.train_x.size),
        coefficient_dtype=np.complex128,
        solver_mode="ridge_lstsq",
        svd_rcond=None,
    )
    models = {
        "ila_spline_baseline": context.baseline,
        "one_direct_update": SparseSplineMemoryDPD.load(
            OUTPUT_DIR / "direct_refined_model.npz"
        ),
        "iterative_direct_refined_spline": SparseSplineMemoryDPD.load(
            OUTPUT_DIR / "iterative_direct_model.npz"
        ),
    }
    with np.load(OUTPUT_DIR / "direct_plus_short_fir.npz") as bundle:
        fir_tail = np.asarray(bundle["tail_coefficients"], dtype=np.complex128)
    ideal = context.gain * context.validation_x
    no_dpd_output = np.asarray(
        alternative_pa.predict(context.validation_x), dtype=np.complex128
    )
    no_dpd_nmse = nmse_db(no_dpd_output, ideal, COMMON_WARMUP)
    rows = []
    for name, model in models.items():
        drive = _candidate_drive(context.validation_x, model)
        output = np.asarray(alternative_pa.predict(drive), dtype=np.complex128)
        score = nmse_db(output, ideal, COMMON_WARMUP)
        rows.append(
            {
                "name": name,
                "nmse_db": score,
                "improvement_over_no_dpd_db": no_dpd_nmse - score,
                "drive_summary": signal_summary(drive, COMMON_WARMUP),
                "segment_improvements_db": segment_improvements(
                    no_dpd_output, output, ideal
                ),
            }
        )
    direct = models["one_direct_update"]
    fir_drive = _candidate_drive(
        context.validation_x, direct, fir_tail=fir_tail
    )
    fir_output = np.asarray(
        alternative_pa.predict(fir_drive), dtype=np.complex128
    )
    fir_score = nmse_db(fir_output, ideal, COMMON_WARMUP)
    rows.append(
        {
            "name": "direct_plus_short_fir",
            "nmse_db": fir_score,
            "improvement_over_no_dpd_db": no_dpd_nmse - fir_score,
            "drive_summary": signal_summary(fir_drive, COMMON_WARMUP),
            "segment_improvements_db": segment_improvements(
                no_dpd_output, fir_output, ideal
            ),
        }
    )
    rows.sort(key=lambda row: row["nmse_db"])
    measured_prediction = np.asarray(
        alternative_pa.predict(context.validation_x), dtype=np.complex128
    )
    result = {
        "stage": "alternative_pa_evaluator_ranking",
        "evaluator": {
            "name": "gmp_both_k2_l9_m2_ridge_1e-10_refit",
            "config": asdict(alternative_config),
            "fit_diagnostics": asdict(diagnostics),
            "measured_output_fidelity_nmse_db": nmse_db(
                measured_prediction, context.validation_y, COMMON_WARMUP
            ),
        },
        "no_dpd_nmse_db": no_dpd_nmse,
        "ranking": rows,
        "ranking_names_best_first": [row["name"] for row in rows],
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_level": "weak independent-topology robustness check",
        "claim_limit": (
            "the alternative evaluator uses the same measured capture and "
            "the same GMP model family; agreement cannot replace an "
            "independently trained neural evaluator or a physical PA test"
        ),
    }
    _write_json(OUTPUT_DIR / "alternative_evaluator.json", result)
    return result


def _fixed_streaming_check(
    evaluator: FixedPointSparseSplineMemoryDPD,
    signal: np.ndarray,
) -> bool:
    """Compare one-record execution with deterministic irregular chunks."""

    full = evaluator.predict_chunk(signal)
    state = evaluator.initial_state()
    pieces = []
    start = 0
    chunk_sizes = (1, 7, 31, 257, 1024, 4093)
    index = 0
    while start < signal.size:
        stop = min(signal.size, start + chunk_sizes[index % len(chunk_sizes)])
        chunk = evaluator.predict_chunk(signal[start:stop], state)
        pieces.append(chunk.output)
        state = chunk.next_state
        start = stop
        index += 1
    streamed = np.concatenate(pieces)
    return bool(
        np.array_equal(streamed, full.output)
        and np.array_equal(state.real_codes, full.next_state.real_codes)
        and np.array_equal(state.imag_codes, full.next_state.imag_codes)
    )


def run_fixed_point(context: Context) -> dict:
    """Evaluate 16/14/12-bit integer arithmetic for the frozen candidate."""

    started = time.perf_counter()
    model = SparseSplineMemoryDPD.load(
        OUTPUT_DIR / "iterative_direct_model.npz"
    )
    float_train_drive = model.predict(context.train_x)
    float_validation_drive = model.predict(context.validation_x)
    ideal = context.gain * context.validation_x
    float_output = np.asarray(
        context.pa.predict(float_validation_drive), dtype=np.complex128
    )
    float_nmse = nmse_db(float_output, ideal, COMMON_WARMUP)
    protocol = {
        "scale_guard_ratio": 1.001,
        "power_bits": 48,
        "accumulator_bits": 56,
        "scalar_accumulator_bits": 56,
        "interpolation_fraction_bits": 16,
    }
    formats = []
    for bits in (16, 14, 12):
        numeric_config, format_record = _make_fixed_config(
            model,
            bits=bits,
            input_peak=float(np.max(np.abs(context.train_x))),
            drive_peak=float(np.max(np.abs(float_train_drive))),
            protocol=protocol,
        )
        evaluator = FixedPointSparseSplineMemoryDPD(model, numeric_config)
        fixed_result = evaluator.predict_chunk(context.validation_x)
        fixed_drive = np.asarray(fixed_result.output, dtype=np.complex128)
        fixed_output = np.asarray(
            context.pa.predict(fixed_drive), dtype=np.complex128
        )
        fixed_nmse = nmse_db(fixed_output, ideal, COMMON_WARMUP)
        rotated = evaluator.predict(1j * context.validation_x)
        phase_nmse = nmse_db(
            rotated,
            1j * fixed_drive,
            int(model.maximum_delay),
        )
        formats.append(
            {
                "bits": bits,
                "formats": format_record,
                "cascade_nmse_db": fixed_nmse,
                "cascade_degradation_vs_float_db": fixed_nmse - float_nmse,
                "drive_nmse_vs_float_db": nmse_db(
                    fixed_drive,
                    float_validation_drive,
                    int(model.maximum_delay),
                ),
                "cascade_nmse_vs_float_db": nmse_db(
                    fixed_output, float_output, COMMON_WARMUP
                ),
                "drive_summary": signal_summary(fixed_drive, COMMON_WARMUP),
                "peak_change_vs_float_db": float(
                    20.0
                    * np.log10(
                        np.max(np.abs(fixed_drive[COMMON_WARMUP:]))
                        / np.max(
                            np.abs(float_validation_drive[COMMON_WARMUP:])
                        )
                    )
                ),
                "stats": fixed_result.stats.to_dict(),
                "coefficient_saturation_count": (
                    evaluator.coefficient_saturation_count
                ),
                "knot_code_collision_count": (
                    evaluator.knot_code_collision_count
                ),
                "maximum_knot_code_shift": evaluator.maximum_knot_code_shift,
                "streaming_chunk_equivalence": _fixed_streaming_check(
                    evaluator, context.validation_x
                ),
                "phase_equivariance_90_degree_nmse_db": phase_nmse,
            }
        )
    result = {
        "stage": "iterative_direct_fixed_point_validation",
        "model": "iterative_direct_model.npz",
        "scale_selection": (
            "input, coefficient and drive scales frozen from train only"
        ),
        "validation_used_for_numeric_scale_or_precision_selection": False,
        "protocol": protocol | {"activation_bits": [16, 14, 12]},
        "float_reference": {
            "cascade_nmse_db": float_nmse,
            "drive_summary": signal_summary(
                float_validation_drive, COMMON_WARMUP
            ),
        },
        "formats": formats,
        "all_streaming_checks_passed": bool(
            all(row["streaming_chunk_equivalence"] for row in formats)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_limit": (
            "bit-accurate Python integer reference through a float PA "
            "surrogate; not RTL/HLS timing or a physical-PA result"
        ),
    }
    _write_json(OUTPUT_DIR / "fixed_point.json", result)
    return result


def build_pareto_summary(context: Context) -> dict:
    direct = json.loads(
        (OUTPUT_DIR / "direct_refinement.json").read_text(encoding="utf-8")
    )
    cross = json.loads(
        (OUTPUT_DIR / "cross_memory.json").read_text(encoding="utf-8")
    )
    gmp = json.loads(
        (OUTPUT_DIR / "gmp_residual.json").read_text(encoding="utf-8")
    )
    fir = json.loads(
        (OUTPUT_DIR / "short_fir.json").read_text(encoding="utf-8")
    )
    iterative = json.loads(
        (OUTPUT_DIR / "iterative_direct.json").read_text(encoding="utf-8")
    )
    knots = json.loads(
        (OUTPUT_DIR / "knot_strategies.json").read_text(encoding="utf-8")
    )
    baseline_cost = context.baseline.operation_count().to_dict()
    no_dpd_nmse = direct["baseline_validation"]["no_dpd_nmse_db"]
    points = [
        {
            "name": "no_dpd",
            "nmse_db": no_dpd_nmse,
            "real_multiplications": 0,
            "real_additions": 0,
            "status": "reference",
        },
        {
            "name": "ila_spline_baseline",
            "nmse_db": direct["baseline_validation"]["nmse_db"],
            "real_multiplications": baseline_cost["real_multiplications"],
            "real_additions": baseline_cost["real_additions"],
            "status": "dominated_by_direct_refinement",
        },
        {
            "name": "direct_refined_spline",
            "nmse_db": direct["refined_validation"]["nmse_db"],
            "real_multiplications": baseline_cost["real_multiplications"],
            "real_additions": baseline_cost["real_additions"],
            "status": "dominated_by_iterative_direct_at_same_inference_cost",
        },
        {
            "name": "iterative_direct_refined_spline",
            "nmse_db": iterative["final_validation"]["nmse_db"],
            "real_multiplications": baseline_cost["real_multiplications"],
            "real_additions": baseline_cost["real_additions"],
            "status": "recommended_fast_path",
        },
        {
            "name": "one_cross_memory_spline_branch",
            "nmse_db": cross["selected"]["validation"]["nmse_db"],
            "real_multiplications": cross["selected"]["operation_count"][
                "real_multiplications"
            ],
            "real_additions": cross["selected"]["operation_count"][
                "real_additions"
            ],
            "status": "rejected_worse_than_baseline",
        },
        {
            "name": "compression_aware_knots_one_direct_update",
            "nmse_db": knots["selected"]["direct_validation"]["nmse_db"],
            "real_multiplications": baseline_cost["real_multiplications"],
            "real_additions": baseline_cost["real_additions"],
            "status": "dominated_by_iterative_quantile_direct",
        },
        {
            "name": "direct_plus_one_gmp_residual",
            "nmse_db": gmp["residual_validation"]["nmse_db"],
            "real_multiplications": baseline_cost["real_multiplications"]
            + gmp["incremental_operation_count"]["real_multiplications"],
            "real_additions": baseline_cost["real_additions"]
            + gmp["incremental_operation_count"]["real_additions"],
            "status": "dominated_by_iterative_direct",
        },
        {
            "name": "direct_plus_short_fir",
            "nmse_db": fir["fir_validation"]["nmse_db"],
            "real_multiplications": baseline_cost["real_multiplications"]
            + fir["incremental_operation_count"]["real_multiplications"],
            "real_additions": baseline_cost["real_additions"]
            + fir["incremental_operation_count"]["real_additions"],
            "status": "optional_subthreshold_gain_over_iterative_direct",
        },
    ]
    # Pareto dominance in NMSE/MUL/ADD only; other hard gates remain explicit.
    frontier = []
    for point in points:
        dominated = any(
            other is not point
            and other["nmse_db"] <= point["nmse_db"]
            and other["real_multiplications"] <= point["real_multiplications"]
            and other["real_additions"] <= point["real_additions"]
            and (
                other["nmse_db"] < point["nmse_db"]
                or other["real_multiplications"] < point["real_multiplications"]
                or other["real_additions"] < point["real_additions"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point["name"])
    fir_gain_over_iterative = float(
        iterative["final_validation"]["nmse_db"]
        - fir["fir_validation"]["nmse_db"]
    )
    recommended = (
        "direct_plus_short_fir"
        if fir_gain_over_iterative >= 0.5
        and fir["fir_validation"]["support_valid"]
        and min(fir["fir_validation"]["segment_improvements_db"])
        >= min(iterative["final_validation"]["segment_improvements_db"])
        else "iterative_direct_refined_spline"
    )
    result = {
        "scope": "BlackBox frozen-PA surrogate, validation-only",
        "points": points,
        "nmse_mul_add_frontier": frontier,
        "internal_extension_gate_db": 0.5,
        "fir_gain_over_iterative_direct_db": fir_gain_over_iterative,
        "recommended_for_next_independent_test": recommended,
        "not_proven": [
            "physical PA improvement",
            "sealed-test generalization",
            "independent-evaluator ranking agreement",
        ],
    }
    _write_json(OUTPUT_DIR / "pareto_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "residual",
            "direct",
            "direct-robustness",
            "cross",
            "gmp-residual",
            "fir",
            "direct-iterative",
            "knots",
            "alternative-evaluator",
            "fixed-point",
        ),
    )
    args = parser.parse_args(argv)
    context = load_context()
    baseline = evaluate_model(context, context.baseline, context.validation_x)
    print(
        f"baseline: {baseline['nmse_db']:.6f} dB; "
        f"improvement vs no DPD {baseline['improvement_over_no_dpd_db']:.6f} dB"
    )
    if args.stage == "residual":
        result = run_residual_analysis(context)
        print("top branch scores:")
        for row in result["top_cross_memory_branch_scores"][:8]:
            print(row)
        print("slow-state diagnostics:", result["slow_state_diagnostics"])
    elif args.stage == "direct":
        result = run_direct_refinement(context)
        print("selected:", result["selected_on_disjoint_train_advisor"])
        print(
            "validation baseline/refined:",
            result["baseline_validation"]["nmse_db"],
            result["refined_validation"]["nmse_db"],
        )
        print(
            "validation gain over baseline DPD:",
            result["validation_gain_over_baseline_dpd_db"],
            "dB",
        )
    elif args.stage == "direct-robustness":
        result = run_direct_robustness(context)
        print("validation gain summary:", result["validation_gain_summary_db"])
        for case in result["cases"]:
            print(
                case["fit_range"],
                "epsilon", case["epsilon"],
                "gain", case["validation_gain_over_baseline_dpd_db"],
                "NMSE", case["validation"]["nmse_db"],
            )
    elif args.stage == "cross":
        result = run_cross_memory(context)
        print("selected:", result["selected"]["added_branch"],
              "ridge", result["selected"]["ridge"])
        print("validation NMSE:", result["selected"]["validation"]["nmse_db"])
        print("gain over baseline DPD:", result["gain_over_baseline_dpd_db"])
        print("gain over direct refined:", result["gain_over_direct_refined_db"])
    elif args.stage == "gmp-residual":
        result = run_gmp_residual(context)
        print("selected:", result["selected"])
        print("direct NMSE:", result["direct_validation"]["nmse_db"])
        print("direct + GMP residual NMSE:", result["residual_validation"]["nmse_db"])
        print("gain over direct:", result["validation_gain_over_direct_refined_db"])
        print("incremental cost:", result["incremental_operation_count"])
    elif args.stage == "fir":
        result = run_short_fir(context)
        print("selected:", result["selected"])
        print("direct NMSE:", result["direct_validation"]["nmse_db"])
        print("direct + FIR NMSE:", result["fir_validation"]["nmse_db"])
        print("gain over direct:", result["validation_gain_over_direct_refined_db"])
        print("incremental cost:", result["incremental_operation_count"])
        pareto = build_pareto_summary(context)
        print("Pareto frontier:", pareto["nmse_mul_add_frontier"])
        print("recommended:", pareto["recommended_for_next_independent_test"])
    elif args.stage == "direct-iterative":
        result = run_iterative_direct(context)
        print("one update NMSE:", result["one_update_validation"]["nmse_db"])
        print("iterative NMSE:", result["final_validation"]["nmse_db"])
        print("gain over one update:", result["validation_gain_over_one_update_db"])
        print("schedule difference:", result["schedule_final_nmse_difference_db"])
        for name, schedule in result["schedules"].items():
            steps = [row["selected"]["step"] for row in schedule["updates"]]
            print(name, schedule["final_validation"]["nmse_db"], "steps", steps)
    elif args.stage == "knots":
        result = run_knot_strategies(context)
        for row in result["trials"]:
            print(
                row["strategy"], row["compression_power"],
                "ILA", row["ila_validation"]["nmse_db"],
                "direct", row["direct_validation"]["nmse_db"],
            )
        print("selected:", result["selected"]["strategy"],
              result["selected"]["compression_power"])
        print("gain over quantile one update:",
              result["gain_over_quantile_one_update_db"])
        print("gain over iterative quantile:",
              result["gain_over_iterative_quantile_db"])
    elif args.stage == "alternative-evaluator":
        result = run_alternative_evaluator(context)
        print(
            "alternative evaluator fidelity:",
            result["evaluator"]["measured_output_fidelity_nmse_db"],
            "dB",
        )
        for row in result["ranking"]:
            print(
                row["name"], row["nmse_db"],
                "improvement", row["improvement_over_no_dpd_db"],
            )
    else:
        result = run_fixed_point(context)
        print("float NMSE:", result["float_reference"]["cascade_nmse_db"])
        for row in result["formats"]:
            print(
                row["bits"], "bit:", row["cascade_nmse_db"],
                "degradation", row["cascade_degradation_vs_float_db"],
                "streaming", row["streaming_chunk_equivalence"],
                "stats", row["stats"],
            )
    print(f"elapsed: {result['elapsed_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
