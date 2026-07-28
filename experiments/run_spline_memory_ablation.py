"""Validation-only ablation for sparse phase-equivariant spline-memory DPD.

The script intentionally keeps model selection separate from test evaluation:
all branch/K/ridge candidates are fit on train and ranked on validation through
a train-fitted memory-polynomial PA surrogate.  The test split is opened only
after the selected candidate for each branch family has been frozen.

Results are surrogate-only.  A physical PA or an independently frozen OpenDPD
neural checkpoint is required before making a superiority claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.alignment import (  # noqa: E402
    align_and_estimate_gain,
    fractional_delay_diagnostic,
)
from baseline.metrics import (  # noqa: E402
    nmse_opendpd_db,
    opendpd_aclr_db,
    opendpd_spectral_evm_db,
    papr_db,
    peak_amplitude,
)
from baseline.pa_models import (  # noqa: E402
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
)
from baseline.spline_memory_dpd import (  # noqa: E402
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    fit_ila_sparse_spline_memory_dpd,
)
from baseline.train_spline import (  # noqa: E402
    _paired_time_metrics,
    _waveform_output_metrics,
    align_split_pair,
    file_sha256,
    gain_from_training,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


def _parse_ints(text: str, minimum: int) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value < minimum for value in values):
        raise argparse.ArgumentTypeError(
            f"values must be comma-separated integers >= {minimum}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def _parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(not np.isfinite(value) or value < 0 for value in values):
        raise argparse.ArgumentTypeError(
            "values must be finite, comma-separated, and non-negative"
        )
    return values


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {"real": number.real, "imag": number.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def _fit_and_score(
    model: SparseSplineMemoryDPD,
    validation_input: np.ndarray,
    validation_output: np.ndarray,
    surrogate: MemoryPolynomialPA,
    gain: complex,
    segment_length: int,
    surrogate_training_peak: float,
) -> dict[str, Any]:
    inverse_input = validation_output / gain
    inverse_estimate = model.predict_segments(inverse_input, segment_length)
    inverse_metrics = _paired_time_metrics(
        inverse_estimate,
        validation_input,
        warmup_samples=model.maximum_delay,
        segment_length=segment_length,
    )
    predistorted = model.predict_segments(validation_input, segment_length)
    cascade_output = surrogate.predict_segments(predistorted, segment_length)
    cascade_metrics = _paired_time_metrics(
        cascade_output,
        gain * validation_input,
        warmup_samples=surrogate.causal_warmup_samples,
        segment_length=segment_length,
    )
    cascade_metrics.update(
        {
            "scope": "surrogate_only",
            "predistorted_waveform": _waveform_output_metrics(predistorted),
            "spline_input_extrapolation_fraction": float(
                np.mean(np.abs(validation_input) > model.knots[-1])
            ),
            "surrogate_training_range_extrapolation_fraction": float(
                np.mean(
                    np.abs(predistorted)
                    > surrogate_training_peak
                )
            ),
        }
    )
    return {
        "inverse_postdistorter_diagnostic": inverse_metrics,
        "surrogate_cascade": cascade_metrics,
    }


def _repository_spectral_metrics(
    signal: np.ndarray,
    reference: np.ndarray,
    spec: dict[str, Any],
) -> dict[str, Any]:
    required = ("input_signal_fs", "bw_main_ch", "n_sub_ch", "nperseg")
    if any(key not in spec for key in required):
        return {"available": False}
    nperseg = int(spec["nperseg"])
    if signal.size % nperseg or reference.size % nperseg:
        return {"available": False, "reason": "length not divisible by nperseg"}
    aclr = opendpd_aclr_db(
        signal,
        fs=float(spec["input_signal_fs"]),
        nperseg=nperseg,
        bandwidth_main=float(spec["bw_main_ch"]),
        n_subchannels=int(spec["n_sub_ch"]),
    )
    return {
        "available": True,
        "nmse_opendpd_mean_segment_db": nmse_opendpd_db(
            signal.reshape(-1, nperseg),
            reference.reshape(-1, nperseg),
        ),
        "opendpd_spectral_evm_db": opendpd_spectral_evm_db(
            signal,
            reference,
            fs=float(spec["input_signal_fs"]),
            bandwidth_main=float(spec["bw_main_ch"]),
            n_subchannels=int(spec["n_sub_ch"]),
            nperseg=nperseg,
        ),
        "opendpd_aclr_db": {
            "left": aclr.left_db,
            "right": aclr.right_db,
            "average": aclr.average_db,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "memory_ablation_report.json"
    trials_path = output / "validation_trials.json"
    if any(path.exists() for path in (report_path, trials_path)) and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing memory-ablation artifacts; use --overwrite"
        )

    raw_train_input, raw_train_output = load_split_pair(dataset, "train")
    raw_val_input, raw_val_output = load_split_pair(dataset, "val")
    spec = load_dataset_spec(dataset)
    segment_length = int(spec.get("nperseg", 0)) or None
    if segment_length is None:
        raise ValueError("dataset spec must provide positive nperseg")
    _, _, delay, _ = align_and_estimate_gain(
        raw_train_input,
        raw_train_output,
        max_abs_delay=args.alignment_max_delay,
    )
    fractional = fractional_delay_diagnostic(
        raw_train_input,
        raw_train_output,
        args.alignment_max_delay,
    )
    train_input, train_output = align_split_pair(
        raw_train_input,
        raw_train_output,
        delay=delay,
    )
    validation_input, validation_output = align_split_pair(
        raw_val_input,
        raw_val_output,
        delay=delay,
    )
    gain, gain_definition = gain_from_training(
        train_input,
        train_output,
        strategy=args.gain_strategy,
    )
    surrogate_started = time.perf_counter()
    surrogate, surrogate_diagnostics = fit_memory_polynomial_pa(
        train_input,
        train_output,
        orders=(1, 3, 5, 7, 9),
        delays=(0, 1, 2, 3, 4),
        ridge=1e-8,
        segment_length=segment_length,
    )
    surrogate_fit_seconds = time.perf_counter() - surrogate_started
    surrogate_path = output / "pa_surrogate.npz"
    surrogate.save(surrogate_path)

    branch_families: dict[str, tuple[SplineMemoryBranch, ...]] = {
        "memoryless": (SplineMemoryBranch(0, 0),),
        "signal_delay_01": (
            SplineMemoryBranch(0, 0),
            SplineMemoryBranch(1, 0),
        ),
        "signal_delay_012": (
            SplineMemoryBranch(0, 0),
            SplineMemoryBranch(1, 0),
            SplineMemoryBranch(2, 0),
        ),
        "envelope_delay_01": (
            SplineMemoryBranch(0, 0),
            SplineMemoryBranch(0, 1),
        ),
        "mixed_signal_envelope": (
            SplineMemoryBranch(0, 0),
            SplineMemoryBranch(1, 0),
            SplineMemoryBranch(1, 1),
        ),
    }
    trials: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    models: dict[str, SparseSplineMemoryDPD] = {}
    for family, branches in branch_families.items():
        best_key: tuple[float, int, float] | None = None
        best_trial: dict[str, Any] | None = None
        best_model: SparseSplineMemoryDPD | None = None
        for knot_count in args.knot_counts:
            for ridge in args.ridges:
                started = time.perf_counter()
                model, diagnostics = fit_ila_sparse_spline_memory_dpd(
                    train_input,
                    train_output,
                    gain,
                    branches=branches,
                    knot_count=knot_count,
                    knot_strategy="quantile",
                    ridge=ridge,
                )
                fit_seconds = time.perf_counter() - started
                scores = _fit_and_score(
                    model,
                    validation_input,
                    validation_output,
                    surrogate,
                    gain,
                    segment_length,
                    surrogate_diagnostics.maximum_training_input_amplitude,
                )
                raw_score = scores["surrogate_cascade"][
                    "complex_nmse_pooled_db"
                ]
                valid = bool(
                    diagnostics.solver_rank == diagnostics.feature_count
                    and np.isfinite(raw_score)
                )
                score = float(raw_score) if valid else np.inf
                trial = {
                    "family": family,
                    "branches": [dataclasses.asdict(branch) for branch in branches],
                    "requested_knot_count": knot_count,
                    "effective_knot_count": model.knot_count,
                    "ridge": ridge,
                    "fit_seconds": fit_seconds,
                    "fit_diagnostics": diagnostics,
                    "validation": scores,
                    "valid_for_selection": valid,
                    "selection_score_db": score,
                }
                trials.append(trial)
                write_json(
                    trials_path,
                    {
                        "schema_version": 1,
                        "dataset": dataset,
                        "split": "validation",
                        "test_accessed": False,
                        "completed_trials": trials,
                    },
                )
                key = (score, knot_count, ridge)
                if valid and (best_key is None or key < best_key):
                    best_key = key
                    best_trial = trial
                    best_model = model
        if best_trial is None or best_model is None:
            raise RuntimeError(f"no valid candidate for branch family {family}")
        model_path = output / f"{family}.npz"
        best_model.save(model_path)
        selected[family] = {
            "trial": best_trial,
            "model_path": model_path,
            "model_sha256": file_sha256(model_path),
        }
        models[family] = best_model

    # Test is intentionally opened only after every family has been frozen.
    raw_test_input, raw_test_output = load_split_pair(dataset, "test")
    test_input, test_output = align_split_pair(
        raw_test_input,
        raw_test_output,
        delay=delay,
    )
    test_results: dict[str, Any] = {}
    for family, model in models.items():
        predistorted = model.predict_segments(test_input, segment_length)
        cascade = surrogate.predict_segments(predistorted, segment_length)
        no_dpd = surrogate.predict_segments(test_input, segment_length)
        test_results[family] = {
            "scope": "surrogate_only",
            "inverse_postdistorter_diagnostic": _paired_time_metrics(
                model.predict_segments(test_output / gain, segment_length),
                test_input,
                warmup_samples=model.maximum_delay,
                segment_length=segment_length,
            ),
            "surrogate_fidelity": _paired_time_metrics(
                no_dpd,
                test_output,
                warmup_samples=surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
            "without_dpd_vs_ideal": _paired_time_metrics(
                no_dpd,
                gain * test_input,
                warmup_samples=surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
            "with_dpd_vs_ideal": _paired_time_metrics(
                cascade,
                gain * test_input,
                warmup_samples=surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
            "predistorted_waveform": {
                "peak_amplitude": peak_amplitude(predistorted),
                "papr_db": papr_db(predistorted),
            },
            "spectral_metrics": _repository_spectral_metrics(
                cascade,
                gain * test_input,
                spec,
            ),
            "operations": model.operation_count().to_dict(),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "sparse_spline_memory_ablation",
        "claims_scope": {
            "physical_pa_result": False,
            "surrogate_only": True,
            "test_used_for_selection": False,
        },
        "command": [sys.executable, *sys.argv],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "directory": dataset,
            "spec": spec,
            "spec_sha256": (
                file_sha256(dataset / "spec.json")
                if (dataset / "spec.json").is_file()
                else None
            ),
            "raw_train_samples": raw_train_input.size,
            "raw_validation_samples": raw_val_input.size,
            "raw_test_samples": raw_test_input.size,
            "train_validation_hashes": {
                name: file_sha256(dataset / f"{name}.csv")
                for name in (
                    "train_input",
                    "train_output",
                    "val_input",
                    "val_output",
                )
            },
            "test_hashes": {
                "test_input": file_sha256(dataset / "test_input.csv"),
                "test_output": file_sha256(dataset / "test_output.csv"),
            },
        },
        "alignment": {
            "frozen_integer_delay_samples": delay,
            "fractional_delay_diagnostic": fractional,
            "fractional_delay_applied": False,
            "test_delay_retuned": False,
        },
        "target_gain": {
            "strategy": args.gain_strategy,
            "value": gain,
            "definition": gain_definition,
        },
        "framing": {
            "segment_length": segment_length,
            "state_reset": "zero at every segment boundary",
        },
        "pa_surrogate": {
            "path": surrogate_path,
            "sha256": file_sha256(surrogate_path),
            "metadata": surrogate.metadata,
            "diagnostics": surrogate_diagnostics,
            "fit_seconds": surrogate_fit_seconds,
        },
        "selection": {
            "split": "validation",
            "metric": "surrogate_cascade.complex_nmse_pooled_db",
            "lower_db_is_better": True,
            "candidate_count": len(trials),
            "families": {
                family: selected[family]["trial"]["selection_score_db"]
                for family in selected
            },
        },
        "validation_trials": trials,
        "selected": selected,
        "test_results": test_results,
        "artifacts": {
            "report": report_path,
            "validation_trials": trials_path,
        },
    }
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knot-counts", type=lambda x: _parse_ints(x, 2), default=(8, 16, 24, 32))
    parser.add_argument("--ridges", type=_parse_floats, default=(1e-8, 1e-6, 1e-4))
    parser.add_argument("--gain-strategy", choices=("complex_ls", "opendpd_peak"), default="complex_ls")
    parser.add_argument("--alignment-max-delay", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
