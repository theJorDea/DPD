"""Evaluate frozen DPD coefficients through a second PA surrogate family.

The alternate surrogate is fit on the training split only and is never used to
select the DPD.  This is a bias/sensitivity check, not a replacement for a
physical PA measurement.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.evaluate_spline import _load_artifacts, _verify_dataset_identity  # noqa: E402
from baseline.pa_models import fit_memory_polynomial_pa  # noqa: E402
from baseline.train_spline import (  # noqa: E402
    _paired_time_metrics,
    align_split_pair,
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.complexfloating, complex)):
        value = complex(value)
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def run(
    dataset: Path,
    training_report: Path,
    model_paths: tuple[Path, ...],
    output: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    training_report = training_report.resolve()
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    report, _, gain = _load_artifacts(
        training_report,
        allow_artifact_mismatch=False,
    )
    identity = _verify_dataset_identity(
        report,
        dataset,
        allow_mismatch=False,
    )
    spec = load_dataset_spec(dataset)
    segment_length = int(spec["nperseg"])
    alignment_delay = int(
        report["alignment"]["frozen_integer_delay_samples"]
    )
    train_input, train_output = load_split_pair(dataset, "train")
    val_input, val_output = load_split_pair(dataset, "val")
    test_input, test_output = load_split_pair(dataset, "test")
    train_input, train_output = align_split_pair(
        train_input,
        train_output,
        delay=alignment_delay,
    )
    val_input, val_output = align_split_pair(
        val_input,
        val_output,
        delay=alignment_delay,
    )
    test_input, test_output = align_split_pair(
        test_input,
        test_output,
        delay=alignment_delay,
    )

    started = time.perf_counter()
    surrogate, diagnostics = fit_memory_polynomial_pa(
        train_input,
        train_output,
        orders=(1, 3, 5, 7, 9, 11),
        delays=(0, 1, 2, 3, 4, 5, 6),
        ridge=1e-8,
        segment_length=segment_length,
    )
    fit_seconds = time.perf_counter() - started
    surrogate_path = output.with_name(output.stem + "_pa_surrogate.npz")
    surrogate.save(surrogate_path)

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "cross_surrogate_sensitivity",
        "scope": {
            "physical_pa_result": False,
            "alternate_surrogate_fit_split": "train",
            "dpd_selection_split": "validation on the primary surrogate",
            "test_used_for_selection": False,
        },
        "dataset": {
            "directory": dataset,
            "identity_check": identity,
            "spec": spec,
            "spec_sha256": file_sha256(dataset / "spec.json"),
        },
        "training_report": training_report,
        "alternate_surrogate": {
            "path": surrogate_path,
            "sha256": file_sha256(surrogate_path),
            "orders": surrogate.orders,
            "delays": surrogate.delays,
            "segment_length": segment_length,
            "fit_seconds": fit_seconds,
            "diagnostics": diagnostics,
        },
        "alignment": {
            "frozen_integer_delay_samples": alignment_delay,
            "test_delay_retuned": False,
        },
        "models": {},
    }
    validation_fidelity = surrogate.predict_segments(val_input, segment_length)
    test_fidelity = surrogate.predict_segments(test_input, segment_length)
    for path in model_paths:
        path = path.resolve()
        # Memoryless and sparse-memory NPZs intentionally expose different
        # loaders; inspect the model_type field without allowing pickle.
        with np.load(path, allow_pickle=False) as archive:
            if "model_type" in archive:
                model_type = str(archive["model_type"])
            elif "signal_delays" in archive:
                model_type = "phase_equivariant_sparse_spline_memory_dpd"
            elif "coefficients" in archive:
                model_type = "complex_linear_spline_dpd"
            else:
                raise ValueError(f"cannot infer model type in {path}")
        if model_type == "complex_linear_spline_dpd":
            from baseline.complex_spline_dpd import ComplexLinearSplineDPD

            model = ComplexLinearSplineDPD.load(path)
        elif model_type == "phase_equivariant_sparse_spline_memory_memory_dpd":
            raise ValueError("unexpected legacy sparse-memory model type")
        elif model_type == "phase_equivariant_sparse_spline_memory_dpd":
            from baseline.spline_memory_dpd import SparseSplineMemoryDPD

            model = SparseSplineMemoryDPD.load(path)
        else:
            raise ValueError(f"unsupported DPD model type in {path}: {model_type}")
        if hasattr(model, "predict_segments"):
            val_predistorted = model.predict_segments(val_input, segment_length)
            test_predistorted = model.predict_segments(test_input, segment_length)
            max_delay = int(getattr(model, "maximum_delay", 0))
        else:
            val_predistorted = model.predict(val_input)
            test_predistorted = model.predict(test_input)
            max_delay = 0
        val_cascade = surrogate.predict_segments(
            val_predistorted,
            segment_length,
        )
        test_cascade = surrogate.predict_segments(
            test_predistorted,
            segment_length,
        )
        key = path.stem
        result["models"][key] = {
            "path": path,
            "sha256": file_sha256(path),
            "model_type": model_type,
            "validation_surrogate_fidelity": _paired_time_metrics(
                validation_fidelity,
                val_output,
                warmup_samples=surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
            "test_surrogate_fidelity": _paired_time_metrics(
                test_fidelity,
                test_output,
                warmup_samples=surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
            "validation_cascade_vs_ideal": _paired_time_metrics(
                val_cascade,
                gain * val_input,
                warmup_samples=surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
            "test_cascade_vs_ideal": _paired_time_metrics(
                test_cascade,
                gain * test_input,
                warmup_samples=surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
            "dpd_warmup_samples": max_delay,
        }
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.dataset,
        args.training_report,
        tuple(args.model),
        args.output_json,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
