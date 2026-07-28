"""Generate train-OOF and validation residual diagnostics for a frozen MP PA.

The selected architecture/ridge is read from a validation-selection manifest.
For discovery, coefficients are refit in leave-one-explicit-frame-out folds
inside the training split.  The validation residual is reported separately,
but it is not an independent confirmation dataset because validation already
selected the MP configuration.  Test files are never opened by this command.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from baseline.pa_models import (
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
    segmented_steady_state_mask,
)
from baseline.residual_analysis import (
    ResidualAnalysisSpec,
    analyze_pa_residuals,
    freeze_residual_reference,
)
from baseline.train_spline import (
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    value = _load_json_object(path, name="residual config")
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("residual config schema_version must equal 1")
    required = {"selection_manifest", "output_dir", "lag_grid"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"residual config is missing keys: {sorted(missing)}")
    return value


def _verify_file(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )


def _parse_lag_grid(value: Any) -> tuple[int, ...]:
    if not isinstance(value, dict):
        raise ValueError("lag_grid must be a JSON object")
    start = int(value["start"])
    stop = int(value["stop"])
    step = int(value.get("step", 1))
    if step <= 0 or stop < start:
        raise ValueError("lag_grid requires positive step and stop >= start")
    lags = tuple(range(start, stop + 1, step))
    if not lags:
        raise ValueError("lag_grid generated no lags")
    return lags


def _number_tuple(
    values: Any,
    *,
    name: str,
    integer: bool,
) -> tuple[int, ...] | tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON list")
    if integer:
        result = tuple(int(value) for value in values)
        if any(value < 0 for value in result):
            raise ValueError(f"{name} must be non-negative")
    else:
        result = tuple(float(value) for value in values)
        if any(not np.isfinite(value) or value <= 0.0 for value in result):
            raise ValueError(f"{name} must be positive and finite")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _build_spec(
    config: dict[str, Any],
    dataset_spec: dict[str, Any],
) -> ResidualAnalysisSpec:
    return ResidualAnalysisSpec(
        sample_rate_hz=float(dataset_spec["input_signal_fs"]),
        psd_nperseg=int(dataset_spec["nperseg"]),
        main_bandwidth_hz=float(dataset_spec["bw_main_ch"]),
        adjacent_bandwidth_hz=float(dataset_spec["bw_sub_ch"]),
        lags=_parse_lag_grid(config["lag_grid"]),
        envelope_lags=_number_tuple(
            config.get(
                "envelope_lags",
                [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
            ),
            name="envelope_lags",
            integer=True,
        ),
        envelope_powers=tuple(
            int(value)
            for value in _number_tuple(
                config.get("envelope_powers", [1, 2, 3]),
                name="envelope_powers",
                integer=True,
            )
        ),
        slow_time_constants_samples=_number_tuple(
            config.get(
                "slow_time_constants_samples",
                [4.0, 16.0, 64.0, 256.0, 1024.0],
            ),
            name="slow_time_constants_samples",
            integer=False,
        ),
        amplitude_quantiles=tuple(
            float(value)
            for value in config.get("amplitude_quantiles", [0.90, 0.95, 0.99])
        ),
        characteristic_bins=int(config.get("characteristic_bins", 32)),
        position_bins=int(config.get("position_bins", 10)),
        amplitude_floor_fraction=float(
            config.get("amplitude_floor_fraction", 1e-6)
        ),
        minimum_time_constants_per_segment=float(
            config.get("minimum_time_constants_per_segment", 20.0)
        ),
        independent_capture_count=int(
            config.get("independent_capture_count", 0)
        ),
    )


def explicit_frame_ids(sample_count: int, nperseg: int) -> np.ndarray:
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if not isinstance(nperseg, int) or nperseg <= 0:
        raise ValueError("nperseg must be a positive integer")
    return np.arange(sample_count, dtype=int) // nperseg


def leave_one_frame_out_predictions(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    *,
    segment_id: np.ndarray,
    orders: tuple[int, ...],
    delays: tuple[int, ...],
    ridge: float,
    nperseg: int,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Fit on all other explicit frames and predict each held-out frame."""

    x = np.asarray(pa_input, dtype=np.complex128)
    y = np.asarray(measured_output, dtype=np.complex128)
    segments = np.asarray(segment_id)
    if x.ndim != 1 or y.ndim != 1 or segments.ndim != 1:
        raise ValueError("OOF input, output, and segment_id must be 1-D")
    if x.shape != y.shape or x.shape != segments.shape:
        raise ValueError("OOF arrays must have equal length")
    unique_segments = np.unique(segments)
    if unique_segments.size < 2:
        raise ValueError("OOF residuals require at least two explicit frames")

    prediction = np.empty(x.shape, dtype=np.complex128)
    fold_reports: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    for held_segment in unique_segments:
        held = segments == held_segment
        fit = ~held
        started = time.perf_counter()
        model, diagnostics = fit_memory_polynomial_pa(
            x[fit],
            y[fit],
            orders=orders,
            delays=delays,
            ridge=ridge,
            segment_length=nperseg,
            coefficient_dtype=np.complex128,
        )
        fit_seconds = time.perf_counter() - started
        total_fit_seconds += fit_seconds
        prediction[held] = model.predict(x[held])
        fold_reports.append(
            {
                "held_segment_id": (
                    held_segment.item()
                    if hasattr(held_segment, "item")
                    else held_segment
                ),
                "fit_sample_count": int(np.count_nonzero(fit)),
                "held_sample_count": int(np.count_nonzero(held)),
                "fit_seconds": fit_seconds,
                "fit_diagnostics": dataclasses.asdict(diagnostics),
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("OOF prediction contains non-finite samples")
    return prediction, fold_reports, total_fit_seconds


def _top_rows(
    rows: list[dict[str, Any]],
    *,
    value_path: tuple[str, ...],
    causal_only: bool,
    count: int = 5,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if causal_only and not row.get("causal_feature_eligible", True):
            continue
        value: Any = row
        for key in value_path:
            value = value[key]
        number = float(value)
        if np.isfinite(number):
            scored.append((abs(number), row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:count]]


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    lag_rows = report["lag_correlations"]
    envelope_rows = report["envelope_correlations"]
    amplitude_summary: list[dict[str, Any]] = []
    for row in report["amplitude_regions"]:
        high = row["high_region"]["nmse_db"]
        complement = row["complement_region"]["nmse_db"]
        amplitude_summary.append(
            {
                "threshold_name": row["threshold_name"],
                "threshold_amplitude": row["threshold_amplitude"],
                "high_nmse_db": high,
                "complement_nmse_db": complement,
                "high_minus_complement_db": (
                    None
                    if high is None or complement is None
                    else float(high - complement)
                ),
            }
        )
    return {
        "global_metrics": report["global_metrics"],
        "top_causal_proper_lag_correlations": _top_rows(
            lag_rows,
            value_path=("proper_complex_correlation", "magnitude"),
            causal_only=True,
        ),
        "top_causal_pseudo_lag_correlations": _top_rows(
            lag_rows,
            value_path=("pseudo_complex_correlation", "magnitude"),
            causal_only=True,
        ),
        "top_radial_envelope_correlations": _top_rows(
            envelope_rows,
            value_path=("corr_radial_envelope",),
            causal_only=False,
        ),
        "top_tangential_envelope_correlations": _top_rows(
            envelope_rows,
            value_path=("corr_tangential_envelope",),
            causal_only=False,
        ),
        "amplitude_region_summary": amplitude_summary,
        "slow_state_branch_eligible": any(
            row["eligible_for_state_branch_selection"]
            for row in report["slow_state_correlations"]
        ),
        "error_psd_integrated_bands": report["error_psd"].get(
            "integrated_bands"
        ),
    }


def analyze_from_config(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_config = Path(config_path).resolve()
    config = _load_config(source_config)
    selection_path = Path(config["selection_manifest"]).resolve()
    selection = _load_json_object(
        selection_path,
        name="selection manifest",
    )
    if selection.get("test_split_accessed") is not False:
        raise ValueError("selection manifest must certify sealed test data")
    if selection.get("model_class") != "complex_memory_polynomial":
        raise ValueError("residual runner currently supports selected MP PA")
    dataset = Path(selection["dataset"]).resolve()
    model_path = Path(selection["selected_model"]).resolve()
    _verify_file(
        model_path,
        str(selection["selected_model_sha256"]),
        label="selected MP PA model",
    )
    for name in ("train_input.csv", "train_output.csv", "val_input.csv", "val_output.csv", "spec.json"):
        _verify_file(
            dataset / name,
            str(selection["dataset_files_sha256"][name]),
            label=f"selection dataset file {name}",
        )

    output = Path(config["output_dir"]).resolve()
    oof_report_path = output / "train_oof_residual_analysis.json"
    validation_report_path = output / "validation_residual_analysis.json"
    prediction_path = output / "residual_predictions.npz"
    manifest_path = output / "residual_manifest.json"
    owned_paths = (
        oof_report_path,
        validation_report_path,
        prediction_path,
        manifest_path,
    )
    existing = [path for path in owned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite residual artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)

    # Complete data access list. Test is intentionally absent.
    train_input, train_output = load_split_pair(dataset, "train")
    validation_input, validation_output = load_split_pair(dataset, "val")
    dataset_spec = load_dataset_spec(dataset)
    spec = _build_spec(config, dataset_spec)
    nperseg = spec.psd_nperseg
    train_segments = explicit_frame_ids(train_input.size, nperseg)
    validation_segments = explicit_frame_ids(
        validation_input.size,
        nperseg,
    )
    selected = selection["selected_trial"]
    orders = tuple(int(value) for value in selected["orders"])
    delays = tuple(int(value) for value in selected["delays"])
    ridge = float(selected["ridge"])
    oof_prediction, fold_reports, total_oof_fit_seconds = (
        leave_one_frame_out_predictions(
            train_input,
            train_output,
            segment_id=train_segments,
            orders=orders,
            delays=delays,
            ridge=ridge,
            nperseg=nperseg,
        )
    )
    frozen_model = MemoryPolynomialPA.load(model_path)
    validation_prediction = frozen_model.predict_segments(
        validation_input,
        nperseg,
    )
    common_warmup = int(selection["common_warmup_samples_per_frame"])
    train_valid = segmented_steady_state_mask(
        train_input.size,
        segment_length=nperseg,
        warmup_samples=common_warmup,
    )
    validation_valid = segmented_steady_state_mask(
        validation_input.size,
        segment_length=nperseg,
        warmup_samples=common_warmup,
    )
    frozen_reference = freeze_residual_reference(train_input, spec)
    oof_report = analyze_pa_residuals(
        train_input,
        train_output,
        oof_prediction,
        segment_id=train_segments,
        valid_mask=train_valid,
        split_role="train_oof",
        spec=spec,
        frozen_reference=frozen_reference,
    )
    validation_report = analyze_pa_residuals(
        validation_input,
        validation_output,
        validation_prediction,
        segment_id=validation_segments,
        valid_mask=validation_valid,
        split_role="validation_confirmation",
        spec=spec,
        frozen_reference=frozen_reference,
    )
    write_json(oof_report_path, oof_report)
    write_json(validation_report_path, validation_report)
    np.savez_compressed(
        prediction_path,
        schema_version=np.asarray(1, dtype=np.int64),
        train_oof_prediction=oof_prediction,
        validation_prediction=validation_prediction,
        train_segment_id=train_segments,
        validation_segment_id=validation_segments,
        train_valid_mask=train_valid,
        validation_valid_mask=validation_valid,
    )
    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "task": "forward_pa_residual_analysis",
        "dataset": dataset,
        "selection_manifest": selection_path,
        "selection_manifest_sha256": file_sha256(selection_path),
        "selected_model": model_path,
        "selected_model_sha256": file_sha256(model_path),
        "accessed_splits": ["train", "validation"],
        "test_split_accessed": False,
        "discovery_split": "leave-one-explicit-frame-out training residual",
        "validation_role": (
            "descriptive confirmation only; this validation split already "
            "selected the MP architecture and ridge"
        ),
        "future_branch_selection_rule": (
            "use train inner blocked CV; do not choose a branch from this "
            "validation report and then reuse the same validation as confirmation"
        ),
        "segment_policy": (
            "nperseg-derived explicit model-reset frames; not claimed to be "
            "independent physical captures"
        ),
        "common_warmup_samples_per_frame": common_warmup,
        "selected_architecture": {
            "orders": list(orders),
            "delays": list(delays),
            "ridge": ridge,
        },
        "oof_fold_count": len(fold_reports),
        "oof_total_fit_seconds": total_oof_fit_seconds,
        "oof_folds": fold_reports,
        "train_oof_summary": _summarize(oof_report),
        "validation_summary": _summarize(validation_report),
        "state_conditioned_model_gate": (
            "locked: independent_capture_count=0 and present records cannot "
            "establish slow thermal/bias state"
        ),
        "train_oof_report": oof_report_path,
        "train_oof_report_sha256": file_sha256(oof_report_path),
        "validation_report": validation_report_path,
        "validation_report_sha256": file_sha256(validation_report_path),
        "predictions": prediction_path,
        "predictions_sha256": file_sha256(prediction_path),
        "config": source_config,
        "config_sha256": file_sha256(source_config),
        "source_sha256": {
            "experiments/analyze_pa_residuals.py": file_sha256(source_path),
            "baseline/residual_analysis.py": file_sha256(
                source_path.parents[1] / "baseline" / "residual_analysis.py"
            ),
            "baseline/pa_models.py": file_sha256(
                source_path.parents[1] / "baseline" / "pa_models.py"
            ),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MP PA residuals on train OOF and validation only; never "
            "open test."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = analyze_from_config(args.config, overwrite=args.overwrite)
    train_nmse = manifest["train_oof_summary"]["global_metrics"][
        "pooled_complex_nmse_db"
    ]
    validation_nmse = manifest["validation_summary"]["global_metrics"][
        "pooled_complex_nmse_db"
    ]
    print(
        "Residual analysis:",
        f"train OOF NMSE={train_nmse:.6f} dB",
        f"validation NMSE={validation_nmse:.6f} dB",
        "state branch gate=locked",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
