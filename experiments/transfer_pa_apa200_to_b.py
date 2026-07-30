"""Pre-test APA capture-transfer benchmark.

The runner is deliberately narrower than the ordinary PA selectors:

* only the source/target ``train`` and ``val`` files listed in the frozen
  preregistration are addressable;
* source model topology and coefficients are immutable;
* target adaptation is coefficient-only and uses fixed prefixes;
* the target held-out split has no path in this module and is never opened or
  hashed.

The command produces a train/validation-only immutable bundle.  A later,
separate release command may consume the frozen bundle; this module is not a
test evaluator.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np

from baseline.alignment import (
    complex_ls_gain,
    estimate_integer_delay,
    overlap_for_delay,
)
from baseline.gmp_pa import (
    GMPConfig,
    GeneralizedMemoryPolynomialPA,
    fit_gmp_pa,
)
from baseline.metrics import (
    nmse_opendpd_db,
    nmse_pooled_db,
    time_domain_rms_evm_db,
)
from baseline.sparse_spline_memory_pa import (
    SparseSplineMemoryPA,
    fit_sparse_spline_memory_pa_segments,
)
from baseline.train_spline import load_split_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ALLOWED_SPLITS = frozenset({"train", "val"})
MODEL_NAMES = ("causal_gmp", "lag9_sparse_spline_memory")
Progress = Callable[[str], None]


def file_sha256(path: str | Path) -> str:
    """Hash one regular file without following a symlink supplied as a file."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"expected regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str | Path, *, name: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{name} must be repository-relative")
    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository") from error
    return resolved


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
        return {"real": float(number.real), "imag": float(number.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _json_ready(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("transfer config must contain one JSON object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("transfer config schema mismatch")
    if config.get("status") != "preregistered_before_target_fit":
        raise ValueError("transfer config is not preregistered")
    if config.get("task") != (
        "forward_pa_capture_transfer_and_limited_coefficient_calibration"
    ):
        raise ValueError("unexpected transfer task")
    for key in (
        "source_dataset",
        "target_dataset",
        "output_dir",
        "dataset_contract",
        "source_models",
        "alignment",
        "framing",
        "calibration",
        "implementation_contract",
        "release_policy",
    ):
        if key not in config:
            raise ValueError(f"transfer config missing {key}")
    sealed = config.get("sealed_split_policy")
    if not isinstance(sealed, dict):
        raise ValueError("sealed_split_policy must be an object")
    for role in ("source", "target"):
        record = sealed.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"sealed policy missing {role}")
        if record.get("held_out_hash_recorded") is not False:
            raise ValueError(f"{role} held-out hash must remain sealed")
    if config["alignment"].get("primary_mode") != "strict_source_reuse":
        raise ValueError("strict source alignment must be the primary mode")
    if config["alignment"].get("source_delay_samples") != 0:
        raise ValueError("source transfer delay must be frozen to zero")
    if config["alignment"].get("fractional_transform") != "disabled":
        raise ValueError("fractional transform must remain disabled")
    counts = tuple(config["calibration"]["sample_counts_per_frame"])
    if not counts or any(int(value) != value or int(value) <= 0 for value in counts):
        raise ValueError("calibration sample counts must be positive integers")
    if tuple(sorted(set(int(value) for value in counts))) != tuple(
        int(value) for value in counts
    ):
        raise ValueError("calibration sample counts must be sorted and unique")
    names = tuple(record.get("name") for record in config["source_models"])
    if names != MODEL_NAMES:
        raise ValueError(f"source models must be exactly {MODEL_NAMES}")
    return config


def _dataset_file_map(config: dict[str, Any]) -> dict[str, Path]:
    source = _project_path(config["source_dataset"], name="source dataset")
    target = _project_path(config["target_dataset"], name="target dataset")
    return {
        "source/spec.json": source / "spec.json",
        "source/train_input.csv": source / "train_input.csv",
        "source/train_output.csv": source / "train_output.csv",
        "source/val_input.csv": source / "val_input.csv",
        "source/val_output.csv": source / "val_output.csv",
        "target/spec.json": target / "spec.json",
        "target/train_input.csv": target / "train_input.csv",
        "target/train_output.csv": target / "train_output.csv",
        "target/val_input.csv": target / "val_input.csv",
        "target/val_output.csv": target / "val_output.csv",
    }


def verify_preregistered_inputs(
    config: dict[str, Any],
    config_path: str | Path,
) -> dict[str, Any]:
    """Verify every source/target pre-test input before loading IQ arrays."""

    expected_config_hash = file_sha256(config_path)
    dataset_paths = _dataset_file_map(config)
    contract = config["dataset_contract"]
    expected_dataset_hashes = {
        "source/spec.json": contract["source_spec_sha256"],
        "source/train_input.csv": contract["source_train_input_sha256"],
        "source/train_output.csv": contract["source_train_output_sha256"],
        "source/val_input.csv": contract["source_val_input_sha256"],
        "source/val_output.csv": contract["source_val_output_sha256"],
        "target/spec.json": contract["target_spec_sha256"],
        "target/train_input.csv": contract["target_train_input_sha256"],
        "target/train_output.csv": contract["target_train_output_sha256"],
        "target/val_input.csv": contract["target_val_input_sha256"],
        "target/val_output.csv": contract["target_val_output_sha256"],
    }
    actual_dataset_hashes: dict[str, str] = {}
    for label, expected in expected_dataset_hashes.items():
        actual = file_sha256(dataset_paths[label])
        if actual != expected:
            raise ValueError(f"dataset hash mismatch for {label}")
        actual_dataset_hashes[label] = actual

    if (
        actual_dataset_hashes["source/train_input.csv"]
        != actual_dataset_hashes["target/train_input.csv"]
        or actual_dataset_hashes["source/val_input.csv"]
        != actual_dataset_hashes["target/val_input.csv"]
    ):
        raise ValueError("source/target input identity contract failed")

    artifact_hashes: dict[str, str] = {}
    for record in config["source_models"]:
        for field in ("model_path", "config_path", "manifest_path"):
            path = _project_path(record[field], name=f"source artifact {field}")
            actual = file_sha256(path)
            expected = record[f"{field[:-5] if field.endswith('_path') else field}_sha256"]
            if actual != expected:
                raise ValueError(f"source artifact hash mismatch for {record['name']}:{field}")
            artifact_hashes[f"{record['name']}/{field}"] = actual

    if file_sha256(config_path) != expected_config_hash:
        raise RuntimeError("transfer config changed during preregistration checks")
    return {
        "config_sha256": expected_config_hash,
        "dataset_hashes": actual_dataset_hashes,
        "artifact_hashes": artifact_hashes,
        "dataset_paths": dataset_paths,
        "test_split_accessed": False,
        "target_held_out_hash_recorded": False,
    }


def _load_allowed_pair(dataset: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only an explicitly allowed pre-test split."""

    if split not in ALLOWED_SPLITS:
        raise ValueError("this runner accepts only preregistered train/val splits")
    return load_split_pair(dataset, split)


def frame_lengths(sample_count: int, nperseg: int) -> tuple[int, ...]:
    if sample_count <= 0 or nperseg <= 0:
        raise ValueError("sample_count and nperseg must be positive")
    full, remainder = divmod(int(sample_count), int(nperseg))
    lengths = [int(nperseg)] * full
    if remainder:
        lengths.append(int(remainder))
    return tuple(lengths)


def split_frames(signal: np.ndarray, lengths: Iterable[int]) -> tuple[np.ndarray, ...]:
    values = np.asarray(signal, dtype=np.complex128)
    lengths_tuple = tuple(int(length) for length in lengths)
    if any(length <= 0 for length in lengths_tuple) or sum(lengths_tuple) != values.size:
        raise ValueError("frame lengths do not partition the signal")
    result: list[np.ndarray] = []
    start = 0
    for length in lengths_tuple:
        result.append(values[start : start + length].copy())
        start += length
    return tuple(result)


def extract_prefix_segments(
    segments: Iterable[np.ndarray],
    sample_count: int,
) -> tuple[np.ndarray, ...]:
    """Take an identical chronological prefix from every complete frame."""

    if not isinstance(sample_count, (int, np.integer)) or int(sample_count) <= 0:
        raise ValueError("sample_count must be a positive integer")
    count = int(sample_count)
    result: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        values = np.asarray(segment, dtype=np.complex128)
        if values.ndim != 1 or values.size < count:
            raise ValueError(
                f"frame {index} is shorter than calibration prefix {count}"
            )
        result.append(values[:count].copy())
    if not result:
        raise ValueError("at least one frame is required")
    return tuple(result)


def common_mask(lengths: Iterable[int], warmup: int) -> np.ndarray:
    lengths_tuple = tuple(int(length) for length in lengths)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    mask = np.zeros(sum(lengths_tuple), dtype=bool)
    start = 0
    for length in lengths_tuple:
        if warmup >= length:
            raise ValueError("warmup consumes a frame")
        mask[start + warmup : start + length] = True
        start += length
    return mask


def metric_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    lengths: Iterable[int],
    *,
    warmup: int,
) -> dict[str, Any]:
    estimate = np.asarray(prediction, dtype=np.complex128)
    reference = np.asarray(target, dtype=np.complex128)
    lengths_tuple = tuple(int(length) for length in lengths)
    if (
        estimate.ndim != 1
        or reference.shape != estimate.shape
        or sum(lengths_tuple) != estimate.size
    ):
        raise ValueError("metric arrays and frame contract disagree")
    if not np.all(np.isfinite(estimate)):
        raise ValueError("prediction contains non-finite values")
    mask = common_mask(lengths_tuple, warmup)
    complete = [
        (start, start + length)
        for start, length in _frame_offsets(lengths_tuple)
        if length == max(lengths_tuple)
    ]
    open_nmse = (
        nmse_opendpd_db(
            np.stack([estimate[start:stop] for start, stop in complete]),
            np.stack([reference[start:stop] for start, stop in complete]),
        )
        if complete
        else None
    )
    error = estimate - reference
    reference_power = float(np.sum(np.abs(reference) ** 2))
    error_power = float(np.sum(np.abs(error) ** 2))
    return {
        "full_record_nmse_db": nmse_pooled_db(estimate, reference),
        "common_interior_nmse_db": nmse_pooled_db(estimate[mask], reference[mask]),
        "opendpd_compatible_nmse_db": open_nmse,
        "time_domain_rms_evm_db": time_domain_rms_evm_db(estimate, reference),
        "common_interior_time_domain_rms_evm_db": time_domain_rms_evm_db(
            estimate[mask], reference[mask]
        ),
        "relative_error_power": error_power / reference_power,
        "sample_count": int(estimate.size),
        "common_sample_count": int(np.count_nonzero(mask)),
        "per_frame_nmse_db": [
            nmse_pooled_db(estimate[start:stop], reference[start:stop])
            for start, stop in _frame_offsets(lengths_tuple)
        ],
        "finite_output": True,
    }


def _frame_offsets(lengths: Iterable[int]) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    start = 0
    for length in tuple(int(value) for value in lengths):
        offsets.append((start, start + length))
        start += length
    return tuple(offsets)


def _operation_count(model: Any) -> Any:
    value = getattr(model, "operation_count")
    return value() if callable(value) else value


def _predict_frames(model: Any, segments: Iterable[np.ndarray]) -> np.ndarray:
    predictions = [np.asarray(model.predict(segment), dtype=np.complex128) for segment in segments]
    if not predictions:
        raise ValueError("at least one frame is required")
    return np.concatenate(predictions)


def _streaming_check(model: Any, segments: Iterable[np.ndarray]) -> dict[str, Any]:
    """Check arbitrary causal chunks against reset-per-frame prediction."""

    max_error = 0.0
    reset_error = 0.0
    frames = tuple(np.asarray(segment, dtype=np.complex128) for segment in segments)
    for frame in frames:
        reference = np.asarray(model.predict(frame), dtype=np.complex128)
        state = (
            None
            if hasattr(model, "predict_streaming_chunk")
            else model.initial_state()
        )
        chunks: list[np.ndarray] = []
        start = 0
        for width in (1, 17, 263, 4096, 8192):
            if start >= frame.size:
                break
            stop = min(frame.size, start + width)
            chunk = frame[start:stop]
            if hasattr(model, "predict_streaming_chunk"):
                output, state = model.predict_streaming_chunk(chunk, state)
            else:
                output, state = model.predict_chunk(chunk, state)
            chunks.append(np.asarray(output, dtype=np.complex128))
            start = stop
        if start < frame.size:
            if hasattr(model, "predict_streaming_chunk"):
                output, state = model.predict_streaming_chunk(frame[start:], state)
            else:
                output, state = model.predict_chunk(frame[start:], state)
            chunks.append(np.asarray(output, dtype=np.complex128))
        streamed = np.concatenate(chunks)
        max_error = max(max_error, float(np.max(np.abs(streamed - reference))))
        reset_error = max(
            reset_error,
            float(np.max(np.abs(_predict_frames(model, (frame,)) - reference))),
        )
    return {
        "streaming_chunk_max_abs_error": max_error,
        "segmented_reset_max_abs_error": reset_error,
        "streaming_chunk_equivalence_passed": bool(max_error <= 1e-11),
        "segmented_reset_equivalence_passed": bool(reset_error <= 1e-12),
    }


def _support_summary(
    segments: Iterable[np.ndarray],
    *,
    support_maximum: float,
) -> dict[str, Any]:
    values = np.concatenate(tuple(np.asarray(segment, dtype=np.complex128) for segment in segments))
    amplitudes = np.abs(values)
    fraction = float(np.mean(amplitudes > float(support_maximum)))
    return {
        "support_maximum": float(support_maximum),
        "maximum_input_amplitude": float(np.max(amplitudes)),
        "fraction_above_support": fraction,
        "support_violation_count": int(np.count_nonzero(amplitudes > support_maximum)),
    }


def _evaluate(
    model: Any,
    input_segments: tuple[np.ndarray, ...],
    target_segments: tuple[np.ndarray, ...],
    *,
    warmup: int,
    support_maximum: float,
) -> tuple[dict[str, Any], np.ndarray]:
    if len(input_segments) != len(target_segments):
        raise ValueError("input and target frame counts differ")
    if any(left.size != right.size for left, right in zip(input_segments, target_segments, strict=True)):
        raise ValueError("input and target frame lengths differ")
    started = time.perf_counter()
    prediction = _predict_frames(model, input_segments)
    inference_seconds = time.perf_counter() - started
    lengths = tuple(int(segment.size) for segment in input_segments)
    metrics = metric_summary(
        prediction,
        np.concatenate(target_segments),
        lengths,
        warmup=warmup,
    )
    operation = _operation_count(model)
    operation_dict = operation.to_dict()
    coefficient_bytes_fp32 = int(operation.stored_real_coefficients) * 4
    constant_bytes_fp32 = int(operation.stored_real_constants) * 4
    state_bytes_fp32 = int(operation.state_real_values) * 4
    metrics.update(
        {
            "inference_wall_clock_seconds": float(inference_seconds),
            "inference_samples_per_second": float(
                prediction.size / inference_seconds
                if inference_seconds > 0.0
                else float("inf")
            ),
            "operation_count": operation_dict,
            "memory": {
                "stored_real_coefficients": int(operation.stored_real_coefficients),
                "stored_real_constants": int(operation.stored_real_constants),
                "state_real_values": int(operation.state_real_values),
                "coefficient_bytes_fp32": coefficient_bytes_fp32,
                "constant_bytes_fp32": constant_bytes_fp32,
                "state_bytes_fp32": state_bytes_fp32,
                "total_model_and_state_bytes_fp32": (
                    coefficient_bytes_fp32
                    + constant_bytes_fp32
                    + state_bytes_fp32
                ),
                "coefficient_bytes_fp64": coefficient_bytes_fp32 * 2,
                "constant_bytes_fp64": constant_bytes_fp32 * 2,
                "state_bytes_fp64": state_bytes_fp32 * 2,
            },
            "support": _support_summary(
                input_segments,
                support_maximum=support_maximum,
            ),
            "streaming": _streaming_check(model, input_segments),
        }
    )
    return metrics, prediction


def _fit_target_model(
    name: str,
    source_model: Any,
    input_prefixes: tuple[np.ndarray, ...],
    output_prefixes: tuple[np.ndarray, ...],
    *,
    sample_count: int,
    record: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    if name == "causal_gmp":
        model, diagnostics = fit_gmp_pa(
            np.concatenate(input_prefixes),
            np.concatenate(output_prefixes),
            config=source_model.config,
            ridge=float(record["source_ridge"]),
            segment_length=int(sample_count),
            solver_mode="ridge_lstsq",
        )
    elif name == "lag9_sparse_spline_memory":
        model, diagnostics = fit_sparse_spline_memory_pa_segments(
            input_prefixes,
            output_prefixes,
            branches=source_model.branches,
            knots=source_model.knots,
            ridge=float(record["source_ridge"]),
        )
    else:
        raise ValueError(f"unknown transfer model {name}")
    fit_seconds = time.perf_counter() - started
    source_operation = _operation_count(source_model).to_dict()
    fitted_operation = _operation_count(model).to_dict()
    if fitted_operation != source_operation:
        raise RuntimeError(f"{name} operation schedule changed during coefficient fit")
    return model, {
        "fit_wall_clock_seconds": float(fit_seconds),
        "fit_diagnostics": dataclasses.asdict(diagnostics),
        "operation_count": fitted_operation,
        "coefficient_hash": _array_sha256(model.coefficients),
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_source_models(config: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for record in config["source_models"]:
        model_path = _project_path(record["model_path"], name="source model")
        if record["name"] == "causal_gmp":
            model = GeneralizedMemoryPolynomialPA.load(model_path)
            expected = record["architecture"]
            actual = dataclasses.asdict(model.config)
            if actual != expected:
                raise ValueError(f"GMP architecture mismatch: {actual} != {expected}")
        elif record["name"] == "lag9_sparse_spline_memory":
            model = SparseSplineMemoryPA.load(model_path)
            expected_branches = tuple(tuple(pair) for pair in record["branches"])
            actual_branches = tuple(
                (branch.signal_delay, branch.envelope_delay)
                for branch in model.branches
            )
            if actual_branches != expected_branches:
                raise ValueError("sparse source branch topology mismatch")
            if model.knot_count != int(record["knot_count"]):
                raise ValueError("sparse source knot count mismatch")
        else:
            raise ValueError(f"unknown source model {record['name']}")
        operation = _operation_count(model)
        actual_operation = operation.to_dict()
        expected_operation = record["operation_budget"]
        key_map = {
            "real_multiplications_per_sample": "real_multiplications",
            "real_additions_per_sample": "real_additions",
            "nonlinear_operations_per_sample": "nonlinear_operations",
            "comparisons_per_sample": "comparisons",
            "lut_accesses_per_sample": "lookups",
            "state_real_values": "state_real_values",
            "stored_real_coefficients": "stored_real_coefficients",
        }
        mismatches = {
            expected_key: (
                actual_operation[actual_key],
                expected_value,
            )
            for expected_key, actual_key in key_map.items()
            if expected_key in expected_operation
            and actual_operation[actual_key] != expected_operation[expected_key]
        }
        if mismatches:
            raise ValueError(
                f"{record['name']} operation contract mismatch: {mismatches}"
            )
        if model.coefficients.size * 2 != int(
            expected_operation["stored_real_coefficients"]
        ):
            raise ValueError(f"{record['name']} coefficient count mismatch")
        models[record["name"]] = model
    return models


def _target_nuisance_diagnostic(
    input_segments: tuple[np.ndarray, ...],
    output_segments: tuple[np.ndarray, ...],
    *,
    max_abs_delay: int,
) -> dict[str, Any]:
    input_all = np.concatenate(input_segments)
    output_all = np.concatenate(output_segments)
    delay = estimate_integer_delay(input_all, output_all, max_abs_delay)
    aligned_input, aligned_output = overlap_for_delay(input_all, output_all, delay)
    gain = complex_ls_gain(aligned_input, aligned_output)
    return {
        "estimated_integer_delay_samples": int(delay),
        "complex_ls_gain_real": float(gain.real),
        "complex_ls_gain_imag": float(gain.imag),
        "complex_ls_gain_abs": float(abs(gain)),
        "complex_ls_gain_phase_rad": float(np.angle(gain)),
        "fit_scope": "target_train_only",
        "frozen_before_target_validation": True,
        "used_for_strict_selection": False,
    }


def _nuisance_zero_shot_evaluate(
    model: Any,
    input_segments: tuple[np.ndarray, ...],
    output_segments: tuple[np.ndarray, ...],
    nuisance: dict[str, Any],
    *,
    warmup: int,
) -> dict[str, Any]:
    delay = int(nuisance["estimated_integer_delay_samples"])
    gain = complex(
        float(nuisance["complex_ls_gain_real"]),
        float(nuisance["complex_ls_gain_imag"]),
    )
    aligned_inputs: list[np.ndarray] = []
    aligned_outputs: list[np.ndarray] = []
    for input_segment, output_segment in zip(
        input_segments,
        output_segments,
        strict=True,
    ):
        left, right = overlap_for_delay(input_segment, output_segment, delay)
        aligned_inputs.append(left)
        aligned_outputs.append(right)
    prediction = _predict_frames(model, tuple(aligned_inputs))
    normalized_prediction = prediction / gain
    normalized_target = np.concatenate(aligned_outputs) / gain
    lengths = tuple(segment.size for segment in aligned_inputs)
    summary = metric_summary(
        normalized_prediction,
        normalized_target,
        lengths,
        warmup=warmup,
    )
    summary["normalization"] = "divide prediction and measured output by target-train complex LS gain"
    summary["delay_applied_to_each_frame"] = delay
    summary["selection_eligible"] = False
    return summary


def _select_calibration_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [
        record
        for record in records
        if record.get("status") == "feasible"
        and int(record["sample_count_per_frame"]) > 0
        and np.isfinite(float(record["validation"]["full_record_nmse_db"]))
        and record["validation"]["support"]["fraction_above_support"] == 0.0
        and record["validation"]["streaming"]["streaming_chunk_equivalence_passed"]
        and record["validation"]["streaming"]["segmented_reset_equivalence_passed"]
    ]
    if not feasible:
        raise RuntimeError("no feasible target calibration record")
    best = min(
        feasible,
        key=lambda record: float(record["validation"]["full_record_nmse_db"]),
    )
    threshold = float(best["validation"]["full_record_nmse_db"]) + 0.25
    within = [
        record
        for record in feasible
        if float(record["validation"]["full_record_nmse_db"]) <= threshold
    ]
    return min(
        within,
        key=lambda record: (
            int(record["sample_count_per_frame"]),
            float(record["validation"]["common_interior_nmse_db"]),
            float(record["fit"]["fit_wall_clock_seconds"]),
        ),
    )


def _acquire_lock(output: Path) -> tuple[Path, bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable transfer output already exists: {output}")
    lock = output.parent / f".{output.name}.lock"
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f"transfer output lock already exists: {lock}")
    payload = json.dumps(
        {"pid": os.getpid(), "token": secrets.token_hex(24)},
        sort_keys=True,
    ).encode("utf-8")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock, payload


def _verify_lock(lock: Path, payload: bytes) -> None:
    if lock.is_symlink() or not lock.is_file() or lock.read_bytes() != payload:
        raise RuntimeError("transfer publication lock was replaced")


def _publish_bundle(
    output: Path,
    *,
    lock: Path,
    lock_payload: bytes,
    manifest: dict[str, Any],
    predictions: dict[str, np.ndarray],
    coefficients: dict[str, np.ndarray],
    execution: dict[str, Any],
) -> dict[str, Any]:
    _verify_lock(lock, lock_payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError("transfer output appeared before publication")
    temporary = output.parent / f".{output.name}.tmp-{secrets.token_hex(12)}"
    temporary.mkdir()
    try:
        np.savez_compressed(
            temporary / "predictions.npz",
            schema_version=np.asarray(1, dtype=np.int64),
            **predictions,
        )
        np.savez_compressed(
            temporary / "calibration_coefficients.npz",
            schema_version=np.asarray(1, dtype=np.int64),
            **coefficients,
        )
        _write_json(temporary / "execution_record.json", execution)
        artifact_names = (
            "predictions.npz",
            "calibration_coefficients.npz",
            "execution_record.json",
        )
        artifact_hashes = {
            name: {
                "path": name,
                "sha256": file_sha256(temporary / name),
            }
            for name in artifact_names
        }
        final_manifest = {
            **manifest,
            "artifacts": artifact_hashes,
            "publication": {
                "immutable_bundle": True,
                "atomic_directory_rename": True,
                "manifest_written_last_inside_temporary_bundle": True,
            },
        }
        _write_json(temporary / "transfer_manifest.json", final_manifest)
        _verify_lock(lock, lock_payload)
        if output.exists() or output.is_symlink():
            raise FileExistsError("transfer output appeared during publication")
        os.replace(temporary, output)
        temporary = None  # type: ignore[assignment]
        return final_manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def run_from_config(
    config_path: str | Path,
    *,
    progress: Progress = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Run source control, zero-shot transfer and target-train curves only."""

    started = time.perf_counter()
    source_config = Path(config_path).resolve()
    initial_config_hash = file_sha256(source_config)
    config = load_config(source_config)
    if file_sha256(source_config) != initial_config_hash:
        raise RuntimeError("transfer config changed while being parsed")
    verified = verify_preregistered_inputs(config, source_config)
    if verified["config_sha256"] != initial_config_hash:
        raise RuntimeError("transfer config snapshot mismatch")
    output = _project_path(config["output_dir"], name="output directory")
    lock, lock_payload = _acquire_lock(output)
    try:
        progress("[integrity] source/target train+val hashes verified before IQ load")
        models = _load_source_models(config)
        source_dataset = _project_path(config["source_dataset"], name="source dataset")
        target_dataset = _project_path(config["target_dataset"], name="target dataset")

        source_train_input, _ = _load_allowed_pair(source_dataset, "train")
        source_val_input, source_val_output = _load_allowed_pair(source_dataset, "val")
        target_train_input, target_train_output = _load_allowed_pair(target_dataset, "train")
        source_contract = tuple(config["dataset_contract"]["source_frame_lengths"])
        target_contract = tuple(config["dataset_contract"]["target_frame_lengths"])
        nperseg = int(config["dataset_contract"]["nperseg"])
        if source_train_input.size != sum(source_contract):
            raise ValueError("source train frame contract mismatch")
        if source_val_input.size != nperseg:
            raise ValueError("source validation frame contract mismatch")
        if target_train_input.size != sum(target_contract):
            raise ValueError("target train frame contract mismatch")
        source_train_segments = split_frames(source_train_input, source_contract)
        source_val_segments = split_frames(source_val_input, (source_val_input.size,))
        source_val_target_segments = split_frames(source_val_output, (source_val_output.size,))
        target_train_segments = split_frames(target_train_input, target_contract)
        target_train_output_segments = split_frames(target_train_output, target_contract)
        if tuple(segment.size for segment in target_train_segments) != target_contract:
            raise ValueError("target train frame lengths are not deterministic")
        if file_sha256(source_config) != initial_config_hash:
            raise RuntimeError("transfer config changed after train load")
        verified_after_train = verify_preregistered_inputs(config, source_config)
        if verified_after_train["dataset_hashes"] != verified["dataset_hashes"]:
            raise RuntimeError("pre-test dataset hashes changed after train load")

        source_support = float(np.max(np.abs(source_train_input)))
        source_results: dict[str, Any] = {}
        source_predictions: dict[str, np.ndarray] = {}
        target_models: dict[str, dict[int, Any]] = {name: {} for name in MODEL_NAMES}
        target_fit_records: dict[str, list[dict[str, Any]]] = {
            name: [] for name in MODEL_NAMES
        }
        coefficient_arrays: dict[str, np.ndarray] = {}

        for record in config["source_models"]:
            name = record["name"]
            model = models[name]
            source_metrics, source_prediction = _evaluate(
                model,
                source_val_segments,
                source_val_target_segments,
                warmup=int(config["framing"]["common_warmup_samples"][name]),
                support_maximum=(
                    float(model.knots[-1])
                    if name == "lag9_sparse_spline_memory"
                    else source_support
                ),
            )
            source_results[name] = {
                "model_role": "frozen_source_validation_control",
                "metrics": source_metrics,
                "coefficient_hash": _array_sha256(model.coefficients),
            }
            source_predictions[f"{name}__source_val"] = source_prediction
            progress(
                f"[source control] {name}: "
                f"{source_metrics['full_record_nmse_db']:.6f} dB"
            )

        nuisance = _target_nuisance_diagnostic(
            target_train_input_segments := target_train_segments,
            target_train_output_segments,
            max_abs_delay=int(config["alignment"]["max_abs_delay_diagnostic"]),
        )
        progress(
            "[target train] nuisance diagnostic frozen: "
            f"delay={nuisance['estimated_integer_delay_samples']} samples"
        )

        calibration_counts = tuple(
            int(value) for value in config["calibration"]["sample_counts_per_frame"]
        )
        for record in config["source_models"]:
            name = record["name"]
            source_model = models[name]
            for count in calibration_counts:
                input_prefixes = extract_prefix_segments(target_train_segments, count)
                output_prefixes = extract_prefix_segments(
                    target_train_output_segments,
                    count,
                )
                try:
                    fitted_model, fit_info = _fit_target_model(
                        name,
                        source_model,
                        input_prefixes,
                        output_prefixes,
                        sample_count=count,
                        record=record,
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    target_fit_records[name].append(
                        {
                            "sample_count_per_frame": count,
                            "status": "infeasible",
                            "reason": str(error),
                            "target_validation_loaded": False,
                        }
                    )
                    continue
                target_models[name][count] = fitted_model
                target_fit_records[name].append(
                    {
                        "sample_count_per_frame": count,
                        "status": "fit_complete_waiting_for_validation",
                        "fit": fit_info,
                        "target_validation_loaded": False,
                    }
                )
                coefficient_arrays[f"{name}__N{count}"] = fitted_model.coefficients
            progress(
                f"[target calibration] {name}: "
                f"{len(target_models[name])}/{len(calibration_counts)} feasible fits"
            )

        # Target validation is intentionally first loaded after every prefix fit.
        target_val_input, target_val_output = _load_allowed_pair(target_dataset, "val")
        if target_val_input.size != nperseg or target_val_output.size != nperseg:
            raise ValueError("target validation frame contract mismatch")
        target_val_segments = split_frames(target_val_input, (nperseg,))
        target_val_output_segments = split_frames(target_val_output, (nperseg,))
        if file_sha256(source_config) != initial_config_hash:
            raise RuntimeError("transfer config changed before validation load")

        target_results: dict[str, Any] = {}
        target_predictions: dict[str, np.ndarray] = {}
        selected_records: dict[str, Any] = {}
        for record in config["source_models"]:
            name = record["name"]
            source_model = models[name]
            support_maximum = (
                float(source_model.knots[-1])
                if name == "lag9_sparse_spline_memory"
                else source_support
            )
            zero_metrics, zero_prediction = _evaluate(
                source_model,
                target_val_segments,
                target_val_output_segments,
                warmup=int(config["framing"]["common_warmup_samples"][name]),
                support_maximum=support_maximum,
            )
            target_predictions[f"{name}__zero_shot_target_val"] = zero_prediction
            nuisance_metrics = _nuisance_zero_shot_evaluate(
                source_model,
                target_val_segments,
                target_val_output_segments,
                nuisance,
                warmup=int(config["framing"]["common_warmup_samples"][name]),
            )
            curves: list[dict[str, Any]] = [
                {
                    "sample_count_per_frame": 0,
                    "status": "feasible",
                    "mode": "none",
                    "validation": zero_metrics,
                    "fit": {
                        "fit_wall_clock_seconds": 0.0,
                        "coefficient_hash": _array_sha256(source_model.coefficients),
                    },
                }
            ]
            for fit_record in target_fit_records[name]:
                count = int(fit_record["sample_count_per_frame"])
                fit_record["target_validation_loaded"] = True
                if fit_record["status"] == "infeasible":
                    curves.append(fit_record)
                    continue
                fitted_model = target_models[name][count]
                validation_metrics, validation_prediction = _evaluate(
                    fitted_model,
                    target_val_segments,
                    target_val_output_segments,
                    warmup=int(config["framing"]["common_warmup_samples"][name]),
                    support_maximum=float(
                        np.max(np.abs(np.concatenate(extract_prefix_segments(
                            target_train_segments,
                            count,
                        ))))
                    ),
                )
                fit_record["status"] = "feasible"
                fit_record["validation"] = validation_metrics
                fit_record["validation_loaded_after_all_prefix_fits"] = True
                curves.append(fit_record)
                target_predictions[f"{name}__N{count}__target_val"] = (
                    validation_prediction
                )
            selected = _select_calibration_record(curves)
            selected_records[name] = selected
            target_results[name] = {
                "zero_shot": {
                    "validation": zero_metrics,
                    "nuisance_diagnostic": nuisance_metrics,
                    "selection_eligible": True,
                },
                "coefficient_only_curves": curves,
                "selected_calibration": selected,
            }
            progress(
                f"[target validation] {name}: zero-shot "
                f"{zero_metrics['full_record_nmse_db']:.6f} dB; "
                f"selected N={selected['sample_count_per_frame']}"
            )

        input_reverification = verify_preregistered_inputs(config, source_config)
        source_paths = {
            "experiments/transfer_pa_apa200_to_b.py": Path(__file__).resolve(),
            "baseline/alignment.py": PROJECT_ROOT / "baseline/alignment.py",
            "baseline/gmp_pa.py": PROJECT_ROOT / "baseline/gmp_pa.py",
            "baseline/metrics.py": PROJECT_ROOT / "baseline/metrics.py",
            "baseline/sparse_spline_memory_pa.py": PROJECT_ROOT
            / "baseline/sparse_spline_memory_pa.py",
            "baseline/spline_memory_dpd.py": PROJECT_ROOT
            / "baseline/spline_memory_dpd.py",
            "baseline/train_spline.py": PROJECT_ROOT / "baseline/train_spline.py",
        }
        source_hashes = {
            name: file_sha256(path) for name, path in source_paths.items()
        }
        execution = {
            "schema_version": 1,
            "command": " ".join(sys.argv),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "runtime_seconds_before_publication": time.perf_counter() - started,
            "source_models_frozen_before_target_fit": True,
            "all_target_prefix_fits_completed_before_validation_load": True,
            "test_split_accessed": False,
            "target_held_out_hash_recorded": False,
        }
        manifest = {
            "schema_version": 1,
            "task": config["task"],
            "status": "pretest_train_validation_only",
            "config": str(source_config.relative_to(PROJECT_ROOT)),
            "config_sha256": verified["config_sha256"],
            "dataset_hashes": verified["dataset_hashes"],
            "source_artifact_hashes": verified["artifact_hashes"],
            "source_control": source_results,
            "target_transfer": target_results,
            "target_train_nuisance_diagnostic": nuisance,
            "source_code_sha256": source_hashes,
            "input_integrity": {
                "all_hashes_verified_before_waveform_load": True,
                "all_inputs_reverified_before_publication": all(
                    input_reverification["dataset_hashes"][key]
                    == verified["dataset_hashes"][key]
                    for key in verified["dataset_hashes"]
                ),
                "source_target_inputs_bit_identical": True,
                "test_never_opened_or_hashed": True,
                "target_held_out_hash_recorded": False,
            },
            "selection_policy": config["calibration"]["validation_selection_rule"],
            "dpd_gate": "closed",
        }
        final_manifest = _publish_bundle(
            output,
            lock=lock,
            lock_payload=lock_payload,
            manifest=manifest,
            predictions=source_predictions | target_predictions,
            coefficients=coefficient_arrays,
            execution=execution,
        )
        progress(f"[publish] immutable pre-test bundle: {output}")
        return final_manifest
    finally:
        if lock.exists() and not lock.is_symlink():
            lock.unlink()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered APA capture-transfer benchmark using only "
            "source/target train and validation splits."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    manifest = run_from_config(arguments.config)
    print(
        "APA transfer pre-test complete:",
        f"models={len(manifest['target_transfer'])}",
        "target-held-out-access=False",
    )


if __name__ == "__main__":
    main()
