"""Evaluate a frozen spline-memory DPD under a sealed fixed-point contract.

Only desired ``train_input.csv`` and ``val_input.csv`` are opened.  Numeric
formats are frozen from the train desired signal, the frozen DPD coefficients,
and the floating DPD train drive before validation waveform values are parsed
or used.  Validation bytes are hash-verified only.  The validation path is
always

    desired x -> fixed DPD -> frozen PA surrogate -> output,

never measured PA output -> inverse model.

The PA surrogate is used only to describe fixed-point cascade degradation.
This runner does not fit, select, estimate gain/alignment, access test data,
or claim a physical-PA/hardware result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.fixed_point_pa import (  # noqa: E402
    FixedPointFormat,
    FixedPointPAStats,
)
from baseline.fixed_point_spline_memory_dpd import (  # noqa: E402
    FixedPointDPDConfig,
    FixedPointSparseSplineMemoryDPD,
)
from baseline.metrics import (  # noqa: E402
    nmse_opendpd_db,
    papr_db,
    peak_amplitude,
)
from baseline.pa_models import MemoryPolynomialPA  # noqa: E402
from baseline.spline_memory_dpd import SparseSplineMemoryDPD  # noqa: E402
from baseline.train_spline import (  # noqa: E402
    _paired_time_metrics,
    file_sha256,
    load_complex_iq_csv,
    write_json,
)
from experiments.replay_frozen_spline_memory_dpd import (  # noqa: E402
    _complex_from_json,
    _hash_map,
    _model_contract,
    _path,
    _regular_file,
    _spectral_config,
    _verify_hash_map,
    _verify_selection_report,
)


SCHEMA_VERSION = 1
RUNNER_SOURCE = "experiments/evaluate_fixed_point_dpd.py"
REPLAY_SOURCE = "experiments/replay_frozen_spline_memory_dpd.py"
SOURCE_FILES = (
    "baseline/__init__.py",
    "baseline/alignment.py",
    "baseline/complex_spline_dpd.py",
    "baseline/complexity.py",
    "baseline/fixed_point_pa.py",
    "baseline/fixed_point_sparse_spline_pa.py",
    "baseline/fixed_point_spline_memory_dpd.py",
    "baseline/gmp_pa.py",
    "baseline/metrics.py",
    "baseline/pa_models.py",
    "baseline/sparse_spline_memory_pa.py",
    "baseline/spline_memory_dpd.py",
    "baseline/spline_hammerstein_pa.py",
    "baseline/train_spline.py",
    REPLAY_SOURCE,
    RUNNER_SOURCE,
)


def validate_config(config: dict[str, Any]) -> None:
    """Reject any split, fit, or numeric decision outside the frozen scope."""

    allowed_keys = {
        "access_policy",
        "alignment_delay_samples",
        "artifact_sha256",
        "dataset",
        "dataset_spec_sha256",
        "expected_model",
        "expected_surrogate",
        "fit_performed",
        "fixed_point_protocol",
        "gain_or_alignment_retuned",
        "model_path",
        "nperseg",
        "schema_version",
        "selected_family",
        "selection_performed",
        "selection_report",
        "source_sha256",
        "split",
        "split_input_sha256",
        "surrogate_path",
        "target_gain",
        "task",
        "train_input_sha256",
        "train_sample_count",
        "validation_sample_count",
    }
    unknown = set(config) - allowed_keys
    if unknown:
        raise ValueError(f"unknown fixed-point DPD config keys: {sorted(unknown)}")
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported fixed-point DPD schema")
    if config.get("task") != "frozen_spline_memory_dpd_fixed_point":
        raise ValueError("unexpected fixed-point DPD task")
    if config.get("split") != "val":
        raise ValueError("fixed-point DPD runner supports validation only")
    if bool(config.get("fit_performed", True)):
        raise ValueError("fit_performed must be false")
    if bool(config.get("selection_performed", True)):
        raise ValueError("selection_performed must be false")
    if bool(config.get("gain_or_alignment_retuned", True)):
        raise ValueError("gain_or_alignment_retuned must be false")
    if not isinstance(config.get("selected_family"), str):
        raise ValueError("selected_family is required")
    if config.get("alignment_delay_samples") != 0:
        raise ValueError("only the frozen zero-delay contract is supported")

    access = config.get("access_policy")
    if not isinstance(access, dict):
        raise ValueError("access_policy is required")
    if set(access) != {
        "allowed_waveform_files",
        "measured_output_opened",
        "test_split_accessed",
    }:
        raise ValueError("access_policy contains unknown or missing keys")
    if access.get("allowed_waveform_files") != [
        "train_input.csv",
        "val_input.csv",
    ]:
        raise ValueError("only train_input.csv and val_input.csv are allowed")
    if access.get("measured_output_opened") is not False:
        raise ValueError("measured_output_opened must be false")
    if access.get("test_split_accessed") is not False:
        raise ValueError("test_split_accessed must be false")

    nperseg = config.get("nperseg")
    if (
        not isinstance(nperseg, int)
        or isinstance(nperseg, bool)
        or nperseg < 2
        or nperseg % 2
    ):
        raise ValueError("nperseg must be an even integer >= 2")
    train_count = config.get("train_sample_count")
    validation_count = config.get("validation_sample_count")
    for name, value in (
        ("train_sample_count", train_count),
        ("validation_sample_count", validation_count),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if int(validation_count) % int(nperseg):
        raise ValueError("validation must contain complete spectral frames")
    if int(train_count) % int(nperseg):
        raise ValueError("train must contain complete spectral frames")

    protocol = config.get("fixed_point_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("fixed_point_protocol is required")
    if set(protocol) != {
        "accumulator_bits",
        "activation_bits",
        "interpolation_fraction_bits",
        "overflow",
        "power_bits",
        "rounding",
        "scalar_accumulator_bits",
        "scale_guard_ratio",
    }:
        raise ValueError(
            "fixed_point_protocol contains unknown or missing keys"
        )
    if protocol.get("activation_bits") != [16, 14, 12]:
        raise ValueError("activation_bits must be exactly [16, 14, 12]")
    if protocol.get("rounding") != "nearest_even":
        raise ValueError("rounding must be nearest_even")
    if protocol.get("overflow") != "saturate_and_count":
        raise ValueError("overflow must be saturate_and_count")
    guard_ratio = protocol.get("scale_guard_ratio")
    if (
        not isinstance(guard_ratio, (int, float))
        or isinstance(guard_ratio, bool)
        or not np.isfinite(float(guard_ratio))
        or float(guard_ratio) < 1.0
    ):
        raise ValueError("scale_guard_ratio must be finite and >= 1")
    for name in (
        "power_bits",
        "accumulator_bits",
        "scalar_accumulator_bits",
        "interpolation_fraction_bits",
    ):
        value = protocol.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"fixed_point_protocol.{name} must be positive")

    for name in (
        "dataset",
        "model_path",
        "surrogate_path",
        "selection_report",
    ):
        _path(config.get(name), field=name)
    _hash_map(config.get("artifact_sha256"), field="artifact_sha256")
    _hash_map(config.get("source_sha256"), field="source_sha256")
    _complex_from_json(config.get("target_gain"), field="target_gain")
    for name in (
        "dataset_spec_sha256",
        "train_input_sha256",
        "split_input_sha256",
    ):
        value = config.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a SHA-256 string")


def _frame_lengths(sample_count: int, segment_length: int) -> tuple[int, ...]:
    if sample_count <= 0 or segment_length <= 0:
        raise ValueError("sample and segment lengths must be positive")
    return tuple(
        min(segment_length, sample_count - start)
        for start in range(0, sample_count, segment_length)
    )


def _segments(
    signal: np.ndarray,
    frame_lengths: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    array = np.asarray(signal)
    if array.ndim != 1 or sum(frame_lengths) != array.size:
        raise ValueError("frame lengths must partition a one-dimensional signal")
    start = 0
    result: list[np.ndarray] = []
    for length in frame_lengths:
        if length <= 0:
            raise ValueError("frame lengths must be positive")
        result.append(array[start : start + length])
        start += length
    return tuple(result)


def _stats_aggregate(stats: list[FixedPointPAStats]) -> dict[str, int]:
    if not stats:
        raise ValueError("at least one fixed-point frame is required")
    additive = {
        "sample_count",
        "input_saturations",
        "power_saturations",
        "scalar_accumulator_saturations",
        "accumulator_saturations",
        "output_saturations",
        "interpolation_saturations",
    }
    per_model_static = {
        "coefficient_saturations",
        "knot_code_collision_count",
        "maximum_knot_code_shift",
    }
    result: dict[str, int] = {}
    for field in stats[0].to_dict():
        values = [int(getattr(item, field)) for item in stats]
        if field in additive:
            result[field] = int(sum(values))
        elif field in per_model_static:
            result[field] = int(max(values))
        else:
            result[field] = int(max(values))
    return result


def _predict_fixed_frames(
    evaluator: FixedPointSparseSplineMemoryDPD,
    signal: np.ndarray,
    frame_lengths: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, int], float]:
    outputs: list[np.ndarray] = []
    stats: list[FixedPointPAStats] = []
    started = time.perf_counter()
    for segment in _segments(signal, frame_lengths):
        result = evaluator.predict_chunk(segment)
        outputs.append(result.output)
        stats.append(result.stats)
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), _stats_aggregate(stats), float(elapsed)


def _paired_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    warmup_samples: int,
    segment_length: int,
) -> dict[str, Any]:
    """Report pooled and OpenDPD-style metrics under one framing contract."""

    estimate_array = np.asarray(estimate)
    reference_array = np.asarray(reference)
    if (
        estimate_array.ndim != 1
        or reference_array.shape != estimate_array.shape
        or estimate_array.size % segment_length
    ):
        raise ValueError("paired metrics require equal complete 1-D frames")
    if warmup_samples >= segment_length:
        raise ValueError("warmup must leave at least one sample per frame")
    estimate_frames = estimate_array.reshape(-1, segment_length)
    reference_frames = reference_array.reshape(-1, segment_length)
    result = _paired_time_metrics(
        estimate_array,
        reference_array,
        warmup_samples=warmup_samples,
        segment_length=segment_length,
    )
    result["complex_nmse_opendpd_full_frame_db"] = nmse_opendpd_db(
        estimate_frames,
        reference_frames,
    )
    result["complex_nmse_opendpd_scored_interior_db"] = nmse_opendpd_db(
        estimate_frames[:, warmup_samples:],
        reference_frames[:, warmup_samples:],
    )
    return result


def _streaming_equivalence(
    evaluator: FixedPointSparseSplineMemoryDPD,
    signal: np.ndarray,
    frame_lengths: tuple[int, ...],
) -> dict[str, Any]:
    checks: list[bool] = []
    for segment in _segments(signal, frame_lengths):
        full = evaluator.predict_chunk(segment)
        boundaries = sorted(
            {
                0,
                max(1, segment.size // 3),
                max(1, 2 * segment.size // 3),
                segment.size,
            }
        )
        state = evaluator.initial_state()
        pieces: list[np.ndarray] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            if stop <= start:
                continue
            chunk = evaluator.predict_chunk(segment[start:stop], state)
            pieces.append(chunk.output)
            state = chunk.next_state
        streamed = np.concatenate(pieces)
        checks.append(
            bool(
                np.array_equal(streamed, full.output)
                and np.array_equal(
                    state.real_codes,
                    full.next_state.real_codes,
                )
                and np.array_equal(
                    state.imag_codes,
                    full.next_state.imag_codes,
                )
            )
        )
    return {
        "streaming_chunk_equivalence_passed": bool(all(checks)),
        "per_frame": checks,
        "state_reset": "zero at every frame; carried inside each frame",
    }


def _phase_equivariance(
    evaluator: FixedPointSparseSplineMemoryDPD,
    signal: np.ndarray,
    reference_output: np.ndarray,
    frame_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Measure the integer kernel's response to an exact 90-degree rotation."""

    rotated_output, stats, _ = _predict_fixed_frames(
        evaluator,
        1j * signal,
        frame_lengths,
    )
    expected = 1j * np.asarray(reference_output)
    return {
        "rotation_radians": float(np.pi / 2.0),
        "bit_exact": bool(np.array_equal(rotated_output, expected)),
        "rotated_vs_expected": _paired_metrics(
            rotated_output,
            expected,
            warmup_samples=evaluator.model.maximum_delay,
            segment_length=max(frame_lengths),
        ),
        "rotated_input_stats": stats,
    }


def _coefficient_peak(model: SparseSplineMemoryDPD) -> float:
    coefficients = np.asarray(model.coefficients)
    return float(
        max(
            np.max(np.abs(coefficients.real), initial=0.0),
            np.max(np.abs(coefficients.imag), initial=0.0),
        )
    )


def _format_record(fmt: FixedPointFormat) -> dict[str, Any]:
    return {
        "bits": int(fmt.bits),
        "fractional_bits": int(fmt.fractional_bits),
        "scale": float(fmt.scale),
        "representable_minimum": float(fmt.representable_minimum),
        "representable_maximum": float(fmt.representable_maximum),
    }


def _make_fixed_config(
    model: SparseSplineMemoryDPD,
    *,
    bits: int,
    input_peak: float,
    drive_peak: float,
    protocol: dict[str, Any],
) -> tuple[FixedPointDPDConfig, dict[str, Any]]:
    guard = float(protocol["scale_guard_ratio"])
    input_format = FixedPointFormat.for_full_scale(
        bits,
        input_peak,
        label="desired_input",
        guard_ratio=guard,
    )
    output_format = FixedPointFormat.for_full_scale(
        bits,
        drive_peak,
        label="predistorted_drive",
        guard_ratio=guard,
    )
    coefficient_format = FixedPointFormat.for_full_scale(
        bits,
        max(_coefficient_peak(model), np.finfo(float).tiny),
        label="dpd_coefficient",
        guard_ratio=guard,
    )
    power_format = FixedPointFormat(
        int(protocol["power_bits"]),
        input_format.fractional_bits,
        label="envelope_power",
    )
    config = FixedPointDPDConfig(
        input_format=input_format,
        coefficient_format=coefficient_format,
        power_format=power_format,
        accumulator_bits=int(protocol["accumulator_bits"]),
        scalar_accumulator_bits=int(protocol["scalar_accumulator_bits"]),
        output_format=output_format,
        interpolation_fraction_bits=int(
            protocol["interpolation_fraction_bits"]
        ),
    )
    return config, {
        "scale_freeze_source": (
            "train desired-input peak, frozen floating-DPD train-drive peak, "
            "and frozen DPD coefficient peak; validation waveform values not "
            "parsed or used (bytes hash-verified only)"
        ),
        "input_peak_from_train": float(input_peak),
        "drive_peak_from_float_train_dpd": float(drive_peak),
        "coefficient_peak_from_frozen_dpd": _coefficient_peak(model),
        "guard_ratio": guard,
        "input": _format_record(input_format),
        "coefficient": _format_record(coefficient_format),
        "power": _format_record(power_format),
        "output": _format_record(output_format),
    }


def _waveform_metrics(signal: np.ndarray) -> dict[str, float]:
    return {
        "maximum_amplitude": peak_amplitude(signal),
        "papr_db": papr_db(signal),
        "average_power": float(np.mean(np.abs(signal) ** 2)),
    }


def _timing_record(seconds: float, sample_count: int) -> dict[str, Any]:
    if seconds <= 0.0 or sample_count <= 0:
        raise ValueError("timing requires positive duration and sample count")
    return {
        "seconds": float(seconds),
        "samples_per_second": float(sample_count / seconds),
        "microseconds_per_sample": float(seconds * 1e6 / sample_count),
        "scope": (
            "single unpinned Python integer-reference diagnostic; "
            "not a hardware or customer latency gate"
        ),
    }


def _amplitude_ratio_db(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator_peak = peak_amplitude(numerator)
    denominator_peak = peak_amplitude(denominator)
    if denominator_peak <= 0.0:
        raise ValueError("reference peak amplitude must be positive")
    if numerator_peak <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(numerator_peak / denominator_peak))


def _dataset_hashes(
    config: dict[str, Any],
    dataset: Path,
) -> dict[Path, str]:
    return {
        dataset / "spec.json": str(config["dataset_spec_sha256"]),
        dataset / "train_input.csv": str(config["train_input_sha256"]),
        dataset / "val_input.csv": str(config["split_input_sha256"]),
    }


def evaluate(
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run train-only scale freeze and validation-only fixed DPD replay."""

    started = time.perf_counter()
    started_unix_time = time.time()
    config_file = _path(str(config_path), field="config")
    _regular_file(config_file, field="config")
    config_hash = file_sha256(config_file)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain one JSON object")
    validate_config(config)
    if file_sha256(config_file) != config_hash:
        raise RuntimeError("config changed while being parsed")

    output = _path(str(output_dir), field="output_dir")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")

    dataset = _path(config["dataset"], field="dataset")
    if not dataset.is_dir() or dataset.is_symlink():
        raise FileNotFoundError("dataset must be a regular directory")
    model_path = _path(config["model_path"], field="model_path")
    surrogate_path = _path(config["surrogate_path"], field="surrogate_path")
    selection_path = _path(
        config["selection_report"],
        field="selection_report",
    )
    artifact_hashes = _hash_map(
        config["artifact_sha256"],
        field="artifact_sha256",
    )
    expected_artifacts = {model_path, surrogate_path, selection_path}
    if set(artifact_hashes) != expected_artifacts:
        raise ValueError(
            "artifact_sha256 must bind model, surrogate and selection report"
        )
    source_hashes = _hash_map(
        config["source_sha256"],
        field="source_sha256",
    )
    expected_sources = {
        _path(path, field="source") for path in SOURCE_FILES
    }
    if set(source_hashes) != expected_sources:
        raise ValueError(
            "source_sha256 does not bind the declared numerical source set"
        )
    _verify_hash_map(artifact_hashes, label="frozen artifact")
    _verify_hash_map(source_hashes, label="source")

    dataset_hashes = _dataset_hashes(config, dataset)
    _verify_hash_map(dataset_hashes, label="input-only dataset file")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _verify_selection_report(
        config,
        selection,
        model_path=model_path,
        surrogate_path=surrogate_path,
    )
    model = SparseSplineMemoryDPD.load(model_path)
    surrogate = MemoryPolynomialPA.load(surrogate_path)
    _model_contract(config, model, surrogate)

    nperseg = int(config["nperseg"])
    train_input = load_complex_iq_csv(dataset / "train_input.csv")
    if train_input.size != int(config["train_sample_count"]):
        raise ValueError("train input sample count disagrees with config")
    train_lengths = _frame_lengths(train_input.size, nperseg)
    float_train_drive = model.predict_segments(train_input, nperseg)
    input_peak = peak_amplitude(train_input)
    drive_peak = peak_amplitude(float_train_drive)
    fixed_by_bits: dict[
        int,
        tuple[
            FixedPointSparseSplineMemoryDPD,
            dict[str, Any],
        ],
    ] = {}
    protocol = config["fixed_point_protocol"]
    for raw_bits in protocol["activation_bits"]:
        bits = int(raw_bits)
        numeric_config, format_record = _make_fixed_config(
            model,
            bits=bits,
            input_peak=input_peak,
            drive_peak=drive_peak,
            protocol=protocol,
        )
        fixed_by_bits[bits] = (
            FixedPointSparseSplineMemoryDPD(model, numeric_config),
            format_record,
        )

    # Close train-side decisions before the validation waveform is parsed.
    _verify_hash_map(artifact_hashes, label="frozen artifact")
    _verify_hash_map(source_hashes, label="source")
    _verify_hash_map(dataset_hashes, label="input-only dataset file")
    if file_sha256(config_file) != config_hash:
        raise RuntimeError("config changed before validation access")

    validation_input = load_complex_iq_csv(dataset / "val_input.csv")
    if validation_input.size != int(config["validation_sample_count"]):
        raise ValueError("validation input sample count disagrees with config")
    validation_lengths = _frame_lengths(validation_input.size, nperseg)
    if any(length != nperseg for length in validation_lengths):
        raise ValueError("validation contains an incomplete spectral frame")
    gain = _complex_from_json(config["target_gain"], field="target_gain")
    ideal_output = gain * validation_input
    float_validation_drive = model.predict_segments(
        validation_input,
        nperseg,
    )
    no_dpd_output = surrogate.predict_segments(validation_input, nperseg)
    float_dpd_output = surrogate.predict_segments(
        float_validation_drive,
        nperseg,
    )
    cascade_warmup = (
        int(model.maximum_delay) + int(surrogate.causal_warmup_samples)
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "frozen_spline_memory_dpd_fixed_point_validation",
        "claims_scope": {
            "surrogate_only": True,
            "physical_pa_result": False,
            "rf_harmonic_claim": False,
            "test_split_accessed": False,
            "measured_output_opened": False,
            "dpd_latency_gate_evaluable": False,
            "integer_arithmetic_bit_accurate_reference": True,
            "rtl_bit_true": False,
            "hardware_latency_or_resources": False,
            "validation_reused_after_historical_float_model_selection": True,
            "precision_selected_by_runner": False,
            "eligible_as_untouched_final_evidence": False,
        },
        "direction": {
            "deployment_path": (
                "desired validation x -> frozen fixed-point DPD -> "
                "frozen float PA surrogate"
            ),
            "measured_output_used_as_dpd_input": False,
            "inverse_diagnostic_run": False,
        },
        "config": {"path": str(config_file), "sha256": config_hash},
        "dataset": {
            "path": str(dataset),
            "allowed_waveform_files_opened": [
                "train_input.csv",
                "val_input.csv",
            ],
            "measured_output_opened": False,
            "test_split_accessed": False,
            "test_file_hashes_recorded": False,
            "train_sample_count": int(train_input.size),
            "validation_sample_count": int(validation_input.size),
            "train_frame_lengths": train_lengths,
            "validation_frame_lengths": validation_lengths,
            "nperseg": nperseg,
            "input_file_sha256": {
                path.name: digest for path, digest in dataset_hashes.items()
            },
        },
        "frozen_decisions": {
            "selected_family": config["selected_family"],
            "model_path": str(model_path),
            "model_sha256": artifact_hashes[model_path],
            "surrogate_path": str(surrogate_path),
            "surrogate_sha256": artifact_hashes[surrogate_path],
            "selection_report": str(selection_path),
            "selection_report_sha256": artifact_hashes[selection_path],
            "target_gain": {
                "real": float(gain.real),
                "imag": float(gain.imag),
            },
            "integer_delay_samples": 0,
            "formats_frozen_before_validation": True,
            "validation_used_for_scale_selection": False,
            "current_runner_fit_or_architecture_selection_performed": False,
            "frozen_float_model_historically_validation_selected": True,
        },
        "protocol": {
            **protocol,
            "dpd_state_reset": "zero history at every nperseg frame",
            "pa_surrogate_state_reset": "zero history at every nperseg frame",
            "cascade_common_warmup_samples_per_frame": cascade_warmup,
        },
        "float_reference": {
            "dpd_operation_count": model.operation_count().to_dict(),
            "no_dpd_vs_ideal": _paired_metrics(
                no_dpd_output,
                ideal_output,
                warmup_samples=cascade_warmup,
                segment_length=nperseg,
            ),
            "float_dpd_vs_ideal": _paired_metrics(
                float_dpd_output,
                ideal_output,
                warmup_samples=cascade_warmup,
                segment_length=nperseg,
            ),
            "predistorted_drive": _waveform_metrics(
                float_validation_drive
            ),
        },
        "formats": {},
        "execution": {
            "python": sys.version,
            "numpy": np.__version__,
            "started_unix_time": started_unix_time,
        },
    }
    waveform_payloads: dict[int, dict[str, np.ndarray]] = {}
    for bits, (evaluator, format_record) in fixed_by_bits.items():
        train_fixed, train_stats, train_seconds = _predict_fixed_frames(
            evaluator,
            train_input,
            train_lengths,
        )
        validation_fixed, validation_stats, validation_seconds = (
            _predict_fixed_frames(
                evaluator,
                validation_input,
                validation_lengths,
            )
        )
        train_stream = _streaming_equivalence(
            evaluator,
            train_input,
            train_lengths,
        )
        validation_stream = _streaming_equivalence(
            evaluator,
            validation_input,
            validation_lengths,
        )
        validation_phase = _phase_equivariance(
            evaluator,
            validation_input,
            validation_fixed,
            validation_lengths,
        )
        if not (
            train_stream["streaming_chunk_equivalence_passed"]
            and validation_stream["streaming_chunk_equivalence_passed"]
        ):
            raise RuntimeError(f"streaming equivalence failed at {bits} bits")
        fixed_output = surrogate.predict_segments(
            validation_fixed,
            nperseg,
        )
        operation = evaluator.operation_count()
        coefficient_bytes = operation.coefficient_bytes(bits)
        constant_bytes = (
            operation.stored_real_constants
            * evaluator.config.power_format.bits
            + 7
        ) // 8
        report["formats"][str(bits)] = {
            **format_record,
            "fixed_schedule_operation_count": operation.to_dict(),
            "coefficient_memory_bytes": coefficient_bytes,
            "constant_memory_bytes": constant_bytes,
            "coefficient_plus_constant_memory_bytes": (
                coefficient_bytes + constant_bytes
            ),
            "state_memory_bytes": (
                operation.state_real_values * bits + 7
            )
            // 8,
            "train": {
                "fixed_vs_float_drive": _paired_metrics(
                    train_fixed,
                    float_train_drive,
                    warmup_samples=model.maximum_delay,
                    segment_length=nperseg,
                ),
                "fixed_drive": _waveform_metrics(train_fixed),
                "python_integer_reference_timing": _timing_record(
                    train_seconds,
                    train_input.size,
                ),
                "stats": train_stats,
                "streaming": train_stream,
            },
            "validation": {
                "fixed_vs_float_drive": _paired_metrics(
                    validation_fixed,
                    float_validation_drive,
                    warmup_samples=model.maximum_delay,
                    segment_length=nperseg,
                ),
                "fixed_vs_float_cascade": _paired_metrics(
                    fixed_output,
                    float_dpd_output,
                    warmup_samples=cascade_warmup,
                    segment_length=nperseg,
                ),
                "fixed_cascade_vs_ideal": _paired_metrics(
                    fixed_output,
                    ideal_output,
                    warmup_samples=cascade_warmup,
                    segment_length=nperseg,
                ),
                "fixed_drive": _waveform_metrics(validation_fixed),
                "peak_change_vs_float_db": _amplitude_ratio_db(
                    validation_fixed,
                    float_validation_drive,
                ),
                "python_integer_reference_timing": _timing_record(
                    validation_seconds,
                    validation_input.size,
                ),
                "stats": validation_stats,
                "streaming": validation_stream,
                "phase_equivariance": validation_phase,
            },
            "selection_or_tuning": {
                "used_for_selection": False,
                "scales_frozen_before_validation": True,
                "validation_used_to_modify_model": False,
                "precision_selected_by_runner": False,
                "precision_candidates_preregistered": [16, 14, 12],
            },
        }
        waveform_payloads[bits] = {
            "desired_input": validation_input,
            "predistorted_drive": validation_fixed,
            "no_dpd_output": no_dpd_output,
            "dpd_output": fixed_output,
        }

    # Close every input/source TOCTOU window before publication.
    _verify_hash_map(artifact_hashes, label="frozen artifact")
    _verify_hash_map(source_hashes, label="source")
    _verify_hash_map(dataset_hashes, label="input-only dataset file")
    if file_sha256(config_file) != config_hash:
        raise RuntimeError("config changed during fixed-point evaluation")

    report["execution"]["runtime_seconds_before_publication"] = (
        time.perf_counter() - started
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{secrets.token_hex(12)}"
    temporary.mkdir()
    try:
        artifacts: dict[str, str] = {}
        float_filename = "waveforms_float.npz"
        float_waveform_path = temporary / float_filename
        np.savez_compressed(
            float_waveform_path,
            schema_version=np.asarray(1, dtype=np.int64),
            desired_input=validation_input,
            predistorted_drive=float_validation_drive,
            no_dpd_output=no_dpd_output,
            dpd_output=float_dpd_output,
        )
        artifacts[float_filename] = file_sha256(float_waveform_path)
        float_spectral = _spectral_config(
            config,
            output_archive=output / float_filename,
            output_archive_sha256=artifacts[float_filename],
        )
        float_spectral["claim_scope"]["fixed_point_dpd"] = False
        float_spectral["claim_scope"]["numeric_representation"] = (
            "floating_reference"
        )
        float_spectral["claim_scope"][
            "validation_reused_after_historical_selection"
        ] = True
        float_spectral["claim_scope"]["untouched_final_test"] = False
        float_spectral["fixed_point_parent"] = {
            "config_sha256": config_hash,
            "model_sha256": artifact_hashes[model_path],
            "activation_bits": None,
        }
        float_spectral_path = temporary / "spectral_config_float.json"
        write_json(float_spectral_path, float_spectral)
        artifacts["spectral_config_float.json"] = file_sha256(
            float_spectral_path
        )

        for bits, payload in waveform_payloads.items():
            filename = f"waveforms_{bits}bit.npz"
            waveform_path = temporary / filename
            np.savez_compressed(
                waveform_path,
                schema_version=np.asarray(1, dtype=np.int64),
                **payload,
            )
            artifacts[filename] = file_sha256(waveform_path)
            spectral = _spectral_config(
                config,
                output_archive=output / filename,
                output_archive_sha256=artifacts[filename],
            )
            spectral["claim_scope"]["fixed_point_dpd"] = True
            spectral["claim_scope"]["activation_bits"] = bits
            spectral["claim_scope"][
                "validation_reused_after_historical_selection"
            ] = True
            spectral["claim_scope"]["untouched_final_test"] = False
            spectral["fixed_point_parent"] = {
                "config_sha256": config_hash,
                "model_sha256": artifact_hashes[model_path],
                "activation_bits": bits,
            }
            spectral_name = f"spectral_config_{bits}bit.json"
            spectral_path = temporary / spectral_name
            write_json(spectral_path, spectral)
            artifacts[spectral_name] = file_sha256(spectral_path)

        report_path = temporary / "fixed_point_report.json"
        write_json(report_path, report)
        artifacts["fixed_point_report.json"] = file_sha256(report_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": (
                "frozen_spline_memory_dpd_fixed_point_validation_bundle"
            ),
            "config_sha256": config_hash,
            "source_sha256": {
                str(path): digest for path, digest in source_hashes.items()
            },
            "input_hashes": {
                "model": artifact_hashes[model_path],
                "surrogate": artifact_hashes[surrogate_path],
                "selection_report": artifact_hashes[selection_path],
                **{
                    path.name: digest
                    for path, digest in dataset_hashes.items()
                },
            },
            "artifacts": artifacts,
            "fit_performed": False,
            "selection_performed": False,
            "measured_output_opened": False,
            "test_split_accessed": False,
            "validation_reused_after_historical_selection": True,
            "precision_selected_by_runner": False,
            "atomic_publication": True,
        }
        manifest_path = temporary / "completion_manifest.json"
        write_json(manifest_path, manifest)

        # Nothing below this check may reopen a frozen source or input.  This
        # closes the publication-time TOCTOU window after spectral configs and
        # every output artifact have already been materialized.
        _verify_hash_map(artifact_hashes, label="frozen artifact")
        _verify_hash_map(source_hashes, label="source")
        _verify_hash_map(dataset_hashes, label="input-only dataset file")
        if file_sha256(config_file) != config_hash:
            raise RuntimeError("config changed before atomic publication")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing publication race at: {output}")
        os.replace(temporary, output)
        temporary = None  # type: ignore[assignment]
        return report | {
            "artifacts": {
                "bundle": str(output),
                "fixed_point_report": str(
                    output / "fixed_point_report.json"
                ),
                "completion_manifest": str(
                    output / "completion_manifest.json"
                ),
            }
        }
    finally:
        if temporary is not None and temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    evaluate(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
