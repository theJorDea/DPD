"""Evaluate frozen PA models under a preregistered fixed-point contract.

This runner is intentionally narrower than the DPD evaluators.  It opens only
the explicitly allowed ``train`` and ``val`` waveform files, freezes every
numeric scale from train data and frozen model coefficients, and then reports
fixed-point degradation for the causal GMP and sparse spline-memory PA
artifacts.  It never calls the ``test`` split loader.

The output is a PA-model arithmetic report, not a claim about RTL timing,
FPGA resources, or physical-PA linearization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.fixed_point_pa import (  # noqa: E402
    FixedPointFormat,
    FixedPointPAConfig,
    FixedPointPAStats,
    FixedPointGMPPA,
)
from baseline.fixed_point_sparse_spline_pa import (  # noqa: E402
    FixedPointSparseSplineMemoryPA,
)
from baseline.gmp_pa import (  # noqa: E402
    GeneralizedMemoryPolynomialPA,
)
from baseline.metrics import nmse_pooled_db  # noqa: E402
from baseline.sparse_spline_memory_pa import SparseSplineMemoryPA  # noqa: E402
from baseline.train_spline import (  # noqa: E402
    load_split_pair,
    write_json,
)
from experiments.sparse_pa_benchmark_support import (  # noqa: E402
    metric_summary,
    partition_lengths,
)
from experiments.select_pa_sparse_spline_memory import (  # noqa: E402
    frame_segments,
)


Progress = Callable[[str], None]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str, *, label: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the project root") from error
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"config must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", -1)) != 1:
        raise ValueError("fixed-point config must be schema_version 1")
    policy = value.get("access_policy")
    if not isinstance(policy, dict):
        raise ValueError("config.access_policy is required")
    if tuple(policy.get("allowed_splits", ())) != ("train", "val"):
        raise ValueError("fixed-point runner allows exactly train and val")
    if policy.get("test_split_accessed") is not False:
        raise ValueError("config must seal test_split_accessed=false")
    contract = value.get("dataset_contract")
    if not isinstance(contract, dict):
        raise ValueError("config.dataset_contract is required")
    required = {
        "dataset",
        "output_dir",
        "frozen_models",
        "fixed_point_protocol",
        "dataset_contract",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"config is missing keys: {missing}")
    if int(contract.get("alignment_delay_samples", 0)) != 0:
        raise ValueError(
            "this frozen runner requires the preregistered zero alignment delay"
        )
    if bool(contract.get("fractional_delay_applied", False)):
        raise ValueError(
            "this frozen runner does not silently apply a fractional delay"
        )
    return value


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256(path)
    if actual != str(expected):
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )


def _load_frozen_models(config: dict[str, Any]) -> dict[str, Any]:
    raw_models = config["frozen_models"]
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("frozen_models must be a non-empty object")
    loaded: dict[str, Any] = {}
    for name, record in raw_models.items():
        if not isinstance(record, dict):
            raise ValueError(f"frozen_models.{name} must be an object")
        path = _project_path(str(record["path"]), label=f"model {name}")
        _verify_hash(path, str(record["sha256"]), label=f"model {name}")
        model_type = str(record["type"])
        if model_type == "causal_gmp":
            model = GeneralizedMemoryPolynomialPA.load(path)
        elif model_type == "sparse_spline_memory_pa":
            model = SparseSplineMemoryPA.load(path)
        else:
            raise ValueError(f"unsupported frozen model type: {model_type}")
        loaded[str(name)] = {
            "record": record,
            "path": path,
            "model": model,
        }
    return loaded


def _validate_dataset_contract(
    dataset: Path,
    contract: dict[str, Any],
) -> None:
    required_hashes = contract.get("required_files_sha256")
    if not isinstance(required_hashes, dict):
        raise ValueError("dataset_contract.required_files_sha256 is required")
    for filename, expected in required_hashes.items():
        if str(filename).startswith("test"):
            raise ValueError("test files are forbidden in fixed-point config")
        path = dataset / str(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        _verify_hash(path, str(expected), label=f"dataset/{filename}")


def _load_allowed_split(
    dataset: Path,
    split: str,
    *,
    allowed: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if split not in allowed:
        raise RuntimeError(f"attempted forbidden split access: {split}")
    return load_split_pair(dataset, split)


def _coefficient_peak(model: Any) -> float:
    coefficients = np.asarray(model.coefficients)
    return float(
        max(
            np.max(np.abs(coefficients.real), initial=0.0),
            np.max(np.abs(coefficients.imag), initial=0.0),
        )
    )


def _model_peak(model: Any, prediction: np.ndarray) -> float:
    return float(np.max(np.abs(prediction), initial=0.0))


def _format_record(fmt: FixedPointFormat) -> dict[str, Any]:
    return {
        "bits": int(fmt.bits),
        "fractional_bits": int(fmt.fractional_bits),
        "scale": float(fmt.scale),
        "representable_minimum": float(fmt.representable_minimum),
        "representable_maximum": float(fmt.representable_maximum),
    }


def _make_fixed_config(
    model: Any,
    *,
    bits: int,
    input_peak: float,
    output_peak: float,
    protocol: dict[str, Any],
) -> tuple[FixedPointPAConfig, dict[str, Any]]:
    guard_ratio = float(protocol["scale_guard_ratio"])
    input_format = FixedPointFormat.for_full_scale(
        bits,
        input_peak,
        label="input",
        guard_ratio=guard_ratio,
    )
    output_format = FixedPointFormat.for_full_scale(
        bits,
        output_peak,
        label="output",
        guard_ratio=guard_ratio,
    )
    coefficient_format = FixedPointFormat.for_full_scale(
        bits,
        max(_coefficient_peak(model), np.finfo(float).tiny),
        label="coefficient",
        guard_ratio=guard_ratio,
    )
    power_format = FixedPointFormat(
        int(protocol["power_bits"]),
        input_format.fractional_bits,
        label="power",
    )
    config = FixedPointPAConfig(
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
        "input_peak_from_train": float(input_peak),
        "output_peak_from_train_and_float_model": float(output_peak),
        "coefficient_peak_from_frozen_model": _coefficient_peak(model),
        "input": _format_record(input_format),
        "coefficient": _format_record(coefficient_format),
        "power": _format_record(power_format),
        "output": _format_record(output_format),
        "guard_ratio": guard_ratio,
    }


def _fixed_evaluator(model: Any, config: FixedPointPAConfig) -> Any:
    if isinstance(model, GeneralizedMemoryPolynomialPA):
        return FixedPointGMPPA(model, config)
    if isinstance(model, SparseSplineMemoryPA):
        return FixedPointSparseSplineMemoryPA(model, config)
    raise TypeError(f"unsupported model class: {type(model)!r}")


def _stats_aggregate(stats: list[FixedPointPAStats]) -> dict[str, int]:
    if not stats:
        raise ValueError("at least one fixed-point frame is required")
    fields = tuple(stats[0].to_dict())
    result: dict[str, int] = {}
    additive = {
        "sample_count",
        "input_saturations",
        "coefficient_saturations",
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
    for field in fields:
        values = [int(getattr(item, field)) for item in stats]
        if field in per_model_static:
            result[field] = int(max(values))
        elif field in additive:
            result[field] = int(sum(values))
        else:
            result[field] = int(max(values))
    return result


def _predict_fixed_frames(
    evaluator: Any,
    signal: np.ndarray,
    frame_lengths: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, int], float]:
    segments = frame_segments(signal, frame_lengths)
    outputs: list[np.ndarray] = []
    stats: list[FixedPointPAStats] = []
    started = time.perf_counter()
    for segment in segments:
        result = evaluator.predict_chunk(segment)
        outputs.append(result.output)
        stats.append(result.stats)
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), _stats_aggregate(stats), float(elapsed)


def _streaming_equivalence(
    evaluator: Any,
    signal: np.ndarray,
    frame_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Check chunk equivalence inside each reset frame."""

    checks: list[bool] = []
    for segment in frame_segments(signal, frame_lengths):
        full = evaluator.predict_chunk(segment)
        split_a = max(1, segment.size // 3)
        split_b = max(split_a + 1, 2 * segment.size // 3)
        first = evaluator.predict_chunk(segment[:split_a])
        second = evaluator.predict_chunk(
            segment[split_a:split_b],
            first.next_state,
        )
        third = evaluator.predict_chunk(
            segment[split_b:],
            second.next_state,
        )
        streamed = np.concatenate(
            (first.output, second.output, third.output)
        )
        checks.append(
            bool(
                np.array_equal(streamed, full.output)
                and np.array_equal(
                    third.next_state.real_codes,
                    full.next_state.real_codes,
                )
                and np.array_equal(
                    third.next_state.imag_codes,
                    full.next_state.imag_codes,
                )
            )
        )
    return {
        "streaming_chunk_equivalence_passed": bool(all(checks)),
        "per_frame": checks,
    }


def _model_warmup(evaluator: Any) -> int:
    return int(evaluator.history_length)


def _model_operation_count(model: Any) -> dict[str, Any]:
    operation = model.operation_count
    operation = operation() if callable(operation) else operation
    return operation.to_dict()


def _fixed_operation_count(evaluator: Any) -> dict[str, Any]:
    return evaluator.operation_count().to_dict()


def run_from_config(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    progress: Progress = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Run the sealed train-scale → validation-description evaluation."""

    started = time.perf_counter()
    config_file = Path(config_path).resolve()
    config_hash = _sha256(config_file)
    config = _load_config(config_file)
    if _sha256(config_file) != config_hash:
        raise RuntimeError("config changed while being parsed")
    dataset = _project_path(str(config["dataset"]), label="dataset")
    contract = config["dataset_contract"]
    _validate_dataset_contract(dataset, contract)
    models = _load_frozen_models(config)
    protocol = config["fixed_point_protocol"]
    bits_list = tuple(int(value) for value in protocol["activation_bits"])
    if not bits_list or len(set(bits_list)) != len(bits_list):
        raise ValueError("activation_bits must be unique and non-empty")
    if any(bits < 4 or bits > 24 for bits in bits_list):
        raise ValueError("activation_bits must be in [4, 24]")
    allowed_splits = ("train", "val")

    progress("[integrity] dataset and frozen model hashes verified")
    train_input, train_output = _load_allowed_split(
        dataset,
        "train",
        allowed=allowed_splits,
    )
    if train_input.size != int(contract["train_sample_count"]):
        raise ValueError("train sample count disagrees with contract")
    train_lengths = tuple(int(x) for x in contract["train_frame_lengths"])
    if sum(train_lengths) != train_input.size:
        raise ValueError("train frame lengths do not sum to train samples")
    frame_length = int(contract["frame_length"])
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")

    # All scale decisions happen here, before validation is opened.
    frozen: dict[str, Any] = {}
    for name, record in models.items():
        model = record["model"]
        float_train = model.predict_segments(train_input, frame_length)
        input_peak = float(np.max(np.abs(train_input), initial=0.0))
        measured_output_peak = float(
            np.max(np.abs(train_output), initial=0.0)
        )
        output_peak = max(measured_output_peak, _model_peak(model, float_train))
        frozen[name] = {
            "float_train_prediction": float_train,
            "input_peak": input_peak,
            "measured_output_peak": measured_output_peak,
            "output_peak": output_peak,
            "coefficient_peak": _coefficient_peak(model),
        }
    progress("[freeze] input/output/coefficient scales frozen from train only")

    # Validation is deliberately loaded only after all scales are immutable.
    validation_input, validation_output = _load_allowed_split(
        dataset,
        "val",
        allowed=allowed_splits,
    )
    if validation_input.size != int(contract["validation_sample_count"]):
        raise ValueError("validation sample count disagrees with contract")
    validation_lengths = tuple(
        int(x) for x in contract["validation_frame_lengths"]
    )
    if sum(validation_lengths) != validation_input.size:
        raise ValueError("validation frame lengths do not sum to val samples")

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "frozen_pa_fixed_point_train_validation",
        "claims_scope": {
            "physical_pa_result": False,
            "dpd_linearization_result": False,
            "rtl_bit_true": False,
            "hardware_latency_or_resources": False,
            "python_runtime_only": True,
        },
        "config": {
            "path": str(config_file),
            "sha256": config_hash,
        },
        "dataset": {
            "path": str(dataset),
            "allowed_splits_opened": ["train", "val"],
            "test_split_accessed": False,
            "test_file_hashes_recorded": False,
            "train_sample_count": int(train_input.size),
            "validation_sample_count": int(validation_input.size),
            "train_frame_lengths": train_lengths,
            "validation_frame_lengths": validation_lengths,
            "alignment_delay_samples": int(
                contract["alignment_delay_samples"]
            ),
            "required_files_sha256": contract["required_files_sha256"],
        },
        "protocol": protocol,
        "models": {},
        "execution": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "started_unix_time": time.time(),
        },
    }

    for name, record in models.items():
        model = record["model"]
        frozen_record = frozen[name]
        model_report: dict[str, Any] = {
            "type": record["record"]["type"],
            "path": str(record["path"]),
            "sha256": str(record["record"]["sha256"]),
            "float_reference_operation_count": _model_operation_count(model),
            "float_train_peak": _model_peak(
                model,
                frozen_record["float_train_prediction"],
            ),
            "train_measured_output_peak": frozen_record[
                "measured_output_peak"
            ],
            "scale_freeze_source": (
                "train input peak, train measured output peak, and frozen "
                "model train prediction peak; no validation values"
            ),
            "formats": {},
        }
        warmup = None
        for bits in bits_list:
            fixed_config, format_record = _make_fixed_config(
                model,
                bits=bits,
                input_peak=frozen_record["input_peak"],
                output_peak=frozen_record["output_peak"],
                protocol=protocol,
            )
            evaluator = _fixed_evaluator(model, fixed_config)
            if warmup is None:
                warmup = _model_warmup(evaluator)
            elif warmup != _model_warmup(evaluator):
                raise RuntimeError(
                    "fixed-point evaluator changed causal history across bit widths"
                )
            train_float = frozen_record["float_train_prediction"]
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
            train_float_metrics = metric_summary(
                train_float,
                train_output,
                frame_lengths=train_lengths,
                common_warmup=_model_warmup(evaluator),
            )
            train_fixed_metrics = metric_summary(
                train_fixed,
                train_output,
                frame_lengths=train_lengths,
                common_warmup=_model_warmup(evaluator),
            )
            validation_float = model.predict_segments(
                validation_input,
                frame_length,
            )
            validation_float_metrics = metric_summary(
                validation_float,
                validation_output,
                frame_lengths=validation_lengths,
                common_warmup=_model_warmup(evaluator),
            )
            validation_fixed_metrics = metric_summary(
                validation_fixed,
                validation_output,
                frame_lengths=validation_lengths,
                common_warmup=_model_warmup(evaluator),
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
            if not (
                train_stream["streaming_chunk_equivalence_passed"]
                and validation_stream["streaming_chunk_equivalence_passed"]
            ):
                raise RuntimeError(
                    f"streaming equivalence failed for {name} {bits}-bit"
                )
            format_report = {
                **format_record,
                "fixed_schedule_operation_count": _fixed_operation_count(
                    evaluator
                ),
                "fixed_schedule_coefficient_memory_bytes": int(
                    evaluator.operation_count().coefficient_bytes(bits)
                ),
                "fixed_schedule_state_memory_bytes": int(
                    (
                        evaluator.operation_count().state_real_values
                        * bits
                        + 7
                    )
                    // 8
                ),
                "causal_history_samples": _model_warmup(evaluator),
                "train": {
                    "fixed_metrics": train_fixed_metrics,
                    "float_reference_metrics": train_float_metrics,
                    "fixed_vs_float_nmse_db": nmse_pooled_db(
                        train_fixed,
                        train_float,
                    ),
                    "inference_seconds": train_seconds,
                    "stats": train_stats,
                    "streaming": train_stream,
                },
                "validation": {
                    "fixed_metrics": validation_fixed_metrics,
                    "float_reference_metrics": validation_float_metrics,
                    "fixed_vs_float_nmse_db": nmse_pooled_db(
                        validation_fixed,
                        validation_float,
                    ),
                    "inference_seconds": validation_seconds,
                    "stats": validation_stats,
                    "streaming": validation_stream,
                },
                "selection_or_tuning": {
                    "used_for_selection": False,
                    "scales_frozen_before_validation": True,
                    "validation_used_to_modify_model": False,
                },
            }
            model_report["formats"][str(bits)] = format_report
            progress(
                f"[{name} {bits}-bit] train fixed NMSE "
                f"{train_fixed_metrics['full_record_nmse_db']:.4f} dB; "
                f"val fixed NMSE "
                f"{validation_fixed_metrics['full_record_nmse_db']:.4f} dB"
            )
        report["models"][name] = model_report

    report["execution"]["runtime_seconds"] = time.perf_counter() - started
    if output_path is not None:
        destination = Path(output_path).resolve()
    else:
        destination = (PROJECT_ROOT / config["output_dir"] / "fixed_point_report.json").resolve()
        try:
            destination.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError("default output escapes the project root") from error
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {destination}; use --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(destination)
    write_json(destination, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen PA models in train/validation only under the "
            "preregistered fixed-point contract."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_from_config(
        args.config,
        output_path=args.output_json,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
