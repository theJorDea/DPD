"""Run fixed-point/FP16-like emulation on a frozen spline test artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.evaluate_spline import _load_artifacts, _verify_dataset_identity
from baseline.fixed_point import (
    FixedPointConfig,
    predict_fixed_point,
    predict_fp16_storage,
)
from baseline.metrics import (
    nmse_pooled_db,
    opendpd_aclr_db,
    opendpd_spectral_evm_db,
    papr_db,
    peak_amplitude,
    time_domain_rms_evm_db,
)
from baseline.pa_models import MemoryPolynomialPA
from baseline.train_spline import (
    _paired_time_metrics,
    align_split_pair,
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


def _time_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    warmup: int,
    segment_length: int,
) -> dict[str, Any]:
    return _paired_time_metrics(
        estimate,
        reference,
        warmup_samples=warmup,
        segment_length=segment_length,
    )


def evaluate(
    dataset: Path,
    training_report_path: Path,
    surrogate_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset = dataset.resolve()
    training_report_path = training_report_path.resolve()
    surrogate_path = surrogate_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}")

    report, model, gain = _load_artifacts(
        training_report_path,
        allow_artifact_mismatch=False,
    )
    model_path = Path(report["artifacts"]["spline_model"])
    if not model_path.is_absolute():
        model_path = training_report_path.parent / model_path
    dataset_identity = _verify_dataset_identity(
        report,
        dataset,
        allow_mismatch=False,
    )
    expected_surrogate = report.get("pa_surrogate")
    if not isinstance(expected_surrogate, dict):
        raise ValueError("training report did not freeze a PA surrogate")
    actual_surrogate_hash = file_sha256(surrogate_path)
    if actual_surrogate_hash != expected_surrogate.get("artifact_sha256"):
        raise ValueError("PA surrogate hash does not match training report")
    surrogate = MemoryPolynomialPA.load(surrogate_path)
    raw_test_input, raw_test_output = load_split_pair(dataset, "test")
    alignment_delay = int(
        report["alignment"]["frozen_integer_delay_samples"]
    )
    test_input, _ = align_split_pair(
        raw_test_input,
        raw_test_output,
        delay=alignment_delay,
    )
    spec = load_dataset_spec(dataset)
    segment_length = int(spec["nperseg"])
    target = gain * test_input
    warmup = surrogate.causal_warmup_samples
    float_dpd = model.predict(test_input)
    float_cascade = surrogate.predict_segments(float_dpd, segment_length)

    coefficient_full_scale = float(
        max(
            np.max(np.abs(model.coefficients.real), initial=0.0),
            np.max(np.abs(model.coefficients.imag), initial=0.0),
        )
    )
    fixed_formats = {
        "fp16_like_storage": None,
        "signed16_reference": FixedPointConfig(
            input_bits=16,
            coefficient_bits=16,
            input_full_scale=1.0,
            coefficient_full_scale=coefficient_full_scale,
            accumulator_bits=40,
            output_bits=16,
            output_full_scale=1.5,
        ),
        "signed12_reference": FixedPointConfig(
            input_bits=12,
            coefficient_bits=12,
            input_full_scale=1.0,
            coefficient_full_scale=coefficient_full_scale,
            accumulator_bits=32,
            output_bits=12,
            output_full_scale=1.5,
        ),
    }

    def spectral_metrics(signal: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
        if not spec:
            return {}
        fs = float(spec["input_signal_fs"])
        bw = float(spec["bw_main_ch"])
        n_sub = int(spec["n_sub_ch"])
        nperseg = int(spec["nperseg"])
        aclr = opendpd_aclr_db(
            signal,
            fs=fs,
            nperseg=nperseg,
            bandwidth_main=bw,
            n_subchannels=n_sub,
        )
        return {
            "opendpd_spectral_evm_db": opendpd_spectral_evm_db(
                signal,
                reference,
                fs=fs,
                bandwidth_main=bw,
                n_subchannels=n_sub,
                nperseg=nperseg,
            ),
            "opendpd_aclr_db": {
                "left": aclr.left_db,
                "right": aclr.right_db,
                "average": aclr.average_db,
            },
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "frozen_spline_fixed_point_emulation",
        "scope": "surrogate_only_when_cascade_metrics_are_present",
        "claims_scope": {
            "physical_pa_result": False,
            "bit_true_rtl": False,
            "hardware_latency_or_resources": False,
            "integer_emulation": (
                "numerical arithmetic reference; magnitude uses rounded "
                "NumPy sqrt and is not an RTL implementation"
            ),
            "fp16_emulation": (
                "component FP16 storage with explicit FP32 interpolation and "
                "product; not a vendor accelerator claim"
            ),
        },
        "dataset": {
            "directory": str(dataset),
            "identity_check": dataset_identity,
            "test_input_sha256": file_sha256(dataset / "test_input.csv"),
            "raw_test_samples": int(raw_test_input.size),
            "aligned_test_samples": int(test_input.size),
        },
        "training_report": str(training_report_path),
        "spline_model_sha256": file_sha256(model_path),
        "pa_surrogate": {
            "path": str(surrogate_path),
            "sha256": actual_surrogate_hash,
            "metadata": surrogate.metadata,
        },
        "selected_configuration": report["selection"]["selected_configuration"],
        "target_gain": gain,
        "test_was_not_used_for_selection": True,
        "alignment": {
            "frozen_integer_delay_samples": alignment_delay,
            "test_delay_retuned": False,
        },
        "framing": {
            "segment_length": segment_length,
            "surrogate_state_reset": "zero at every segment boundary",
            "warmup_samples_per_segment": warmup,
        },
        "quantization_scales_frozen_before_test_scoring": {
            "input_full_scale": 1.0,
            "coefficient_full_scale": coefficient_full_scale,
            "output_full_scale": 1.5,
            "scale_source": (
                "protocol constant for input/output; coefficient scale from "
                "frozen training-selected model only"
            ),
        },
        "complex128_floating_reference": {
            "dpd_output": {
                "peak_amplitude": peak_amplitude(float_dpd),
                "papr_db": papr_db(float_dpd),
            },
            "cascade_metrics": _time_metrics(
                float_cascade,
                target,
                warmup,
                segment_length,
            ),
            "spectral_metrics": spectral_metrics(float_cascade, target),
        },
        "formats": {},
    }

    for name, config in fixed_formats.items():
        if config is None:
            quantized_dpd = predict_fp16_storage(model, test_input)
            metadata: dict[str, Any] = {
                "emulation": (
                    "real/imaginary values rounded to float16 storage and "
                    "promoted to float32 arithmetic"
                )
            }
        else:
            fixed = predict_fixed_point(model, test_input, config)
            quantized_dpd = fixed.output
            metadata = {
                "input_bits": config.input_bits,
                "coefficient_bits": config.coefficient_bits,
                "interpolation_fraction_bits": config.interpolation_fraction_bits,
                "accumulator_bits": config.accumulator_bits,
                "input_saturations": fixed.input_saturations,
                "coefficient_saturations": fixed.coefficient_saturations,
                "accumulator_saturations": fixed.accumulator_saturations,
                "output_saturations": fixed.output_saturations,
                "maximum_accumulator_magnitude": fixed.maximum_accumulator_magnitude,
                "knot_code_collision_count": fixed.knot_code_collision_count,
                "maximum_knot_code_shift": fixed.maximum_knot_code_shift,
                "input_scale": fixed.input_scale,
                "coefficient_scale": fixed.coefficient_scale,
                "output_scale": fixed.output_scale,
                "output_bits": config.output_bits,
                "output_full_scale": config.output_full_scale,
            }
        cascade = surrogate.predict_segments(quantized_dpd, segment_length)
        rotation = np.exp(1j * 0.731)
        if config is None:
            rotated_dpd = predict_fp16_storage(model, test_input * rotation)
            rotation_saturations = None
        else:
            rotated_fixed = predict_fixed_point(
                model,
                test_input * rotation,
                config,
            )
            rotated_dpd = rotated_fixed.output
            rotation_saturations = {
                "input": rotated_fixed.input_saturations,
                "coefficient": rotated_fixed.coefficient_saturations,
                "accumulator": rotated_fixed.accumulator_saturations,
                "output": rotated_fixed.output_saturations,
            }
        metadata.update(
            {
                "output_bits": (
                    config.output_bits if config is not None else None
                ),
                "output_full_scale": (
                    config.output_full_scale if config is not None else None
                ),
                "dpd_output_vs_float_nmse_db": nmse_pooled_db(
                    quantized_dpd,
                    float_dpd,
                ),
                "dpd_output_peak_amplitude": peak_amplitude(quantized_dpd),
                "dpd_output_papr_db": papr_db(quantized_dpd),
                "phase_equivariance_nmse_db": nmse_pooled_db(
                    rotated_dpd,
                    quantized_dpd * rotation,
                ),
                "phase_equivariance_rotation_rad": 0.731,
                "phase_equivariance_rotation_saturations": rotation_saturations,
                "cascade_metrics": _time_metrics(
                    cascade,
                    target,
                    warmup,
                    segment_length,
                ),
                "spectral_metrics": spectral_metrics(cascade, target),
            }
        )
        result["formats"][name] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--pa-surrogate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.dataset,
        args.training_report,
        args.pa_surrogate,
        args.output_json,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
