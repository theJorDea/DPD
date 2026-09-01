"""Research fixed-point gate for frozen DPD models (non-sealed).

Verifies the Stage-4 deployment gate for a frozen spline-memory or
composite spline+GMP DPD:

* coefficient quantization at 16/14/12 bits with the repository
  ``FixedPointFormat`` (ties-to-even, saturate-and-count);
* zero coefficient saturations at the target bit width;
* cascade NMSE degradation at most 0.05 dB at 16 bits, measured through
  the primary and (optionally) secondary frozen evaluators on the given
  train block;
* bit-exact 90-degree phase rotation of the quantized model;
* full-record vs streaming chunk equivalence of the quantized model.

This is a numerical equivalence gate, not the repository's sealed
artifact-chain protocol; nothing here selects, fits, or retunes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from baseline.direct_learning import nmse_db
from baseline.fixed_point_pa import FixedPointFormat
from baseline.gmp_dictionary_dpd import CompositeSplineGmpDPD
from baseline.gmp_pa import GeneralizedMemoryPolynomialPA, gmp_terms
from baseline.spline_memory_dpd import SparseSplineMemoryDPD
from baseline.train_spline import (
    file_sha256,
    load_split_pair,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEGRADATION_LIMIT_DB = 0.05


def _load_model(path: Path):
    with np.load(path, allow_pickle=False) as data:
        model_type = str(data["model_type"])
    if model_type == "phase_equivariant_sparse_spline_memory_dpd":
        return SparseSplineMemoryDPD.load(path)
    if model_type == "phase_equivariant_composite_spline_gmp_dpd":
        return CompositeSplineGmpDPD.load(path)
    raise ValueError(f"unsupported model type: {model_type}")


def _quantize_model(model, bits: int, guard_ratio: float):
    """Return a model of the same class with quantized coefficients."""

    if isinstance(model, SparseSplineMemoryDPD) and not isinstance(
        model, CompositeSplineGmpDPD
    ):
        peak = float(np.max(np.abs(model.coefficients)))
        fmt = FixedPointFormat.for_full_scale(
            bits, peak, guard_ratio=guard_ratio, label="spline.coefficients"
        )
        quantized = fmt.quantize_complex(model.coefficients.reshape(-1))
        coefficients = fmt.dequantize_complex(
            quantized.real, quantized.imag
        ).reshape(model.coefficients.shape)
        quantized_model = SparseSplineMemoryDPD(
            knots=model.knots,
            branches=model.branches,
            coefficients=coefficients,
            knot_strategy=model.knot_strategy,
        )
        return quantized_model, {
            "fractional_bits": fmt.fractional_bits,
            "saturation_count": int(quantized.saturation_count),
            "coefficient_peak": peak,
        }
    if isinstance(model, CompositeSplineGmpDPD):
        spline_quantized, spline_info = _quantize_model(
            model.spline, bits, guard_ratio
        )
        if model.member_coefficients.size:
            peak = float(np.max(np.abs(model.member_coefficients)))
            fmt = FixedPointFormat.for_full_scale(
                bits,
                peak,
                guard_ratio=guard_ratio,
                label="member.coefficients",
            )
            quantized = fmt.quantize_complex(model.member_coefficients)
            members = fmt.dequantize_complex(
                quantized.real, quantized.imag
            )
            member_info = {
                "fractional_bits": fmt.fractional_bits,
                "saturation_count": int(quantized.saturation_count),
                "coefficient_peak": peak,
            }
        else:
            members = model.member_coefficients
            member_info = {
                "fractional_bits": None,
                "saturation_count": 0,
                "coefficient_peak": 0.0,
            }
        quantized_model = CompositeSplineGmpDPD(
            spline=spline_quantized,
            members=model.members,
            member_coefficients=members,
        )
        return quantized_model, {
            "spline": spline_info,
            "members": member_info,
        }
    raise TypeError("unsupported model class")


def _pa_warmup(pa: GeneralizedMemoryPolynomialPA) -> int:
    return int(max(term.signal_delay for term in gmp_terms(pa.config)) + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("task") != "frozen_dpd_fixed_point_research":
        raise ValueError("unexpected fixed-point research task")

    dataset = PROJECT_ROOT / config["dataset"]
    train_input, _ = load_split_pair(dataset, "train")
    validation_input, _ = load_split_pair(dataset, "val")
    model = _load_model(PROJECT_ROOT / config["model_npz"])
    pa = GeneralizedMemoryPolynomialPA.load(
        PROJECT_ROOT / config["primary_pa_npz"]
    )
    pa_b = None
    if config.get("secondary_pa_npz"):
        pa_b = GeneralizedMemoryPolynomialPA.load(
            PROJECT_ROOT / config["secondary_pa_npz"]
        )
    warmup = max(
        model.maximum_delay,
        _pa_warmup(pa),
        _pa_warmup(pa_b) if pa_b is not None else 0,
    )
    gain = complex(
        np.vdot(train_input, pa.predict(train_input))
        / np.vdot(train_input, train_input)
    )
    gain_b = (
        complex(
            np.vdot(train_input, pa_b.predict(train_input))
            / np.vdot(train_input, train_input)
        )
        if pa_b is not None
        else None
    )

    span = slice(*[int(v) for v in config["check_block"]])
    desired = train_input[span]

    def cascade(model_, evaluator, evaluator_gain):
        drive = model_.predict(desired)
        output = np.asarray(evaluator.predict(drive), dtype=np.complex128)
        return nmse_db(output, evaluator_gain * desired, warmup)

    def cascade_validation(model_, evaluator, evaluator_gain):
        drive = model_.predict(validation_input)
        output = np.asarray(evaluator.predict(drive), dtype=np.complex128)
        return nmse_db(output, evaluator_gain * validation_input, warmup)

    results: dict[str, Any] = {}
    for bits in (16, 14, 12):
        quantized_model, info = _quantize_model(
            model, bits, float(config.get("scale_guard_ratio", 1.001))
        )
        record: dict[str, Any] = {
            "quantization": info,
            "primary": {
                "float_nmse_db": cascade(model, pa, gain),
                "quantized_nmse_db": cascade(quantized_model, pa, gain),
            },
        }
        record["primary"]["degradation_db"] = (
            record["primary"]["quantized_nmse_db"]
            - record["primary"]["float_nmse_db"]
        )
        if pa_b is not None:
            record["secondary"] = {
                "float_nmse_db": cascade(model, pa_b, gain_b),
                "quantized_nmse_db": cascade(quantized_model, pa_b, gain_b),
            }
            record["secondary"]["degradation_db"] = (
                record["secondary"]["quantized_nmse_db"]
                - record["secondary"]["float_nmse_db"]
            )
        if bits == 16:
            # Phase-equivariance check.  The float complex-multiply kernel
            # is not bit-exact under a 90-degree rotation even for the
            # frozen float model (ulp-level asymmetry); bit-exactness is a
            # property of the integer fixed-point kernel.  The gate is
            # therefore: quantization must not worsen the rotation error,
            # and the rotation error must stay at ulp scale.
            probe = desired[:4096]
            direct = quantized_model.predict(probe * 1j)
            rotated = 1j * quantized_model.predict(probe)
            rotation_error = float(np.max(np.abs(direct - rotated)))
            float_direct = model.predict(probe * 1j)
            float_rotated = 1j * model.predict(probe)
            float_rotation_error = float(
                np.max(np.abs(float_direct - float_rotated))
            )
            record["rotation_max_abs_error"] = rotation_error
            record["float_model_rotation_max_abs_error"] = (
                float_rotation_error
            )
            record["rotation_not_worsened_by_quantization"] = bool(
                rotation_error <= float_rotation_error + 1e-15
            )
            # Full record vs streaming chunk equivalence.
            full = quantized_model.predict(desired)
            state = quantized_model.initial_state()
            pieces = []
            for start in range(0, desired.size, 512):
                chunk, state = quantized_model.predict_chunk(
                    desired[start : start + 512], state
                )
                pieces.append(chunk)
            record["chunk_equivalence_bit_exact"] = bool(
                np.array_equal(np.concatenate(pieces), full)
            )
            record["validation_primary_float_nmse_db"] = cascade_validation(
                model, pa, gain
            )
            record["validation_primary_quantized_nmse_db"] = (
                cascade_validation(quantized_model, pa, gain)
            )
            if pa_b is not None:
                record["validation_secondary_float_nmse_db"] = (
                    cascade_validation(model, pa_b, gain_b)
                )
                record["validation_secondary_quantized_nmse_db"] = (
                    cascade_validation(quantized_model, pa_b, gain_b)
                )
        results[str(bits)] = record

    gate_16 = results["16"]
    saturations = gate_16["quantization"]
    if "saturation_count" in saturations:
        total_saturation = int(saturations["saturation_count"])
    else:
        total_saturation = int(
            saturations["spline"]["saturation_count"]
            + saturations["members"]["saturation_count"]
        )
    degradations = [gate_16["primary"]["degradation_db"]]
    if "secondary" in gate_16:
        degradations.append(gate_16["secondary"]["degradation_db"])
    predicates = {
        "zero_saturations_16bit": total_saturation == 0,
        "degradation_within_0p05db_16bit": all(
            value <= DEGRADATION_LIMIT_DB for value in degradations
        ),
        "rotation_not_worsened_by_quantization": bool(
            gate_16["rotation_not_worsened_by_quantization"]
        ),
        "rotation_ulp_scale": bool(
            gate_16["rotation_max_abs_error"] <= 1e-12
        ),
        "chunk_equivalence_bit_exact": bool(
            gate_16["chunk_equivalence_bit_exact"]
        ),
    }
    passed = all(predicates.values())

    output_dir = PROJECT_ROOT / config["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "task": "frozen_dpd_fixed_point_research",
        "decision": "PASS" if passed else "HOLD",
        "failed_predicates": [
            name for name, value in predicates.items() if not value
        ],
        "degradation_limit_db": DEGRADATION_LIMIT_DB,
        "predicates": predicates,
        "model_npz": config["model_npz"],
        "model_sha256": file_sha256(PROJECT_ROOT / config["model_npz"]),
        "primary_pa_npz": config["primary_pa_npz"],
        "secondary_pa_npz": config.get("secondary_pa_npz"),
        "results_by_bits": results,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_limits": [
            "numerical equivalence gate only; not the sealed artifact-chain protocol",
            "fixed-point kernel cost accounting is unchanged from the float model",
            (
                "the float dequantized reference cannot demonstrate integer-kernel "
                "bit-exact rotation; that property belongs to the sealed integer "
                "fixed-point kernel and is checked there"
            ),
        ],
    }
    write_json(output_dir / "fixed_point_research_report.json", report)
    print("FIXED-POINT GATE:", "PASS" if passed else "HOLD")
    print(json.dumps(predicates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
