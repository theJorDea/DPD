"""Evaluate a trained spline without changing its configuration.

The command has explicit modes so that an ILA inverse diagnostic cannot be
mistaken for a deployment test:

``inverse-diagnostic``
    Feed ``measured_pa_output / gain`` to the fitted postdistorter and compare
    its estimate with the known PA input.  If a surrogate is supplied, also
    report the inverse--forward *circular reconstruction* of the measured
    output.

``surrogate-cascade``
    Feed the desired test input ``x_test`` to the DPD, then feed that result to
    an explicitly supplied PA surrogate and compare with ``gain*x_test``.
    This is the correct direction, but remains surrogate-only.

``both``
    Compute both sets of metrics.  A surrogate is mandatory.

No parameter, gain, alignment, or checkpoint is fitted from test data here.
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

from .complex_spline_dpd import ComplexLinearSplineDPD
from .metrics import (
    nmse_opendpd_db,
    opendpd_aclr_db,
    opendpd_spectral_evm_db,
    papr_db,
    peak_amplitude,
)
from .pa_models import MemoryPolynomialPA
from .train_spline import (
    _json_ready,
    _paired_time_metrics,
    _waveform_output_metrics,
    align_split_pair,
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


def _as_gain(value: Any) -> complex:
    if isinstance(value, dict):
        return complex(float(value["real"]), float(value["imag"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    return complex(value)


def inverse_postdistorter_signals(
    model: ComplexLinearSplineDPD,
    known_pa_input: np.ndarray,
    measured_pa_output: np.ndarray,
    gain: complex,
) -> dict[str, np.ndarray]:
    """Return signals for the ILA inverse diagnostic.

    The function intentionally names the model input ``postdistorter_input``:
    this is not a valid desired-signal deployment path.
    """

    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("gain must be finite and non-zero")
    known = np.asarray(known_pa_input, dtype=np.complex128)
    measured = np.asarray(measured_pa_output, dtype=np.complex128)
    if known.ndim != 1 or measured.ndim != 1 or known.shape != measured.shape:
        raise ValueError("known_pa_input and measured_pa_output must match 1-D")
    postdistorter_input = measured / gain
    estimated_pa_input = model.predict(postdistorter_input)
    return {
        "known_pa_input": known,
        "measured_pa_output": measured,
        "postdistorter_input": postdistorter_input,
        "estimated_pa_input": estimated_pa_input,
    }


def desired_input_cascade_signals(
    model: ComplexLinearSplineDPD,
    desired_signal: np.ndarray,
    gain: complex,
    pa_surrogate: MemoryPolynomialPA,
    *,
    segment_length: int | None = None,
) -> dict[str, np.ndarray]:
    """Return the correctly directed desired -> DPD -> surrogate PA cascade."""

    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("gain must be finite and non-zero")
    desired = np.asarray(desired_signal, dtype=np.complex128)
    if desired.ndim != 1:
        raise ValueError("desired_signal must be one-dimensional")
    predistorted = model.predict(desired)
    surrogate_output = (
        pa_surrogate.predict(predistorted)
        if segment_length is None
        else pa_surrogate.predict_segments(predistorted, segment_length)
    )
    return {
        "desired_signal": desired,
        "dpd_input": desired,
        "predistorted_signal": predistorted,
        "surrogate_pa_output": surrogate_output,
        "ideal_output": gain * desired,
    }


def _repository_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    signal_for_aclr: np.ndarray | None,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Compute repository-style spectral metrics when the spec is complete."""

    required = (
        "input_signal_fs",
        "bw_main_ch",
        "n_sub_ch",
        "nperseg",
    )
    if any(key not in spec for key in required):
        return {
            "available": False,
            "reason": f"dataset spec lacks one of {required}",
        }
    fs = float(spec["input_signal_fs"])
    bandwidth = float(spec["bw_main_ch"])
    channels = int(spec["n_sub_ch"])
    nperseg = int(spec["nperseg"])
    if estimate.size % nperseg or reference.size % nperseg:
        return {
            "available": False,
            "reason": "record length is not divisible by spec nperseg",
            "nperseg": nperseg,
        }
    estimate_segments = estimate.reshape(-1, nperseg)
    reference_segments = reference.reshape(-1, nperseg)
    result: dict[str, Any] = {
        "available": True,
        "fs_hz": fs,
        "bandwidth_main_hz": bandwidth,
        "n_subchannels": channels,
        "nperseg": nperseg,
        "nmse_opendpd_mean_segment_db": nmse_opendpd_db(
            estimate_segments,
            reference_segments,
        ),
        "opendpd_spectral_evm_db": opendpd_spectral_evm_db(
            estimate,
            reference,
            fs=fs,
            bandwidth_main=bandwidth,
            n_subchannels=channels,
            nperseg=nperseg,
        ),
    }
    if signal_for_aclr is not None:
        aclr = opendpd_aclr_db(
            signal_for_aclr,
            fs=fs,
            nperseg=nperseg,
            bandwidth_main=bandwidth,
            n_subchannels=channels,
        )
        result["opendpd_aclr_db"] = {
            "left": aclr.left_db,
            "right": aclr.right_db,
            "average": aclr.average_db,
        }
    return result


def _signal_summary(signal: np.ndarray) -> dict[str, Any]:
    return {
        "waveform": _waveform_output_metrics(signal),
        "sample_count": int(signal.size),
        "finite": bool(np.all(np.isfinite(signal))),
    }


def _paired_result(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    warmup_samples: int,
    segment_length: int | None,
    spec: dict[str, Any],
    signal_for_aclr: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "time_domain_full": _paired_time_metrics(estimate, reference),
        "time_domain_after_causal_warmup": _paired_time_metrics(
            estimate,
            reference,
            warmup_samples=warmup_samples,
            segment_length=segment_length,
        ),
        "repository_spectral_metrics_full_record": _repository_metrics(
            estimate,
            reference,
            signal_for_aclr=signal_for_aclr,
            spec=spec,
        ),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen complex spline on the OpenDPD test split."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument(
        "--pa-surrogate",
        type=Path,
        help="explicit train-fitted MemoryPolynomialPA .npz for cascade modes",
    )
    parser.add_argument(
        "--mode",
        choices=("inverse-diagnostic", "surrogate-cascade", "both"),
        default="both",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-npz",
        type=Path,
        help="optional waveform archive; arrays are labelled by direction",
    )
    parser.add_argument("--allow-artifact-mismatch", action="store_true")
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help=(
            "permit spec/train/validation hashes to differ from the training "
            "report; intended only for a separately documented audit"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_artifacts(
    training_report_path: Path,
    *,
    allow_artifact_mismatch: bool,
) -> tuple[dict[str, Any], ComplexLinearSplineDPD, complex]:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("training report must contain a JSON object")
    model_path = Path(report["artifacts"]["spline_model"])
    if not model_path.is_absolute():
        model_path = training_report_path.parent / model_path
    model = ComplexLinearSplineDPD.load(model_path)
    expected_hash = report["artifacts"].get("spline_model_sha256")
    actual_hash = file_sha256(model_path)
    if (
        expected_hash
        and actual_hash != expected_hash
        and not allow_artifact_mismatch
    ):
        raise ValueError(
            "spline model hash does not match training report; use "
            "--allow-artifact-mismatch only for an explicitly documented audit"
        )
    gain = _as_gain(report["target_gain"]["value"])
    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("training report contains an invalid target gain")
    return report, model, gain


def _verify_dataset_identity(
    training_report: dict[str, Any],
    dataset: Path,
    *,
    allow_mismatch: bool,
) -> dict[str, Any]:
    """Verify immutable train/validation provenance without opening test data."""

    expected = training_report.get("dataset", {})
    mismatches: list[str] = []
    spec_path = dataset / "spec.json"
    expected_spec_hash = expected.get("spec_sha256")
    actual_spec_hash = file_sha256(spec_path) if spec_path.is_file() else None
    if expected_spec_hash != actual_spec_hash:
        mismatches.append("spec.json SHA-256")

    expected_files = expected.get("input_file_sha256", {})
    actual_files: dict[str, str] = {}
    for label in ("train_input", "train_output", "val_input", "val_output"):
        path = dataset / f"{label}.csv"
        if path.is_file():
            actual_files[label] = file_sha256(path)
        if expected_files.get(label) != actual_files.get(label):
            mismatches.append(f"{label}.csv SHA-256")

    if mismatches and not allow_mismatch:
        raise ValueError(
            "evaluation dataset does not match frozen training provenance "
            f"({', '.join(mismatches)}); use --allow-dataset-mismatch only "
            "for an explicitly documented audit"
        )
    return {
        "matched": not mismatches,
        "mismatches": mismatches,
        "expected_spec_sha256": expected_spec_hash,
        "actual_spec_sha256": actual_spec_hash,
        "train_validation_hashes_checked": True,
        "override_used": bool(mismatches and allow_mismatch),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_json = args.output_json.resolve()
    output_npz = args.output_npz.resolve() if args.output_npz else None
    if output_json.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_json}")
    if output_npz is not None and output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_npz}")
    needs_surrogate = args.mode in {"surrogate-cascade", "both"}
    if needs_surrogate and args.pa_surrogate is None:
        raise ValueError(
            "--pa-surrogate is required for surrogate-cascade and both modes"
        )

    training_report, model, gain = _load_artifacts(
        args.training_report.resolve(),
        allow_artifact_mismatch=args.allow_artifact_mismatch,
    )
    dataset = args.dataset.resolve()
    dataset_identity = _verify_dataset_identity(
        training_report,
        dataset,
        allow_mismatch=args.allow_dataset_mismatch,
    )
    pa_surrogate = (
        MemoryPolynomialPA.load(args.pa_surrogate.resolve())
        if args.pa_surrogate is not None
        else None
    )
    expected_surrogate = training_report.get("pa_surrogate")
    supplied_surrogate_hash = (
        file_sha256(args.pa_surrogate.resolve())
        if args.pa_surrogate is not None
        else None
    )
    expected_surrogate_hash = (
        expected_surrogate.get("artifact_sha256")
        if isinstance(expected_surrogate, dict)
        else None
    )
    if pa_surrogate is not None:
        mismatch_reason: str | None = None
        if expected_surrogate_hash is None:
            mismatch_reason = (
                "the frozen training selection did not record a PA surrogate"
            )
        elif supplied_surrogate_hash != expected_surrogate_hash:
            mismatch_reason = "supplied PA surrogate SHA-256 differs from training"
        if mismatch_reason and not args.allow_artifact_mismatch:
            raise ValueError(
                f"{mismatch_reason}; use --allow-artifact-mismatch only for "
                "an explicitly documented audit"
            )

    raw_test_input, raw_test_output = load_split_pair(dataset, "test")
    alignment_record = training_report.get("alignment")
    if not isinstance(alignment_record, dict):
        if not args.allow_artifact_mismatch:
            raise ValueError(
                "training report has no frozen alignment record; use "
                "--allow-artifact-mismatch only for a legacy artifact audit"
            )
        alignment_delay = 0
    else:
        alignment_delay = int(
            alignment_record["frozen_integer_delay_samples"]
        )
    test_input, test_output = align_split_pair(
        raw_test_input,
        raw_test_output,
        delay=alignment_delay,
    )
    spec = load_dataset_spec(dataset)
    segment_length = (
        int(spec["nperseg"])
        if pa_surrogate is not None
        and "nperseg" in spec
        and int(spec["nperseg"]) > 0
        else None
    )
    warmup = pa_surrogate.causal_warmup_samples if pa_surrogate else 0

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "complex_linear_spline_test_evaluation",
        "claims_scope": {
            "physical_pa_result": False,
            "surrogate_cascade_result": (
                "surrogate-only; the supplied PA is a fitted behavioral model"
                if pa_surrogate is not None
                else "not computed"
            ),
            "inverse_result": (
                "postdistorter/inverse diagnostic only; not a deployment score"
                if args.mode in {"inverse-diagnostic", "both"}
                else "not computed"
            ),
        },
        "command": [sys.executable, "-m", "baseline.evaluate_spline", *sys.argv[1:]],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "directory": dataset,
            "split": "test",
            "spec": spec,
            "identity_check": dataset_identity,
            "raw_sample_count": int(raw_test_input.size),
            "sample_count": int(test_input.size),
            "test_input_sha256": file_sha256(
                dataset / "test_input.csv"
            ),
            "test_output_sha256": file_sha256(
                dataset / "test_output.csv"
            ),
        },
        "frozen_training_artifact": {
            "training_report": args.training_report.resolve(),
            "model": training_report["artifacts"]["spline_model"],
            "target_gain": gain,
            "selection": training_report.get("selection"),
            "test_was_not_used_for_selection": True,
        },
        "evaluation_policy": {
            "dpd_cascade_input": "desired test_input x_test",
            "cascade_target": "training_gain * x_test",
            "inverse_input": "measured test_output / training_gain",
            "causal_warmup_samples_for_time_metrics": warmup,
            "surrogate_segment_length": segment_length,
            "surrogate_state_reset_policy": (
                "not applicable"
                if pa_surrogate is None
                else "zero history at each dataset nperseg boundary"
            ),
            "frozen_integer_delay_samples": alignment_delay,
            "test_alignment_retuned": False,
            "spectral_metrics_include_record_startup_samples": True,
        },
        "pa_surrogate_artifact": (
            {
                "path": args.pa_surrogate.resolve(),
                "sha256": supplied_surrogate_hash,
                "expected_training_sha256": expected_surrogate_hash,
                "matches_training_artifact": (
                    supplied_surrogate_hash == expected_surrogate_hash
                    if expected_surrogate_hash is not None
                    else False
                ),
                "override_used": bool(
                    pa_surrogate is not None
                    and supplied_surrogate_hash != expected_surrogate_hash
                    and args.allow_artifact_mismatch
                ),
                "metadata": pa_surrogate.metadata,
                "causal_warmup_samples": pa_surrogate.causal_warmup_samples,
            }
            if pa_surrogate is not None
            else None
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "test_desired_input": test_input,
        "test_measured_pa_output": test_output,
    }

    if args.mode in {"inverse-diagnostic", "both"}:
        inverse = inverse_postdistorter_signals(
            model,
            test_input,
            test_output,
            gain,
        )
        inverse_result: dict[str, Any] = {
            "direction": (
                "measured_pa_output/gain -> spline postdistorter -> "
                "estimated_pa_input"
            ),
            "input_summary": _signal_summary(inverse["postdistorter_input"]),
            "estimated_input_summary": _signal_summary(
                inverse["estimated_pa_input"]
            ),
            "estimated_input_vs_known_input": _paired_result(
                inverse["estimated_pa_input"],
                test_input,
                warmup_samples=0,
                segment_length=None,
                spec=spec,
            ),
        }
        arrays.update(
            {
                "inverse_postdistorter_input_y_over_gain": inverse[
                    "postdistorter_input"
                ],
                "inverse_estimated_pa_input": inverse["estimated_pa_input"],
            }
        )
        if pa_surrogate is not None:
            circular_output = (
                pa_surrogate.predict(inverse["estimated_pa_input"])
                if segment_length is None
                else pa_surrogate.predict_segments(
                    inverse["estimated_pa_input"],
                    segment_length,
                )
            )
            inverse_result["circular_inverse_forward_reconstruction"] = {
                "warning": (
                    "This reconstructs the already observed y_test and does "
                    "not test a new desired x_test."
                ),
                "output_vs_measured_output": _paired_result(
                    circular_output,
                    test_output,
                    warmup_samples=warmup,
                    segment_length=segment_length,
                    spec=spec,
                    signal_for_aclr=circular_output,
                ),
            }
            arrays["inverse_circular_reconstructed_output"] = circular_output
        report["inverse_postdistorter_diagnostic"] = inverse_result

    if args.mode in {"surrogate-cascade", "both"}:
        assert pa_surrogate is not None
        cascade = desired_input_cascade_signals(
            model,
            test_input,
            gain,
            pa_surrogate,
            segment_length=segment_length,
        )
        surrogate_without_dpd = (
            pa_surrogate.predict(test_input)
            if segment_length is None
            else pa_surrogate.predict_segments(test_input, segment_length)
        )
        cascade_result: dict[str, Any] = {
            "scope": "surrogate_only",
            "direction": "desired x_test -> spline DPD -> supplied PA surrogate",
            "dpd_input_assertion": "exactly test_input; measured test_output is not fed to DPD",
            "dpd_input_summary": _signal_summary(cascade["dpd_input"]),
            "predistorted_summary": _signal_summary(
                cascade["predistorted_signal"]
            ),
            "surrogate_fidelity_on_measured_test_input": _paired_result(
                surrogate_without_dpd,
                test_output,
                warmup_samples=warmup,
                segment_length=segment_length,
                spec=spec,
                signal_for_aclr=surrogate_without_dpd,
            ),
            "without_dpd_vs_ideal": _paired_result(
                surrogate_without_dpd,
                gain * test_input,
                warmup_samples=warmup,
                segment_length=segment_length,
                spec=spec,
                signal_for_aclr=surrogate_without_dpd,
            ),
            "with_dpd_vs_ideal": _paired_result(
                cascade["surrogate_pa_output"],
                cascade["ideal_output"],
                warmup_samples=warmup,
                segment_length=segment_length,
                spec=spec,
                signal_for_aclr=cascade["surrogate_pa_output"],
            ),
            "spline_input_extrapolation_fraction": float(
                np.mean(np.abs(test_input) > model.knots[-1])
            ),
        }
        if isinstance(expected_surrogate, dict):
            fit_diagnostics = expected_surrogate.get("fit_diagnostics", {})
            training_peak = fit_diagnostics.get(
                "maximum_training_input_amplitude"
            )
            if training_peak is not None:
                cascade_result[
                    "surrogate_training_range_extrapolation_fraction"
                ] = float(
                    np.mean(
                        np.abs(cascade["predistorted_signal"])
                        > float(training_peak)
                    )
                )
        if "pa_surrogate" in training_report:
            surrogate_record = training_report["pa_surrogate"]
            if surrogate_record:
                expected_hash = surrogate_record.get("artifact_sha256")
                if expected_hash:
                    cascade_result["supplied_surrogate_matches_training_artifact"] = (
                        file_sha256(args.pa_surrogate.resolve()) == expected_hash
                    )
        report["surrogate_only_predistorter_cascade"] = cascade_result
        arrays.update(
            {
                "desired_test_input_to_dpd": cascade["dpd_input"],
                "predistorted_test_signal": cascade["predistorted_signal"],
                "surrogate_cascade_output": cascade["surrogate_pa_output"],
                "ideal_test_output": cascade["ideal_output"],
                "surrogate_without_dpd_output": surrogate_without_dpd,
            }
        )

    report["artifacts"] = {
        "waveform_npz": output_npz,
        "evaluation_json": output_json,
    }
    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_npz,
            schema_version=np.asarray(1, dtype=np.int64),
            **arrays,
        )
        report["artifacts"]["waveform_npz_sha256"] = file_sha256(output_npz)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
