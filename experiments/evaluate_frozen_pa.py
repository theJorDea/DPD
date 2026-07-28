"""Evaluate a validation-selected PA model on the previously sealed test split.

Integrity checks are intentionally completed before ``test_input.csv`` or
``test_output.csv`` is opened.  This command never refits coefficients,
alignment, gain, framing, characteristic bins, or hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from baseline.complexity import OperationCount
from baseline.pa_benchmark import (
    PAEvaluationProtocol,
    evaluate_pa_predictor,
    prepare_pa_split,
)
from baseline.pa_models import MemoryPolynomialPA
from baseline.train_spline import (
    file_sha256,
    load_split_pair,
    write_json,
)


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("selection manifest must contain one JSON object")
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("unsupported selection manifest schema")
    if value.get("task") != "forward_pa_identification_model_selection":
        raise ValueError("manifest is not a forward PA selection artifact")
    if value.get("model_class") != "complex_memory_polynomial":
        raise ValueError("this runner currently supports the frozen MP PA model")
    if value.get("selection_split") != "validation":
        raise ValueError("model must have been selected on validation")
    if value.get("test_split_accessed") is not False:
        raise ValueError("selection manifest does not certify sealed test data")
    return value


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )


def verify_selection_before_test_access(
    manifest_path: str | Path,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    PAEvaluationProtocol,
    MemoryPolynomialPA,
    OperationCount,
]:
    """Verify every frozen selection artifact without opening test data."""

    source_manifest = Path(manifest_path).resolve()
    manifest = _read_manifest(source_manifest)
    dataset = Path(manifest["dataset"]).resolve()
    config = Path(manifest["config"]).resolve()
    model_path = Path(manifest["selected_model"]).resolve()

    _verify_hash(
        config,
        str(manifest["config_sha256"]),
        label="selection config",
    )
    _verify_hash(
        model_path,
        str(manifest["selected_model_sha256"]),
        label="selected PA model",
    )
    for relative_name, expected in manifest["dataset_files_sha256"].items():
        if relative_name.startswith("test_"):
            raise ValueError(
                "selection manifest must not contain a test-file hash"
            )
        _verify_hash(
            dataset / relative_name,
            str(expected),
            label=f"selection dataset file {relative_name}",
        )
    for source_name, expected in manifest["source_sha256"].items():
        _verify_hash(
            Path(source_name).resolve(),
            str(expected),
            label=f"selection source {source_name}",
        )

    protocol_value = dict(manifest["protocol"])
    protocol_value["characteristic_bin_edges"] = tuple(
        protocol_value["characteristic_bin_edges"]
    )
    protocol = PAEvaluationProtocol(**protocol_value)
    model = MemoryPolynomialPA.load(model_path)
    selected = manifest["selected_trial"]
    if tuple(selected["orders"]) != model.orders:
        raise ValueError("selected order set does not match frozen model")
    if tuple(selected["delays"]) != model.delays:
        raise ValueError("selected delays do not match frozen model")

    operations_value = dict(
        selected["operation_count_per_complex_sample"]
    )
    operations_value["notes"] = tuple(operations_value.get("notes", ()))
    operation_count = OperationCount(**operations_value)
    if (
        operation_count.stored_real_coefficients
        != model.stored_real_coefficients
    ):
        raise ValueError(
            "operation-count coefficient storage does not match frozen model"
        )
    return (
        manifest,
        source_manifest,
        dataset,
        protocol,
        model,
        operation_count,
    )


def evaluate_from_manifest(
    manifest_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify selection, open test once, and write immutable test artifacts."""

    (
        selection,
        source_manifest,
        dataset,
        protocol,
        model,
        operation_count,
    ) = verify_selection_before_test_access(manifest_path)

    if output_directory is None:
        output = source_manifest.parent
    else:
        output = Path(output_directory).resolve()
    evaluation_path = output / "test_evaluation.json"
    waveform_path = output / "test_prediction.npz"
    test_manifest_path = output / "test_manifest.json"
    owned_paths = (evaluation_path, waveform_path, test_manifest_path)
    existing = [path for path in owned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing test artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)

    # This is deliberately the first test-split access in the control path.
    test_input, measured_test_output = load_split_pair(dataset, "test")
    selected = selection["selected_trial"]
    evaluation, prediction = evaluate_pa_predictor(
        model.predict,
        test_input,
        measured_test_output,
        protocol=protocol,
        model_label=(
            f"complex_mp_{selected['family']}_"
            f"o{len(model.orders)}_q{len(model.delays)}"
        ),
        split="test",
        purpose="final_report",
        common_warmup_samples=int(
            selection["common_warmup_samples_per_frame"]
        ),
        operation_count=operation_count,
        trainable_real_parameter_count=model.stored_real_coefficients,
        fit_seconds=float(selected["fit_seconds"]),
        precision_label="numpy_complex128",
    )
    write_json(evaluation_path, evaluation.to_dict())
    _, measured_test_output_aligned = prepare_pa_split(
        test_input,
        measured_test_output,
        protocol,
    )
    np.savez_compressed(
        waveform_path,
        schema_version=np.asarray(1, dtype=np.int64),
        split=np.asarray("test"),
        direction=np.asarray("x_test -> frozen PA model -> y_hat_test"),
        predicted_pa_output=prediction,
        complex_residual=prediction - measured_test_output_aligned,
    )
    report = {
        "schema_version": 1,
        "task": "frozen_forward_pa_test_evaluation",
        "selection_manifest": source_manifest,
        "selection_manifest_sha256": file_sha256(source_manifest),
        "selected_model": selection["selected_model"],
        "selected_model_sha256": selection["selected_model_sha256"],
        "dataset": dataset,
        "test_files_sha256": {
            "test_input.csv": file_sha256(dataset / "test_input.csv"),
            "test_output.csv": file_sha256(dataset / "test_output.csv"),
        },
        "integrity_checks_completed_before_test_access": True,
        "refit_performed": False,
        "post_prediction_gain_fit": False,
        "post_prediction_delay_fit": False,
        "test_evaluation": evaluation_path,
        "test_evaluation_sha256": file_sha256(evaluation_path),
        "test_prediction": waveform_path,
        "test_prediction_sha256": file_sha256(waveform_path),
        "selected_validation_score_db": selected["selection_score_db"],
        "test_full_record_pooled_nmse_db": evaluation.full_record_metrics[
            "complex_nmse_pooled_db"
        ],
        "test_opendpd_nmse_db": evaluation.opendpd_compatible_metrics[
            "nmse_mean_segment_db"
        ],
        "command": " ".join(sys.argv),
    }
    write_json(test_manifest_path, report)
    return report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Integrity-check a validation-selected MP PA model, then evaluate "
            "the frozen model once on test."
        )
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    report = evaluate_from_manifest(
        args.selection_manifest,
        output_directory=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        "Frozen PA test:",
        f"pooled NMSE={report['test_full_record_pooled_nmse_db']:.6f} dB",
        f"OpenDPD NMSE={report['test_opendpd_nmse_db']:.6f} dB",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
