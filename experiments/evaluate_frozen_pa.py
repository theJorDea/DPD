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
from baseline.gmp_pa import (
    GMPConfig,
    GeneralizedMemoryPolynomialPA,
)
from baseline.metrics import nmse_pooled_db, time_domain_rms_evm_db
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

FrozenPAModel = MemoryPolynomialPA | GeneralizedMemoryPolynomialPA
SUPPORTED_MODEL_CLASSES = {
    "complex_memory_polynomial",
    "complex_generalized_memory_polynomial",
}


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("selection manifest must contain one JSON object")
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("unsupported selection manifest schema")
    if value.get("task") != "forward_pa_identification_model_selection":
        raise ValueError("manifest is not a forward PA selection artifact")
    if value.get("model_class") not in SUPPORTED_MODEL_CLASSES:
        raise ValueError(
            "unsupported frozen PA model_class; expected complex MP or GMP"
        )
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


def _load_and_verify_model(
    manifest: dict[str, Any],
    model_path: Path,
) -> tuple[FrozenPAModel, OperationCount]:
    """Load the declared model and verify its selected topology and cost."""

    selected = manifest.get("selected_trial")
    if not isinstance(selected, dict):
        raise ValueError("selection manifest is missing selected_trial")
    model_class = manifest["model_class"]
    if model_class == "complex_memory_polynomial":
        model: FrozenPAModel = MemoryPolynomialPA.load(model_path)
        if tuple(selected.get("orders", ())) != model.orders:
            raise ValueError("selected order set does not match frozen MP model")
        if tuple(selected.get("delays", ())) != model.delays:
            raise ValueError("selected delays do not match frozen MP model")
    else:
        model = GeneralizedMemoryPolynomialPA.load(model_path)
        config_value = selected.get("gmp_config")
        if not isinstance(config_value, dict):
            raise ValueError("selected GMP trial is missing gmp_config")
        expected_keys = {
            "ka",
            "la",
            "kb",
            "lb",
            "mb",
            "kc",
            "lc",
            "mc",
            "leading_policy",
        }
        if set(config_value) != expected_keys:
            raise ValueError(
                "selected gmp_config must contain the exact frozen GMP fields"
            )
        try:
            selected_config = GMPConfig(**config_value)
        except (TypeError, ValueError) as error:
            raise ValueError("selected gmp_config is invalid") from error
        if selected_config != model.config:
            raise ValueError(
                "selected gmp_config does not match frozen GMP model"
            )

    operations_value = selected.get("operation_count_per_complex_sample")
    if not isinstance(operations_value, dict):
        raise ValueError("selected trial is missing operation count")
    operations_value = dict(operations_value)
    operations_value["notes"] = tuple(operations_value.get("notes", ()))
    try:
        operation_count = OperationCount(**operations_value)
    except TypeError as error:
        raise ValueError("selected operation count has an invalid schema") from error
    if (
        operation_count.stored_real_coefficients
        != model.stored_real_coefficients
    ):
        raise ValueError(
            "operation-count coefficient storage does not match frozen model"
        )
    if (
        isinstance(model, GeneralizedMemoryPolynomialPA)
        and operation_count != model.operation_count
    ):
        raise ValueError(
            "recorded GMP operation count does not match frozen GMP topology"
        )
    return model, operation_count


def _boundary_exclusions(
    manifest: dict[str, Any],
    model: FrozenPAModel,
    protocol: PAEvaluationProtocol,
) -> tuple[int, int]:
    """Validate the common boundary policy frozen during model selection."""

    raw_warmup = manifest.get("common_warmup_samples_per_frame")
    if (
        not isinstance(raw_warmup, int)
        or isinstance(raw_warmup, bool)
        or raw_warmup < 0
    ):
        raise ValueError(
            "common_warmup_samples_per_frame must be a non-negative integer"
        )
    raw_cooldown = manifest.get(
        "common_future_cooldown_samples_per_frame",
        0,
    )
    if (
        not isinstance(raw_cooldown, int)
        or isinstance(raw_cooldown, bool)
        or raw_cooldown < 0
    ):
        raise ValueError(
            "common_future_cooldown_samples_per_frame must be a "
            "non-negative integer"
        )
    warmup = int(raw_warmup)
    cooldown = int(raw_cooldown)
    if warmup + cooldown >= protocol.nperseg:
        raise ValueError(
            "common warmup/cooldown consumes every complete test frame"
        )
    if isinstance(model, MemoryPolynomialPA):
        required_warmup = model.causal_warmup_samples
        required_cooldown = 0
    else:
        required_warmup = model.config.causal_warmup_samples
        required_cooldown = model.config.lookahead_samples
    if warmup < required_warmup:
        raise ValueError(
            "frozen common warmup is shorter than selected model memory"
        )
    if cooldown < required_cooldown:
        raise ValueError(
            "frozen common cooldown is shorter than selected GMP lookahead"
        )
    return warmup, cooldown


def verify_selection_before_test_access(
    manifest_path: str | Path,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    PAEvaluationProtocol,
    FrozenPAModel,
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
    try:
        protocol = PAEvaluationProtocol(**protocol_value)
    except TypeError as error:
        raise ValueError("selection protocol has an invalid schema") from error
    model, operation_count = _load_and_verify_model(manifest, model_path)
    _boundary_exclusions(manifest, model, protocol)
    return (
        manifest,
        source_manifest,
        dataset,
        protocol,
        model,
        operation_count,
    )


def _model_label(
    selection: dict[str, Any],
    model: FrozenPAModel,
) -> str:
    selected = selection["selected_trial"]
    if isinstance(model, MemoryPolynomialPA):
        return (
            f"complex_mp_{selected['family']}_"
            f"o{len(model.orders)}_q{len(model.delays)}"
        )
    return (
        f"complex_gmp_{selected['topology']}_"
        f"ka{model.config.ka}_l{model.config.la}_"
        f"{model.config.leading_policy}"
    )


def _segmented_interior_mask(
    sample_count: int,
    *,
    segment_length: int,
    warmup_samples: int,
    cooldown_samples: int,
) -> np.ndarray:
    """Return the train-frozen common interior of every test frame."""

    mask = np.zeros(sample_count, dtype=bool)
    for start in range(0, sample_count, segment_length):
        stop = min(start + segment_length, sample_count)
        interior_start = min(start + warmup_samples, stop)
        interior_stop = max(start, stop - cooldown_samples)
        if interior_start < interior_stop:
            mask[interior_start:interior_stop] = True
    if not np.any(mask):
        raise ValueError("common warmup/cooldown consumes every test sample")
    return mask


def _common_interior_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    segment_length: int,
    warmup_samples: int,
    cooldown_samples: int,
) -> dict[str, float | int | str]:
    estimate = np.asarray(prediction)
    target = np.asarray(reference)
    if estimate.ndim != 1 or estimate.shape != target.shape:
        raise ValueError("test prediction/reference must be equal-length vectors")
    mask = _segmented_interior_mask(
        estimate.size,
        segment_length=segment_length,
        warmup_samples=warmup_samples,
        cooldown_samples=cooldown_samples,
    )
    error_power = float(np.mean(np.abs(estimate[mask] - target[mask]) ** 2))
    reference_power = float(np.mean(np.abs(target[mask]) ** 2))
    return {
        "role": (
            "boundary-excluded diagnostic using the validation-frozen common "
            "support; full-record pooled NMSE remains the primary test score"
        ),
        "complex_nmse_pooled_db": nmse_pooled_db(
            estimate[mask],
            target[mask],
        ),
        "time_domain_rms_sample_evm_db": time_domain_rms_evm_db(
            estimate[mask],
            target[mask],
        ),
        "mse": error_power,
        "reference_power": reference_power,
        "relative_error_power": error_power / reference_power,
        "warmup_samples_per_frame": warmup_samples,
        "cooldown_samples_per_frame": cooldown_samples,
        "scored_sample_count": int(np.count_nonzero(mask)),
        "discarded_sample_count": int(mask.size - np.count_nonzero(mask)),
    }


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
    common_warmup, common_cooldown = _boundary_exclusions(
        selection,
        model,
        protocol,
    )

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
        model_label=_model_label(selection, model),
        split="test",
        purpose="final_report",
        common_warmup_samples=common_warmup,
        operation_count=operation_count,
        trainable_real_parameter_count=model.stored_real_coefficients,
        fit_seconds=float(selected["fit_seconds"]),
        precision_label="numpy_complex128",
    )
    _, measured_test_output_aligned = prepare_pa_split(
        test_input,
        measured_test_output,
        protocol,
    )
    common_interior = _common_interior_metrics(
        prediction,
        measured_test_output_aligned,
        segment_length=protocol.nperseg,
        warmup_samples=common_warmup,
        cooldown_samples=common_cooldown,
    )
    evaluation_payload = evaluation.to_dict()
    evaluation_payload["common_interior_metrics"] = common_interior
    evaluation_payload["primary_test_metric"] = (
        "full_record_metrics.complex_nmse_pooled_db"
    )
    write_json(evaluation_path, evaluation_payload)
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
        "model_class": selection["model_class"],
        "model_label": evaluation.model_label,
        "selected_model_configuration": (
            selected["gmp_config"]
            if selection["model_class"]
            == "complex_generalized_memory_polynomial"
            else {
                "orders": selected["orders"],
                "delays": selected["delays"],
            }
        ),
        "operation_count_per_complex_sample": operation_count.to_dict(),
        "dataset": dataset,
        "test_files_sha256": {
            "test_input.csv": file_sha256(dataset / "test_input.csv"),
            "test_output.csv": file_sha256(dataset / "test_output.csv"),
        },
        "integrity_checks_completed_before_test_access": True,
        "refit_performed": False,
        "post_prediction_gain_fit": False,
        "post_prediction_delay_fit": False,
        "primary_test_metric": (
            "test_full_record_pooled_nmse_db; all aligned test samples, "
            "including zero-context frame boundaries"
        ),
        "common_boundary_policy": {
            "warmup_samples_per_frame": common_warmup,
            "cooldown_samples_per_frame": common_cooldown,
            "source": "validation selection manifest; not refit on test",
        },
        "test_common_interior": common_interior,
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
        "test_common_interior_pooled_nmse_db": common_interior[
            "complex_nmse_pooled_db"
        ],
        "command": " ".join(sys.argv),
    }
    write_json(test_manifest_path, report)
    return report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Integrity-check a validation-selected MP/GMP PA model, then "
            "evaluate the frozen model once on test."
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
