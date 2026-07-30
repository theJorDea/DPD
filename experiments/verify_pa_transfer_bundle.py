"""Integrity and metric verifier for a published APA transfer pre-test bundle.

This verifier has the same sealed split boundary as the producer: it resolves
only source/target ``train`` and ``val`` files named by the preregistered
configuration.  It never searches a dataset directory and has no held-out
split loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from baseline.train_spline import load_split_pair
from experiments.transfer_pa_apa200_to_b import (
    PROJECT_ROOT,
    _load_source_models,
    _array_sha256,
    _project_path,
    file_sha256,
    load_config,
    metric_summary,
    split_frames,
    verify_preregistered_inputs,
)


TOLERANCE = 1e-9


def _assert_close(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        if actual != expected:
            raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")
        return
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=TOLERANCE):
        raise RuntimeError(f"{label} mismatch: {actual} != {expected}")


def _verify_metric_record(
    prediction: np.ndarray,
    target: np.ndarray,
    lengths: tuple[int, ...],
    *,
    warmup: int,
    reported: dict[str, Any],
    label: str,
) -> None:
    actual = metric_summary(prediction, target, lengths, warmup=warmup)
    for key in (
        "full_record_nmse_db",
        "common_interior_nmse_db",
        "opendpd_compatible_nmse_db",
        "time_domain_rms_evm_db",
        "common_interior_time_domain_rms_evm_db",
        "relative_error_power",
    ):
        _assert_close(actual[key], reported.get(key), f"{label}.{key}")
    if actual["per_frame_nmse_db"] != reported.get("per_frame_nmse_db"):
        for index, (left, right) in enumerate(
            zip(
                actual["per_frame_nmse_db"],
                reported.get("per_frame_nmse_db", ()),
                strict=False,
            )
        ):
            _assert_close(left, right, f"{label}.per_frame_nmse_db[{index}]")


def verify_bundle(bundle: str | Path) -> dict[str, Any]:
    bundle_path = Path(bundle).resolve()
    manifest_path = bundle_path / "transfer_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pretest_train_validation_only":
        raise RuntimeError("bundle is not a pre-test transfer bundle")
    integrity = manifest.get("input_integrity", {})
    if integrity.get("test_never_opened_or_hashed") is not True:
        raise RuntimeError("bundle does not seal held-out access")
    if integrity.get("target_held_out_hash_recorded") is not False:
        raise RuntimeError("bundle records a target held-out hash")

    for artifact_name, record in manifest["artifacts"].items():
        path = bundle_path / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"bundle artifact hash mismatch: {artifact_name}")

    config_path = _project_path(manifest["config"], name="transfer config")
    config = load_config(config_path)
    if file_sha256(config_path) != manifest["config_sha256"]:
        raise RuntimeError("transfer config hash differs from bundle")
    verified = verify_preregistered_inputs(config, config_path)
    if verified["dataset_hashes"] != manifest["dataset_hashes"]:
        raise RuntimeError("dataset hashes differ from bundle")
    if verified["artifact_hashes"] != manifest["source_artifact_hashes"]:
        raise RuntimeError("source artifact hashes differ from bundle")

    prediction_path = bundle_path / "predictions.npz"
    coefficient_path = bundle_path / "calibration_coefficients.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        predictions = {
            name: np.asarray(archive[name], dtype=np.complex128).copy()
            for name in archive.files
            if name != "schema_version"
        }
    with np.load(coefficient_path, allow_pickle=False) as archive:
        coefficients = {
            name: np.asarray(archive[name], dtype=np.complex128).copy()
            for name in archive.files
            if name != "schema_version"
        }
    if any("test" in name.lower() for name in predictions | coefficients):
        raise RuntimeError("sealed bundle contains a held-out prediction payload")

    source_dataset = _project_path(config["source_dataset"], name="source dataset")
    target_dataset = _project_path(config["target_dataset"], name="target dataset")
    source_train_input, _ = load_split_pair(source_dataset, "train")
    source_val_input, source_val_output = load_split_pair(source_dataset, "val")
    target_val_input, target_val_output = load_split_pair(target_dataset, "val")
    source_val_lengths = (source_val_input.size,)
    target_val_lengths = (target_val_input.size,)
    source_val_target = np.asarray(source_val_output, dtype=np.complex128)
    target_val_target = np.asarray(target_val_output, dtype=np.complex128)
    source_support = float(np.max(np.abs(source_train_input)))

    models = _load_source_models(config)
    checked_records = 0
    for record in config["source_models"]:
        name = record["name"]
        warmup = int(config["framing"]["common_warmup_samples"][name])
        source_key = f"{name}__source_val"
        target_key = f"{name}__zero_shot_target_val"
        _verify_metric_record(
            predictions[source_key],
            source_val_target,
            source_val_lengths,
            warmup=warmup,
            reported=manifest["source_control"][name]["metrics"],
            label=f"source_control.{name}",
        )
        _verify_metric_record(
            predictions[target_key],
            target_val_target,
            target_val_lengths,
            warmup=warmup,
            reported=manifest["target_transfer"][name]["zero_shot"]["validation"],
            label=f"target_transfer.{name}.zero_shot",
        )
        checked_records += 2
        support_maximum = (
            float(models[name].knots[-1])
            if name == "lag9_sparse_spline_memory"
            else source_support
        )
        for curve in manifest["target_transfer"][name]["coefficient_only_curves"]:
            count = int(curve["sample_count_per_frame"])
            if count == 0 or curve.get("status") != "feasible":
                continue
            key = f"{name}__N{count}__target_val"
            if key not in predictions:
                raise RuntimeError(f"missing prediction payload: {key}")
            reported = curve["validation"]
            _verify_metric_record(
                predictions[key],
                target_val_target,
                target_val_lengths,
                warmup=warmup,
                reported=reported,
                label=f"target_transfer.{name}.N{count}",
            )
            coefficient_key = f"{name}__N{count}"
            if coefficient_key not in coefficients:
                raise RuntimeError(f"missing coefficient payload: {coefficient_key}")
            expected_hash = curve["fit"]["coefficient_hash"]
            array_hash = _array_sha256(coefficients[coefficient_key])
            if array_hash != expected_hash:
                raise RuntimeError(f"coefficient hash mismatch: {coefficient_key}")
            checked_records += 1

    return {
        "bundle": str(bundle_path.relative_to(PROJECT_ROOT)),
        "checked_metric_records": checked_records,
        "artifact_hashes_verified": True,
        "dataset_hashes_verified": True,
        "source_artifact_hashes_verified": True,
        "test_never_opened_or_hashed": True,
        "target_held_out_hash_recorded": False,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an APA transfer pre-test bundle without held-out access."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    return parser


def main() -> None:
    result = verify_bundle(_argument_parser().parse_args().bundle)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
