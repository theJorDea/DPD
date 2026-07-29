"""Integrity, metric, and publication support for the sparse PA benchmark.

This module contains no candidate fitting and no split loader.  It keeps the
hash checks, frame-aware metrics, streaming checks, and immutable bundle
publication independently testable from the production experiment runner.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
from typing import Any, Iterable

import numpy as np

from baseline.metrics import (
    nmse_opendpd_db,
    nmse_pooled_db,
    time_domain_rms_evm_db,
)
from baseline.residual_analysis import ResidualAnalysisSpec
from baseline.sparse_spline_memory_pa import SparseSplineMemoryPA
from baseline.train_spline import write_json
from experiments.select_pa_sparse_spline_memory import (
    common_mask,
    file_sha256,
    frame_segments,
    project_path,
    strip_prediction,
)


REFERENCE_REPRODUCTION_TOLERANCE_DB = 1e-9


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def frame_ids(frame_lengths: Iterable[int]) -> np.ndarray:
    lengths = tuple(int(length) for length in frame_lengths)
    return np.concatenate(
        [
            np.full(length, index, dtype=np.int64)
            for index, length in enumerate(lengths)
        ]
    )


def partition_lengths(sample_count: int, segment_length: int) -> tuple[int, ...]:
    if sample_count <= 0 or segment_length <= 0:
        raise ValueError("sample and segment counts must be positive")
    full, remainder = divmod(int(sample_count), int(segment_length))
    lengths = [segment_length] * full
    if remainder:
        lengths.append(remainder)
    return tuple(lengths)


def metric_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    frame_lengths: tuple[int, ...],
    common_warmup: int,
) -> dict[str, Any]:
    estimate = np.asarray(prediction, dtype=np.complex128)
    measured = np.asarray(target, dtype=np.complex128)
    if estimate.shape != measured.shape or estimate.shape != (sum(frame_lengths),):
        raise ValueError("metric arrays do not match the explicit frame contract")
    interior = common_mask(frame_lengths, common_warmup)
    estimate_frames = frame_segments(estimate, frame_lengths)
    target_frames = frame_segments(measured, frame_lengths)
    complete_length = max(frame_lengths)
    complete = [
        (left, right)
        for left, right in zip(estimate_frames, target_frames, strict=True)
        if left.size == complete_length
    ]
    if complete:
        open_nmse: float | None = nmse_opendpd_db(
            np.stack([pair[0] for pair in complete]),
            np.stack([pair[1] for pair in complete]),
        )
    else:
        open_nmse = None
    return {
        "full_record_nmse_db": nmse_pooled_db(estimate, measured),
        "common_interior_nmse_db": nmse_pooled_db(
            estimate[interior], measured[interior]
        ),
        "full_record_time_domain_rms_evm_db": time_domain_rms_evm_db(
            estimate, measured
        ),
        "common_interior_time_domain_rms_evm_db": time_domain_rms_evm_db(
            estimate[interior], measured[interior]
        ),
        "opendpd_compatible_nmse_db": open_nmse,
        "opendpd_complete_frame_count": len(complete),
        "scored_sample_count_full": int(estimate.size),
        "scored_sample_count_common": int(np.count_nonzero(interior)),
        "per_frame": [
            {
                "frame_id": index,
                "sample_count": int(left.size),
                "full_record_nmse_db": nmse_pooled_db(left, right),
            }
            for index, (left, right) in enumerate(
                zip(estimate_frames, target_frames, strict=True)
            )
        ],
    }


def reverify_inputs(
    verified: dict[str, Any],
    config_path: Path,
    *,
    scope: str,
) -> dict[str, bool]:
    config = verified["config"]
    checks: dict[str, bool] = {
        "config": file_sha256(config_path) == verified["config_sha256"]
    }
    dataset = verified["dataset"]
    for name, expected in verified["dataset_hashes"].items():
        checks[f"dataset/{name}"] = file_sha256(dataset / name) == expected
    for label, record in config["evidence"].items():
        path = project_path(str(record["path"]), name=f"evidence/{label}")
        checks[f"evidence/{label}"] = file_sha256(path) == record["sha256"]
    for name, expected in config["preimplementation_source_sha256"].items():
        checks[f"source/{name}"] = (
            file_sha256(project_path(name, name="source")) == expected
        )
    failed = sorted(label for label, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"{scope} input reverification failed: {failed}")
    return checks


def load_frozen_evidence(verified: dict[str, Any]) -> dict[str, Any]:
    """Load metadata and train OOF prediction only after hash verification."""

    config = verified["config"]
    evidence = config["evidence"]
    sph_manifest_path = project_path(
        evidence["sph_selection_manifest"]["path"],
        name="SPH selection manifest",
    )
    sph_residual_path = project_path(
        evidence["sph_train_residual_analysis"]["path"],
        name="SPH train residual report",
    )
    prediction_path = project_path(
        evidence["gmp_residual_predictions"]["path"],
        name="GMP residual predictions",
    )
    sph_manifest = json.loads(sph_manifest_path.read_text(encoding="utf-8"))
    sph_residual = json.loads(sph_residual_path.read_text(encoding="utf-8"))
    for label, path in (
        ("sph_selection_manifest", sph_manifest_path),
        ("sph_train_residual_analysis", sph_residual_path),
    ):
        if file_sha256(path) != evidence[label]["sha256"]:
            raise RuntimeError(f"{label} changed while being loaded")
    if not isinstance(sph_manifest, dict) or not isinstance(sph_residual, dict):
        raise ValueError("frozen SPH evidence must contain JSON objects")
    if sph_manifest.get("test_split_accessed") is not False:
        raise ValueError("SPH evidence did not seal test access")
    if sph_residual.get("test_access_permitted") is not False:
        raise ValueError("SPH residual evidence permits test access")
    if "spec" not in sph_residual or "frozen_reference" not in sph_residual:
        raise ValueError("SPH residual evidence lacks its frozen analysis contract")

    with np.load(prediction_path, allow_pickle=False) as archive:
        if any(name.lower().startswith("test") for name in archive.files):
            raise ValueError("GMP reference archive contains a test payload")
        if str(archive["model_class"]) != "complex_generalized_memory_polynomial":
            raise ValueError("GMP reference archive has the wrong model class")
        prediction = np.asarray(
            archive["train_oof_prediction"], dtype=np.complex128
        ).copy()
        segment_id = np.asarray(
            archive["train_segment_id"], dtype=np.int64
        ).copy()
    expected_count = int(config["dataset_contract"]["train_sample_count"])
    expected_ids = frame_ids(config["dataset_contract"]["frame_lengths"])
    if prediction.shape != (expected_count,) or not np.all(np.isfinite(prediction)):
        raise ValueError("GMP train OOF prediction is invalid")
    if not np.array_equal(segment_id, expected_ids):
        raise ValueError("GMP train OOF segment IDs disagree with framing")
    if file_sha256(prediction_path) != evidence["gmp_residual_predictions"]["sha256"]:
        raise RuntimeError("GMP reference archive changed while being loaded")
    return {
        "gmp_train_oof_prediction": prediction,
        "residual_spec": ResidualAnalysisSpec(**sph_residual["spec"]),
        "residual_frozen_reference": sph_residual["frozen_reference"],
        "sph_manifest": sph_manifest,
    }


def reference_reproduction(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    frame_lengths: tuple[int, ...],
    common_warmup: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics = metric_summary(
        prediction,
        target,
        frame_lengths=frame_lengths,
        common_warmup=common_warmup,
    )
    expected = config["reference_models"]["matched_gmp_oof"]
    deltas = {
        "full_record_nmse_db": (
            float(metrics["full_record_nmse_db"])
            - float(expected["full_record_nmse_db"])
        ),
        "common_interior_nmse_db": (
            float(metrics["common_interior_nmse_db"])
            - float(expected["common_interior_nmse_db"])
        ),
    }
    if any(
        abs(value) > REFERENCE_REPRODUCTION_TOLERANCE_DB
        for value in deltas.values()
    ):
        raise ValueError(f"frozen GMP reference reproduction failed: {deltas}")
    return {
        "metrics": metrics,
        "frozen_expected": expected,
        "delta_db": deltas,
        "tolerance_db": REFERENCE_REPRODUCTION_TOLERANCE_DB,
        "passed": True,
    }


def streaming_and_reset_checks(
    model: SparseSplineMemoryPA,
    segments: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    probe = segments[0]
    one_shot = model.predict(probe)
    state = model.initial_state()
    chunks: list[np.ndarray] = []
    starts = (0, 1, 17, 263, min(4096, probe.size), probe.size)
    boundaries = sorted(set(value for value in starts if 0 <= value <= probe.size))
    if boundaries[-1] != probe.size:
        boundaries.append(probe.size)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop > start:
            output, state = model.predict_chunk(probe[start:stop], state)
            chunks.append(output)
    streamed = np.concatenate(chunks)
    independent = np.concatenate([model.predict(segment) for segment in segments])
    flattened = np.concatenate(segments)
    segmented = model.predict_segments(flattened, segments[0].size)
    return {
        "streaming_chunk_equivalence_passed": bool(np.array_equal(one_shot, streamed)),
        "maximum_streaming_error": float(
            np.max(np.abs(one_shot - streamed), initial=0.0)
        ),
        "segmented_reset_equivalence_passed": bool(
            np.array_equal(independent, segmented)
        ),
        "maximum_segmented_reset_error": float(
            np.max(np.abs(independent - segmented), initial=0.0)
        ),
    }


def support_summary(
    model: SparseSplineMemoryPA,
    segments: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    radius = np.abs(np.concatenate(segments))
    above = radius > model.knots[-1]
    return {
        "coordinate": "amplitude",
        "fit_maximum": float(model.knots[-1]),
        "observed_maximum": float(np.max(radius)),
        "count_above_fit_maximum": int(np.count_nonzero(above)),
        "fraction_above_fit_maximum": float(np.mean(above)),
    }


def acquire_lock(output: Path) -> tuple[Path, bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable sparse PA output already exists: {output}")
    lock = output.parent / f".{output.name}.lock"
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f"sparse PA output lock already exists: {lock}")
    payload = json.dumps(
        {"pid": os.getpid(), "token": secrets.token_hex(24)}, sort_keys=True
    ).encode("utf-8")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock, payload


def verify_lock(lock: Path, payload: bytes) -> None:
    if lock.is_symlink() or not lock.is_file() or lock.read_bytes() != payload:
        raise RuntimeError("sparse PA publication lock was replaced or modified")


def staged_ledger(search: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "forward_pa_non_factorized_sparse_spline_memory_selection",
        "selection_samples": "train leave-one-explicit-frame-out only",
        "validation_loaded": False,
        "test_split_accessed": False,
        "budget_summary": search["budget_summary"],
        "retained_families": list(search["retained_families"]),
        "stage_recipe_sha256": {
            stage: [row["recipe_sha256"] for row in rows]
            for stage, rows in search["stage_results"].items()
        },
        "unique_trials": {
            digest: strip_prediction(row) for digest, row in search["cache"].items()
        },
        "counts": {
            name: int(search[name])
            for name in (
                "cache_hits",
                "unique_recipe_evaluations",
                "completed_oof_fit_calls",
                "stage_recipe_associations",
            )
        },
        "decision": search["decision"],
    }


def publish_bundle(
    output: Path,
    *,
    lock: Path,
    lock_payload: bytes,
    model: SparseSplineMemoryPA,
    manifest_payload: dict[str, Any],
    staged_ledger_payload: dict[str, Any],
    predictions: dict[str, np.ndarray],
    residual_reports: dict[str, dict[str, Any]],
    execution: dict[str, Any],
) -> dict[str, Any]:
    verify_lock(lock, lock_payload)
    if output.exists() or output.is_symlink():
        raise FileExistsError("sparse PA output appeared before publication")
    temporary = output.parent / f".{output.name}.tmp-{secrets.token_hex(12)}"
    temporary.mkdir()
    try:
        model.save(temporary / "selected_sparse_pa.npz")
        np.savez_compressed(
            temporary / "predictions.npz",
            schema_version=np.asarray(1, dtype=np.int64),
            model_type=np.asarray(
                "phase_equivariant_non_factorized_sparse_spline_memory_pa"
            ),
            **predictions,
        )
        write_json(temporary / "staged_trials.json", staged_ledger_payload)
        write_json(
            temporary / "train_oof_residual_analysis.json",
            residual_reports["train_oof"],
        )
        write_json(
            temporary / "validation_reused_residual_analysis.json",
            residual_reports["validation_reused"],
        )
        write_json(temporary / "execution_record.json", execution)
        artifact_names = (
            "selected_sparse_pa.npz",
            "predictions.npz",
            "staged_trials.json",
            "train_oof_residual_analysis.json",
            "validation_reused_residual_analysis.json",
            "execution_record.json",
        )
        artifacts = {
            name: {"path": name, "sha256": file_sha256(temporary / name)}
            for name in artifact_names
        }
        manifest = {
            **manifest_payload,
            "artifacts": artifacts,
            "publication": {
                "immutable_bundle": True,
                "atomic_directory_rename": True,
                "completion_manifest_written_last_inside_temporary_bundle": True,
            },
        }
        write_json(temporary / "selection_manifest.json", manifest)
        verify_lock(lock, lock_payload)
        if output.exists() or output.is_symlink():
            raise FileExistsError("sparse PA output appeared during publication")
        os.replace(temporary, output)
        temporary = None  # type: ignore[assignment]
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
