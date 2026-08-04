"""Prepare the external ``BlackBoxData.mat`` capture for local experiments.

The source MAT file is deliberately not copied into the repository.  This
command validates its three top-level complex vectors and exports the ``x/y``
pair into the split-CSV convention already used by the project.  Selection
data and the held-out test are placed in separate directories so a selector
can be given only the former.  ``eRef`` is recorded only as an unresolved
diagnostic channel; it is never used as a model input or target.

The default chronological split preserves the indices supplied with the data:

* discard ``[0, 5000)``;
* train ``[5000, 97000)``;
* validation ``[97000, 120000)``;
* held-out test ``[120000, N)``.

No sampling rate, RF carrier, frame length, occupied bandwidth, or spectral
acceptance region is invented by this adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA_ROOT = PROJECT_ROOT / "data" / "private"
OWNED_FILES = (
    "selection/train_input.csv",
    "selection/train_output.csv",
    "selection/val_input.csv",
    "selection/val_output.csv",
    "selection/spec.json",
    "selection/selection_view.json",
    "sealed/test_input.csv",
    "sealed/test_output.csv",
    "sealed/test_release.json",
    "preparation_manifest.json",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _complex_vector(value: Any, *, name: str) -> tuple[np.ndarray, list[int]]:
    raw = np.asarray(value)
    source_shape = [int(item) for item in raw.shape]
    if raw.ndim not in {1, 2} or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty MATLAB vector")
    if raw.ndim == 2 and 1 not in raw.shape:
        raise ValueError(f"{name} must be a MATLAB row or column vector")
    if not np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain complex I/Q samples")
    vector = np.asarray(raw.reshape(-1), dtype=np.complex128)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return vector, source_shape


def load_blackbox_mat(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the three documented container variables."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        import scipy.io
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "BlackBox MAT import requires the optional scipy dependency"
        ) from error

    payload = scipy.io.loadmat(source)
    missing = {name for name in ("x", "y", "eRef") if name not in payload}
    if missing:
        raise ValueError(f"MAT file is missing variables: {sorted(missing)}")

    vectors: dict[str, np.ndarray] = {}
    shapes: dict[str, list[int]] = {}
    for name in ("x", "y", "eRef"):
        vectors[name], shapes[name] = _complex_vector(payload[name], name=name)
    lengths = {vector.size for vector in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("x, y, and eRef must have equal sample counts")
    return {
        "source": source.resolve(),
        "source_sha256": file_sha256(source),
        "vectors": vectors,
        "source_shapes": shapes,
    }


def _validate_boundaries(
    sample_count: int,
    *,
    train_start: int,
    validation_start: int,
    test_start: int,
) -> None:
    boundaries = (train_start, validation_start, test_start)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
        raise TypeError("split boundaries must be integers")
    if not 0 <= train_start < validation_start < test_start < sample_count:
        raise ValueError(
            "split boundaries must satisfy "
            "0 <= train_start < validation_start < test_start < sample_count"
        )


def _write_iq_csv(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.column_stack((signal.real, signal.imag))
    np.savetxt(
        path,
        values,
        delimiter=",",
        header="I,Q",
        comments="",
        fmt="%.17g",
    )


def _is_beneath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_location(output: Path, *, allow_public_output: bool) -> None:
    if (
        _is_beneath(output, PROJECT_ROOT)
        and not _is_beneath(output, PRIVATE_DATA_ROOT)
        and not allow_public_output
    ):
        raise ValueError(
            "prepared capture data inside the repository must stay under "
            f"{PRIVATE_DATA_ROOT}; pass allow_public_output=True only after "
            "confirming publication rights"
        )


def prepare_blackbox_data(
    mat_path: str | Path,
    output_dir: str | Path,
    *,
    train_start: int = 5_000,
    validation_start: int = 97_000,
    test_start: int = 120_000,
    allow_public_output: bool = False,
) -> dict[str, Any]:
    """Export deterministic chronological splits and an evidence manifest."""

    loaded = load_blackbox_mat(mat_path)
    vectors = loaded["vectors"]
    sample_count = int(vectors["x"].size)
    _validate_boundaries(
        sample_count,
        train_start=train_start,
        validation_start=validation_start,
        test_start=test_start,
    )

    final_output = Path(output_dir).resolve()
    _validate_output_location(
        final_output,
        allow_public_output=allow_public_output,
    )
    if final_output.exists():
        raise FileExistsError(
            "prepared datasets are immutable; choose a new output directory: "
            f"{final_output}"
        )

    ranges = {
        "train": (train_start, validation_start),
        "val": (validation_start, test_start),
        "test": (test_start, sample_count),
    }
    training_input = vectors["x"][train_start:validation_start]
    training_peak = float(np.max(np.abs(training_input)))
    if training_peak <= 0.0:
        raise ValueError("training x must have non-zero peak amplitude")

    eref = vectors["eRef"]
    y = vectors["y"]
    eref_power = float(np.mean(np.abs(eref) ** 2))
    y_power = float(np.mean(np.abs(y) ** 2))
    if y_power <= 0.0:
        raise ValueError("y must have non-zero average power")
    eref_relative_power = eref_power / y_power
    eref_relative_db = (
        None
        if eref_relative_power == 0.0
        else float(10.0 * np.log10(eref_relative_power))
    )

    final_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.tmp-",
            dir=final_output.parent,
        )
    )
    try:
        return _prepare_into_staging(
            loaded=loaded,
            vectors=vectors,
            sample_count=sample_count,
            output=output,
            ranges=ranges,
            train_start=train_start,
            validation_start=validation_start,
            test_start=test_start,
            training_peak=training_peak,
            eref_power=eref_power,
            eref_relative_power=eref_relative_power,
            eref_relative_db=eref_relative_db,
            final_output=final_output,
        )
    finally:
        published = final_output.exists() and not output.exists()
        if not published and output.exists():
            shutil.rmtree(output)


def _prepare_into_staging(
    *,
    loaded: dict[str, Any],
    vectors: dict[str, np.ndarray],
    sample_count: int,
    output: Path,
    ranges: dict[str, tuple[int, int]],
    train_start: int,
    validation_start: int,
    test_start: int,
    training_peak: float,
    eref_power: float,
    eref_relative_power: float,
    eref_relative_db: float | None,
    final_output: Path,
) -> dict[str, Any]:
    """Build a complete bundle in a private staging directory, then publish."""

    for split in ("train", "val"):
        start, stop = ranges[split]
        _write_iq_csv(
            output / "selection" / f"{split}_input.csv",
            vectors["x"][start:stop],
        )
        _write_iq_csv(
            output / "selection" / f"{split}_output.csv",
            vectors["y"][start:stop],
        )
    test_begin, test_end = ranges["test"]
    _write_iq_csv(
        output / "sealed" / "test_input.csv",
        vectors["x"][test_begin:test_end],
    )
    _write_iq_csv(
        output / "sealed" / "test_output.csv",
        vectors["y"][test_begin:test_end],
    )

    spec = {
        "schema_version": SCHEMA_VERSION,
        "dataset_label": "BlackBoxData external capture",
        "sample_rate_status": "unknown_not_present_in_mat",
        "rf_carrier_status": "unknown_not_present_in_mat",
        "waveform_status": "unknown_not_present_in_mat",
        "frame_length_status": "unknown_not_present_in_mat",
        "spectral_regions_status": "unknown_not_present_in_mat",
        "sequence_policy": "each split is one independent chronological record",
        "normalization_policy": (
            "CSV values remain in source units; model runners must divide x and y "
            "by selection_view.normalization_contract.training_input_peak"
        ),
    }
    _write_json(output / "selection" / "spec.json", spec)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "blackbox_mat_split_manifest",
        "source": {
            "filename": loaded["source"].name,
            "sha256": loaded["source_sha256"],
            "mat_variables": {
                name: {
                    "source_shape": loaded["source_shapes"][name],
                    "sample_count": sample_count,
                    "export_dtype": "complex128",
                }
                for name in ("x", "y", "eRef")
            },
        },
        "semantics": {
            "x": "provisional PA/black-box complex input",
            "y": "provisional corresponding complex output",
            "status": "must_be_confirmed_by_data_owner",
            "eRef": "unknown auxiliary/reference channel",
            "eRef_used_as_model_input_or_target": False,
        },
        "split_contract": {
            "indexing": "zero_based_half_open",
            "discarded": {"start": 0, "stop": train_start, "count": train_start},
            "train": {
                "start": train_start,
                "stop": validation_start,
                "count": validation_start - train_start,
            },
            "validation": {
                "start": validation_start,
                "stop": test_start,
                "count": test_start - validation_start,
            },
            "test": {
                "start": test_start,
                "stop": sample_count,
                "count": sample_count - test_start,
            },
            "chronological": True,
            "overlap_samples": 0,
            "test_allowed_for_model_selection": False,
            "mat_container_loaded_as_one_file_during_export": True,
        },
        "normalization_contract": {
            "csv_values_scaled": False,
            "training_input_peak": training_peak,
            "recommended_common_scale_for_x_and_y": training_peak,
            "scale_fitted_from": "train_input_only",
        },
        "eRef_diagnostic": {
            "rms": float(np.sqrt(eref_power)),
            "power_relative_to_y": eref_relative_power,
            "power_relative_to_y_db": eref_relative_db,
            "interpretation": "unknown_do_not_use_without_owner_definition",
        },
        "missing_metadata": [
            "sample_rate_hz",
            "rf_carrier_hz",
            "waveform_definition",
            "frame_boundaries",
            "occupied_bandwidth_hz",
            "adjacent_or_harmonic_regions_hz",
        ],
    }
    selection_files = (
        "train_input.csv",
        "train_output.csv",
        "val_input.csv",
        "val_output.csv",
        "spec.json",
    )
    selection_hashes = {
        name: file_sha256(output / "selection" / name)
        for name in selection_files
    }
    selection_view = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "blackbox_selection_view",
        "source_filename": loaded["source"].name,
        "source_sha256": loaded["source_sha256"],
        "available_splits": ["train", "validation"],
        "test_split_available": False,
        "test_path_or_hash_included": False,
        "split_contract": {
            "indexing": "zero_based_half_open",
            "train": manifest["split_contract"]["train"],
            "validation": manifest["split_contract"]["validation"],
        },
        "normalization_contract": manifest["normalization_contract"],
        "semantics": manifest["semantics"],
        "missing_metadata": manifest["missing_metadata"],
        "files_sha256": selection_hashes,
    }
    _write_json(output / "selection" / "selection_view.json", selection_view)

    test_hashes = {
        name: file_sha256(output / "sealed" / name)
        for name in ("test_input.csv", "test_output.csv")
    }
    test_release = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "blackbox_sealed_test_release",
        "source_filename": loaded["source"].name,
        "source_sha256": loaded["source_sha256"],
        "split_contract": {
            "indexing": "zero_based_half_open",
            "test": manifest["split_contract"]["test"],
        },
        "files_sha256": test_hashes,
        "release_policy": "open_only_after_model_and_protocol_are_frozen",
    }
    _write_json(output / "sealed" / "test_release.json", test_release)

    manifest["views"] = {
        "selection": {
            "directory": "selection",
            "manifest": "selection/selection_view.json",
            "manifest_sha256": file_sha256(
                output / "selection" / "selection_view.json"
            ),
        },
        "sealed_test": {
            "directory": "sealed",
            "manifest": "sealed/test_release.json",
            "manifest_sha256": file_sha256(output / "sealed" / "test_release.json"),
        },
    }
    manifest["exported_files_sha256"] = {
        **{f"selection/{name}": digest for name, digest in selection_hashes.items()},
        "selection/selection_view.json": manifest["views"]["selection"][
            "manifest_sha256"
        ],
        **{f"sealed/{name}": digest for name, digest in test_hashes.items()},
        "sealed/test_release.json": manifest["views"]["sealed_test"][
            "manifest_sha256"
        ],
    }
    _write_json(output / "preparation_manifest.json", manifest)
    os.replace(output, final_output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and split an external BlackBoxData.mat capture."
    )
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=5_000)
    parser.add_argument("--validation-start", type=int, default=97_000)
    parser.add_argument("--test-start", type=int, default=120_000)
    parser.add_argument(
        "--allow-public-output",
        action="store_true",
        help=(
            "allow generated capture samples inside the repository but outside "
            "data/private; use only after confirming publication rights"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_blackbox_data(
        args.mat,
        args.output_dir,
        train_start=args.train_start,
        validation_start=args.validation_start,
        test_start=args.test_start,
        allow_public_output=args.allow_public_output,
    )
    split = manifest["split_contract"]
    print(
        "Prepared BlackBoxData:",
        f"train={split['train']['count']}",
        f"validation={split['validation']['count']}",
        f"test={split['test']['count']}",
        f"source_sha256={manifest['source']['sha256']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
