"""Run the sealed validation-only spline-memory DPD surrogate demo.

This is a presentation/reproduction wrapper around the existing frozen
runners.  It performs no fit or model selection and never opens a measured PA
output.  Both datasets follow the deployment direction

    desired validation x -> frozen DPD -> frozen PA surrogate.

The wrapper verifies input/source hashes, explicit claim boundaries, reference
metrics within declared floating-point tolerances, operation schedules, fixed
point saturation counters and streaming equivalence.  A completion manifest
is published last; its absence marks an incomplete run.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_fixed_point_dpd import (  # noqa: E402
    evaluate as evaluate_fixed_point,
)
from experiments.evaluate_frozen_dpd_spectrum import (  # noqa: E402
    evaluate as evaluate_spectrum,
)
from experiments.replay_frozen_spline_memory_dpd import (  # noqa: E402
    replay as replay_float,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = PROJECT_ROOT / "experiments/configs/surrogate_demo.json"
RUNNER_SOURCE = "experiments/run_surrogate_demo.py"
EXPECTED_DATASETS = ("DPA_200MHz", "APA_200MHz")
EXPECTED_FORMATS = ("16", "14", "12")
CHILD_STAGES = (
    "float_replay",
    "float_spectrum",
    "fixed_point",
    "fixed_spectrum_16bit",
    "fixed_spectrum_14bit",
    "fixed_spectrum_12bit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    if "\\" in value:
        raise ValueError(f"{field} must use POSIX repository path separators")
    raw = Path(value)
    if (
        raw.is_absolute()
        or raw in {Path("."), Path("..")}
        or ".." in raw.parts
        or any(component in {"", "."} for component in raw.parts)
    ):
        raise ValueError(f"{field} must stay beneath the repository")
    return raw


def _read_regular_file_beneath(
    root: Path,
    relative_path: Path,
    *,
    field: str,
) -> bytes:
    """Read one regular file without following any path-component symlink."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError(
            "sealed surrogate reproduction requires O_NOFOLLOW and O_DIRECTORY"
        )
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        directory_descriptors.append(os.open(root, directory_flags))
        for component in relative_path.parts[:-1]:
            directory_descriptors.append(
                os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptors[-1],
                )
            )
        file_descriptor = os.open(
            relative_path.parts[-1],
            file_flags,
            dir_fd=directory_descriptors[-1],
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field} must be one regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"every {field} path component must be a real non-symlink "
                "directory or file"
            ) from error
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _read_repo_regular_file(value: object, *, field: str) -> tuple[Path, bytes]:
    relative_path = _repo_relative_path(value, field=field)
    payload = _read_regular_file_beneath(
        PROJECT_ROOT,
        relative_path,
        field=field,
    )
    return PROJECT_ROOT / relative_path, payload


def _absolute_without_symlink_resolution(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _read_regular_path(value: str | Path, *, field: str) -> tuple[Path, bytes]:
    """Read an arbitrary absolute/relative path through component handles."""

    absolute = _absolute_without_symlink_resolution(value)
    if absolute.anchor != os.path.sep or len(absolute.parts) < 2:
        raise ValueError(f"{field} must name one regular file")
    relative = Path(*absolute.parts[1:])
    return absolute, _read_regular_file_beneath(
        Path(os.path.sep),
        relative,
        field=field,
    )


def _hash(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(12)}"
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Publish one owned immutable input copy without following a symlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_owned_output_root(path: Path) -> tuple[int, int]:
    """Create the one output directory and return its immutable identity."""

    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite demo output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(path)
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("created surrogate demo output is not a directory")
    return int(metadata.st_dev), int(metadata.st_ino)


def _assert_owned_output_root(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    """Fail if the output directory name no longer denotes our inode."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise RuntimeError("surrogate demo output directory disappeared") from error
    actual_identity = (int(metadata.st_dev), int(metadata.st_ino))
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or actual_identity != expected_identity
    ):
        raise RuntimeError("surrogate demo output directory identity changed")


def _validate_operation(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    integer_fields = {
        "comparisons",
        "lookups",
        "nonlinear_operations",
        "real_additions",
        "real_divisions",
        "real_memory_reads",
        "real_memory_writes",
        "real_multiplications",
        "state_real_values",
        "stored_real_coefficients",
        "stored_real_constants",
    }
    if set(value) != integer_fields:
        raise ValueError(f"{field} has an incomplete operation contract")
    result: dict[str, Any] = {}
    for name in sorted(integer_fields):
        item = value[name]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{field}.{name} must be a non-negative integer")
        result[name] = item
    return result


def _validate_metric_map(value: object, *, field: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a non-empty object")
    return {
        str(name): _finite(item, field=f"{field}.{name}")
        for name, item in value.items()
    }


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "task",
        "claims_scope",
        "environment",
        "tolerances",
        "files_sha256",
        "datasets",
    }
    if set(config) != required:
        raise ValueError("surrogate demo config has unknown or missing keys")
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported surrogate demo schema")
    if config.get("task") != "frozen_spline_memory_surrogate_demo":
        raise ValueError("unexpected surrogate demo task")
    if config.get("claims_scope") != {
        "surrogate_only": True,
        "physical_pa_result": False,
        "rf_harmonic_claim": False,
        "test_split_accessed": False,
        "model_selection_performed": False,
        "validation_reused_after_historical_model_selection": True,
    }:
        raise ValueError("claims_scope must preserve the validation-only boundary")

    environment = config.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "numpy_version"
    }:
        raise ValueError("environment must freeze exactly numpy_version")
    if not isinstance(environment["numpy_version"], str):
        raise ValueError("environment.numpy_version must be a string")

    tolerances = config.get("tolerances")
    if not isinstance(tolerances, dict) or set(tolerances) != {
        "amplitude_absolute",
        "db_absolute",
    }:
        raise ValueError("tolerances must define amplitude_absolute and db_absolute")
    for name, value in tolerances.items():
        numeric = _finite(value, field=f"tolerances.{name}")
        if numeric <= 0.0:
            raise ValueError(f"tolerances.{name} must be positive")

    files = config.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("files_sha256 must be a non-empty object")
    for raw_path, expected in files.items():
        _repo_relative_path(raw_path, field="files_sha256 path")
        _hash(expected, field=f"files_sha256[{raw_path}]")
    if RUNNER_SOURCE not in files:
        raise ValueError("files_sha256 must bind the demo runner")

    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or tuple(datasets) != EXPECTED_DATASETS:
        raise ValueError("datasets must be ordered DPA_200MHz then APA_200MHz")
    for name, dataset in datasets.items():
        field = f"datasets.{name}"
        if not isinstance(dataset, dict) or set(dataset) != {
            "fixed_operation_contract",
            "fixed_point_config",
            "float_operation_contract",
            "float_replay_config",
            "reference",
        }:
            raise ValueError(f"{field} has unknown or missing keys")
        replay_path = _repo_relative_path(
            dataset["float_replay_config"], field=f"{field}.float_replay_config"
        )
        fixed_path = _repo_relative_path(
            dataset["fixed_point_config"], field=f"{field}.fixed_point_config"
        )
        if replay_path.as_posix() not in files:
            raise ValueError(f"{field} float config is not hash-bound")
        if fixed_path.as_posix() not in files:
            raise ValueError(f"{field} fixed config is not hash-bound")
        _validate_operation(
            dataset["float_operation_contract"],
            field=f"{field}.float_operation_contract",
        )
        _validate_operation(
            dataset["fixed_operation_contract"],
            field=f"{field}.fixed_operation_contract",
        )
        reference = dataset["reference"]
        if not isinstance(reference, dict) or set(reference) != {
            "fixed",
            "float",
        }:
            raise ValueError(f"{field}.reference must define float and fixed")
        _validate_metric_map(reference["float"], field=f"{field}.reference.float")
        fixed_reference = reference["fixed"]
        if (
            not isinstance(fixed_reference, dict)
            or tuple(fixed_reference) != EXPECTED_FORMATS
        ):
            raise ValueError(f"{field}.reference.fixed has wrong formats")
        for bits, metrics in fixed_reference.items():
            _validate_metric_map(metrics, field=f"{field}.reference.fixed.{bits}")


def _load_verified_frozen_files(
    config: dict[str, Any],
) -> tuple[dict[str, str], dict[str, bytes]]:
    verified: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for raw_path, expected_value in config["files_sha256"].items():
        _, payload = _read_repo_regular_file(raw_path, field="frozen file")
        expected = _hash(expected_value, field=f"files_sha256[{raw_path}]")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(
                f"frozen file SHA-256 mismatch for {raw_path}: "
                f"expected {expected}, found {actual}"
            )
        verified[raw_path] = actual
        payloads[raw_path] = payload
    return verified, payloads


def _verify_frozen_files(config: dict[str, Any]) -> dict[str, str]:
    verified, _ = _load_verified_frozen_files(config)
    return verified


def _assert_close(
    actual: object,
    expected: object,
    *,
    tolerance: float,
    field: str,
) -> float:
    actual_value = _finite(actual, field=f"actual {field}")
    expected_value = _finite(expected, field=f"expected {field}")
    difference = abs(actual_value - expected_value)
    if difference > tolerance:
        raise ValueError(
            f"reference metric mismatch for {field}: actual={actual_value}, "
            f"expected={expected_value}, tolerance={tolerance}"
        )
    return actual_value


def _metric_row(
    replay_report: dict[str, Any],
    spectral_report: dict[str, Any],
) -> dict[str, float]:
    return {
        "no_dpd_nmse_db": replay_report["metrics"]["without_dpd_vs_ideal"][
            "complex_nmse_pooled_db"
        ],
        "dpd_nmse_db": replay_report["metrics"]["with_dpd_vs_ideal"][
            "complex_nmse_pooled_db"
        ],
        "predistorted_peak": replay_report["metrics"][
            "predistorted_peak_amplitude"
        ],
        "predistorted_papr_db": replay_report["metrics"][
            "predistorted_papr_db"
        ],
        "main_power_change_db": spectral_report["main_region"][
            "main_power_change_db"
        ]["value"],
        "left_relative_improvement_db": spectral_report["regions"][
            "left_adjacent"
        ]["relative_leakage_improvement_db"]["value"],
        "right_relative_improvement_db": spectral_report["regions"][
            "right_adjacent"
        ]["relative_leakage_improvement_db"]["value"],
        "left_absolute_suppression_db": spectral_report["regions"][
            "left_adjacent"
        ]["absolute_suppression_db"]["value"],
        "right_absolute_suppression_db": spectral_report["regions"][
            "right_adjacent"
        ]["absolute_suppression_db"]["value"],
    }


def _fixed_metric_row(
    format_report: dict[str, Any],
    spectral_report: dict[str, Any],
) -> dict[str, float]:
    validation = format_report["validation"]
    return {
        "cascade_nmse_db": validation["fixed_cascade_vs_ideal"][
            "complex_nmse_pooled_db"
        ],
        "drive_vs_float_nmse_db": validation["fixed_vs_float_drive"][
            "complex_nmse_pooled_db"
        ],
        "predistorted_peak": validation["fixed_drive"]["maximum_amplitude"],
        "predistorted_papr_db": validation["fixed_drive"]["papr_db"],
        "left_absolute_suppression_db": spectral_report["regions"][
            "left_adjacent"
        ]["absolute_suppression_db"]["value"],
        "right_absolute_suppression_db": spectral_report["regions"][
            "right_adjacent"
        ]["absolute_suppression_db"]["value"],
    }


def _verify_metrics(
    actual: dict[str, float],
    expected: dict[str, Any],
    *,
    tolerances: dict[str, float],
    field: str,
) -> dict[str, float]:
    if set(actual) != set(expected):
        raise ValueError(f"{field} reference metric keys do not match")
    result: dict[str, float] = {}
    for name, value in actual.items():
        tolerance = (
            tolerances["amplitude_absolute"]
            if name == "predistorted_peak"
            else tolerances["db_absolute"]
        )
        result[name] = _assert_close(
            value,
            expected[name],
            tolerance=tolerance,
            field=f"{field}.{name}",
        )
    return result


def _operation_without_notes(value: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(item)
        for name, item in value.items()
        if name != "notes"
    }


def _assert_operation(
    actual: dict[str, Any], expected: dict[str, Any], *, field: str
) -> dict[str, int]:
    normalized = _operation_without_notes(actual)
    if normalized != expected:
        raise ValueError(
            f"operation contract mismatch for {field}: "
            f"actual={normalized}, expected={expected}"
        )
    return normalized


def _assert_fixed_integrity(format_report: dict[str, Any], *, field: str) -> None:
    selection = format_report["selection_or_tuning"]
    if selection != {
        "precision_candidates_preregistered": [16, 14, 12],
        "precision_selected_by_runner": False,
        "scales_frozen_before_validation": True,
        "used_for_selection": False,
        "validation_used_to_modify_model": False,
    }:
        raise ValueError(f"{field} precision/scale selection contract changed")
    for split in ("train", "validation"):
        split_report = format_report[split]
        if split_report["streaming"]["streaming_chunk_equivalence_passed"] is not True:
            raise ValueError(f"{field}.{split} streaming equivalence failed")
        for name, value in split_report["stats"].items():
            if "saturations" in name or name == "knot_code_collision_count":
                if value != 0:
                    raise ValueError(f"{field}.{split}.{name} is nonzero")
    phase = format_report["validation"]["phase_equivariance"]
    if phase["bit_exact"] is not True:
        raise ValueError(f"{field}.validation phase equivariance is not bit-exact")
    for name, value in phase["rotated_input_stats"].items():
        if "saturations" in name or name == "knot_code_collision_count":
            if value != 0:
                raise ValueError(
                    f"{field}.validation.phase_equivariance.{name} is nonzero"
                )


def _assert_claim_boundaries(
    replay_report: dict[str, Any],
    spectral_report: dict[str, Any],
    fixed_report: dict[str, Any],
    fixed_spectra: dict[str, dict[str, Any]],
    *,
    field: str,
) -> None:
    if replay_report["claims_scope"] != {
        "physical_pa_result": False,
        "surrogate_only": True,
        "test_used_for_selection": False,
        "historical_test_access": False,
    }:
        raise ValueError(f"{field} float replay claim boundary changed")
    if replay_report["dataset"]["split"] != "val":
        raise ValueError(f"{field} float replay is not validation-only")
    if replay_report["dataset"]["measured_output_opened"] is not False:
        raise ValueError(f"{field} float replay opened measured output")
    direction = replay_report["direction"]
    if (
        direction["measured_output_used_as_dpd_input"] is not False
        or direction["inverse_diagnostic_run"] is not False
        or not direction["deployment_path"].startswith("desired input x ->")
    ):
        raise ValueError(f"{field} float replay direction is invalid")
    for name, report in {"float": spectral_report, **fixed_spectra}.items():
        claims = report["claims_scope"]
        if (
            claims["surrogate_only"] is not True
            or claims["physical_pa_measurement"] is not False
            or claims["rf_harmonic_claim"] is not False
            or report["split_role"] != "validation"
        ):
            raise ValueError(f"{field}.{name} spectral claim boundary changed")
    claims = fixed_report["claims_scope"]
    required_false = (
        "dpd_latency_gate_evaluable",
        "eligible_as_untouched_final_evidence",
        "hardware_latency_or_resources",
        "measured_output_opened",
        "physical_pa_result",
        "precision_selected_by_runner",
        "rf_harmonic_claim",
        "rtl_bit_true",
        "test_split_accessed",
    )
    if claims["surrogate_only"] is not True or any(
        claims[name] is not False for name in required_false
    ):
        raise ValueError(f"{field} fixed-point claim boundary changed")
    if fixed_report["dataset"]["allowed_waveform_files_opened"] != [
        "train_input.csv",
        "val_input.csv",
    ]:
        raise ValueError(f"{field} fixed-point waveform access changed")
    fixed_direction = fixed_report["direction"]
    if (
        fixed_direction["measured_output_used_as_dpd_input"] is not False
        or fixed_direction["inverse_diagnostic_run"] is not False
        or not fixed_direction["deployment_path"].startswith(
            "desired validation x ->"
        )
    ):
        raise ValueError(f"{field} fixed-point direction is invalid")


def _expected_child_manifest_paths() -> set[str]:
    return {
        (
            Path("datasets")
            / dataset.lower()
            / stage
            / "completion_manifest.json"
        ).as_posix()
        for dataset in EXPECTED_DATASETS
        for stage in CHILD_STAGES
    }


def _child_manifest_hashes(output: Path) -> dict[str, str]:
    paths = sorted(output.rglob("completion_manifest.json"))
    relative_paths = {
        path.relative_to(output).as_posix()
        for path in paths
    }
    expected_paths = _expected_child_manifest_paths()
    if relative_paths != expected_paths:
        missing = sorted(expected_paths - relative_paths)
        unexpected = sorted(relative_paths - expected_paths)
        raise RuntimeError(
            "surrogate demo child manifest set changed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in paths
    }


def run(config_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_file, config_bytes = _read_regular_path(
        config_path,
        field="surrogate demo config",
    )
    config = json.loads(config_bytes.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("surrogate demo config must contain one JSON object")
    validate_config(config)
    if np.__version__ != config["environment"]["numpy_version"]:
        raise RuntimeError(
            "NumPy version mismatch: "
            f"expected {config['environment']['numpy_version']}, "
            f"found {np.__version__}"
        )
    verified_files, frozen_payloads = _load_verified_frozen_files(config)

    output = _absolute_without_symlink_resolution(output_root)
    output_identity = _create_owned_output_root(output)
    completed = False
    try:
        tolerances = {
            name: float(value) for name, value in config["tolerances"].items()
        }
        summary_datasets: dict[str, Any] = {}
        for dataset_name, dataset_config in config["datasets"].items():
            slug = dataset_name.lower()
            dataset_output = output / "datasets" / slug
            replay_output = dataset_output / "float_replay"
            spectrum_output = dataset_output / "float_spectrum"
            fixed_output = dataset_output / "fixed_point"

            replay_relative = _repo_relative_path(
                dataset_config["float_replay_config"],
                field=f"{dataset_name} float replay config",
            )
            fixed_relative = _repo_relative_path(
                dataset_config["fixed_point_config"],
                field=f"{dataset_name} fixed-point config",
            )
            replay_payload = frozen_payloads[replay_relative.as_posix()]
            fixed_payload = frozen_payloads[fixed_relative.as_posix()]
            replay_config = output / "sealed_inputs" / slug / "float_replay.json"
            fixed_config = output / "sealed_inputs" / slug / "fixed_point.json"
            _write_bytes_exclusive(replay_config, replay_payload)
            _write_bytes_exclusive(fixed_config, fixed_payload)
            replay_report = replay_float(replay_config, replay_output)
            spectral_report = evaluate_spectrum(
                replay_output / "spectral_config.json",
                spectrum_output,
                release_test=False,
            )
            fixed_report = evaluate_fixed_point(fixed_config, fixed_output)
            fixed_spectra: dict[str, dict[str, Any]] = {}
            for bits in EXPECTED_FORMATS:
                mode = f"{bits}bit"
                fixed_spectra[bits] = evaluate_spectrum(
                    fixed_output / f"spectral_config_{mode}.json",
                    dataset_output / f"fixed_spectrum_{mode}",
                    release_test=False,
                )

            _assert_claim_boundaries(
                replay_report,
                spectral_report,
                fixed_report,
                fixed_spectra,
                field=dataset_name,
            )
            float_operation = _assert_operation(
                fixed_report["float_reference"]["dpd_operation_count"],
                dataset_config["float_operation_contract"],
                field=f"{dataset_name}.float",
            )
            float_metrics = _verify_metrics(
                _metric_row(replay_report, spectral_report),
                dataset_config["reference"]["float"],
                tolerances=tolerances,
                field=f"{dataset_name}.float",
            )
            fixed_metrics: dict[str, dict[str, float]] = {}
            fixed_operation: dict[str, int] | None = None
            for bits in EXPECTED_FORMATS:
                format_report = fixed_report["formats"][bits]
                _assert_fixed_integrity(
                    format_report, field=f"{dataset_name}.fixed.{bits}"
                )
                operation = _assert_operation(
                    format_report["fixed_schedule_operation_count"],
                    dataset_config["fixed_operation_contract"],
                    field=f"{dataset_name}.fixed.{bits}",
                )
                if fixed_operation is not None and operation != fixed_operation:
                    raise ValueError(
                        f"{dataset_name} fixed operation schedule varies by format"
                    )
                fixed_operation = operation
                fixed_metrics[bits] = _verify_metrics(
                    _fixed_metric_row(format_report, fixed_spectra[bits]),
                    dataset_config["reference"]["fixed"][bits],
                    tolerances=tolerances,
                    field=f"{dataset_name}.fixed.{bits}",
                )
            assert fixed_operation is not None
            summary_datasets[dataset_name] = {
                "all_checks_passed": True,
                "claim_scope": "validation_replay_surrogate_only",
                "direction": (
                    "desired validation x -> frozen DPD -> frozen PA surrogate"
                ),
                "float": {
                    "metrics": float_metrics,
                    "operation_count": float_operation,
                },
                "fixed_point": {
                    "formats": fixed_metrics,
                    "operation_count": fixed_operation,
                    "precision_selected": False,
                    "saturation_or_collision_count": 0,
                    "streaming_chunk_equivalence": True,
                },
                "artifacts": {
                    "float_replay": str(replay_output.relative_to(output)),
                    "float_spectrum": str(spectrum_output.relative_to(output)),
                    "fixed_point": str(fixed_output.relative_to(output)),
                    "fixed_spectra": {
                        bits: str(
                            (
                                dataset_output / f"fixed_spectrum_{bits}bit"
                            ).relative_to(output)
                        )
                        for bits in EXPECTED_FORMATS
                    },
                },
            }

        _, final_config_bytes = _read_regular_path(
            config_file,
            field="surrogate demo config",
        )
        if final_config_bytes != config_bytes:
            raise RuntimeError("surrogate demo config changed during execution")
        if _verify_frozen_files(config) != verified_files:
            raise RuntimeError("frozen demo files changed during execution")
        _assert_owned_output_root(output, output_identity)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "frozen_spline_memory_surrogate_demo",
            "all_checks_passed": True,
            "claims_scope": config["claims_scope"],
            "method": {
                "equation": "z[n] = sum_{m in {0,1,2}} x[n-m] C_m(|x[n]|)",
                "basis": "complex local-linear spline with quantile knots",
                "phase_equivariant": True,
            },
            "environment": {
                "numpy": np.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "execution": {
                "wall_seconds_before_summary_publication": (
                    time.perf_counter() - started
                ),
                "scope": (
                    "host Python reproduction time; not DPD inference latency"
                ),
            },
            "provenance": {
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "frozen_files_sha256": verified_files,
            },
            "tolerances": config["tolerances"],
            "datasets": summary_datasets,
        }
        summary_path = output / "summary.json"
        _write_json_atomic(summary_path, summary)

        child_manifests = _child_manifest_hashes(output)
        _, final_config_bytes = _read_regular_path(
            config_file,
            field="surrogate demo config",
        )
        if final_config_bytes != config_bytes:
            raise RuntimeError("surrogate demo config changed before completion")
        if _verify_frozen_files(config) != verified_files:
            raise RuntimeError("frozen demo files changed before completion")
        _assert_owned_output_root(output, output_identity)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "frozen_spline_memory_surrogate_demo_bundle",
            "all_checks_passed": True,
            "atomic_summary_publication": True,
            "completion_manifest_published_last": True,
            "incomplete_output_cleanup_policy": (
                "preserve partial root for forensics; absence of this manifest "
                "means incomplete"
            ),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "frozen_files_sha256": verified_files,
            "child_completion_manifests_sha256": child_manifests,
            "wall_seconds_before_completion_publication": (
                time.perf_counter() - started
            ),
            "artifacts": {"summary.json": sha256_file(summary_path)},
            "fit_performed": False,
            "selection_performed": False,
            "measured_output_opened": False,
            "test_split_accessed": False,
            "physical_pa_result": False,
            "rf_harmonic_claim": False,
            "surrogate_only": True,
            "precision_selected": False,
            "validation_reused_after_historical_model_selection": True,
        }
        _write_json_atomic(output / "completion_manifest.json", manifest)
        _assert_owned_output_root(output, output_identity)
        completed = True
        return summary | {
            "artifacts": {
                "summary": str(summary_path),
                "completion_manifest": str(output / "completion_manifest.json"),
            }
        }
    finally:
        # Never recursively delete by pathname after a failure.  A concurrent
        # rename/replacement could otherwise redirect cleanup to an unrelated
        # directory.  The missing completion manifest is the failure marker.
        if not completed:
            pass


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed validation-only DPD surrogate demo."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = run(args.config, args.output_root)
    for dataset, report in result["datasets"].items():
        metrics = report["float"]["metrics"]
        print(
            f"{dataset}: NMSE {metrics['no_dpd_nmse_db']:.3f} -> "
            f"{metrics['dpd_nmse_db']:.3f} dB; adjacent relative "
            f"L/R +{metrics['left_relative_improvement_db']:.3f}/"
            f"+{metrics['right_relative_improvement_db']:.3f} dB"
        )
    print("PASS: validation-only surrogate demo; no physical-PA claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
