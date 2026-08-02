"""Sealed train/validation-only reproduction runner for OpenDPD PA models.

The upstream ``Project.build_dataloaders`` loads all three split pairs and the
upstream training loop can evaluate the test loader after every epoch.  That is
not acceptable for the two-loop benchmark in this repository.  This module
keeps the upstream model class, optimizer recipe, frame/segment datasets and
validation NMSE convention, but owns the data access and training loop:

* only ``spec.json``, ``train_input.csv``, ``train_output.csv``,
  ``val_input.csv`` and ``val_output.csv`` are permitted;
* no path containing ``test_input.csv`` or ``test_output.csv`` is resolved;
* checkpoint selection uses validation NMSE only;
* deterministic model/optimizer/scheduler/RNG state is journaled after every
  completed epoch and may be resumed only under the exact frozen contract;
* the resulting artifact is explicitly marked as train/validation-only.

The runner is intentionally usable as a module without PyTorch installed.
PyTorch and the vendored OpenDPD package are imported lazily by the execution
functions, so the NumPy-only unit-test environment can still verify the sealed
contract.

This is a PA behavioral-model runner.  It has no DPD ``1000 real MUL`` gate;
that timing gate applies only to the later deployed DPD datapath.
"""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPENDPD_ROOT = PROJECT_ROOT / "vendor" / "OpenDPD"
ALLOWED_DATASET_FILES: tuple[str, ...] = (
    "spec.json",
    "train_input.csv",
    "train_output.csv",
    "val_input.csv",
    "val_output.csv",
)
FORBIDDEN_DATASET_FILES: tuple[str, ...] = (
    "test_input.csv",
    "test_output.csv",
)
SUPPORTED_BACKBONES = {"gru", "tres_gru", "tres_deltagru"}
SCHEMA_VERSION = 1
TASK = "opendpd_pa_train_validation_only"
CONFIG_STATUS = "preregistered_train_validation_only"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RESUME_SCHEMA_VERSION = 1
RESUME_MANIFEST = "run_manifest.json"
RESUME_DIRECTORY = "resume"
RESUME_STATES_DIRECTORY = "states"
RESUME_JOURNAL_DIRECTORY = "journal"
STATE_FILE_PATTERN = re.compile(
    r"^state_epoch_(?P<epoch>[0-9]{6})_(?P<digest>[0-9a-f]{16})\.pt$"
)
JOURNAL_FILE_PATTERN = re.compile(r"^epoch_(?P<epoch>[0-9]{6})\.json$")
LOCKED_REQUIREMENT_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$"
)


def _display_path(path: str | Path) -> str:
    """Prefer a repository-relative path, but support external temp configs."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON used for config/source provenance."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_forbidden_dataset_path(value: Any) -> bool:
    """Detect test split filenames in a config before any waveform access."""

    if isinstance(value, Mapping):
        return any(
            _contains_forbidden_dataset_path(key)
            or _contains_forbidden_dataset_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_dataset_path(item) for item in value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        return any(token in normalized for token in FORBIDDEN_DATASET_FILES)
    return False


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _validate_git_commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or GIT_COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be one full lowercase Git commit ID")
    return value


def _validate_environment_lock_config(value: Any) -> None:
    """Validate optional environment-lock metadata without opening the file."""

    if not isinstance(value, dict):
        raise ValueError("environment_lock must be an object")
    expected_keys = {"path", "sha256", "verify_installed_versions"}
    if set(value) != expected_keys:
        raise ValueError(
            "environment_lock must contain exactly path, sha256 and "
            "verify_installed_versions"
        )
    path_value = value["path"]
    if (
        not isinstance(path_value, str)
        or not path_value
        or "\\" in path_value
    ):
        raise ValueError(
            "environment_lock.path must be a non-empty POSIX repository path"
        )
    path = Path(path_value)
    if path.is_absolute() or path in {Path("."), Path("..")} or ".." in path.parts:
        raise ValueError(
            "environment_lock.path must be repository-relative and cannot escape"
        )
    _validate_sha256(value["sha256"], label="environment_lock.sha256")
    if value["verify_installed_versions"] is not True:
        raise ValueError(
            "environment_lock.verify_installed_versions must be true"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a sealed runner configuration.

    Validation happens before resolving a dataset split.  In particular, a
    configuration cannot smuggle a test filename into an otherwise valid
    dataset file list.
    """

    config_path = Path(path).resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("OpenDPD runner config must contain one JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("OpenDPD runner config schema_version mismatch")
    if value.get("task") != TASK:
        raise ValueError("unexpected OpenDPD runner task")
    if value.get("status") != CONFIG_STATUS:
        raise ValueError(
            "config must be preregistered before train/validation execution"
        )
    scope = value.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("config.scope must be an object")
    if scope.get("test_split_access_permitted") is not False:
        raise ValueError("config must prohibit test split access")
    if _contains_forbidden_dataset_path(value):
        raise ValueError("config contains a forbidden test split filename")

    dataset_files = value.get("dataset_files")
    if tuple(dataset_files or ()) != ALLOWED_DATASET_FILES:
        raise ValueError(
            "dataset_files must exactly whitelist spec/train/val files"
        )
    dataset_hashes = value.get("dataset_files_sha256")
    if not isinstance(dataset_hashes, dict) or set(dataset_hashes) != set(
        ALLOWED_DATASET_FILES
    ):
        raise ValueError(
            "dataset_files_sha256 must bind every and only allowed dataset file"
        )
    for name, digest in dataset_hashes.items():
        _validate_sha256(digest, label=f"dataset_files_sha256[{name!r}]")
    dataset_dir = value.get("dataset_dir")
    if not isinstance(dataset_dir, str) or not dataset_dir:
        raise ValueError("config.dataset_dir must be a non-empty string")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("config.candidates must be a non-empty list")
    candidate_names = [
        candidate.get("name")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        _validate_candidate(candidate)
    framing = value.get("framing")
    if not isinstance(framing, dict):
        raise ValueError("config.framing must be an object")
    if framing.get("train_mode") not in {
        "upstream_flat_windows",
        "upstream_segment_windows",
    }:
        raise ValueError("unsupported framing.train_mode")
    if framing.get("validation_mode") != "upstream_segments":
        raise ValueError("validation_mode must be upstream_segments")
    for key in ("frame_length", "frame_stride", "nperseg"):
        number = framing.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError(f"framing.{key} must be a positive integer")
    segment_lengths = framing.get("train_segment_lengths")
    if segment_lengths is not None:
        if (
            not isinstance(segment_lengths, list)
            or not segment_lengths
            or any(
                not isinstance(length, int)
                or isinstance(length, bool)
                or length < 1
                for length in segment_lengths
            )
        ):
            raise ValueError(
                "framing.train_segment_lengths must be positive integers"
            )
    training = value.get("training")
    if not isinstance(training, dict):
        raise ValueError("config.training must be an object")
    for key in ("n_epochs", "batch_size", "batch_size_eval"):
        number = training.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if not isinstance(training.get("seed"), int) or isinstance(
        training.get("seed"), bool
    ):
        raise ValueError("training.seed must be an integer")
    if not isinstance(training.get("deterministic"), bool):
        raise ValueError("training.deterministic must be boolean")
    if training["deterministic"] is not True:
        raise ValueError(
            "sealed resumable OpenDPD training requires deterministic=true"
        )
    if training.get("device") not in {"cpu", "cuda"}:
        raise ValueError("training.device must be cpu or cuda")
    for key in ("lr", "lr_end", "decay_factor", "patience", "grad_clip_val"):
        if key not in training:
            raise ValueError(f"config.training is missing {key}")
    for key in ("lr", "lr_end", "decay_factor", "grad_clip_val"):
        number = training[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not np.isfinite(number)
            or number < 0
        ):
            raise ValueError(f"training.{key} must be finite and non-negative")
    if training["lr"] <= 0 or training["lr_end"] <= 0:
        raise ValueError("training learning rates must be positive")
    if not 0 < training["decay_factor"] <= 1:
        raise ValueError("training.decay_factor must be in (0, 1]")
    if not isinstance(training["patience"], int) or isinstance(
        training["patience"], bool
    ) or training["patience"] < 0:
        raise ValueError("training.patience must be a non-negative integer")
    if "weight_decay" in training:
        number = training["weight_decay"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not np.isfinite(number)
            or number < 0
        ):
            raise ValueError("training.weight_decay must be finite and non-negative")
    if "torch_num_threads" in training:
        number = training["torch_num_threads"]
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError("training.torch_num_threads must be positive integer")
    if training.get("optimizer") != "adamw" or training.get("loss") != "mse":
        raise ValueError("runner currently reproduces only AdamW + MSE")
    if value.get("selection_metric") != "validation_opendpd_nmse_db":
        raise ValueError("checkpoint selection metric must be validation NMSE")
    if "environment_lock" in value:
        _validate_environment_lock_config(value["environment_lock"])
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("config.source must be an object")
    _validate_git_commit(
        source.get("opendpd_commit"),
        label="source.opendpd_commit",
    )
    source_hashes = source.get("files_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("source.files_sha256 must be an object")
    required_sources = _required_source_names(value)
    if set(source_hashes) != required_sources:
        raise ValueError(
            "source.files_sha256 must exactly bind the runner, OpenDPD core "
            f"and selected backbones; required={sorted(required_sources)}"
        )
    for name, digest in source_hashes.items():
        _validate_source_name(name)
        _validate_sha256(digest, label=f"source.files_sha256[{name!r}]")
    return value


def _normalize_distribution_name(name: str) -> str:
    """Return the PEP 503 comparison form used for locked distributions."""

    return re.sub(r"[-_.]+", "-", name).lower()


def _read_regular_file_beneath_project(relative_path: Path) -> bytes:
    """Read one regular repository file through no-follow directory handles."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError(
            "sealed environment-lock verification requires O_NOFOLLOW "
            "and O_DIRECTORY"
        )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        directory_descriptors.append(os.open(PROJECT_ROOT, directory_flags))
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
            raise ValueError("environment lock must be one regular file")
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
                "each environment_lock.path component must be a real "
                "non-symlink directory or file"
            ) from error
        raise
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def verify_environment_lock(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Verify the optional requirements lock and every listed package version.

    Extra installed distributions are allowed because the editable ``opendpd``
    package is imported from the separately hash-bound vendored source tree.
    Every distribution named by the lock, however, must be present at exactly
    the recorded version before source or waveform data are accessed.
    """

    value = config.get("environment_lock")
    if value is None:
        return None
    _validate_environment_lock_config(value)
    relative_path = Path(str(value["path"]))
    payload = _read_regular_file_beneath_project(relative_path)
    actual_hash = hashlib.sha256(payload).hexdigest()
    expected_hash = str(value["sha256"])
    if actual_hash != expected_hash:
        raise RuntimeError(
            "environment lock SHA-256 mismatch: "
            f"expected {expected_hash}, found {actual_hash}"
        )
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("environment lock must be UTF-8 text") from error

    requirements: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = LOCKED_REQUIREMENT_PATTERN.fullmatch(stripped)
        if match is None:
            raise ValueError(
                "environment lock supports only exact name==version entries; "
                f"invalid line {line_number}"
            )
        declared_name = match.group("name")
        normalized_name = _normalize_distribution_name(declared_name)
        if normalized_name in requirements:
            raise ValueError(
                "environment lock contains duplicate distribution "
                f"{normalized_name!r}"
            )
        requirements[normalized_name] = (
            declared_name,
            match.group("version"),
        )
    if not requirements:
        raise ValueError("environment lock contains no distributions")

    inventory: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for normalized_name in sorted(requirements):
        declared_name, expected_version = requirements[normalized_name]
        try:
            actual_version = importlib_metadata.version(declared_name)
        except importlib_metadata.PackageNotFoundError:
            mismatches.append(
                f"{declared_name}: expected {expected_version}, not installed"
            )
            continue
        inventory[normalized_name] = {
            "declared_name": declared_name,
            "expected_version": expected_version,
            "installed_version": actual_version,
        }
        if actual_version != expected_version:
            mismatches.append(
                f"{declared_name}: expected {expected_version}, "
                f"found {actual_version}"
            )
    if mismatches:
        raise RuntimeError(
            "installed environment does not match lock: " + "; ".join(mismatches)
        )

    return {
        "path": relative_path.as_posix(),
        "sha256": actual_hash,
        "verify_installed_versions": True,
        "locked_distribution_count": len(requirements),
        "installed_locked_distributions": inventory,
        "installed_locked_inventory_sha256": sha256_json(inventory),
        "extra_installed_distributions_permitted": True,
        "verified_before_source_and_waveform_access": True,
    }


def _validate_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("each candidate must be an object")
    name = candidate.get("name")
    backbone = candidate.get("backbone")
    hidden_size = candidate.get("hidden_size")
    if not isinstance(name, str) or not name:
        raise ValueError("candidate.name must be a non-empty string")
    if (
        name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise ValueError("candidate.name must be one safe directory component")
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"unsupported PA backbone {backbone!r}; "
            f"choose from {sorted(SUPPORTED_BACKBONES)}"
        )
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
        raise ValueError("candidate.hidden_size must be an integer")
    if hidden_size < 1:
        raise ValueError("candidate.hidden_size must be positive")
    if backbone == "tres_deltagru":
        for key in ("thx", "thh"):
            if key not in candidate:
                raise ValueError(f"DeltaGRU candidate is missing {key}")
    for key in ("thx", "thh"):
        if key in candidate:
            value = candidate[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"candidate.{key} must be finite and non-negative")


def _required_source_names(config: Mapping[str, Any]) -> set[str]:
    names = {
        "experiments/train_opendpd_pa.py",
        "vendor/OpenDPD/models.py",
        "vendor/OpenDPD/modules/data_collector.py",
        "vendor/OpenDPD/backbones/__init__.py",
        "vendor/OpenDPD/backbones/rvtdcnn.py",
    }
    for candidate in config.get("candidates", ()):
        if isinstance(candidate, Mapping) and isinstance(
            candidate.get("backbone"), str
        ):
            names.add(f"vendor/OpenDPD/backbones/{candidate['backbone']}.py")
    if any(
        isinstance(candidate, Mapping)
        and candidate.get("backbone") == "tres_deltagru"
        for candidate in config.get("candidates", ())
    ):
        names.update(
            {
                "vendor/OpenDPD/backbones/triton_deltagru.py",
                "vendor/OpenDPD/quant/__init__.py",
                "vendor/OpenDPD/quant/modules/__init__.py",
                "vendor/OpenDPD/quant/modules/ops.py",
            }
        )
    return names


def _validate_source_name(name: Any) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("source file name must be a non-empty string")
    path = (PROJECT_ROOT / name).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"source path escapes project root: {name}") from error
    if path.name in FORBIDDEN_DATASET_FILES:
        raise ValueError(f"source manifest contains forbidden dataset path: {name}")
    return path


def verify_source_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the vendored commit and exact source payload before waveforms."""

    source = config["source"]
    expected_commit = str(source["opendpd_commit"])
    actual_commit = subprocess.check_output(
        ["git", "-C", str(OPENDPD_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            "vendored OpenDPD commit mismatch: "
            f"expected {expected_commit}, found {actual_commit}"
        )
    dirty_status = subprocess.check_output(
        ["git", "-C", str(OPENDPD_ROOT), "status", "--porcelain"],
        text=True,
    )
    if dirty_status.strip():
        raise RuntimeError(
            "vendored OpenDPD worktree is dirty; refusing an unbound source"
        )
    actual_hashes: dict[str, str] = {}
    for name, expected in source["files_sha256"].items():
        path = _validate_source_name(name)
        if not path.is_file():
            raise FileNotFoundError(f"bound source file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"source SHA-256 mismatch for {name}: "
                f"expected {expected}, found {actual}"
            )
        actual_hashes[name] = actual
    return {
        "vendored_commit": actual_commit,
        "vendored_worktree_clean": True,
        "files": actual_hashes,
        "verified_before_waveform_access": True,
    }


def resolve_dataset_dir(config: Mapping[str, Any], config_path: str | Path) -> Path:
    """Resolve the declared dataset directory without inspecting other files."""

    source = Path(config_path).resolve().parent
    dataset_dir = Path(str(config["dataset_dir"]))
    if not dataset_dir.is_absolute():
        dataset_dir = source / dataset_dir
    if dataset_dir.is_symlink():
        raise ValueError("dataset_dir must not be a symlink")
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    return dataset_dir


def verify_allowed_inputs(
    config: Mapping[str, Any],
    config_path: str | Path,
) -> tuple[Path, dict[str, str]]:
    """Verify only whitelisted source files before waveform loading.

    The returned hash map deliberately contains no test-file key.  This is
    checked by unit tests and recorded in every output report.
    """

    if _contains_forbidden_dataset_path(config):
        raise ValueError("forbidden test split filename appears in config")
    dataset_dir = resolve_dataset_dir(config, config_path)
    hashes: dict[str, str] = {}
    for name in ALLOWED_DATASET_FILES:
        path = dataset_dir / name
        if path.is_symlink():
            raise ValueError(f"dataset file must not be a symlink: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"required train/validation file missing: {path}")
        if path.resolve().parent != dataset_dir:
            raise ValueError(f"dataset file resolves outside dataset directory: {name}")
        actual = sha256_file(path)
        expected = str(config["dataset_files_sha256"][name])
        if actual != expected:
            raise RuntimeError(
                f"dataset SHA-256 mismatch for {name}: "
                f"expected {expected}, found {actual}"
            )
        hashes[name] = actual
    return dataset_dir, hashes


def load_allowed_split(
    dataset_dir: str | Path,
    split: str,
    *,
    expected_hashes: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly one allowed split; reject ``test`` before path creation."""

    if split not in {"train", "val"}:
        raise RuntimeError(
            "forbidden split requested; this runner can load only train or val"
        )
    root = Path(dataset_dir)
    input_path = root / f"{split}_input.csv"
    output_path = root / f"{split}_output.csv"
    # Read each file into one immutable byte snapshot and parse that snapshot.
    # This closes the hash/read TOCTOU window: a replacement between the
    # preflight hash and parsing is rejected rather than silently consumed.
    def load_bound(path: Path) -> np.ndarray:
        payload = path.read_bytes()
        if expected_hashes is not None:
            expected = expected_hashes[path.name]
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"dataset changed between verification and load: {path.name}"
                )
        header = payload.splitlines()[0].decode("utf-8").strip()
        if tuple(part.strip().lower() for part in header.split(",")) != ("i", "q"):
            raise ValueError(f"{path} must have exactly the header I,Q")
        values = np.loadtxt(
            io.BytesIO(payload),
            delimiter=",",
            skiprows=1,
            dtype=np.float64,
        )
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError(f"{path} must contain at least one two-column IQ row")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains NaN or infinite values")
        return values[:, 0] + 1j * values[:, 1]

    features = load_bound(input_path)
    targets = load_bound(output_path)
    if features.shape != targets.shape:
        raise ValueError(f"{split} input/output lengths differ")
    return features, targets


def _import_opendpd():
    """Import vendored OpenDPD modules without requiring them at test import."""

    import importlib

    vendor = str(OPENDPD_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    models = importlib.import_module("models")
    data_collector = importlib.import_module("modules.data_collector")
    for module_name, module in (
        ("models", models),
        ("modules.data_collector", data_collector),
    ):
        module_path = getattr(module, "__file__", None)
        if module_path is None or not Path(module_path).resolve().is_relative_to(
            OPENDPD_ROOT
        ):
            raise RuntimeError(
                f"{module_name} was imported outside vendored OpenDPD: "
                f"{module_path!r}; run in a clean Python process"
            )
    return models, data_collector


def _iq_pairs(signal: np.ndarray) -> np.ndarray:
    return np.column_stack((signal.real, signal.imag)).astype(np.float32)


def build_dataloaders(
    train_input: np.ndarray,
    train_output: np.ndarray,
    val_input: np.ndarray,
    val_output: np.ndarray,
    *,
    framing: Mapping[str, Any],
    batch_size: int,
    batch_size_eval: int,
    seed: int,
    train_segment_lengths: Sequence[int] | None = None,
):
    """Build upstream-compatible train-frame and validation-segment loaders."""

    # Importing the classes is safe: unlike ``load_dataset``, constructors
    # below receive already loaded arrays and never inspect a dataset folder.
    _, data_collector = _import_opendpd()
    train_x = _iq_pairs(train_input)
    train_y = _iq_pairs(train_output)
    val_x = _iq_pairs(val_input)
    val_y = _iq_pairs(val_output)

    train_mode = framing["train_mode"]
    if train_mode == "upstream_flat_windows":
        train_set = data_collector.IQFrameDataset(
            train_x,
            train_y,
            frame_length=int(framing["frame_length"]),
            stride=int(framing["frame_stride"]),
        )
    elif train_mode == "upstream_segment_windows":
        # This is an explicit alternative protocol.  Segmenting is performed
        # here, before framing, rather than silently relying on OpenDPD's
        # currently unused ``--use_segments`` flag.
        segmenter = data_collector.IQSegmentDataset
        segmented = segmenter(
            train_x,
            train_y,
            nperseg=int(framing["nperseg"]),
        )
        import torch
        from torch.utils.data import TensorDataset

        frame_features = []
        frame_targets = []
        for segment_features, segment_targets in zip(
            segmented.features.numpy(),
            segmented.targets.numpy(),
        ):
            segment_set = data_collector.IQFrameDataset(
                segment_features,
                segment_targets,
                frame_length=int(framing["frame_length"]),
                stride=int(framing["frame_stride"]),
            )
            frame_features.append(segment_set.features)
            frame_targets.append(segment_set.targets)
        if not frame_features:
            raise ValueError("segmented train input produced no segments")
        # TensorDataset preserves the same sample ordering as concatenating
        # each segment's local upstream frames and never crosses a boundary.
        train_set = TensorDataset(
            torch.cat(frame_features, dim=0),
            torch.cat(frame_targets, dim=0),
        )
    else:  # pragma: no cover - config validation catches this
        raise ValueError(f"unsupported train framing mode: {train_mode}")

    val_set = data_collector.IQSegmentDataset(
        val_x,
        val_y,
        nperseg=int(framing["nperseg"]),
    )
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        pin_memory=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(batch_size_eval),
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )
    return train_loader, val_loader, {
        "train_examples": int(len(train_set)),
        "validation_segments": int(len(val_set)),
        "train_mode": train_mode,
        "validation_mode": str(framing["validation_mode"]),
        "train_boundary_policy": (
            "flat_concatenated_upstream"
            if train_mode == "upstream_flat_windows"
            else "reset_per_segment"
        ),
        "train_cross_boundary_window_count": (
            _count_cross_boundary_windows(
                train_segment_lengths,
                total_samples=int(train_x.shape[0]),
                frame_length=int(framing["frame_length"]),
                stride=int(framing["frame_stride"]),
            )
            if train_mode == "upstream_flat_windows"
            and train_segment_lengths is not None
            else 0 if train_mode == "upstream_segment_windows" else None
        ),
    }


def _count_cross_boundary_windows(
    segment_lengths: Sequence[int],
    *,
    total_samples: int,
    frame_length: int,
    stride: int,
) -> int:
    """Count flat upstream windows crossing declared segment boundaries."""

    if sum(int(length) for length in segment_lengths) != total_samples:
        raise ValueError(
            "train_segment_lengths must sum to the loaded train sequence length"
        )
    boundaries = np.cumsum(np.asarray(segment_lengths, dtype=np.int64))[:-1]
    starts = range(0, total_samples - frame_length + 1, stride)
    return int(
        sum(
            1
            for start in starts
            if np.any((boundaries > start) & (boundaries < start + frame_length))
        )
    )


def opendpd_nmse_db(prediction: np.ndarray, target: np.ndarray) -> float:
    """Reproduce ``vendor/OpenDPD/utils/metrics.py::NMSE`` exactly."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("expected [segments, samples, 2] IQ arrays")
    error_power = np.mean(
        np.square(target[..., 0] - prediction[..., 0])
        + np.square(target[..., 1] - prediction[..., 1]),
        axis=-1,
    )
    target_power = np.mean(
        np.square(target[..., 0]) + np.square(target[..., 1]),
        axis=-1,
    )
    if np.any(error_power <= 0.0) or np.any(target_power <= 0.0):
        raise ValueError("NMSE requires positive per-segment powers")
    return float(np.mean(10.0 * np.log10(error_power / target_power)))


def pooled_nmse_db(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return pooled complex NMSE for the same validation predictions."""

    error = prediction - target
    numerator = float(np.sum(np.square(error)))
    denominator = float(np.sum(np.square(target)))
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("pooled NMSE requires positive powers")
    return float(10.0 * np.log10(numerator / denominator))


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(bool(deterministic))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _candidate_model(candidate: Mapping[str, Any]):
    models, _ = _import_opendpd()
    return models.CoreModel(
        input_size=2,
        hidden_size=int(candidate["hidden_size"]),
        num_layers=1,
        backbone_type=str(candidate["backbone"]),
        window_size=4,
        num_dvr_units=3,
        thx=float(candidate.get("thx", 0.0)),
        thh=float(candidate.get("thh", 0.0)),
    )


def _limited_batches(loader: Iterable[Any], maximum: int | None):
    if maximum is None:
        yield from loader
        return
    for index, batch in enumerate(loader):
        if index >= maximum:
            break
        yield batch


def _train_epoch(
    model: Any,
    loader: Iterable[Any],
    *,
    optimizer: Any,
    criterion: Any,
    device: Any,
    grad_clip_val: float,
    max_batches: int | None,
) -> float:
    import torch

    model.train()
    losses: list[float] = []
    for features, targets in _limited_batches(loader, max_batches):
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = criterion(prediction, targets)
        loss.backward()
        if grad_clip_val:
            torch.nn.utils.clip_grad_norm_(
                tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
                float(grad_clip_val),
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _evaluate(
    model: Any,
    loader: Iterable[Any],
    *,
    criterion: Any,
    device: Any,
    max_batches: int | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    with torch.inference_mode():
        for features, targets in _limited_batches(loader, max_batches):
            features = features.to(device)
            targets = targets.to(device)
            prediction = model(features)
            losses.append(float(criterion(prediction, targets).cpu()))
            predictions.append(prediction.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    if not predictions:
        raise RuntimeError("validation loader produced no batches")
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets_list, axis=0)
    return (
        float(np.mean(losses)),
        prediction_array,
        target_array,
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _save_checkpoint_atomic(state_dict: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        torch.save(state_dict, temporary_name)
        with Path(temporary_name).open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> str:
    """Publish one immutable JSON object without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace immutable artifact: {path}"
            ) from error
        _fsync_directory(path.parent)
        return sha256_file(path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be one regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value


def _resume_paths(output: Path) -> tuple[Path, Path, Path]:
    resume_root = output / RESUME_DIRECTORY
    return (
        resume_root,
        resume_root / RESUME_STATES_DIRECTORY,
        resume_root / RESUME_JOURNAL_DIRECTORY,
    )


def _ensure_resume_directories(output: Path) -> tuple[Path, Path, Path]:
    paths = _resume_paths(output)
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"resume directory must not be a symlink: {path}")
        existed = path.exists()
        path.mkdir(exist_ok=True)
        if not path.is_dir():
            raise RuntimeError(f"resume path is not a directory: {path}")
        if not existed:
            _fsync_directory(path.parent)
    return paths


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        return None
    return platform.processor() or None


def _runtime_signature(torch: Any, device: Any) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "python": sys.version,
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "device": str(device),
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "float32_matmul_precision": str(
            torch.get_float32_matmul_precision()
        ),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_LAUNCH_BLOCKING",
                "NVIDIA_TF32_OVERRIDE",
                "OPENDPD_DISABLE_TRITON_DELTAGRU",
            )
        },
    }
    if device.type == "cuda":
        index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        signature["cuda"] = {
            "torch_cuda_version": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": int(torch.cuda.device_count()),
            "selected_device_index": index,
            "selected_device_name": str(torch.cuda.get_device_name(index)),
            "selected_device_capability": list(
                torch.cuda.get_device_capability(index)
            ),
        }
        try:
            driver_lines = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=5,
            ).splitlines()
            signature["cuda"]["driver_version"] = driver_lines[index].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            signature["cuda"]["driver_version"] = None
    else:
        signature["cuda"] = None
    return signature


def _build_resume_contract(
    config: Mapping[str, Any],
    config_path: str | Path,
    candidate: Mapping[str, Any],
    *,
    output: Path,
    environment: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    dataset_dir: Path,
    dataset_hashes: Mapping[str, str],
    requested_epochs: int,
    effective_epochs: int,
    max_epochs: int | None,
    max_train_batches: int | None,
    max_val_batches: int | None,
    runtime_signature: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "task": TASK,
        "config": {
            "path": _display_path(config_path),
            "sha256": sha256_file(config_path),
            "canonical_sha256": sha256_json(config),
        },
        "candidate": dict(candidate),
        "candidate_sha256": sha256_json(dict(candidate)),
        "output_directory": _display_path(output),
        "dataset": {
            "directory": _display_path(dataset_dir),
            "files_sha256": dict(dataset_hashes),
            "test_file_hashes_recorded": False,
        },
        "environment_lock": (
            None if environment is None else dict(environment)
        ),
        "source": dict(source),
        "recipe": {
            "requested_epochs": int(requested_epochs),
            "effective_epochs": int(effective_epochs),
            "max_epochs_argument": max_epochs,
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_val_batches,
            "training": dict(config["training"]),
            "framing": dict(config["framing"]),
        },
        "runtime_signature": dict(runtime_signature),
        "scope": {
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
            "selection_split": "validation",
        },
    }


def _initialize_or_verify_run_manifest(
    output: Path,
    contract: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    contract_hash = sha256_json(contract)
    manifest = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_resumable_run_manifest",
        "task": TASK,
        "status": "in_progress_until_completion_manifest",
        "resume_contract": dict(contract),
        "resume_contract_sha256": contract_hash,
        "test_split_accessed": False,
        "test_path_resolved": False,
        "test_file_hashes_recorded": False,
    }
    path = output / RESUME_MANIFEST
    if resume:
        observed = _read_json_object(path, label="resume run manifest")
        if observed != manifest:
            raise RuntimeError(
                "resume contract mismatch; config/candidate/source/data/runtime "
                "or execution limits changed"
            )
    else:
        _write_json_exclusive_atomic(path, manifest)
    _ensure_resume_directories(output)
    return manifest


def _capture_rng_state(
    torch: Any,
    train_loader: Any,
    *,
    include_cuda: bool,
) -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    generator = getattr(train_loader, "generator", None)
    if generator is None:
        raise RuntimeError("train DataLoader has no resumable shuffle generator")
    return {
        "python": {
            "version": int(python_state[0]),
            "state": [int(value) for value in python_state[1]],
            "gauss_next": (
                None
                if python_state[2] is None
                else float(python_state[2])
            ),
        },
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": [int(value) for value in numpy_state[1]],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            list(torch.cuda.get_rng_state_all())
            if include_cuda
            else []
        ),
        "train_loader_generator": generator.get_state(),
    }


def _restore_rng_state(
    torch: Any,
    train_loader: Any,
    state: Mapping[str, Any],
) -> None:
    python_state = state["python"]
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["state"]),
            python_state["gauss_next"],
        )
    )
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("resume state requires CUDA RNG on a CPU runtime")
        torch.cuda.set_rng_state_all(cuda_states)
    generator = getattr(train_loader, "generator", None)
    if generator is None:
        raise RuntimeError("train DataLoader has no resumable shuffle generator")
    generator.set_state(state["train_loader_generator"])


def _make_resume_state(
    torch: Any,
    *,
    contract_hash: str,
    completed_epochs: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    best_metric: float,
    best_epoch: int | None,
    best_state: Mapping[str, Any] | None,
    history: Sequence[Mapping[str, Any]],
    productive_fit_seconds: float,
    train_loader: Any,
) -> dict[str, Any]:
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_epoch_resume_state",
        "task": TASK,
        "resume_contract_sha256": contract_hash,
        "completed_epochs": int(completed_epochs),
        "current_model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
        "best_validation_opendpd_nmse_db": (
            None if not np.isfinite(best_metric) else float(best_metric)
        ),
        "best_epoch": best_epoch,
        "best_model_state_dict": (
            None if best_state is None else copy.deepcopy(best_state)
        ),
        "history": [dict(row) for row in history],
        "productive_fit_seconds": float(productive_fit_seconds),
        "rng_state": _capture_rng_state(
            torch,
            train_loader,
            include_cuda=any(
                bool(parameter.is_cuda)
                for parameter in model.parameters()
            ),
        ),
        "test_split_accessed": False,
        "test_path_resolved": False,
        "test_file_hashes_recorded": False,
    }


def _validate_resume_state(
    state: Any,
    *,
    contract_hash: str,
    completed_epochs: int,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise RuntimeError("resume checkpoint must contain one mapping")
    if (
        state.get("schema_version") != RESUME_SCHEMA_VERSION
        or state.get("artifact_type") != "opendpd_pa_epoch_resume_state"
        or state.get("task") != TASK
    ):
        raise RuntimeError("resume checkpoint schema/task mismatch")
    if state.get("resume_contract_sha256") != contract_hash:
        raise RuntimeError("resume checkpoint contract hash mismatch")
    if state.get("completed_epochs") != completed_epochs:
        raise RuntimeError("resume checkpoint completed-epoch count mismatch")
    if any(
        state.get(key) is not False
        for key in (
            "test_split_accessed",
            "test_path_resolved",
            "test_file_hashes_recorded",
        )
    ):
        raise RuntimeError("resume checkpoint violates the sealed test scope")
    history = state.get("history")
    if not isinstance(history, list) or len(history) != completed_epochs:
        raise RuntimeError("resume checkpoint history length mismatch")
    for index, row in enumerate(history):
        if not isinstance(row, dict) or row.get("epoch") != index:
            raise RuntimeError("resume checkpoint history is not contiguous")
    productive = state.get("productive_fit_seconds")
    if (
        isinstance(productive, bool)
        or not isinstance(productive, (int, float))
        or not np.isfinite(productive)
        or productive < 0
    ):
        raise RuntimeError("resume checkpoint productive time is invalid")
    best_epoch = state.get("best_epoch")
    best_metric = state.get("best_validation_opendpd_nmse_db")
    best_state = state.get("best_model_state_dict")
    if completed_epochs == 0:
        if best_epoch is not None or best_metric is not None or best_state is not None:
            raise RuntimeError("initial resume checkpoint contains a best model")
    elif (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 0 <= best_epoch < completed_epochs
        or isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not np.isfinite(best_metric)
        or not isinstance(best_state, Mapping)
    ):
        raise RuntimeError("resume checkpoint best-model metadata is invalid")
    required_mappings = (
        "current_model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
    )
    if any(not isinstance(state.get(key), Mapping) for key in required_mappings):
        raise RuntimeError("resume checkpoint is missing state mappings")
    return state


def _save_resume_state_content_addressed(
    state: Mapping[str, Any],
    *,
    states_dir: Path,
    completed_epochs: int,
) -> tuple[Path, str]:
    import torch

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".state_epoch_{completed_epochs:06d}.",
        suffix=".tmp",
        dir=str(states_dir),
    )
    os.close(fd)
    try:
        torch.save(dict(state), temporary_name)
        with Path(temporary_name).open("rb") as stream:
            os.fsync(stream.fileno())
        digest = sha256_file(temporary_name)
        path = states_dir / (
            f"state_epoch_{completed_epochs:06d}_{digest[:16]}.pt"
        )
        if path.is_symlink() or path.exists():
            raise FileExistsError(
                f"refusing to replace immutable resume checkpoint: {path}"
            )
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace immutable resume checkpoint: {path}"
            ) from error
        _fsync_directory(states_dir)
        return path, digest
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _journal_record(
    *,
    contract: Mapping[str, Any],
    contract_hash: str,
    completed_epochs: int,
    state_path: Path,
    state_sha256: str,
    state: Mapping[str, Any],
    previous_journal_sha256: str | None,
    session_id: str,
    recovered_orphan: bool,
) -> dict[str, Any]:
    output = Path(str(contract["output_directory"]))
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    relative_state = state_path.resolve().relative_to(output.resolve())
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_append_only_epoch_journal",
        "task": TASK,
        "status": (
            "initial_state"
            if completed_epochs == 0
            else "completed_epoch"
        ),
        "resume_contract_sha256": contract_hash,
        "config_sha256": contract["config"]["sha256"],
        "candidate_sha256": contract["candidate_sha256"],
        "dataset_manifest_sha256": sha256_json(
            contract["dataset"]["files_sha256"]
        ),
        "source_manifest_sha256": sha256_json(contract["source"]),
        "completed_epochs": int(completed_epochs),
        "configured_epochs": int(contract["recipe"]["requested_epochs"]),
        "contracted_epochs": int(contract["recipe"]["effective_epochs"]),
        "history_length": len(state["history"]),
        "last_history_row": (
            None if not state["history"] else dict(state["history"][-1])
        ),
        "best_epoch": state["best_epoch"],
        "best_validation_opendpd_nmse_db": state[
            "best_validation_opendpd_nmse_db"
        ],
        "productive_fit_seconds": float(state["productive_fit_seconds"]),
        "state_path": relative_state.as_posix(),
        "state_sha256": state_sha256,
        "previous_journal_sha256": previous_journal_sha256,
        "session_id": session_id,
        "recovered_after_interrupted_journal_publication": bool(
            recovered_orphan
        ),
        "test_split_accessed": False,
        "test_path_resolved": False,
        "test_file_hashes_recorded": False,
    }


def _publish_journal_record(
    output: Path,
    record: Mapping[str, Any],
) -> tuple[Path, str]:
    _, _, journal_dir = _resume_paths(output)
    completed_epochs = int(record["completed_epochs"])
    path = journal_dir / f"epoch_{completed_epochs:06d}.json"
    digest = _write_json_exclusive_atomic(path, record)
    return path, digest


def _load_resume_layout(
    output: Path,
    *,
    contract: Mapping[str, Any],
    contract_hash: str,
) -> tuple[list[dict[str, Any]], Path | None]:
    _, states_dir, journal_dir = _ensure_resume_directories(output)
    journal_files = sorted(
        path
        for path in journal_dir.iterdir()
        if not path.name.startswith(".")
    )
    records: list[dict[str, Any]] = []
    expected_state_paths: set[Path] = set()
    previous_digest: str | None = None
    for expected_epoch, path in enumerate(journal_files):
        match = JOURNAL_FILE_PATTERN.fullmatch(path.name)
        if match is None or int(match.group("epoch")) != expected_epoch:
            raise RuntimeError("resume journal files are not contiguous")
        record = _read_json_object(path, label="resume journal entry")
        if (
            record.get("schema_version") != RESUME_SCHEMA_VERSION
            or record.get("artifact_type")
            != "opendpd_pa_append_only_epoch_journal"
            or record.get("task") != TASK
            or record.get("resume_contract_sha256") != contract_hash
            or record.get("completed_epochs") != expected_epoch
            or record.get("history_length") != expected_epoch
            or record.get("configured_epochs")
            != int(contract["recipe"]["requested_epochs"])
            or record.get("contracted_epochs")
            != int(contract["recipe"]["effective_epochs"])
            or record.get("previous_journal_sha256") != previous_digest
            or record.get("test_split_accessed") is not False
            or record.get("test_path_resolved") is not False
            or record.get("test_file_hashes_recorded") is not False
        ):
            raise RuntimeError("resume journal entry failed contract validation")
        relative = record.get("state_path")
        if not isinstance(relative, str):
            raise RuntimeError("resume journal state path is invalid")
        relative_path = Path(relative)
        expected_prefix = (
            Path(RESUME_DIRECTORY) / RESUME_STATES_DIRECTORY
        )
        if (
            relative_path.is_absolute()
            or relative_path.parent != expected_prefix
            or ".." in relative_path.parts
        ):
            raise RuntimeError("resume journal state path escapes its directory")
        state_path = output / relative_path
        if state_path.is_symlink() or not state_path.is_file():
            raise RuntimeError("resume journal checkpoint is missing or a symlink")
        state_match = STATE_FILE_PATTERN.fullmatch(state_path.name)
        expected_digest = record.get("state_sha256")
        if (
            state_match is None
            or int(state_match.group("epoch")) != expected_epoch
            or not isinstance(expected_digest, str)
            or SHA256_PATTERN.fullmatch(expected_digest) is None
            or state_match.group("digest") != expected_digest[:16]
            or sha256_file(state_path) != expected_digest
        ):
            raise RuntimeError("resume journal checkpoint hash/name mismatch")
        if record.get("config_sha256") != contract["config"]["sha256"]:
            raise RuntimeError("resume journal config hash mismatch")
        if record.get("candidate_sha256") != contract["candidate_sha256"]:
            raise RuntimeError("resume journal candidate hash mismatch")
        if record.get("dataset_manifest_sha256") != sha256_json(
            contract["dataset"]["files_sha256"]
        ):
            raise RuntimeError("resume journal dataset hash mismatch")
        if record.get("source_manifest_sha256") != sha256_json(
            contract["source"]
        ):
            raise RuntimeError("resume journal source hash mismatch")
        previous_digest = sha256_file(path)
        expected_state_paths.add(state_path)
        records.append(record)

    state_files = sorted(
        path
        for path in states_dir.iterdir()
        if not path.name.startswith(".")
    )
    extras = [path for path in state_files if path not in expected_state_paths]
    if len(extras) > 1:
        raise RuntimeError("multiple unjournaled resume checkpoints found")
    orphan: Path | None = None
    if extras:
        orphan = extras[0]
        if orphan.is_symlink() or not orphan.is_file():
            raise RuntimeError("unjournaled resume checkpoint is not a regular file")
        match = STATE_FILE_PATTERN.fullmatch(orphan.name)
        expected_epoch = len(records)
        if (
            match is None
            or int(match.group("epoch")) != expected_epoch
            or sha256_file(orphan)[:16] != match.group("digest")
        ):
            raise RuntimeError("unjournaled resume checkpoint is not the next epoch")
    return records, orphan


def _load_torch_resume_state(
    path: Path,
    *,
    device: Any,
    contract_hash: str,
    completed_epochs: int,
) -> dict[str, Any]:
    import torch

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("resume checkpoint must be a regular non-symlink file")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception as error:
        raise RuntimeError(f"cannot safely load resume checkpoint: {path}") from error
    return _validate_resume_state(
        state,
        contract_hash=contract_hash,
        completed_epochs=completed_epochs,
    )


def _restore_training_state(
    torch: Any,
    state: Mapping[str, Any],
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    train_loader: Any,
) -> tuple[
    int,
    float,
    int | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    float,
]:
    model.load_state_dict(state["current_model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    _restore_rng_state(torch, train_loader, state["rng_state"])
    completed = int(state["completed_epochs"])
    metric = state["best_validation_opendpd_nmse_db"]
    return (
        completed,
        float("inf") if metric is None else float(metric),
        state["best_epoch"],
        (
            None
            if state["best_model_state_dict"] is None
            else copy.deepcopy(state["best_model_state_dict"])
        ),
        [dict(row) for row in state["history"]],
        float(state["productive_fit_seconds"]),
    )


def _run_candidate_locked(
    config: Mapping[str, Any],
    config_path: str | Path,
    candidate: Mapping[str, Any],
    *,
    output_dir: str | Path,
    resume: bool = False,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Train or exactly resume one sealed train/validation-only candidate."""

    _validate_candidate(candidate)
    if _contains_forbidden_dataset_path(config):
        raise ValueError("forbidden test split filename appears in config")
    for label, value in (
        ("max_epochs", max_epochs),
        ("max_train_batches", max_train_batches),
        ("max_val_batches", max_val_batches),
    ):
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{label} must be a positive integer")

    output_argument = Path(output_dir)
    if output_argument.is_symlink():
        raise ValueError("OpenDPD output directory must not be a symlink")
    output = output_argument.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not resume:
        raise FileExistsError(f"refusing to reuse OpenDPD output directory: {output}")
    if resume:
        if not output.is_dir() or output.is_symlink():
            raise RuntimeError("resume requires one existing regular output directory")
        if not (output / RESUME_MANIFEST).is_file():
            raise RuntimeError(
                "existing OpenDPD directory is not resumable: "
                "run_manifest.json is missing; the legacy empty run cannot "
                "be recovered"
            )
        if (output / "completion_manifest.json").exists():
            raise RuntimeError("completed OpenDPD output cannot be resumed")

    file_config = load_config(config_path)
    if sha256_json(file_config) != sha256_json(dict(config)):
        raise RuntimeError(
            "run_candidate config mapping differs from its bound config file"
        )
    environment = verify_environment_lock(config)
    source = verify_source_inputs(config)
    dataset_dir, dataset_hashes = verify_allowed_inputs(config, config_path)

    import torch

    training = config["training"]
    framing = config["framing"]
    seed = int(training["seed"])
    deterministic = bool(training.get("deterministic", False))
    _seed_everything(seed, deterministic=deterministic)
    torch.set_num_threads(int(training.get("torch_num_threads", 1)))
    device = torch.device(str(training.get("device", "cpu")))
    if device.type != "cpu":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"requested device is unavailable: {device}")
    requested_epochs = int(training["n_epochs"])
    epochs = requested_epochs if max_epochs is None else min(
        requested_epochs, int(max_epochs)
    )
    if epochs < 1:
        raise ValueError("max_epochs must leave at least one epoch")

    contract = _build_resume_contract(
        config,
        config_path,
        candidate,
        output=output,
        environment=environment,
        source=source,
        dataset_dir=dataset_dir,
        dataset_hashes=dataset_hashes,
        requested_epochs=requested_epochs,
        effective_epochs=epochs,
        max_epochs=max_epochs,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        runtime_signature=_runtime_signature(torch, device),
    )
    contract_hash = sha256_json(contract)
    if not resume:
        try:
            output.mkdir()
            _fsync_directory(output.parent)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to reuse OpenDPD output directory: {output}"
            ) from error
    _initialize_or_verify_run_manifest(
        output,
        contract,
        resume=resume,
    )

    train_input, train_output = load_allowed_split(
        dataset_dir,
        "train",
        expected_hashes=dataset_hashes,
    )
    val_input, val_output = load_allowed_split(
        dataset_dir,
        "val",
        expected_hashes=dataset_hashes,
    )

    train_loader, val_loader, loader_info = build_dataloaders(
        train_input,
        train_output,
        val_input,
        val_output,
        framing=framing,
        batch_size=int(training["batch_size"]),
        batch_size_eval=int(training["batch_size_eval"]),
        seed=seed,
        train_segment_lengths=framing.get("train_segment_lengths"),
    )
    model = _candidate_model(candidate).to(device)
    parameter_count = _parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["decay_factor"]),
        patience=int(training["patience"]),
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=float(training["lr_end"]),
        eps=1e-8,
    )
    criterion = torch.nn.MSELoss()

    best_metric = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    productive_fit_seconds = 0.0
    completed_epochs = 0
    session_id = f"{os.getpid()}-{time.time_ns()}"

    records, orphan = _load_resume_layout(
        output,
        contract=contract,
        contract_hash=contract_hash,
    )
    if orphan is not None:
        orphan_state = _load_torch_resume_state(
            orphan,
            device=device,
            contract_hash=contract_hash,
            completed_epochs=len(records),
        )
        previous_digest = (
            None
            if not records
            else sha256_file(
                output
                / RESUME_DIRECTORY
                / RESUME_JOURNAL_DIRECTORY
                / f"epoch_{len(records) - 1:06d}.json"
            )
        )
        orphan_record = _journal_record(
            contract=contract,
            contract_hash=contract_hash,
            completed_epochs=len(records),
            state_path=orphan,
            state_sha256=sha256_file(orphan),
            state=orphan_state,
            previous_journal_sha256=previous_digest,
            session_id=session_id,
            recovered_orphan=True,
        )
        _publish_journal_record(output, orphan_record)
        records.append(orphan_record)

    if not records:
        initial_state = _make_resume_state(
            torch,
            contract_hash=contract_hash,
            completed_epochs=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
            productive_fit_seconds=productive_fit_seconds,
            train_loader=train_loader,
        )
        _, states_dir, _ = _resume_paths(output)
        state_path, state_digest = _save_resume_state_content_addressed(
            initial_state,
            states_dir=states_dir,
            completed_epochs=0,
        )
        initial_record = _journal_record(
            contract=contract,
            contract_hash=contract_hash,
            completed_epochs=0,
            state_path=state_path,
            state_sha256=state_digest,
            state=initial_state,
            previous_journal_sha256=None,
            session_id=session_id,
            recovered_orphan=False,
        )
        _publish_journal_record(output, initial_record)
        records.append(initial_record)
        current_state = initial_state
    else:
        state_path = output / str(records[-1]["state_path"])
        current_state = _load_torch_resume_state(
            state_path,
            device=device,
            contract_hash=contract_hash,
            completed_epochs=int(records[-1]["completed_epochs"]),
        )

    (
        completed_epochs,
        best_metric,
        best_epoch,
        best_state,
        history,
        productive_fit_seconds,
    ) = _restore_training_state(
        torch,
        current_state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
    )
    if completed_epochs > epochs:
        raise RuntimeError("resume state exceeds the contracted epoch count")
    resumed_from_completed_epochs = completed_epochs

    for epoch in range(completed_epochs, epochs):
        epoch_start = time.perf_counter()
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_val=float(training["grad_clip_val"]),
            max_batches=max_train_batches,
        )
        val_loss, val_prediction, val_target = _evaluate(
            model,
            val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )
        val_nmse = opendpd_nmse_db(val_prediction, val_target)
        val_pooled = pooled_nmse_db(val_prediction, val_target)
        if not all(
            np.isfinite(value)
            for value in (train_loss, val_loss, val_nmse, val_pooled)
        ):
            raise RuntimeError("non-finite OpenDPD epoch metric")
        scheduler.step(val_nmse)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_opendpd_nmse_db": val_nmse,
            "validation_pooled_nmse_db": val_pooled,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if val_nmse < best_metric:
            best_metric = val_nmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        productive_fit_seconds += time.perf_counter() - epoch_start
        completed_epochs = epoch + 1
        epoch_state = _make_resume_state(
            torch,
            contract_hash=contract_hash,
            completed_epochs=completed_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
            productive_fit_seconds=productive_fit_seconds,
            train_loader=train_loader,
        )
        _, states_dir, journal_dir = _resume_paths(output)
        state_path, state_digest = _save_resume_state_content_addressed(
            epoch_state,
            states_dir=states_dir,
            completed_epochs=completed_epochs,
        )
        previous_digest = sha256_file(
            journal_dir / f"epoch_{completed_epochs - 1:06d}.json"
        )
        record = _journal_record(
            contract=contract,
            contract_hash=contract_hash,
            completed_epochs=completed_epochs,
            state_path=state_path,
            state_sha256=state_digest,
            state=epoch_state,
            previous_journal_sha256=previous_digest,
            session_id=session_id,
            recovered_orphan=False,
        )
        _publish_journal_record(output, record)
        records.append(record)
        print(
            f"[{candidate['name']}] epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6g} val_nmse={val_nmse:.6f} dB",
            flush=True,
        )
    if best_state is None or best_epoch is None:
        raise RuntimeError("no validation-selected checkpoint was produced")

    config_hash = str(contract["config"]["sha256"])
    if sha256_file(config_path) != config_hash:
        raise RuntimeError(
            "config changed during OpenDPD training; final publication aborted"
        )
    final_environment = verify_environment_lock(config)
    if final_environment != environment:
        raise RuntimeError(
            "environment-lock provenance changed during OpenDPD training; "
            "final publication aborted"
        )
    final_source = verify_source_inputs(config)
    if final_source != source:
        raise RuntimeError(
            "source provenance changed during OpenDPD training; "
            "final publication aborted"
        )
    final_dataset_dir, final_dataset_hashes = verify_allowed_inputs(
        config,
        config_path,
    )
    if (
        final_dataset_dir != dataset_dir
        or final_dataset_hashes != dataset_hashes
    ):
        raise RuntimeError(
            "dataset provenance changed during OpenDPD training; "
            "final publication aborted"
        )

    checkpoint = output / f"{candidate['name']}.pt"
    _save_checkpoint_atomic(best_state, checkpoint)
    source["opendpd_root"] = _display_path(OPENDPD_ROOT)
    bounded_execution = any(
        contract["recipe"][key] is not None
        for key in (
            "max_epochs_argument",
            "max_train_batches",
            "max_validation_batches",
        )
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "status": (
            "runtime_preflight_not_quality"
            if bounded_execution
            else "completed_train_validation_only"
        ),
        "config": {
            "path": _display_path(config_path),
            "sha256": config_hash,
        },
        "scope": {
            "pa_behavioral_model": True,
            "dpd_latency_gate_applicable": False,
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
            "selection_split": "validation",
            "selection_metric": "validation_opendpd_nmse_db",
        },
        "dataset": {
            "directory": _display_path(dataset_dir),
            "files_sha256": dataset_hashes,
            "test_file_hashes_recorded": False,
        },
        "environment_lock": environment,
        "source": source,
        "candidate": dict(candidate),
        "model": {
            "parameter_count": parameter_count,
            "checkpoint": _display_path(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
        "recipe": {
            "device": str(device),
            "seed": seed,
            "deterministic": deterministic,
            "epochs_requested": requested_epochs,
            "epochs_executed": epochs,
            "max_epochs_argument": max_epochs,
            "batch_size": int(training["batch_size"]),
            "batch_size_eval": int(training["batch_size_eval"]),
            "frame_length": int(framing["frame_length"]),
            "frame_stride": int(framing["frame_stride"]),
            "nperseg": int(framing["nperseg"]),
            "train_framing": framing["train_mode"],
            "validation_framing": framing["validation_mode"],
            "optimizer": "AdamW",
            "weight_decay": float(training.get("weight_decay", 0.01)),
            "initial_lr": float(training["lr"]),
            "minimum_lr": float(training["lr_end"]),
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": float(training["decay_factor"]),
            "scheduler_patience": int(training["patience"]),
            "loss": "MSE",
            "grad_clip_val": float(training["grad_clip_val"]),
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_val_batches,
        },
        "loader": loader_info,
        "selection": {
            "best_epoch": best_epoch,
            "best_validation_opendpd_nmse_db": best_metric,
            "test_used_for_selection": False,
        },
        "history": history,
        "resume": {
            "enabled": True,
            "invocation_used_resume_flag": bool(resume),
            "resumed_from_completed_epochs": resumed_from_completed_epochs,
            "initial_state_saved": True,
            "checkpoint_after_every_completed_epoch": True,
            "incomplete_epoch_replayed_from_last_completed_state": True,
            "run_manifest": _display_path(output / RESUME_MANIFEST),
            "run_manifest_sha256": sha256_file(output / RESUME_MANIFEST),
            "resume_contract_sha256": contract_hash,
            "journal_entry_count": len(records),
            "session_count_observed_in_journal": len(
                {str(record["session_id"]) for record in records}
            ),
            "final_state": str(records[-1]["state_path"]),
            "final_state_sha256": str(records[-1]["state_sha256"]),
            "final_journal": _display_path(
                output
                / RESUME_DIRECTORY
                / RESUME_JOURNAL_DIRECTORY
                / f"epoch_{epochs:06d}.json"
            ),
            "final_journal_sha256": sha256_file(
                output
                / RESUME_DIRECTORY
                / RESUME_JOURNAL_DIRECTORY
                / f"epoch_{epochs:06d}.json"
            ),
        },
        "runtime": {
            "fit_seconds": productive_fit_seconds,
            "fit_seconds_definition": (
                "sum of completed train+validation epoch bodies; excludes "
                "checkpoint/journal publication and discarded partial epochs"
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_threads": int(torch.get_num_threads()),
            "cuda_available": bool(torch.cuda.is_available()),
            "resume_runtime_signature": contract["runtime_signature"],
        },
    }
    _write_json_atomic(output / "training_report.json", report)
    completion = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "artifact_type": "opendpd_pa_training_completion_manifest",
        "task": TASK,
        "status": report["status"],
        "quality_result": report["status"] == "completed_train_validation_only",
        "resume_contract_sha256": contract_hash,
        "artifacts": {
            "run_manifest": {
                "path": _display_path(output / RESUME_MANIFEST),
                "sha256": sha256_file(output / RESUME_MANIFEST),
            },
            "final_resume_state": {
                "path": str(records[-1]["state_path"]),
                "sha256": str(records[-1]["state_sha256"]),
            },
            "final_journal": {
                "path": report["resume"]["final_journal"],
                "sha256": report["resume"]["final_journal_sha256"],
            },
            "selected_checkpoint": {
                "path": _display_path(checkpoint),
                "sha256": sha256_file(checkpoint),
            },
            "training_report": {
                "path": _display_path(output / "training_report.json"),
                "sha256": sha256_file(output / "training_report.json"),
            },
        },
        "scope": {
            "test_split_accessed": False,
            "test_path_resolved": False,
            "test_file_hashes_recorded": False,
            "selection_split": "validation",
        },
        "published_last": True,
    }
    _write_json_exclusive_atomic(
        output / "completion_manifest.json",
        completion,
    )
    return report


def _run_lock_path(output: Path) -> Path:
    return output.parent / f".{output.name}.opendpd-run.lock"


def _acquire_run_lock(output: Path) -> tuple[int, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    path = _run_lock_path(output)
    if path.is_symlink():
        raise RuntimeError(f"OpenDPD run lock must not be a symlink: {path}")
    existed = path.exists()
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        os.close(descriptor)
        raise RuntimeError(
            f"another OpenDPD process already owns the run lock: {path}"
        ) from error
    if not existed:
        _fsync_directory(path.parent)
    return descriptor, path


def run_candidate(
    config: Mapping[str, Any],
    config_path: str | Path,
    candidate: Mapping[str, Any],
    *,
    output_dir: str | Path,
    resume: bool = False,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Serialize one candidate run and delegate to the sealed implementation."""

    output = Path(output_dir).resolve()
    descriptor, _ = _acquire_run_lock(output)
    try:
        return _run_candidate_locked(
            config,
            config_path,
            candidate,
            output_dir=output_dir,
            resume=resume,
            max_epochs=max_epochs,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def run_config(
    config_path: str | Path,
    *,
    candidate_names: Sequence[str] | None = None,
    output_root: str | Path | None = None,
    resume: bool = False,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Run selected candidates from one config, one output directory each."""

    config = load_config(config_path)
    selected = set(candidate_names or ())
    candidates = [
        candidate
        for candidate in config["candidates"]
        if not selected or candidate["name"] in selected
    ]
    if selected and len(candidates) != len(selected):
        known = {candidate["name"] for candidate in config["candidates"]}
        raise ValueError(f"unknown candidate names: {sorted(selected - known)}")
    if not candidates:
        raise ValueError("no candidates selected")
    root = (
        Path(output_root).resolve()
        if output_root is not None
        else (Path(config_path).resolve().parent / "results")
    )
    reports = []
    for candidate in candidates:
        reports.append(
            run_candidate(
                config,
                config_path,
                candidate,
                output_dir=root / candidate["name"],
                resume=resume,
                max_epochs=max_epochs,
                max_train_batches=max_train_batches,
                max_val_batches=max_val_batches,
            )
        )
    return reports


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sealed OpenDPD PA train/validation-only runner"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume only from a hash-matched append-only epoch journal; "
            "completed or legacy empty outputs are rejected"
        ),
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    reports = run_config(
        args.config,
        candidate_names=args.candidates,
        output_root=args.output_root,
        resume=args.resume,
        max_epochs=args.max_epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    for report in reports:
        print(
            f"published {report['model']['checkpoint']} "
            f"(best validation NMSE "
            f"{report['selection']['best_validation_opendpd_nmse_db']:.6f} dB)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
