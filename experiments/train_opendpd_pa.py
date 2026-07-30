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
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
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
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


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
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run_candidate(
    config: Mapping[str, Any],
    config_path: str | Path,
    candidate: Mapping[str, Any],
    *,
    output_dir: str | Path,
    max_epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Train one candidate and atomically publish a sealed report/checkpoint."""

    _validate_candidate(candidate)
    if _contains_forbidden_dataset_path(config):
        raise ValueError("forbidden test split filename appears in config")
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to reuse OpenDPD output directory: {output}")
    source = verify_source_inputs(config)
    dataset_dir, dataset_hashes = verify_allowed_inputs(config, config_path)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to reuse OpenDPD output directory: {output}"
        ) from error
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
    requested_epochs = int(training["n_epochs"])
    epochs = requested_epochs if max_epochs is None else min(
        requested_epochs, int(max_epochs)
    )
    if epochs < 1:
        raise ValueError("max_epochs must leave at least one epoch")

    best_metric = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(epochs):
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
        print(
            f"[{candidate['name']}] epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6g} val_nmse={val_nmse:.6f} dB",
            flush=True,
        )
    elapsed = time.perf_counter() - start
    if best_state is None or best_epoch is None:
        raise RuntimeError("no validation-selected checkpoint was produced")

    checkpoint = output / f"{candidate['name']}.pt"
    _save_checkpoint_atomic(best_state, checkpoint)
    source["opendpd_root"] = _display_path(OPENDPD_ROOT)
    config_hash = sha256_file(config_path)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "status": (
            "runtime_preflight_not_quality"
            if (
                max_epochs is not None
                or max_train_batches is not None
                or max_val_batches is not None
            )
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
            "selection_split": "validation",
            "selection_metric": "validation_opendpd_nmse_db",
        },
        "dataset": {
            "directory": _display_path(dataset_dir),
            "files_sha256": dataset_hashes,
            "test_file_hashes_recorded": False,
        },
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
        "runtime": {
            "fit_seconds": elapsed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_threads": int(torch.get_num_threads()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    _write_json_atomic(output / "training_report.json", report)
    return report


def run_config(
    config_path: str | Path,
    *,
    candidate_names: Sequence[str] | None = None,
    output_root: str | Path | None = None,
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
