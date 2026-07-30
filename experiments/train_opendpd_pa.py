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
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.train_spline import load_complex_iq_csv


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
    dataset_dir = value.get("dataset_dir")
    if not isinstance(dataset_dir, str) or not dataset_dir:
        raise ValueError("config.dataset_dir must be a non-empty string")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("config.candidates must be a non-empty list")
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
    training = value.get("training")
    if not isinstance(training, dict):
        raise ValueError("config.training must be an object")
    for key in ("n_epochs", "batch_size", "batch_size_eval"):
        number = training.get(key)
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("lr", "lr_end", "decay_factor", "patience", "grad_clip_val"):
        if key not in training:
            raise ValueError(f"config.training is missing {key}")
    if training.get("optimizer") != "adamw" or training.get("loss") != "mse":
        raise ValueError("runner currently reproduces only AdamW + MSE")
    if value.get("selection_metric") != "validation_opendpd_nmse_db":
        raise ValueError("checkpoint selection metric must be validation NMSE")
    return value


def _validate_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("each candidate must be an object")
    name = candidate.get("name")
    backbone = candidate.get("backbone")
    hidden_size = candidate.get("hidden_size")
    if not isinstance(name, str) or not name:
        raise ValueError("candidate.name must be a non-empty string")
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


def resolve_dataset_dir(config: Mapping[str, Any], config_path: str | Path) -> Path:
    """Resolve the declared dataset directory without inspecting other files."""

    source = Path(config_path).resolve().parent
    dataset_dir = Path(str(config["dataset_dir"]))
    if not dataset_dir.is_absolute():
        dataset_dir = (source / dataset_dir).resolve()
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
        if not path.is_file():
            raise FileNotFoundError(f"required train/validation file missing: {path}")
        hashes[name] = sha256_file(path)
    return dataset_dir, hashes


def load_allowed_split(
    dataset_dir: str | Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly one allowed split; reject ``test`` before path creation."""

    if split not in {"train", "val"}:
        raise RuntimeError(
            "forbidden split requested; this runner can load only train or val"
        )
    root = Path(dataset_dir)
    input_path = root / f"{split}_input.csv"
    output_path = root / f"{split}_output.csv"
    # ``load_complex_iq_csv`` validates the exact I,Q schema and finite values.
    features = load_complex_iq_csv(input_path)
    targets = load_complex_iq_csv(output_path)
    if features.shape != targets.shape:
        raise ValueError(f"{split} input/output lengths differ")
    return features, targets


def _import_opendpd():
    """Import vendored OpenDPD modules without requiring them at test import."""

    import importlib

    vendor = str(OPENDPD_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return importlib.import_module("models"), importlib.import_module(
        "modules.data_collector"
    )


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
        train_set = data_collector.IQFrameDataset(
            segmented.features.numpy(),
            segmented.targets.numpy(),
            frame_length=int(framing["frame_length"]),
            stride=int(framing["frame_stride"]),
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
    }


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


def _source_provenance() -> dict[str, Any]:
    """Collect source hashes without reading any dataset test path."""

    source_files = [
        OPENDPD_ROOT / "models.py",
        OPENDPD_ROOT / "modules" / "data_collector.py",
    ]
    # Candidate-specific backbone files are added by ``run_candidate``.
    result = {
        "opendpd_root": _display_path(OPENDPD_ROOT),
        "files": {
            _display_path(path): sha256_file(path)
            for path in source_files
        },
    }
    try:
        import subprocess

        result["vendored_commit"] = subprocess.check_output(
            ["git", "-C", str(OPENDPD_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["vendored_commit"] = None
    return result


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
    dataset_dir, dataset_hashes = verify_allowed_inputs(config, config_path)
    train_input, train_output = load_allowed_split(dataset_dir, "train")
    val_input, val_output = load_allowed_split(dataset_dir, "val")

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

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty OpenDPD output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{candidate['name']}.pt"
    _save_checkpoint_atomic(best_state, checkpoint)
    source = _source_provenance()
    backbone_path = OPENDPD_ROOT / "backbones" / f"{candidate['backbone']}.py"
    if backbone_path.is_file():
        source["files"][_display_path(backbone_path)] = sha256_file(
            backbone_path
        )
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
