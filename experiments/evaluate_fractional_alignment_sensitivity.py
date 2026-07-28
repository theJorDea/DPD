"""Preregistered A0/A1 fractional-alignment sensitivity experiment.

This runner compares two target-preprocessing variants without selecting an
architecture, ridge, solver, or delay from their scores:

``A0``
    zero fractional shift, passed through the same FIR implementation and
    cropped by the same guard as A1;

``A1``
    the fractional-delay diagnostic already frozen in a training-only MP
    selection manifest.

Both variants use one selected MP recipe from that manifest and one fixed
causal GMP recipe embedded directly in the hash-recorded runner config.  A
path-only GMP indirection is intentionally unsupported.  Each original
``nperseg`` frame is transformed independently.  Models are assessed with
leave-one-original-frame-out training predictions and with a full-training
fit on validation.  Test files are neither opened nor hashed.

The result is a sensitivity analysis.  It cannot establish that the
correlation-derived delay belongs to the measurement path rather than to the
physical PA response.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import secrets
import shlex
import sys
import time
from typing import Any, Literal

import numpy as np

from baseline.complexity import memory_polynomial_inference_cost
from baseline.fractional_alignment import (
    FractionalAlignmentConfig,
    FrozenFractionalAlignment,
    apply_fractional_alignment_frames,
    freeze_fractional_alignment,
)
from baseline.gmp_pa import (
    GMPConfig,
    GeneralizedMemoryPolynomialPA,
    fit_gmp_pa,
)
from baseline.metrics import (
    nmse_opendpd_db,
    nmse_pooled_db,
    time_domain_rms_evm_db,
)
from baseline.pa_models import (
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
)
from baseline.train_spline import (
    file_sha256,
    load_dataset_spec,
    load_split_pair,
    write_json,
)


ModelKind = Literal["mp", "gmp"]

DECISION_RULE = {
    "primary_metric": "common_causal_interior",
    "gmp_a1_minus_a0_max_db": -0.25,
    "mp_corroboration_a1_minus_a0_max_db": 0.0,
    "required_splits": ["train_oof", "validation"],
    "require_full_record_same_sign": True,
    "fallback_variant": "a0",
    "accepted_a1_scope": (
        "sensitivity_protocol_not_proven_feedback_deembedding"
    ),
}


@dataclass(frozen=True, slots=True)
class FixedMPRecipe:
    orders: tuple[int, ...]
    delays: tuple[int, ...]
    ridge: float
    solver_mode: str = "augmented_complex_lstsq"

    @property
    def causal_warmup_samples(self) -> int:
        return max(self.delays)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_class": "complex_memory_polynomial",
            "orders": list(self.orders),
            "delays": list(self.delays),
            "ridge": self.ridge,
            "solver_mode": self.solver_mode,
        }


@dataclass(frozen=True, slots=True)
class FixedGMPRecipe:
    config: GMPConfig
    ridge: float
    solver_mode: Literal["ridge_lstsq", "truncated_svd"]
    svd_rcond: float | None

    @property
    def causal_warmup_samples(self) -> int:
        return self.config.causal_warmup_samples

    def to_dict(self) -> dict[str, object]:
        return {
            "model_class": "complex_generalized_memory_polynomial",
            "gmp_config": dataclasses.asdict(self.config),
            "ridge": self.ridge,
            "solver_mode": self.solver_mode,
            "svd_rcond": self.svd_rcond,
        }


@dataclass(frozen=True, slots=True)
class FramePairBatch:
    """One transformed split with explicit, unflattened frame boundaries."""

    input_frames: tuple[np.ndarray, ...]
    output_frames: tuple[np.ndarray, ...]
    original_frame_lengths: tuple[int, ...]
    effective_frame_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.input_frames or len(self.input_frames) != len(
            self.output_frames
        ):
            raise ValueError("frame batch requires paired non-empty frames")
        if len(self.input_frames) != len(self.original_frame_lengths):
            raise ValueError("original frame-length metadata does not match frames")
        if len(self.input_frames) != len(self.effective_frame_lengths):
            raise ValueError("effective frame-length metadata does not match frames")
        for frame_input, frame_output, effective_length in zip(
            self.input_frames,
            self.output_frames,
            self.effective_frame_lengths,
        ):
            if frame_input.shape != frame_output.shape:
                raise ValueError("transformed input/output frame shapes differ")
            if frame_input.ndim != 1 or frame_input.size != effective_length:
                raise ValueError("effective frame-length metadata is inconsistent")

    @property
    def sample_count(self) -> int:
        return int(sum(self.effective_frame_lengths))

    def flattened(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate(self.input_frames),
            np.concatenate(self.output_frames),
        )

    def summary(self) -> dict[str, object]:
        return {
            "frame_count": len(self.input_frames),
            "original_frame_lengths": list(self.original_frame_lengths),
            "effective_frame_lengths": list(self.effective_frame_lengths),
            "effective_sample_count": self.sample_count,
        }


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )


def _snapshot_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        label: file_sha256(path)
        for label, path in paths.items()
    }


def _verify_frozen_hashes(
    paths: dict[str, Path],
    frozen_hashes: dict[str, str],
    *,
    scope: str,
) -> None:
    if set(paths) != set(frozen_hashes):
        raise ValueError(f"{scope} path/hash labels differ")
    for label, path in paths.items():
        _verify_hash(
            path,
            frozen_hashes[label],
            label=f"{scope} {label}",
        )


def _path_entry_exists(path: Path) -> bool:
    """Return true for regular entries and dangling symbolic links."""

    return path.exists() or path.is_symlink()


def _acquire_bundle_lock(lock_path: Path) -> bytes:
    """Atomically acquire one same-directory single-writer bundle lock."""

    token = secrets.token_hex(32)
    payload = (
        json.dumps(
            {
                "pid": os.getpid(),
                "token": token,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"sensitivity bundle lock already exists: {lock_path}"
        ) from error
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("bundle lock write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _verify_owned_bundle_lock(
    lock_path: Path,
    expected_payload: bytes,
    *,
    scope: str,
) -> None:
    """Fail closed unless the lock is still the exact one acquired here."""

    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError(f"{scope}: owned bundle lock is missing/replaced")
    actual_payload = lock_path.read_bytes()
    if actual_payload != expected_payload:
        raise RuntimeError(f"{scope}: owned bundle lock token/pid changed")


def _atomic_publish_model(
    model: MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
    *,
    temporary_path: Path,
    final_path: Path,
) -> None:
    """Publish one NPZ atomically without replacing an existing artifact."""

    if temporary_path.suffix != ".npz":
        raise ValueError("model publication temp path must end in .npz")
    if _path_entry_exists(temporary_path):
        raise FileExistsError(
            f"model publication temp already exists: {temporary_path}"
        )
    if _path_entry_exists(final_path):
        raise FileExistsError(
            f"immutable model artifact already exists: {final_path}"
        )
    model.save(temporary_path)
    if not temporary_path.is_file():
        raise RuntimeError("model save did not create the expected NPZ temp")
    if _path_entry_exists(final_path):
        raise FileExistsError(
            f"immutable model artifact appeared during publication: {final_path}"
        )
    os.replace(temporary_path, final_path)


def _atomic_publish_json(
    report: dict[str, Any],
    *,
    temporary_path: Path,
    final_path: Path,
) -> None:
    """Publish the final JSON last through a same-directory atomic replace."""

    if _path_entry_exists(temporary_path):
        raise FileExistsError(
            f"report publication temp already exists: {temporary_path}"
        )
    if _path_entry_exists(final_path):
        raise FileExistsError(
            f"immutable report artifact already exists: {final_path}"
        )
    write_json(temporary_path, report)
    if not temporary_path.is_file():
        raise RuntimeError("report writer did not create the expected temp")
    if _path_entry_exists(final_path):
        raise FileExistsError(
            f"immutable report artifact appeared during publication: {final_path}"
        )
    os.replace(temporary_path, final_path)


def _load_runner_config(path: Path) -> dict[str, Any]:
    config = _load_json_object(path, name="sensitivity config")
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("sensitivity config schema_version must equal 1")
    if "fixed_gmp_config" in config or "fixed_gmp_config_sha256" in config:
        raise ValueError(
            "path-only fixed GMP indirection is unsupported; embed "
            "fixed_gmp_recipe directly"
        )
    required = {
        "dataset",
        "selection_manifest",
        "selection_manifest_sha256",
        "fixed_gmp_recipe",
        "output_dir",
        "alignment_filter",
        "decision_rule",
        "max_real_multiplications_per_sample",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(
            f"sensitivity config is missing keys: {sorted(missing)}"
        )
    filter_config = config["alignment_filter"]
    if not isinstance(filter_config, dict):
        raise ValueError("alignment_filter must be one JSON object")
    if set(filter_config) != {"tap_count", "kaiser_beta"}:
        raise ValueError(
            "alignment_filter must contain exactly tap_count and kaiser_beta"
        )
    if not isinstance(config["fixed_gmp_recipe"], dict):
        raise ValueError("fixed_gmp_recipe must be one inline JSON object")
    if (
        not isinstance(config["dataset"], str)
        or not config["dataset"].strip()
    ):
        raise ValueError("dataset must be a non-empty portable path string")
    operation_budget = config["max_real_multiplications_per_sample"]
    if (
        not isinstance(operation_budget, Integral)
        or isinstance(operation_budget, (bool, np.bool_))
        or int(operation_budget) != 1000
    ):
        raise ValueError(
            "max_real_multiplications_per_sample must equal the "
            "preregistered exclusive limit 1000"
        )
    decision_rule = config["decision_rule"]
    if not isinstance(decision_rule, dict):
        raise ValueError("decision_rule must be one JSON object")
    if set(decision_rule) != set(DECISION_RULE):
        raise ValueError(
            "decision_rule keys must exactly match the preregistered contract"
        )
    for key, expected in DECISION_RULE.items():
        actual = decision_rule[key]
        if isinstance(expected, float):
            if (
                isinstance(actual, (bool, np.bool_))
                or not isinstance(actual, Real)
                or float(actual) != expected
            ):
                raise ValueError(
                    f"decision_rule.{key} must equal preregistered "
                    f"value {expected}"
                )
        elif isinstance(expected, bool):
            if not isinstance(actual, bool) or actual is not expected:
                raise ValueError(
                    f"decision_rule.{key} must equal preregistered "
                    f"value {expected!r}"
                )
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"decision_rule.{key} must equal preregistered "
                f"value {expected!r}"
            )
    return config


def _integer_tuple(
    values: Any,
    *,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON list")
    result: list[int] = []
    for value in values:
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or int(value) < minimum
        ):
            raise ValueError(
                f"every {name} entry must be an integer >= {minimum}"
            )
        result.append(int(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return tuple(result)


def _nonnegative_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _parse_selection_manifest(
    selection: dict[str, Any],
) -> tuple[FixedMPRecipe, dict[str, Any]]:
    if int(selection.get("schema_version", -1)) != 1:
        raise ValueError("unsupported MP selection manifest schema")
    if selection.get("task") != "forward_pa_identification_model_selection":
        raise ValueError("selection manifest is not a forward PA selection")
    if selection.get("model_class") != "complex_memory_polynomial":
        raise ValueError("selection manifest must contain a selected MP model")
    if selection.get("selection_split") != "validation":
        raise ValueError("MP architecture must have been selected on validation")
    if selection.get("test_split_accessed") is not False:
        raise ValueError("MP selection manifest must certify sealed test data")

    selected = selection.get("selected_trial")
    if not isinstance(selected, dict):
        raise ValueError("selection manifest has no selected_trial object")
    recipe = FixedMPRecipe(
        orders=_integer_tuple(
            selected.get("orders"),
            name="selected MP orders",
            minimum=1,
        ),
        delays=_integer_tuple(
            selected.get("delays"),
            name="selected MP delays",
            minimum=0,
        ),
        ridge=_nonnegative_real(
            selected.get("ridge"),
            name="selected MP ridge",
        ),
    )

    protocol = selection.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("selection manifest has no protocol object")
    if int(protocol.get("alignment_delay_samples", 1)) != 0:
        raise ValueError(
            "A0/A1 runner currently requires the selected integer delay to be "
            "zero so both variants have identical effective frame lengths"
        )
    if protocol.get("fractional_delay_applied") is not False:
        raise ValueError("selection data must not already apply fractional delay")
    if protocol.get("fractional_delay_reliable") is not True:
        raise ValueError(
            "A1 requires a training-frozen reliable fractional diagnostic"
        )
    diagnostic = protocol.get("fractional_delay_estimate_samples")
    if isinstance(diagnostic, (bool, np.bool_)) or not isinstance(
        diagnostic,
        Real,
    ):
        raise ValueError("fractional delay estimate must be a real scalar")
    diagnostic = float(diagnostic)
    if not math.isfinite(diagnostic) or not -0.5 <= diagnostic < 0.5:
        raise ValueError(
            "fractional diagnostic must lie in [-0.5, 0.5) after zero "
            "integer alignment"
        )
    offset = float(protocol.get("fractional_delay_offset_samples", np.nan))
    if not math.isfinite(offset) or not math.isclose(
        diagnostic,
        offset,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "fractional estimate and offset are inconsistent for zero "
            "integer alignment"
        )
    nperseg = protocol.get("nperseg")
    if (
        not isinstance(nperseg, Integral)
        or isinstance(nperseg, (bool, np.bool_))
        or int(nperseg) <= 1
    ):
        raise ValueError("selection protocol nperseg must exceed one")
    return recipe, protocol


def _parse_fixed_gmp_recipe(value: dict[str, Any]) -> FixedGMPRecipe:
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("fixed GMP config schema_version must equal 1")
    raw_config = value.get("gmp_config")
    if not isinstance(raw_config, dict):
        raise ValueError("fixed GMP config requires one gmp_config object")
    required_dimensions = {
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
    if set(raw_config) != required_dimensions:
        raise ValueError(
            "gmp_config must contain exactly the fixed GMP dimensions and "
            "leading_policy"
        )
    gmp_config = GMPConfig(**raw_config)
    if gmp_config.leading_policy != "causal_leading":
        raise ValueError("fractional sensitivity requires a fixed causal GMP")
    if gmp_config.lookahead_samples != 0:
        raise ValueError("fixed causal GMP must have zero lookahead")

    solver_mode = value.get("solver_mode")
    if solver_mode not in {"ridge_lstsq", "truncated_svd"}:
        raise ValueError(
            "fixed GMP solver_mode must be ridge_lstsq or truncated_svd"
        )
    ridge = _nonnegative_real(value.get("ridge"), name="fixed GMP ridge")
    raw_rcond = value.get("svd_rcond")
    if solver_mode == "ridge_lstsq":
        if raw_rcond is not None:
            raise ValueError("ridge_lstsq fixed GMP requires svd_rcond=null")
        svd_rcond = None
    else:
        if ridge != 0.0:
            raise ValueError("truncated_svd fixed GMP requires ridge=0")
        if isinstance(raw_rcond, (bool, np.bool_)) or not isinstance(
            raw_rcond,
            Real,
        ):
            raise ValueError(
                "truncated_svd fixed GMP requires a real svd_rcond"
            )
        svd_rcond = float(raw_rcond)
        if not math.isfinite(svd_rcond) or not 0.0 < svd_rcond < 1.0:
            raise ValueError("svd_rcond must satisfy finite 0 < value < 1")
    return FixedGMPRecipe(
        config=gmp_config,
        ridge=ridge,
        solver_mode=solver_mode,
        svd_rcond=svd_rcond,
    )


def _split_original_frames(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    *,
    nperseg: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    x = np.asarray(pa_input, dtype=np.complex128)
    y = np.asarray(measured_output, dtype=np.complex128)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0:
        raise ValueError("split input/output must be non-empty 1-D sequences")
    if x.shape != y.shape:
        raise ValueError("split input/output shapes differ")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("split input/output contains non-finite values")
    if (
        not isinstance(nperseg, Integral)
        or isinstance(nperseg, (bool, np.bool_))
        or int(nperseg) <= 1
    ):
        raise ValueError("nperseg must be an integer greater than one")
    nperseg = int(nperseg)
    input_frames = tuple(
        x[start:min(start + nperseg, x.size)]
        for start in range(0, x.size, nperseg)
    )
    output_frames = tuple(
        y[start:min(start + nperseg, y.size)]
        for start in range(0, y.size, nperseg)
    )
    return input_frames, output_frames


def prepare_alignment_variants(
    pa_input: np.ndarray,
    measured_output: np.ndarray,
    *,
    original_nperseg: int,
    a0_transform: FrozenFractionalAlignment,
    a1_transform: FrozenFractionalAlignment,
) -> dict[str, FramePairBatch]:
    """Prepare equal-support A0/A1 pairs independently per original frame."""

    if a0_transform.config.observed_delay_samples != 0.0:
        raise ValueError("A0 transform must have exactly zero delay")
    if a0_transform.config.tap_count != a1_transform.config.tap_count:
        raise ValueError("A0 and A1 must use identical FIR tap counts")
    if a0_transform.config.kaiser_beta != a1_transform.config.kaiser_beta:
        raise ValueError("A0 and A1 must use identical Kaiser beta")
    if a0_transform.guard_samples != a1_transform.guard_samples:
        raise ValueError("A0 and A1 must use the same symmetric guard")
    if (
        a0_transform.integer_delay_samples != 0
        or a1_transform.integer_delay_samples != 0
    ):
        raise ValueError(
            "A0/A1 equal-support comparison requires zero integer components"
        )

    input_frames, output_frames = _split_original_frames(
        pa_input,
        measured_output,
        nperseg=original_nperseg,
    )
    a0_inputs, a0_outputs = apply_fractional_alignment_frames(
        input_frames,
        output_frames,
        a0_transform,
    )
    a1_inputs, a1_outputs = apply_fractional_alignment_frames(
        input_frames,
        output_frames,
        a1_transform,
    )
    if len(a0_inputs) != len(a1_inputs):
        raise RuntimeError("A0/A1 frame counts differ")
    for a0_input, a1_input, a0_output, a1_output in zip(
        a0_inputs,
        a1_inputs,
        a0_outputs,
        a1_outputs,
    ):
        if not np.array_equal(a0_input, a1_input):
            raise RuntimeError("A0/A1 input support is not bit-identical")
        if a0_output.shape != a1_output.shape:
            raise RuntimeError("A0/A1 output support differs")

    original_lengths = tuple(int(frame.size) for frame in input_frames)
    effective_lengths = tuple(int(frame.size) for frame in a0_inputs)
    return {
        "a0": FramePairBatch(
            a0_inputs,
            a0_outputs,
            original_lengths,
            effective_lengths,
        ),
        "a1": FramePairBatch(
            a1_inputs,
            a1_outputs,
            original_lengths,
            effective_lengths,
        ),
    }


def _energy_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(estimate, dtype=np.complex128)
    target = np.asarray(reference, dtype=np.complex128)
    if prediction.ndim != 1 or prediction.shape != target.shape:
        raise ValueError("metric arrays must be equal-length 1-D vectors")
    error_energy = float(np.sum(np.abs(prediction - target) ** 2))
    reference_energy = float(np.sum(np.abs(target) ** 2))
    if reference_energy <= 0.0:
        raise ValueError("metric reference must have positive energy")
    sample_count = int(target.size)
    return {
        "sample_count": sample_count,
        "error_energy": error_energy,
        "reference_energy": reference_energy,
        "mse": error_energy / sample_count,
        "reference_power": reference_energy / sample_count,
        "relative_error_power": error_energy / reference_energy,
        "complex_nmse_pooled_db": nmse_pooled_db(prediction, target),
        "time_domain_rms_sample_evm_db": time_domain_rms_evm_db(
            prediction,
            target,
        ),
    }


def _prediction_metrics(
    prediction_frames: tuple[np.ndarray, ...],
    reference_frames: tuple[np.ndarray, ...],
    *,
    common_warmup_samples: int,
    expected_frame_length: int,
    opendpd_padded_prediction_frames: tuple[np.ndarray, ...] | None = None,
) -> dict[str, object]:
    if not prediction_frames or len(prediction_frames) != len(reference_frames):
        raise ValueError("prediction/reference frame counts must match")
    if (
        not isinstance(expected_frame_length, Integral)
        or isinstance(expected_frame_length, (bool, np.bool_))
        or int(expected_frame_length) <= 0
    ):
        raise ValueError("expected_frame_length must be a positive integer")
    expected_frame_length = int(expected_frame_length)
    if (
        not isinstance(common_warmup_samples, Integral)
        or isinstance(common_warmup_samples, (bool, np.bool_))
        or int(common_warmup_samples) < 0
        or int(common_warmup_samples) >= expected_frame_length
    ):
        raise ValueError(
            "common_warmup_samples must satisfy "
            "0 <= warmup < expected_frame_length"
        )
    common_warmup_samples = int(common_warmup_samples)
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    interior_predictions: list[np.ndarray] = []
    interior_references: list[np.ndarray] = []
    if opendpd_padded_prediction_frames is None:
        if any(
            np.asarray(frame).size != expected_frame_length
            for frame in prediction_frames
        ):
            raise ValueError(
                "partial frames require predictions obtained by running the "
                "model on right-zero-padded inputs; zero-padding an already "
                "computed prediction is not OpenDPD-compatible"
            )
        opendpd_padded_prediction_frames = prediction_frames
    if len(opendpd_padded_prediction_frames) != len(prediction_frames):
        raise ValueError("padded prediction frame count differs")

    padded_predictions: list[np.ndarray] = []
    padded_references: list[np.ndarray] = []
    frame_nmse: list[float] = []
    complete_frame_count = 0
    actual_nonpadding_sample_count = 0
    predicted_tail_error_energy = 0.0
    predicted_tail_nonzero_sample_count = 0
    for prediction, reference, padded_prediction in zip(
        prediction_frames,
        reference_frames,
        opendpd_padded_prediction_frames,
    ):
        prediction_array = np.asarray(prediction, dtype=np.complex128)
        reference_array = np.asarray(reference, dtype=np.complex128)
        padded_prediction_array = np.asarray(
            padded_prediction,
            dtype=np.complex128,
        )
        if (
            prediction_array.ndim != 1
            or prediction_array.shape != reference_array.shape
        ):
            raise ValueError("every prediction must match its reference frame")
        if prediction_array.size > expected_frame_length:
            raise ValueError(
                "a transformed frame exceeds expected_frame_length"
            )
        if padded_prediction_array.shape != (expected_frame_length,):
            raise ValueError(
                "every OpenDPD padded prediction must have exactly "
                "expected_frame_length samples"
            )
        if not np.array_equal(
            padded_prediction_array[: prediction_array.size],
            prediction_array,
        ):
            raise ValueError(
                "OpenDPD padded prediction prefix differs from the primary "
                "actual-length prediction"
            )
        if common_warmup_samples >= prediction_array.size:
            raise ValueError("common warmup consumes a transformed frame")
        predictions.append(prediction_array)
        references.append(reference_array)
        interior_predictions.append(
            prediction_array[common_warmup_samples:]
        )
        interior_references.append(reference_array[common_warmup_samples:])
        frame_nmse.append(nmse_pooled_db(prediction_array, reference_array))
        actual_nonpadding_sample_count += int(prediction_array.size)
        if prediction_array.size == expected_frame_length:
            complete_frame_count += 1
        padding = expected_frame_length - int(prediction_array.size)
        padded_predictions.append(padded_prediction_array)
        padded_references.append(
            np.pad(reference_array, (0, padding), mode="constant")
        )
        predicted_tail = padded_prediction_array[prediction_array.size :]
        predicted_tail_error_energy += float(
            np.sum(np.abs(predicted_tail) ** 2)
        )
        predicted_tail_nonzero_sample_count += int(
            np.count_nonzero(predicted_tail)
        )

    full = _energy_metrics(
        np.concatenate(predictions),
        np.concatenate(references),
    )
    interior = _energy_metrics(
        np.concatenate(interior_predictions),
        np.concatenate(interior_references),
    )
    interior.update(
        {
            "warmup_samples_per_original_frame": common_warmup_samples,
            "discarded_sample_count": (
                common_warmup_samples * len(prediction_frames)
            ),
        }
    )
    padded_prediction_array = np.stack(padded_predictions)
    padded_reference_array = np.stack(padded_references)
    opendpd_nmse = nmse_opendpd_db(
        padded_prediction_array,
        padded_reference_array,
    )
    opendpd_interior_nmse = nmse_opendpd_db(
        padded_prediction_array[:, common_warmup_samples:],
        padded_reference_array[:, common_warmup_samples:],
    )
    segment_count = len(prediction_frames)
    padded_sample_count = segment_count * expected_frame_length
    zero_padding_sample_count = (
        padded_sample_count - actual_nonpadding_sample_count
    )
    interior_segment_length = expected_frame_length - common_warmup_samples
    interior_padded_sample_count = segment_count * interior_segment_length
    interior_actual_nonpadding_sample_count = (
        actual_nonpadding_sample_count
        - segment_count * common_warmup_samples
    )
    discarded_warmup_samples = segment_count * common_warmup_samples
    return {
        "full_record": full,
        "common_causal_interior": interior,
        "opendpd_compatible": {
            "nmse_mean_segment_db": opendpd_nmse,
            "complete_frame_count": complete_frame_count,
            "segment_count_including_zero_padded_partial": segment_count,
            "segment_length_samples": expected_frame_length,
            "scored_sample_count": padded_sample_count,
            "padded_sample_count": padded_sample_count,
            "actual_nonpadding_scored_sample_count": (
                actual_nonpadding_sample_count
            ),
            "zero_padding_sample_count": zero_padding_sample_count,
            "right_zero_padded_input_sample_count": (
                zero_padding_sample_count
            ),
            "right_zero_padded_reference_sample_count": (
                zero_padding_sample_count
            ),
            "predicted_causal_tail_error_energy": (
                predicted_tail_error_energy
            ),
            "predicted_causal_tail_nonzero_sample_count": (
                predicted_tail_nonzero_sample_count
            ),
            "discarded_partial_tail_samples": 0,
            "partial_frame_count": segment_count - complete_frame_count,
            "partial_frame_policy": (
                "right-zero-pad transformed model input to effective "
                "nperseg, run inference on the full padded input, and "
                "right-zero-pad the reference; delayed prediction tail is "
                "scored rather than forced to zero"
            ),
            "definition": (
                "arithmetic mean in dB of one NMSE ratio per original "
                "transformed frame after right-zero-padding a partial final "
                "model input and reference; causal model output from delayed "
                "state in the padded tail remains part of the error"
            ),
        },
        "opendpd_compatible_common_causal_interior": {
            "nmse_mean_segment_db": opendpd_interior_nmse,
            "complete_frame_count": complete_frame_count,
            "segment_count_including_zero_padded_partial": segment_count,
            "segment_length_after_warmup_samples": interior_segment_length,
            "warmup_samples_per_frame": common_warmup_samples,
            "scored_sample_count": interior_padded_sample_count,
            "padded_sample_count": interior_padded_sample_count,
            "actual_nonpadding_scored_sample_count": (
                interior_actual_nonpadding_sample_count
            ),
            "zero_padding_sample_count": zero_padding_sample_count,
            "right_zero_padded_input_sample_count": (
                zero_padding_sample_count
            ),
            "right_zero_padded_reference_sample_count": (
                zero_padding_sample_count
            ),
            "predicted_causal_tail_error_energy": (
                predicted_tail_error_energy
            ),
            "predicted_causal_tail_nonzero_sample_count": (
                predicted_tail_nonzero_sample_count
            ),
            "discarded_warmup_samples_from_actual_frames": (
                discarded_warmup_samples
            ),
            "discarded_partial_tail_samples": 0,
            "partial_frame_count": segment_count - complete_frame_count,
            "partial_frame_policy": (
                "right-zero-pad to effective nperseg before applying the "
                "same leading common warmup to every frame"
            ),
            "definition": (
                "arithmetic mean in dB of one NMSE ratio per zero-padded "
                "original transformed frame after identical leading common "
                "warmup"
            ),
        },
        "mean_per_original_frame_nmse_db": float(np.mean(frame_nmse)),
        "per_original_frame_nmse_db": frame_nmse,
        "frame_count": len(prediction_frames),
        "gain_fit_after_prediction": False,
        "delay_fit_after_prediction": False,
    }


def _fit_fixed_model(
    kind: ModelKind,
    recipe: FixedMPRecipe | FixedGMPRecipe,
    input_frames: tuple[np.ndarray, ...],
    output_frames: tuple[np.ndarray, ...],
    *,
    effective_nperseg: int,
) -> tuple[
    MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
    dict[str, Any],
    float,
]:
    x = np.concatenate(input_frames)
    y = np.concatenate(output_frames)
    started = time.perf_counter()
    if kind == "mp":
        if not isinstance(recipe, FixedMPRecipe):
            raise TypeError("MP fit requires FixedMPRecipe")
        model, diagnostics = fit_memory_polynomial_pa(
            x,
            y,
            orders=recipe.orders,
            delays=recipe.delays,
            ridge=recipe.ridge,
            segment_length=effective_nperseg,
            coefficient_dtype=np.complex128,
        )
    elif kind == "gmp":
        if not isinstance(recipe, FixedGMPRecipe):
            raise TypeError("GMP fit requires FixedGMPRecipe")
        model, diagnostics = fit_gmp_pa(
            x,
            y,
            config=recipe.config,
            ridge=recipe.ridge,
            segment_length=effective_nperseg,
            coefficient_dtype=np.complex128,
            solver_mode=recipe.solver_mode,
            svd_rcond=recipe.svd_rcond,
        )
    else:
        raise ValueError(f"unsupported model kind: {kind}")
    fit_seconds = time.perf_counter() - started
    return model, dataclasses.asdict(diagnostics), fit_seconds


def _predict_frames(
    model: MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
    input_frames: tuple[np.ndarray, ...],
    *,
    expected_frame_length: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Predict actual prefixes and exact zero-input-padded OpenDPD frames."""

    actual_predictions: list[np.ndarray] = []
    padded_predictions: list[np.ndarray] = []
    for frame in input_frames:
        frame_array = np.asarray(frame, dtype=np.complex128)
        if frame_array.ndim != 1 or not (
            0 < frame_array.size <= expected_frame_length
        ):
            raise ValueError("input frame has invalid effective length")
        padded_input = np.pad(
            frame_array,
            (0, expected_frame_length - frame_array.size),
            mode="constant",
        )
        padded_prediction = np.asarray(
            model.predict(padded_input),
            dtype=np.complex128,
        )
        if padded_prediction.shape != (expected_frame_length,):
            raise RuntimeError("model returned an unexpected padded shape")
        actual_predictions.append(
            padded_prediction[: frame_array.size].copy()
        )
        padded_predictions.append(padded_prediction)
    return tuple(actual_predictions), tuple(padded_predictions)


def _oof_predictions(
    kind: ModelKind,
    recipe: FixedMPRecipe | FixedGMPRecipe,
    batch: FramePairBatch,
    *,
    effective_nperseg: int,
    common_warmup_samples: int,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    list[dict[str, Any]],
    float,
]:
    frame_count = len(batch.input_frames)
    if frame_count < 2:
        raise ValueError(
            "inner frame OOF requires at least two original training frames"
        )
    predictions: list[np.ndarray | None] = [None] * frame_count
    padded_predictions: list[np.ndarray | None] = [None] * frame_count
    fold_reports: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    for held_frame in range(frame_count):
        fit_inputs = tuple(
            frame
            for index, frame in enumerate(batch.input_frames)
            if index != held_frame
        )
        fit_outputs = tuple(
            frame
            for index, frame in enumerate(batch.output_frames)
            if index != held_frame
        )
        model, diagnostics, fit_seconds = _fit_fixed_model(
            kind,
            recipe,
            fit_inputs,
            fit_outputs,
            effective_nperseg=effective_nperseg,
        )
        total_fit_seconds += fit_seconds
        held_predictions, held_padded_predictions = _predict_frames(
            model,
            (batch.input_frames[held_frame],),
            expected_frame_length=effective_nperseg,
        )
        held_prediction = held_predictions[0]
        held_padded_prediction = held_padded_predictions[0]
        predictions[held_frame] = held_prediction
        padded_predictions[held_frame] = held_padded_prediction
        fold_reports.append(
            {
                "held_original_frame_index": held_frame,
                "fit_original_frame_indices": [
                    index for index in range(frame_count) if index != held_frame
                ],
                "fit_sample_count": int(
                    sum(frame.size for frame in fit_inputs)
                ),
                "held_sample_count": int(
                    batch.input_frames[held_frame].size
                ),
                "fit_seconds": fit_seconds,
                "fit_diagnostics": diagnostics,
                "held_frame_metrics": _prediction_metrics(
                    (held_prediction,),
                    (batch.output_frames[held_frame],),
                    common_warmup_samples=common_warmup_samples,
                    expected_frame_length=effective_nperseg,
                    opendpd_padded_prediction_frames=(
                        held_padded_prediction,
                    ),
                ),
            }
        )
    if any(prediction is None for prediction in predictions) or any(
        prediction is None for prediction in padded_predictions
    ):
        raise RuntimeError("OOF prediction was not filled for every frame")
    return (
        tuple(
            np.asarray(prediction, dtype=np.complex128)
            for prediction in predictions
        ),
        tuple(
            np.asarray(prediction, dtype=np.complex128)
            for prediction in padded_predictions
        ),
        fold_reports,
        total_fit_seconds,
    )


def _evaluate_fixed_recipe(
    kind: ModelKind,
    recipe: FixedMPRecipe | FixedGMPRecipe,
    train_batch: FramePairBatch,
    validation_batch: FramePairBatch,
    *,
    effective_nperseg: int,
    common_warmup_samples: int,
) -> tuple[
    dict[str, Any],
    MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
]:
    (
        oof_prediction,
        oof_padded_prediction,
        folds,
        oof_fit_seconds,
    ) = _oof_predictions(
        kind,
        recipe,
        train_batch,
        effective_nperseg=effective_nperseg,
        common_warmup_samples=common_warmup_samples,
    )
    model, full_fit_diagnostics, full_fit_seconds = _fit_fixed_model(
        kind,
        recipe,
        train_batch.input_frames,
        train_batch.output_frames,
        effective_nperseg=effective_nperseg,
    )
    train_fit_prediction, train_fit_padded_prediction = _predict_frames(
        model,
        train_batch.input_frames,
        expected_frame_length=effective_nperseg,
    )
    validation_prediction, validation_padded_prediction = _predict_frames(
        model,
        validation_batch.input_frames,
        expected_frame_length=effective_nperseg,
    )
    if kind == "mp":
        assert isinstance(recipe, FixedMPRecipe)
        operation_count = memory_polynomial_inference_cost(
            recipe.orders,
            recipe.delays,
        )
        stored_real_coefficients = model.stored_real_coefficients
    else:
        assert isinstance(model, GeneralizedMemoryPolynomialPA)
        operation_count = model.operation_count
        stored_real_coefficients = model.stored_real_coefficients

    return (
        {
            "fixed_recipe": recipe.to_dict(),
            "architecture_tuning_performed": False,
            "full_training_fit": {
                "fit_seconds": full_fit_seconds,
                "fit_diagnostics": full_fit_diagnostics,
                "resubstitution_metrics": _prediction_metrics(
                    train_fit_prediction,
                    train_batch.output_frames,
                    common_warmup_samples=common_warmup_samples,
                    expected_frame_length=effective_nperseg,
                    opendpd_padded_prediction_frames=(
                        train_fit_padded_prediction
                    ),
                ),
            },
            "train_inner_original_frame_oof": {
                "scope": (
                    "coefficient_fit_oof_conditional_on_full_train_"
                    "frozen_delay"
                ),
                "delay_estimation_nested_within_oof": False,
                "delay_scope_note": (
                    "the fractional-delay diagnostic was frozen once from "
                    "the complete training split; each fold excludes its held "
                    "frame only from coefficient fitting"
                ),
                "fold_count": len(folds),
                "total_fit_seconds": oof_fit_seconds,
                "metrics": _prediction_metrics(
                    oof_prediction,
                    train_batch.output_frames,
                    common_warmup_samples=common_warmup_samples,
                    expected_frame_length=effective_nperseg,
                    opendpd_padded_prediction_frames=(
                        oof_padded_prediction
                    ),
                ),
                "folds": folds,
            },
            "validation_confirmation": {
                "fit_or_tuning_on_validation": False,
                "recipe_evidence_status": (
                    "mp_recipe_previously_selected_on_same_validation_"
                    "split_not_independent_corroboration"
                    if kind == "mp"
                    else
                    "fixed_gmp_preregistered_no_tuning_within_runner"
                ),
                "mp_recipe_selected_on_this_validation_split": kind == "mp",
                "fixed_gmp_preregistered_before_runner": kind == "gmp",
                "metrics": _prediction_metrics(
                    validation_prediction,
                    validation_batch.output_frames,
                    common_warmup_samples=common_warmup_samples,
                    expected_frame_length=effective_nperseg,
                    opendpd_padded_prediction_frames=(
                        validation_padded_prediction
                    ),
                ),
            },
            "complexity": {
                "operation_count_per_complex_sample": (
                    operation_count.to_dict()
                ),
                "stored_real_coefficients": stored_real_coefficients,
                "complex_multiply_convention": "4 real MUL + 2 real ADD",
            },
        },
        model,
    )


def _comparison_deltas(results: dict[str, Any]) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    paths = {
        "train_oof_full_record": (
            "train_inner_original_frame_oof",
            "metrics",
            "full_record",
        ),
        "train_oof_common_interior": (
            "train_inner_original_frame_oof",
            "metrics",
            "common_causal_interior",
        ),
        "validation_full_record": (
            "validation_confirmation",
            "metrics",
            "full_record",
        ),
        "validation_common_interior": (
            "validation_confirmation",
            "metrics",
            "common_causal_interior",
        ),
    }
    for kind in ("mp", "gmp"):
        per_model: dict[str, float] = {}
        for label, path in paths.items():
            a0_value: Any = results["a0"][kind]
            a1_value: Any = results["a1"][kind]
            for key in path:
                a0_value = a0_value[key]
                a1_value = a1_value[key]
            a0_score = float(a0_value["complex_nmse_pooled_db"])
            a1_score = float(a1_value["complex_nmse_pooled_db"])
            per_model[f"{label}_a1_minus_a0_db"] = a1_score - a0_score
        comparisons[kind] = per_model
    return {
        "definition": "A1 NMSE dB minus A0 NMSE dB; negative favors A1",
        "models": comparisons,
    }


def _evaluate_decision_rule(
    comparison_deltas: dict[str, object],
    decision_rule: dict[str, Any],
) -> dict[str, object]:
    """Evaluate the preregistered rule after every fixed fit has completed."""

    models = comparison_deltas.get("models")
    if not isinstance(models, dict):
        raise ValueError("comparison deltas have no models object")
    specifications = (
        (
            "gmp_train_oof_common_causal_interior",
            "gmp",
            "train_oof_common_interior_a1_minus_a0_db",
            float(decision_rule["gmp_a1_minus_a0_max_db"]),
        ),
        (
            "gmp_validation_common_causal_interior",
            "gmp",
            "validation_common_interior_a1_minus_a0_db",
            float(decision_rule["gmp_a1_minus_a0_max_db"]),
        ),
        (
            "mp_train_oof_common_causal_interior_corroboration",
            "mp",
            "train_oof_common_interior_a1_minus_a0_db",
            float(
                decision_rule["mp_corroboration_a1_minus_a0_max_db"]
            ),
        ),
        (
            "mp_validation_common_causal_interior_corroboration",
            "mp",
            "validation_common_interior_a1_minus_a0_db",
            float(
                decision_rule["mp_corroboration_a1_minus_a0_max_db"]
            ),
        ),
        (
            "gmp_train_oof_full_record_no_sign_reversal",
            "gmp",
            "train_oof_full_record_a1_minus_a0_db",
            0.0,
        ),
        (
            "gmp_validation_full_record_no_sign_reversal",
            "gmp",
            "validation_full_record_a1_minus_a0_db",
            0.0,
        ),
        (
            "mp_train_oof_full_record_no_sign_reversal",
            "mp",
            "train_oof_full_record_a1_minus_a0_db",
            0.0,
        ),
        (
            "mp_validation_full_record_no_sign_reversal",
            "mp",
            "validation_full_record_a1_minus_a0_db",
            0.0,
        ),
    )
    predicates: dict[str, dict[str, object]] = {}
    for name, model, metric, threshold in specifications:
        model_deltas = models.get(model)
        if not isinstance(model_deltas, dict) or metric not in model_deltas:
            raise ValueError(
                f"comparison deltas lack decision metric {model}.{metric}"
            )
        actual = float(model_deltas[metric])
        finite = math.isfinite(actual)
        predicates[name] = {
            "model": model,
            "metric": metric,
            "actual_a1_minus_a0_db": actual,
            "operator": "<=",
            "threshold_db": threshold,
            "metric_value_is_finite": finite,
            "passed": bool(finite and actual <= threshold),
        }
    all_passed = all(
        bool(predicate["passed"]) for predicate in predicates.values()
    )
    return {
        "rule_evaluated_after_all_fixed_results": True,
        "rule_used_for_fit_or_tuning": False,
        "predicate_count": len(predicates),
        "predicates": predicates,
        "all_predicates_passed": all_passed,
        "recommended_protocol_variant": "a1" if all_passed else "a0",
        "fallback_variant_when_any_predicate_fails": "a0",
        "recommendation_scope": (
            "post-fit protocol recommendation for this sensitivity "
            "experiment only"
        ),
        "accepted_a1_caveat": (
            "even a recommended A1 remains a sensitivity protocol and is "
            "not proven feedback-path de-embedding"
        ),
    }


def evaluate_from_config(config_path: str | Path) -> dict[str, Any]:
    """Run the sealed train/validation-only A0/A1 sensitivity experiment."""

    source_config = Path(config_path).resolve()
    initial_config_sha256 = file_sha256(source_config)
    config = _load_runner_config(source_config)
    _verify_hash(
        source_config,
        initial_config_sha256,
        label="sensitivity runner config after parsing",
    )

    source_path = Path(__file__).resolve()
    repository_root = source_path.parents[1]

    def repository_path(value: str | Path) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (repository_root / candidate).resolve()

    selection_path = repository_path(config["selection_manifest"])
    _verify_hash(
        selection_path,
        str(config["selection_manifest_sha256"]),
        label="MP selection manifest",
    )
    initial_selection_sha256 = file_sha256(selection_path)

    selection = _load_json_object(
        selection_path,
        name="MP selection manifest",
    )
    _verify_hash(
        selection_path,
        initial_selection_sha256,
        label="MP selection manifest after parsing",
    )
    mp_recipe, selection_protocol = _parse_selection_manifest(selection)
    gmp_recipe = _parse_fixed_gmp_recipe(
        config["fixed_gmp_recipe"]
    )

    source_dependency_paths = {
        "experiments/evaluate_fractional_alignment_sensitivity.py": (
            source_path
        ),
        "baseline/alignment.py": repository_root / "baseline" / "alignment.py",
        "baseline/complexity.py": (
            repository_root / "baseline" / "complexity.py"
        ),
        "baseline/fractional_alignment.py": (
            repository_root / "baseline" / "fractional_alignment.py"
        ),
        "baseline/gmp_pa.py": repository_root / "baseline" / "gmp_pa.py",
        "baseline/metrics.py": repository_root / "baseline" / "metrics.py",
        "baseline/pa_models.py": (
            repository_root / "baseline" / "pa_models.py"
        ),
        "baseline/train_spline.py": (
            repository_root / "baseline" / "train_spline.py"
        ),
    }
    initial_source_hashes = _snapshot_hashes(source_dependency_paths)

    operation_budget = int(
        config["max_real_multiplications_per_sample"]
    )
    mp_operation_count = memory_polynomial_inference_cost(
        mp_recipe.orders,
        mp_recipe.delays,
    )
    gmp_counter_model = GeneralizedMemoryPolynomialPA(
        gmp_recipe.config,
        np.zeros(
            gmp_recipe.config.coefficient_count,
            dtype=np.complex128,
        ),
    )
    gmp_operation_count = gmp_counter_model.operation_count
    preflight_operation_counts = {
        "mp": mp_operation_count.to_dict(),
        "gmp": gmp_operation_count.to_dict(),
    }
    for kind, operation_count in (
        ("mp", mp_operation_count),
        ("gmp", gmp_operation_count),
    ):
        real_multiplications = int(operation_count.real_multiplications)
        if real_multiplications >= operation_budget:
            raise ValueError(
                f"{kind.upper()} recipe requires {real_multiplications} "
                "real multiplications/sample and violates the exclusive "
                f"<{operation_budget} deployment limit"
            )

    output_directory = repository_path(config["output_dir"])
    report_path = output_directory / "fractional_alignment_sensitivity.json"
    report_temporary_path = (
        output_directory
        / ".fractional_alignment_sensitivity.publishing.json"
    )
    lock_path = (
        output_directory / ".fractional_alignment_sensitivity.lock"
    )
    model_paths = {
        (variant, kind): output_directory / f"{variant}_{kind}_pa.npz"
        for variant in ("a0", "a1")
        for kind in ("mp", "gmp")
    }
    model_temporary_paths = {
        (variant, kind): (
            output_directory / f".{variant}_{kind}_pa.publishing.npz"
        )
        for variant in ("a0", "a1")
        for kind in ("mp", "gmp")
    }
    owned_paths = (
        report_path,
        report_temporary_path,
        *model_paths.values(),
        *model_temporary_paths.values(),
    )
    existing = [
        path
        for path in (*owned_paths, lock_path)
        if _path_entry_exists(path)
    ]
    if existing:
        raise FileExistsError(
            "immutable sensitivity bundle has an existing owned "
            "lock/final/temp artifact: "
            + ", ".join(str(path) for path in existing)
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_payload = _acquire_bundle_lock(lock_path)
    initial_lock_sha256 = file_sha256(lock_path)
    appeared_after_lock = [
        path for path in owned_paths if _path_entry_exists(path)
    ]
    if appeared_after_lock:
        raise FileExistsError(
            "immutable sensitivity bundle acquired an owned final/temp "
            "artifact while taking its lock: "
            + ", ".join(str(path) for path in appeared_after_lock)
        )

    dataset = repository_path(config["dataset"])
    legacy_dataset_value = selection.get("dataset")
    legacy_dataset_path = (
        repository_path(legacy_dataset_value)
        if isinstance(legacy_dataset_value, str)
        and legacy_dataset_value.strip()
        else None
    )
    dataset_hashes = selection.get("dataset_files_sha256")
    if not isinstance(dataset_hashes, dict):
        raise ValueError("selection manifest has no dataset file hashes")
    if any(Path(name).name.startswith("test_") for name in dataset_hashes):
        raise ValueError(
            "selection manifest used by sensitivity runner must not contain "
            "test-file hashes"
        )
    required_dataset_files = {
        "train_input.csv",
        "train_output.csv",
        "val_input.csv",
        "val_output.csv",
        "spec.json",
    }
    missing_hashes = required_dataset_files - set(dataset_hashes)
    if missing_hashes:
        raise ValueError(
            f"selection manifest lacks dataset hashes: {sorted(missing_hashes)}"
        )
    dataset_paths: dict[str, Path] = {}
    frozen_dataset_hashes: dict[str, str] = {}
    for name in sorted(required_dataset_files):
        expected = str(dataset_hashes[name])
        dataset_paths[name] = dataset / name
        frozen_dataset_hashes[name] = expected
    _verify_frozen_hashes(
        dataset_paths,
        frozen_dataset_hashes,
        scope="pre-load dataset",
    )

    # These are the only waveform split accesses in this module.
    train_input, train_output = load_split_pair(dataset, "train")
    validation_input, validation_output = load_split_pair(dataset, "val")
    dataset_spec = load_dataset_spec(dataset)
    _verify_frozen_hashes(
        dataset_paths,
        frozen_dataset_hashes,
        scope="post-load dataset",
    )
    original_nperseg = int(selection_protocol["nperseg"])
    if int(dataset_spec.get("nperseg", -1)) != original_nperseg:
        raise ValueError("dataset spec and selected protocol nperseg differ")

    filter_config = config["alignment_filter"]
    a0_transform = freeze_fractional_alignment(
        FractionalAlignmentConfig(
            observed_delay_samples=0.0,
            tap_count=filter_config["tap_count"],
            kaiser_beta=filter_config["kaiser_beta"],
        )
    )
    diagnostic_delay = float(
        selection_protocol["fractional_delay_estimate_samples"]
    )
    a1_transform = freeze_fractional_alignment(
        FractionalAlignmentConfig(
            observed_delay_samples=diagnostic_delay,
            tap_count=filter_config["tap_count"],
            kaiser_beta=filter_config["kaiser_beta"],
        )
    )
    effective_nperseg = original_nperseg - 2 * a0_transform.guard_samples
    if effective_nperseg <= 1:
        raise ValueError("fractional alignment guard consumes original nperseg")

    train_variants = prepare_alignment_variants(
        train_input,
        train_output,
        original_nperseg=original_nperseg,
        a0_transform=a0_transform,
        a1_transform=a1_transform,
    )
    validation_variants = prepare_alignment_variants(
        validation_input,
        validation_output,
        original_nperseg=original_nperseg,
        a0_transform=a0_transform,
        a1_transform=a1_transform,
    )
    for split_variants in (train_variants, validation_variants):
        if split_variants["a0"].effective_frame_lengths != split_variants[
            "a1"
        ].effective_frame_lengths:
            raise RuntimeError("A0/A1 effective frame lengths differ")
        if split_variants["a0"].effective_frame_lengths[0] != effective_nperseg:
            raise RuntimeError("first transformed frame has unexpected length")

    common_warmup = max(
        mp_recipe.causal_warmup_samples,
        gmp_recipe.causal_warmup_samples,
    )
    all_effective_lengths = (
        train_variants["a0"].effective_frame_lengths
        + validation_variants["a0"].effective_frame_lengths
    )
    if common_warmup >= min(all_effective_lengths):
        raise ValueError("common model warmup consumes a transformed frame")

    recipes: dict[ModelKind, FixedMPRecipe | FixedGMPRecipe] = {
        "mp": mp_recipe,
        "gmp": gmp_recipe,
    }
    results: dict[str, dict[str, Any]] = {"a0": {}, "a1": {}}
    fitted_models: dict[
        tuple[str, str],
        MemoryPolynomialPA | GeneralizedMemoryPolynomialPA,
    ] = {}
    for variant in ("a0", "a1"):
        for kind in ("mp", "gmp"):
            result, model = _evaluate_fixed_recipe(
                kind,
                recipes[kind],
                train_variants[variant],
                validation_variants[variant],
                effective_nperseg=effective_nperseg,
                common_warmup_samples=common_warmup,
            )
            results[variant][kind] = result
            fitted_models[(variant, kind)] = model

    comparison_deltas = _comparison_deltas(results)
    decision_rule_evaluation = _evaluate_decision_rule(
        comparison_deltas,
        config["decision_rule"],
    )

    # A mutation during fitting invalidates the complete bundle.  This check
    # happens before publishing even the first model artifact.
    _verify_hash(
        source_config,
        initial_config_sha256,
        label="sensitivity runner config before model publication",
    )
    _verify_hash(
        selection_path,
        initial_selection_sha256,
        label="MP selection manifest before model publication",
    )
    _verify_frozen_hashes(
        dataset_paths,
        frozen_dataset_hashes,
        scope="pre-publication dataset",
    )
    _verify_frozen_hashes(
        source_dependency_paths,
        initial_source_hashes,
        scope="pre-publication source dependency",
    )
    _verify_owned_bundle_lock(
        lock_path,
        lock_payload,
        scope="before model publication",
    )
    appeared = [path for path in owned_paths if _path_entry_exists(path)]
    if appeared:
        raise FileExistsError(
            "immutable sensitivity bundle acquired an owned final/temp "
            "artifact during evaluation: "
            + ", ".join(str(path) for path in appeared)
        )

    model_artifacts: dict[str, dict[str, str]] = {}
    for (variant, kind), model in fitted_models.items():
        path = model_paths[(variant, kind)]
        _atomic_publish_model(
            model,
            temporary_path=model_temporary_paths[(variant, kind)],
            final_path=path,
        )
        key = f"{variant}_{kind}"
        model_artifacts[key] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        results[variant][kind]["frozen_full_training_model"] = (
            model_artifacts[key]
        )

    quoted_config = shlex.quote(str(source_config))
    report: dict[str, Any] = {
        "schema_version": 1,
        "task": "fractional_alignment_sensitivity_a0_a1",
        "scope": "measured_forward_pa_identification",
        "interpretation": (
            "sensitivity analysis only; A1 does not establish that the "
            "correlation-derived delay is measurement-path ground truth"
        ),
        "preregistration": {
            "variants": ["a0_zero_shift", "a1_train_frozen_diagnostic"],
            "architecture_tuning_performed": False,
            "delay_tuning_performed": False,
            "ridge_tuning_performed": False,
            "validation_role": {
                "a0_a1_runner_tuning": False,
                "mp": (
                    "the MP recipe was already selected on this same "
                    "validation split, so its A0/A1 validation result is "
                    "corroborative but not independent model-selection "
                    "evidence"
                ),
                "gmp": (
                    "the fixed GMP recipe was preregistered in the runner "
                    "config and is not tuned inside this runner"
                ),
            },
        },
        "decision_rule": {
            **config["decision_rule"],
            "contract_role": (
                "validated preregistration metadata; does not alter fits, "
                "rerun candidates, or tune thresholds"
            ),
            "full_record_same_sign_definition": (
                "A1-minus-A0 full-record NMSE must be <= 0 dB for MP and "
                "fixed GMP on both train OOF and validation"
            ),
            "application": (
                "A1 is eligible only when fixed-GMP train-OOF and validation "
                "common-interior deltas are each <= -0.25 dB, MP deltas on "
                "both splits are each <= 0 dB, and no corresponding "
                "full-record delta changes sign; otherwise retain A0"
            ),
        },
        "accessed_splits": ["train", "validation"],
        "test_split_accessed": False,
        "test_file_hashes_recorded": False,
        "config": str(source_config),
        "config_sha256": initial_config_sha256,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": initial_selection_sha256,
        "dataset": str(dataset),
        "dataset_resolution": {
            "runner_config_value": str(config["dataset"]),
            "runner_config_resolved_path": str(dataset),
            "selection_manifest_legacy_value": legacy_dataset_value,
            "selection_manifest_legacy_resolved_path": (
                str(legacy_dataset_path)
                if legacy_dataset_path is not None
                else None
            ),
            "authoritative_path": "runner_config_resolved_path",
            "legacy_path_used_for_io": False,
            "override_by_hash_explanation": (
                "the runner-config dataset path is authoritative for I/O; "
                "its train/validation/spec file identities must exactly "
                "match the hashes frozen in the selection manifest, so a "
                "stale absolute manifest path may be portably overridden "
                "without changing the selected data"
            ),
        },
        "dataset_label": selection.get("dataset_label"),
        "dataset_spec": dataset_spec,
        "dataset_files_sha256": frozen_dataset_hashes,
        "deployment_operation_budget": {
            "metric": "real_multiplications_per_complex_sample",
            "limit": operation_budget,
            "operator": "<",
            "pre_waveform_enforced": True,
            "fixed_recipe_counts": preflight_operation_counts,
        },
        "framing": {
            "original_nperseg": original_nperseg,
            "fir_guard_samples_each_side": a0_transform.guard_samples,
            "effective_nperseg": effective_nperseg,
            "policy": (
                "split each waveform by original nperseg, independently "
                "transform each frame, then reset each PA model at every "
                "effective frame"
            ),
            "train": train_variants["a0"].summary(),
            "validation": validation_variants["a0"].summary(),
        },
        "train_frozen_delay_source": {
            "source": (
                "selection_manifest.protocol."
                "fractional_delay_estimate_samples"
            ),
            "estimated_delay_samples": diagnostic_delay,
            "peak_score": selection_protocol.get(
                "fractional_delay_peak_score"
            ),
            "reliable_flag": selection_protocol.get(
                "fractional_delay_reliable"
            ),
            "reestimated_by_this_runner": False,
        },
        "transforms": {
            "a0": a0_transform.to_metadata(),
            "a1": a1_transform.to_metadata(),
            "input_support_bit_identical_between_variants": True,
            "guard_identical_between_variants": True,
        },
        "fixed_model_recipes": {
            "mp": mp_recipe.to_dict(),
            "gmp": gmp_recipe.to_dict(),
            "gmp_recipe_storage": (
                "inline in hash-recorded sensitivity runner config"
            ),
            "same_recipe_for_a0_and_a1": True,
            "common_causal_warmup_samples_per_original_frame": common_warmup,
        },
        "results": results,
        "a1_minus_a0": comparison_deltas,
        "decision_rule_evaluation": decision_rule_evaluation,
        "model_artifacts": model_artifacts,
        "determinism": {
            "stochastic_fitting": False,
            "seed": None,
            "variant_iteration_order": ["a0", "a1"],
            "model_iteration_order": ["mp", "gmp"],
            "oof_iteration_order": "ascending original training frame index",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_sha256": {
            **initial_source_hashes,
        },
        "input_integrity": {
            "hashes_frozen_before_waveform_load": True,
            "dataset_reverified_after_waveform_load": True,
            "all_inputs_reverified_before_model_publication": True,
            "report_uses_initial_frozen_hashes": True,
        },
        "publication": {
            "immutable_bundle": True,
            "atomic_single_writer_lock": {
                "path": str(lock_path),
                "owner_pid": os.getpid(),
                "owner_payload_sha256": initial_lock_sha256,
                "creation": (
                    "os.open(O_CREAT|O_EXCL|O_WRONLY, mode=0600) before "
                    "waveform access"
                ),
                "verified_before_model_publication": True,
                "verified_before_report_publication": True,
                "success_lifecycle": (
                    "remove only after atomic report publication and one "
                    "final exact-owner verification"
                ),
                "failure_lifecycle": (
                    "leave lock in place as an explicit incomplete-run "
                    "marker; manual review is required before retry"
                ),
            },
            "report_published_last": True,
            "per_artifact_protocol": (
                "same-directory uniquely-owned publishing temp followed by "
                "os.replace; any pre-existing final or exact temp is fatal"
            ),
            "automatic_cleanup_on_failure": False,
        },
        "commands": {
            "invocation": " ".join(shlex.quote(value) for value in sys.argv),
            "reproduce": (
                "python -m "
                "experiments.evaluate_fractional_alignment_sensitivity "
                f"--config {quoted_config}"
            ),
        },
        "report_path": str(report_path),
    }

    # Detect input or just-published model mutation before the report becomes
    # the bundle's completion marker.
    _verify_hash(
        source_config,
        initial_config_sha256,
        label="sensitivity runner config before report publication",
    )
    _verify_hash(
        selection_path,
        initial_selection_sha256,
        label="MP selection manifest before report publication",
    )
    _verify_frozen_hashes(
        dataset_paths,
        frozen_dataset_hashes,
        scope="pre-report dataset",
    )
    _verify_frozen_hashes(
        source_dependency_paths,
        initial_source_hashes,
        scope="pre-report source dependency",
    )
    _verify_frozen_hashes(
        {
            key: Path(value["path"])
            for key, value in model_artifacts.items()
        },
        {
            key: value["sha256"]
            for key, value in model_artifacts.items()
        },
        scope="pre-report frozen model artifact",
    )
    _verify_owned_bundle_lock(
        lock_path,
        lock_payload,
        scope="before report publication",
    )
    _atomic_publish_json(
        report,
        temporary_path=report_temporary_path,
        final_path=report_path,
    )
    _verify_owned_bundle_lock(
        lock_path,
        lock_payload,
        scope="after report publication",
    )
    lock_path.unlink()
    return report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a train/validation-only, fixed-architecture A0/A1 "
            "fractional-alignment sensitivity experiment."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    report = evaluate_from_config(args.config)
    for model in ("mp", "gmp"):
        deltas = report["a1_minus_a0"]["models"][model]
        print(
            f"{model.upper()} A1-A0:",
            "train OOF interior="
            f"{deltas['train_oof_common_interior_a1_minus_a0_db']:.6f} dB,",
            "validation interior="
            f"{deltas['validation_common_interior_a1_minus_a0_db']:.6f} dB",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
