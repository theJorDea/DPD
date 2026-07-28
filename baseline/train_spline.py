"""Train and validation-select the complex linear-spline ILA baseline.

This command deliberately does not open ``test_input.csv`` or
``test_output.csv``.  Candidate configuration is selected only on the
validation split.  Test evaluation is a separate, explicit action provided by
``baseline.evaluate_spline``.

Two validation questions are kept separate:

1. ``inverse_postdistorter_diagnostic`` checks
   ``D(measured_y / gain) ~= known_x``.  It is useful for ILA calibration but
   is not a DPD deployment score.
2. ``surrogate_only_predistorter_cascade`` checks
   ``PA_surrogate(D(x_desired)) ~= gain*x_desired``.  It uses the correct DPD
   input direction, but remains a surrogate-only result.

No result produced by this module is a physical-PA DPD measurement.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .alignment import (
    align_and_estimate_gain,
    complex_ls_gain,
    fractional_delay_diagnostic,
    overlap_for_delay,
)
from .complex_spline_dpd import (
    ComplexLinearSplineDPD,
    fit_ila_postdistorter,
)
from .metrics import (
    nmse_pooled_db,
    papr_db,
    peak_amplitude,
    time_domain_rms_evm_db,
)
from .pa_models import (
    MemoryPolynomialFitDiagnostics,
    MemoryPolynomialPA,
    fit_memory_polynomial_pa,
    segmented_steady_state_mask,
)


def load_complex_iq_csv(path: str | Path) -> np.ndarray:
    """Load an OpenDPD two-column ``I,Q`` CSV using NumPy only."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    columns = tuple(part.strip().lower() for part in header.split(","))
    if columns != ("i", "q"):
        raise ValueError(
            f"{source} must have exactly the header I,Q; found {header!r}"
        )
    values = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError(f"{source} must contain at least one two-column IQ row")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{source} contains NaN or infinite values")
    return values[:, 0] + 1j * values[:, 1]


def load_split_pair(
    dataset_directory: str | Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one named OpenDPD split without touching any other split."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    directory = Path(dataset_directory)
    pa_input = load_complex_iq_csv(directory / f"{split}_input.csv")
    pa_output = load_complex_iq_csv(directory / f"{split}_output.csv")
    if pa_input.shape != pa_output.shape:
        raise ValueError(f"{split} input and output lengths differ")
    return pa_input, pa_output


def align_split_pair(
    pa_input: np.ndarray,
    pa_output: np.ndarray,
    *,
    delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one frozen integer delay convention to any split.

    The delay is estimated from training only and then reused verbatim for
    validation/test.  This prevents each split from receiving a separately
    optimized crop.  A zero delay is a no-op but still goes through the same
    explicit path.
    """

    return overlap_for_delay(pa_input, pa_output, int(delay))


def load_dataset_spec(dataset_directory: str | Path) -> dict[str, Any]:
    """Load optional OpenDPD ``spec.json`` without adding defaults silently."""

    path = Path(dataset_directory) / "spec.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_list(text: str, *, minimum: int, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if not values or any(value < minimum for value in values):
        raise argparse.ArgumentTypeError(
            f"{name} entries must all be integers >= {minimum}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} entries must be unique")
    return values


def _parse_float_list(
    text: str,
    *,
    minimum: float,
    name: str,
) -> tuple[float, ...]:
    try:
        values = tuple(
            float(part.strip()) for part in text.split(",") if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from error
    if (
        not values
        or any(not np.isfinite(value) or value < minimum for value in values)
    ):
        raise argparse.ArgumentTypeError(
            f"{name} entries must all be finite and >= {minimum}"
        )
    return values


def _parse_strategy_list(text: str) -> tuple[str, ...]:
    allowed = {
        "uniform_amplitude",
        "uniform_power",
        "quantile",
        "compression_aware",
    }
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = set(values) - allowed
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            "knot strategies must be comma-separated members of "
            f"{sorted(allowed)}; unknown={sorted(unknown)}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("knot strategies must be unique")
    return values


def gain_from_training(
    pa_input: np.ndarray,
    pa_output: np.ndarray,
    *,
    strategy: str,
    explicit_gain: complex | None = None,
) -> tuple[complex, str]:
    """Estimate or validate a target gain using the training split only."""

    if strategy == "complex_ls":
        gain = complex_ls_gain(pa_input, pa_output)
        definition = (
            "sum(conj(x_train)*y_train)/sum(abs(x_train)**2); "
            "training split only"
        )
    elif strategy == "opendpd_peak":
        input_peak = float(np.max(np.abs(pa_input)))
        if input_peak <= 0.0:
            raise ValueError("peak gain is undefined for zero training input")
        gain = complex(float(np.max(np.abs(pa_output))) / input_peak, 0.0)
        definition = (
            "max(abs(y_train))/max(abs(x_train)); training split only; "
            "matches vendor/OpenDPD/utils/util.py set_target_gain"
        )
    elif strategy == "explicit":
        if explicit_gain is None:
            raise ValueError("explicit gain strategy requires a supplied gain")
        gain = complex(explicit_gain)
        definition = "fixed user-supplied complex gain"
    else:
        raise ValueError(f"unknown gain strategy: {strategy}")
    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("target gain must be finite and non-zero")
    return gain, definition


def _paired_time_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    warmup_samples: int = 0,
    segment_length: int | None = None,
) -> dict[str, Any]:
    estimate_array = np.asarray(estimate, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    if estimate_array.shape != reference_array.shape:
        raise ValueError("paired metric inputs must have the same shape")
    if not isinstance(warmup_samples, int) or warmup_samples < 0:
        raise ValueError("warmup_samples must be a non-negative integer")
    if segment_length is None:
        if warmup_samples >= estimate_array.size:
            raise ValueError("warmup_samples consumes the full record")
        score_mask = np.arange(estimate_array.size) >= warmup_samples
        warmup_policy = "once_at_record_start"
    else:
        score_mask = segmented_steady_state_mask(
            estimate_array.size,
            segment_length=segment_length,
            warmup_samples=warmup_samples,
        )
        warmup_policy = "discard_at_every_segment_start"
    estimate_scored = estimate_array[score_mask]
    reference_scored = reference_array[score_mask]
    error = estimate_scored - reference_scored
    mse = float(np.mean(np.abs(error) ** 2))
    reference_power = float(np.mean(np.abs(reference_scored) ** 2))
    if reference_power <= 0.0:
        raise ValueError("metric reference has zero power")
    return {
        "sample_count": int(estimate_scored.size),
        "discarded_causal_warmup_samples_total": int(
            estimate_array.size - estimate_scored.size
        ),
        "causal_warmup_samples_per_segment": warmup_samples,
        "segment_length": segment_length,
        "warmup_policy": warmup_policy,
        "mse": mse,
        "relative_error_power": mse / reference_power,
        "complex_nmse_pooled_db": nmse_pooled_db(
            estimate_scored,
            reference_scored,
        ),
        "time_domain_rms_evm_db": time_domain_rms_evm_db(
            estimate_scored,
            reference_scored,
        ),
    }


def _waveform_output_metrics(signal: np.ndarray) -> dict[str, float]:
    return {
        "papr_db": papr_db(signal),
        "maximum_amplitude": peak_amplitude(signal),
        "average_power": float(np.mean(np.abs(signal) ** 2)),
    }


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return _json_ready(value._asdict())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.complexfloating, complex)):
        number = complex(value)
        return {"real": float(number.real), "imag": float(number.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, report: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(
            _json_ready(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a complex ILA spline on train, select on val, and never load "
            "test. Invoke as: python -m baseline.train_spline ..."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=False,
        help="OpenDPD split-CSV dataset directory",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "optional JSON configuration; explicit command-line values take "
            "precedence"
        ),
    )
    parser.add_argument(
        "--knot-counts",
        type=lambda value: _parse_int_list(
            value,
            minimum=2,
            name="knot counts",
        ),
        default=(16,),
        help="comma-separated K values (default: 16)",
    )
    parser.add_argument(
        "--knot-strategies",
        type=_parse_strategy_list,
        default=("uniform_amplitude",),
        help="comma-separated placement strategies",
    )
    parser.add_argument(
        "--ridges",
        type=lambda value: _parse_float_list(
            value,
            minimum=0.0,
            name="ridge values",
        ),
        default=(1e-8,),
        help="comma-separated normalized ridge strengths",
    )
    parser.add_argument(
        "--smoothnesses",
        type=lambda value: _parse_float_list(
            value,
            minimum=0.0,
            name="smoothness values",
        ),
        default=(0.0,),
        help="comma-separated normalized second-difference penalties",
    )
    parser.add_argument("--compression-power", type=float, default=2.0)
    parser.add_argument(
        "--alignment-max-delay",
        type=int,
        default=32,
        help=(
            "training-only integer-delay search radius; the selected delay is "
            "frozen for validation/test (default: 32 samples)"
        ),
    )
    parser.add_argument(
        "--alignment-delay",
        type=int,
        help=(
            "optional pre-established integer delay. If supplied, no delay "
            "search is performed"
        ),
    )
    parser.add_argument(
        "--gain-strategy",
        choices=("complex_ls", "opendpd_peak", "explicit"),
        default="complex_ls",
    )
    parser.add_argument("--gain-real", type=float)
    parser.add_argument("--gain-imag", type=float, default=0.0)
    parser.add_argument(
        "--fit-pa-surrogate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "fit a train-only memory-polynomial PA for a correctly directed, "
            "explicitly surrogate-only validation cascade"
        ),
    )
    parser.add_argument(
        "--pa-orders",
        type=lambda value: _parse_int_list(value, minimum=1, name="PA orders"),
        default=(1, 3, 5, 7),
    )
    parser.add_argument(
        "--pa-delays",
        type=lambda value: _parse_int_list(value, minimum=0, name="PA delays"),
        default=(0, 1, 2),
    )
    parser.add_argument("--pa-ridge", type=float, default=1e-8)
    parser.add_argument(
        "--selection-metric",
        choices=("auto", "inverse_nmse", "surrogate_cascade_nmse"),
        default="auto",
        help="validation-only metric; lower dB is better",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this command's existing model/report artifacts",
    )
    return parser


def _config_defaults(config_path: Path) -> dict[str, Any]:
    """Translate a JSON config into argparse defaults with strict keys.

    Keeping this translation here means a sweep can be launched from a
    committed, machine-readable config while still allowing one-off CLI
    overrides.  Lists are converted to tuples so the product order is stable.
    """

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("--config must contain one JSON object")
    allowed = {
        "dataset",
        "output_dir",
        "knot_counts",
        "knot_strategies",
        "ridges",
        "smoothnesses",
        "compression_power",
        "alignment_max_delay",
        "alignment_delay",
        "gain_strategy",
        "gain_real",
        "gain_imag",
        "fit_pa_surrogate",
        "pa_orders",
        "pa_delays",
        "pa_ridge",
        "selection_metric",
        "overwrite",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            f"unknown train_spline config keys: {sorted(unknown)}"
        )
    defaults: dict[str, Any] = dict(config)
    for key in (
        "knot_counts",
        "knot_strategies",
        "ridges",
        "smoothnesses",
        "pa_orders",
        "pa_delays",
    ):
        if key in defaults:
            value = defaults[key]
            if isinstance(value, str):
                # Reuse the same validation as the command-line parser.
                if key in {"knot_counts", "pa_orders", "pa_delays"}:
                    minimum = 2 if key == "knot_counts" else (1 if key == "pa_orders" else 0)
                    defaults[key] = _parse_int_list(
                        value,
                        minimum=minimum,
                        name=key,
                    )
                elif key == "knot_strategies":
                    defaults[key] = _parse_strategy_list(value)
                else:
                    defaults[key] = _parse_float_list(
                        value,
                        minimum=0.0,
                        name=key,
                    )
            else:
                defaults[key] = tuple(value)
    for key in ("dataset", "output_dir"):
        if key in defaults:
            defaults[key] = Path(defaults[key])
    return defaults


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _argument_parser()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(raw_argv)
    if pre_args.config is not None:
        parser.set_defaults(**_config_defaults(pre_args.config))
    args = parser.parse_args(raw_argv)
    if args.dataset is None or args.output_dir is None:
        parser.error("--dataset and --output-dir are required (or must be in --config)")
    if args.compression_power <= 1.0:
        parser.error("--compression-power must be greater than one")
    if args.alignment_max_delay < 0:
        parser.error("--alignment-max-delay must be non-negative")
    if (
        args.alignment_delay is not None
        and abs(args.alignment_delay) > args.alignment_max_delay
        and args.alignment_max_delay != 0
    ):
        parser.error(
            "--alignment-delay must lie within --alignment-max-delay when "
            "a search radius is configured"
        )
    if args.pa_ridge < 0.0 or not np.isfinite(args.pa_ridge):
        parser.error("--pa-ridge must be finite and non-negative")
    return args


def _fit_surrogate(
    args: argparse.Namespace,
    train_input: np.ndarray,
    train_output: np.ndarray,
    *,
    segment_length: int | None,
) -> tuple[
    MemoryPolynomialPA | None,
    MemoryPolynomialFitDiagnostics | None,
    float | None,
]:
    if not args.fit_pa_surrogate:
        return None, None, None
    start = time.perf_counter()
    model, diagnostics = fit_memory_polynomial_pa(
        train_input,
        train_output,
        orders=args.pa_orders,
        delays=args.pa_delays,
        ridge=args.pa_ridge,
        segment_length=segment_length,
    )
    return model, diagnostics, time.perf_counter() - start


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run train/validation selection and write auditable artifacts."""

    dataset = args.dataset.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / "spline_model.npz"
    report_path = output_directory / "training_report.json"
    trials_path = output_directory / "validation_trials.json"
    surrogate_path = output_directory / "pa_surrogate.npz"
    manifest_path = output_directory / "deployment_manifest.json"
    owned_paths = [model_path, report_path, trials_path, manifest_path]
    if args.fit_pa_surrogate:
        owned_paths.append(surrogate_path)
    existing = [path for path in owned_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    # Deliberately load only train and validation.  The test filenames do not
    # occur in this control path.  Alignment is estimated once on train and
    # the integer crop is frozen for every later split.
    raw_train_input, raw_train_output = load_split_pair(dataset, "train")
    raw_val_input, raw_val_output = load_split_pair(dataset, "val")
    spec = load_dataset_spec(dataset)

    if args.alignment_delay is None:
        _, _, alignment_delay, _ = align_and_estimate_gain(
            raw_train_input,
            raw_train_output,
            max_abs_delay=args.alignment_max_delay,
        )
        fractional_diagnostic = fractional_delay_diagnostic(
            raw_train_input,
            raw_train_output,
            args.alignment_max_delay,
        )
    else:
        alignment_delay = int(args.alignment_delay)
        # Keep the diagnostic visible even when an operator supplied a known
        # delay.  It is never used to modify the selected integer delay.
        fractional_diagnostic = fractional_delay_diagnostic(
            raw_train_input,
            raw_train_output,
            max(args.alignment_max_delay, abs(alignment_delay)),
        )
    train_input, train_output = align_split_pair(
        raw_train_input,
        raw_train_output,
        delay=alignment_delay,
    )
    val_input, val_output = align_split_pair(
        raw_val_input,
        raw_val_output,
        delay=alignment_delay,
    )

    explicit_gain = None
    if args.gain_strategy == "explicit":
        if args.gain_real is None:
            raise ValueError("--gain-real is required for explicit gain")
        explicit_gain = complex(args.gain_real, args.gain_imag)
    elif args.gain_real is not None:
        raise ValueError("--gain-real/--gain-imag are only valid for explicit gain")
    gain, gain_definition = gain_from_training(
        train_input,
        train_output,
        strategy=args.gain_strategy,
        explicit_gain=explicit_gain,
    )

    segment_length = (
        int(spec["nperseg"])
        if "nperseg" in spec and int(spec["nperseg"]) > 0
        else None
    )
    pa_surrogate, pa_diagnostics, pa_fit_seconds = _fit_surrogate(
        args,
        train_input,
        train_output,
        segment_length=segment_length,
    )
    if (
        args.selection_metric == "surrogate_cascade_nmse"
        and pa_surrogate is None
    ):
        raise ValueError(
            "surrogate_cascade_nmse selection requires --fit-pa-surrogate"
        )
    selection_metric = args.selection_metric
    if selection_metric == "auto":
        selection_metric = (
            "surrogate_cascade_nmse"
            if pa_surrogate is not None
            else "inverse_nmse"
        )

    trials: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_model: ComplexLinearSplineDPD | None = None
    best_trial: dict[str, Any] | None = None
    configurations: Iterable[tuple[str, int, float, float]] = itertools.product(
        args.knot_strategies,
        args.knot_counts,
        args.ridges,
        args.smoothnesses,
    )
    for strategy, knot_count, ridge, smoothness in configurations:
        start = time.perf_counter()
        model, diagnostics = fit_ila_postdistorter(
            train_input,
            train_output,
            gain,
            knot_count=knot_count,
            knot_strategy=strategy,
            ridge=ridge,
            smoothness=smoothness,
            compression_power=args.compression_power,
        )
        fit_seconds = time.perf_counter() - start
        normalized_val_output = val_output / gain
        inverse_estimate = model.predict(normalized_val_output)
        inverse_metrics = _paired_time_metrics(inverse_estimate, val_input)
        inverse_metrics["calibration_input_extrapolation_fraction"] = float(
            np.mean(np.abs(normalized_val_output) > model.knots[-1])
        )

        cascade_metrics: dict[str, Any] | None = None
        if pa_surrogate is not None:
            predistorted = model.predict(val_input)
            cascade_output = (
                pa_surrogate.predict(predistorted)
                if segment_length is None
                else pa_surrogate.predict_segments(
                    predistorted,
                    segment_length,
                )
            )
            warmup = pa_surrogate.causal_warmup_samples
            cascade_metrics = _paired_time_metrics(
                cascade_output,
                gain * val_input,
                warmup_samples=warmup,
                segment_length=segment_length,
            )
            cascade_metrics.update(
                {
                    "scope": "surrogate_only",
                    "dpd_input": "validation desired x",
                    "target": "gain*x_validation",
                    "predistorted_waveform": _waveform_output_metrics(
                        predistorted
                    ),
                    "spline_input_extrapolation_fraction": float(
                        np.mean(np.abs(val_input) > model.knots[-1])
                    ),
                    "surrogate_training_range_extrapolation_fraction": float(
                        np.mean(
                            np.abs(predistorted)
                            > pa_diagnostics.maximum_training_input_amplitude
                        )
                    ),
                }
            )

        raw_selection_score = (
            inverse_metrics["complex_nmse_pooled_db"]
            if selection_metric == "inverse_nmse"
            else cascade_metrics["complex_nmse_pooled_db"]
        )
        full_rank = (
            diagnostics.data_design_rank == diagnostics.knot_count
            and diagnostics.minimum_nonzero_feature_samples > 0
        )
        valid_for_selection = bool(
            full_rank
            and np.isfinite(diagnostics.data_design_condition_number)
            and np.isfinite(diagnostics.augmented_design_condition_number)
            and (
                np.isfinite(raw_selection_score)
                or raw_selection_score == -np.inf
            )
        )
        selection_score = raw_selection_score if valid_for_selection else np.inf
        sortable_score = (
            float(selection_score)
            if np.isfinite(selection_score) or selection_score == -np.inf
            else np.inf
        )
        trial = {
            "configuration": {
                "knot_strategy": strategy,
                "requested_knot_count": int(knot_count),
                "effective_knot_count": int(model.knot_count),
                "ridge": float(ridge),
                "smoothness": float(smoothness),
                "compression_power": float(args.compression_power),
            },
            "fit_seconds": fit_seconds,
            "fit_diagnostics": diagnostics,
            "validation_inverse_postdistorter_diagnostic": inverse_metrics,
            "validation_surrogate_only_predistorter_cascade": cascade_metrics,
            "raw_selection_score_db": raw_selection_score,
            "valid_for_selection": valid_for_selection,
            "selection_exclusion_reason": (
                None
                if valid_for_selection
                else (
                    "rank-deficient or unoccupied data design"
                    if not full_rank
                    else "non-finite condition number or score"
                )
            ),
            "selection_score_db": selection_score,
        }
        trials.append(trial)
        # Persist every validation row immediately.  This file is intentionally
        # independent of the final model/report, so an interrupted sweep still
        # leaves a complete numerical audit up to its last finished fit.
        write_json(
            trials_path,
            {
                "schema_version": 1,
                "dataset": dataset,
                "dataset_spec_sha256": (
                    file_sha256(dataset / "spec.json")
                    if (dataset / "spec.json").is_file()
                    else None
                ),
                "frozen_integer_delay_samples": alignment_delay,
                "selection_metric": selection_metric,
                "split": "validation",
                "test_accessed": False,
                "completed_trials": trials,
            },
        )
        key = (
            sortable_score,
            strategy,
            int(knot_count),
            float(ridge),
            float(smoothness),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_model = model
            best_trial = trial

    if (
        best_model is None
        or best_trial is None
        or not best_trial["valid_for_selection"]
    ):
        raise RuntimeError(
            "no full-rank finite spline configuration was available for "
            "validation selection"
        )

    pa_validation: dict[str, Any] | None = None
    if pa_surrogate is not None:
        pa_validation_prediction = (
            pa_surrogate.predict(val_input)
            if segment_length is None
            else pa_surrogate.predict_segments(val_input, segment_length)
        )
        pa_validation = {
            "scope": "surrogate_fidelity_on_held_out_validation_measurements",
            "state_reset_policy": (
                "once_at_record_start"
                if segment_length is None
                else "zero_history_at_every_dataset_nperseg_boundary"
            ),
            "metrics_full_record": _paired_time_metrics(
                pa_validation_prediction,
                val_output,
            ),
            "metrics_after_causal_warmup": _paired_time_metrics(
                pa_validation_prediction,
                val_output,
                warmup_samples=pa_surrogate.causal_warmup_samples,
                segment_length=segment_length,
            ),
        }

    best_model.save(model_path)
    if pa_surrogate is not None:
        pa_surrogate.save(surrogate_path)

    input_files = {
        "train_input": dataset / "train_input.csv",
        "train_output": dataset / "train_output.csv",
        "val_input": dataset / "val_input.csv",
        "val_output": dataset / "val_output.csv",
    }
    spec_path = dataset / "spec.json"
    deployment_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "complex_linear_spline_deployment_manifest",
        "model_path": model_path,
        "model_sha256": file_sha256(model_path),
        "model_equation": "z[n] = x[n] * C(abs(x[n]))",
        "phase_equivariance": "exact in floating point",
        "knots": {
            "coordinate": "amplitude",
            "strategy": best_model.knot_strategy,
            "count": best_model.knot_count,
            "minimum": float(best_model.knots[0]),
            "maximum": float(best_model.knots[-1]),
            "endpoint_policy": "clamp complex correction C outside knot range",
        },
        "target_gain": {
            "strategy": args.gain_strategy,
            "value": gain,
            "definition": gain_definition,
        },
        "alignment": {
            "integer_delay_samples": alignment_delay,
            "delay_sign_convention": (
                "positive means PA output lags PA input; crop input[:-d], "
                "output[d:]"
            ),
            "fractional_delay": (
                "diagnostic only; no fractional resampling was applied"
            ),
        },
        "normalization": {
            "sample_iq_scaling": "none",
            "calibration_observation": "aligned measured PA output / target gain",
            "input_full_scale": (
                "not frozen here; fixed-point export must explicitly choose "
                "and record it"
            ),
        },
        "feedback_path_precondition": (
            "input/output CSVs are assumed already corrected for DC offset, "
            "IQ imbalance and feedback frequency response; this pipeline does "
            "not estimate those corrections"
        ),
        "dataset": {
            "directory": dataset,
            "spec_sha256": file_sha256(spec_path) if spec_path.is_file() else None,
        },
    }
    write_json(manifest_path, deployment_manifest)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "complex_linear_spline_ila_training",
        "claims_scope": {
            "inverse_metric": (
                "postdistorter diagnostic only; not proof of predistorter "
                "linearization"
            ),
            "cascade_metric": (
                "surrogate-only; not a physical-PA measurement"
                if pa_surrogate is not None
                else "not computed because no explicit PA surrogate was fitted"
            ),
        },
        "command": [sys.executable, "-m", "baseline.train_spline", *sys.argv[1:]],
        "determinism": {
            "algorithm": "closed-form NumPy complex ridge regression",
            "random_seed": None,
            "stochastic_operations": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "directory": dataset,
            "spec": spec,
            "spec_sha256": file_sha256(spec_path) if spec_path.is_file() else None,
            "raw_train_samples": raw_train_input.size,
            "raw_validation_samples": raw_val_input.size,
            "train_samples": train_input.size,
            "validation_samples": val_input.size,
            "test_accessed": False,
            "input_file_sha256": {
                label: file_sha256(path) for label, path in input_files.items()
            },
        },
        "alignment": {
            "estimated_from_split": (
                "train" if args.alignment_delay is None else "explicit"
            ),
            "search_max_abs_delay_samples": args.alignment_max_delay,
            "frozen_integer_delay_samples": alignment_delay,
            "delay_sign_convention": (
                "observed[n] ~= gain*reference[n-delay]; positive means PA "
                "output lags PA input"
            ),
            "fractional_delay_diagnostic": fractional_diagnostic,
            "fractional_delay_applied": False,
            "validation_delay_retuned": False,
            "feedback_path_correction": {
                "dc_offset": "not estimated; input data assumed pre-corrected",
                "iq_imbalance": "not estimated; input data assumed pre-corrected",
                "frequency_response": (
                    "not estimated; input data assumed pre-equalized"
                ),
            },
        },
        "target_gain": {
            "strategy": args.gain_strategy,
            "value": gain,
            "definition": gain_definition,
        },
        "selection": {
            "split": "validation",
            "metric": selection_metric,
            "lower_db_is_better": True,
            "candidate_count": len(trials),
            "selected_configuration": best_trial["configuration"],
            "selected_score_db": best_trial["selection_score_db"],
        },
        "trials": trials,
        "selected_validation_result": best_trial,
        "pa_surrogate": (
            {
                "scope": "surrogate_only",
                "artifact": surrogate_path,
                "artifact_sha256": file_sha256(surrogate_path),
                "orders": pa_surrogate.orders,
                "delays": pa_surrogate.delays,
                "metadata": pa_surrogate.metadata,
                "fit_seconds": pa_fit_seconds,
                "fit_diagnostics": pa_diagnostics,
                "validation": pa_validation,
            }
            if pa_surrogate is not None
            else None
        ),
        "artifacts": {
            "spline_model": model_path,
            "spline_model_sha256": file_sha256(model_path),
            "deployment_manifest": manifest_path,
            "deployment_manifest_sha256": file_sha256(manifest_path),
            "training_report": report_path,
            "validation_trials": trials_path,
        },
        "test_evaluation_instruction": (
            "Run baseline.evaluate_spline separately. Do not choose a model "
            "after inspecting its test result."
        ),
    }
    write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
