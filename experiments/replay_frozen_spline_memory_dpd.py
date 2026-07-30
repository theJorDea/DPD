"""Replay one already-selected spline-memory DPD without refitting.

The runner is intentionally narrower than ``run_spline_memory_ablation.py``:
it loads only the desired split input, a hash-bound frozen DPD, and a
hash-bound train-fitted PA surrogate.  It never opens measured split output,
never estimates gain/delay, and never ranks candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.pa_models import MemoryPolynomialPA  # noqa: E402
from baseline.spline_memory_dpd import SparseSplineMemoryDPD  # noqa: E402
from baseline.train_spline import (  # noqa: E402
    _paired_time_metrics,
    file_sha256,
    load_complex_iq_csv,
    load_dataset_spec,
    write_json,
)

SCHEMA_VERSION = 1
MODEL_SOURCE = "baseline/spline_memory_dpd.py"
SURROGATE_SOURCE = "baseline/pa_models.py"
CSV_SOURCE = "baseline/train_spline.py"
REPLAY_SOURCE = "experiments/replay_frozen_spline_memory_dpd.py"
SPECTRAL_RUNNER_SOURCE = "experiments/evaluate_frozen_dpd_spectrum.py"
SPECTRAL_MODULE_SOURCE = "baseline/spectral_regions.py"
METRICS_SOURCE = "baseline/metrics.py"


def _path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return Path(os.path.abspath(path))


def _regular_file(path: Path, *, field: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{field} must be a regular file: {path}")


def _hash_map(value: object, *, field: str) -> dict[Path, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field} must be a non-empty object")
    result: dict[Path, str] = {}
    for raw_path, expected in value.items():
        path = _path(raw_path, field=f"{field} path")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{field} contains an invalid SHA-256")
        result[path] = expected
    return result


def _verify_hash_map(
    paths: dict[Path, str],
    *,
    label: str,
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for path, expected in paths.items():
        _regular_file(path, field=label)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label} SHA-256 mismatch for {path}: "
                f"expected {expected}, found {actual}"
            )
        verified[str(path)] = actual
    return verified


def _complex_from_json(value: object, *, field: str) -> complex:
    if not isinstance(value, dict) or set(value) != {"real", "imag"}:
        raise ValueError(f"{field} must contain real and imag")
    try:
        result = complex(float(value["real"]), float(value["imag"]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is invalid") from error
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _json_complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def validate_config(config: dict[str, Any], *, legacy_test_replay: bool) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported frozen replay schema")
    if config.get("task") != "legacy_frozen_spline_memory_dpd_replay":
        raise ValueError("unexpected frozen replay task")
    split = config.get("split")
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if split == "test" and not legacy_test_replay:
        raise ValueError("test replay requires explicit --legacy-test-replay")
    if bool(config.get("fit_performed", True)):
        raise ValueError("fit_performed must be false")
    if bool(config.get("selection_performed", True)):
        raise ValueError("selection_performed must be false")
    if bool(config.get("gain_or_alignment_retuned", True)):
        raise ValueError("gain_or_alignment_retuned must be false")
    if not isinstance(config.get("selected_family"), str):
        raise ValueError("selected_family is required")
    if config.get("alignment_delay_samples") != 0:
        raise ValueError(
            "this input-only replay requires the frozen zero-delay contract"
        )
    nperseg = config.get("nperseg")
    if (
        not isinstance(nperseg, int)
        or isinstance(nperseg, bool)
        or nperseg <= 0
    ):
        raise ValueError("nperseg must be a positive integer")
    _path(config.get("dataset"), field="dataset")
    for field in ("model_path", "surrogate_path", "selection_report"):
        _path(config.get(field), field=field)
    _hash_map(config.get("artifact_sha256"), field="artifact_sha256")
    _hash_map(config.get("source_sha256"), field="source_sha256")
    _complex_from_json(config.get("target_gain"), field="target_gain")
    if split == "test" and config.get("historical_test_access") is not True:
        raise ValueError("test replay must declare historical_test_access=true")


def _verify_selection_report(
    config: dict[str, Any],
    report: dict[str, Any],
    *,
    model_path: Path,
    surrogate_path: Path,
) -> None:
    if report.get("selection", {}).get("split") != "validation":
        raise ValueError("frozen selection report was not validation-selected")
    if report.get("claims_scope", {}).get("test_used_for_selection") is not False:
        raise ValueError("selection report does not certify test-free selection")
    family = config["selected_family"]
    selected = report.get("selected", {}).get(family)
    if not isinstance(selected, dict):
        raise ValueError("selection report does not contain selected family")
    if selected.get("model_sha256") != file_sha256(model_path):
        raise ValueError("selected model hash disagrees with selection report")
    surrogate_record = report.get("pa_surrogate")
    if not isinstance(surrogate_record, dict):
        raise ValueError("selection report lacks PA surrogate record")
    if surrogate_record.get("sha256") != file_sha256(surrogate_path):
        raise ValueError("surrogate hash disagrees with selection report")
    if report.get("alignment", {}).get("frozen_integer_delay_samples") != 0:
        raise ValueError("selection report delay is not the frozen zero-delay contract")
    gain = _complex_from_json(report["target_gain"]["value"], field="report gain")
    configured_gain = _complex_from_json(
        config["target_gain"],
        field="configured target_gain",
    )
    if gain != configured_gain:
        raise ValueError("configured gain is not byte-level decision equivalent")


def _model_contract(
    config: dict[str, Any],
    model: SparseSplineMemoryDPD,
    surrogate: MemoryPolynomialPA,
) -> None:
    expected_model = config.get("expected_model")
    if not isinstance(expected_model, dict):
        raise ValueError("expected_model contract is required")
    expected_branches = tuple(
        (int(item["signal_delay"]), int(item["envelope_delay"]))
        for item in expected_model.get("branches", [])
    )
    actual_branches = tuple(
        (branch.signal_delay, branch.envelope_delay)
        for branch in model.branches
    )
    if actual_branches != expected_branches:
        raise ValueError("frozen DPD branches do not match expected contract")
    if model.knot_count != int(expected_model["knot_count"]):
        raise ValueError("frozen DPD knot count does not match expected contract")
    if model.knot_strategy != expected_model["knot_strategy"]:
        raise ValueError("frozen DPD knot strategy does not match expected contract")
    expected_surrogate = config.get("expected_surrogate")
    if not isinstance(expected_surrogate, dict):
        raise ValueError("expected_surrogate contract is required")
    if tuple(surrogate.orders) != tuple(expected_surrogate["orders"]):
        raise ValueError("surrogate orders do not match expected contract")
    if tuple(surrogate.delays) != tuple(expected_surrogate["delays"]):
        raise ValueError("surrogate delays do not match expected contract")


def _spectral_config(
    config: dict[str, Any],
    *,
    output_archive: Path,
    output_archive_sha256: str,
) -> dict[str, Any]:
    spec = load_dataset_spec(_path(config["dataset"], field="dataset"))
    fs_hz = float(spec["input_signal_fs"])
    bandwidth_main = float(spec["bw_main_ch"])
    bandwidth_adjacent = float(spec["bw_sub_ch"])
    main_half = bandwidth_main / 2.0
    source_paths = (
        SPECTRAL_RUNNER_SOURCE,
        SPECTRAL_MODULE_SOURCE,
        METRICS_SOURCE,
    )
    source_hashes = {
        path: file_sha256(_path(path, field="source")) for path in source_paths
    }
    return {
        "schema_version": 1,
        "task": "frozen_dpd_spectral_evaluation",
        "split_role": (
            "legacy_test"
            if config["split"] == "test"
            else "validation"
        ),
        "selection_performed": False,
        "fit_performed": False,
        "gain_or_alignment_retuned": False,
        "claim_scope": {
            "surrogate_only": True,
            "physical_pa_measurement": False,
            "rf_harmonic_claim": False,
            "descriptive_previously_opened_test": (
                config["split"] == "test"
            ),
        },
        "waveform_archive": str(output_archive),
        "waveform_archive_sha256": output_archive_sha256,
        "waveform_keys": {
            "desired_input": "desired_input",
            "predistorted_drive": "predistorted_drive",
            "no_dpd_output": "no_dpd_output",
            "dpd_output": "dpd_output",
        },
        "source_sha256": source_hashes,
        "fs_hz": fs_hz,
        "nperseg": int(config["nperseg"]),
        "framing": {
            "frame_origin_samples": 0,
            "complete_frames_only": True,
            "state_reset_policy": (
                "DPD and PA reset to zero history at each nperseg frame"
            ),
            "crop_policy": "input archive contains exact complete frames",
        },
        "main_region": {
            "name": "main",
            "low_hz": -main_half,
            "high_hz": main_half,
        },
        "regions": [
            {
                "name": "left_adjacent",
                "low_hz": -main_half - bandwidth_adjacent,
                "high_hz": -main_half,
            },
            {
                "name": "right_adjacent",
                "low_hz": main_half,
                "high_hz": main_half + bandwidth_adjacent,
            },
        ],
        "quantiles": [0.0, 0.5, 0.95, 1.0],
    }


def replay(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    legacy_test_replay: bool = False,
) -> dict[str, Any]:
    config_file = _path(str(config_path), field="config")
    _regular_file(config_file, field="config")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain one JSON object")
    validate_config(config, legacy_test_replay=legacy_test_replay)

    output = _path(str(output_dir), field="output_dir")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    dataset = _path(config["dataset"], field="dataset")
    if not dataset.is_dir() or dataset.is_symlink():
        raise FileNotFoundError("dataset must be a regular directory")
    model_path = _path(config["model_path"], field="model_path")
    surrogate_path = _path(config["surrogate_path"], field="surrogate_path")
    report_path = _path(config["selection_report"], field="selection_report")
    artifact_hashes = _hash_map(config["artifact_sha256"], field="artifact_sha256")
    source_hashes = _hash_map(config["source_sha256"], field="source_sha256")
    expected_artifacts = {model_path, surrogate_path, report_path}
    if set(artifact_hashes) != expected_artifacts:
        raise ValueError(
            "artifact_sha256 must bind exactly model, surrogate and "
            "selection_report"
        )
    expected_sources = {
        _path(path, field="source")
        for path in (MODEL_SOURCE, SURROGATE_SOURCE, CSV_SOURCE, REPLAY_SOURCE)
    }
    if set(source_hashes) != expected_sources:
        raise ValueError(
            "source_sha256 must bind exactly the frozen replay source set"
        )
    _verify_hash_map(artifact_hashes, label="frozen artifact")
    _verify_hash_map(source_hashes, label="source")
    spec_path = dataset / "spec.json"
    input_path = dataset / f"{config['split']}_input.csv"
    _regular_file(spec_path, field="dataset spec")
    _regular_file(input_path, field="split input")
    if config.get("dataset_spec_sha256") != file_sha256(spec_path):
        raise ValueError("dataset spec SHA-256 mismatch")
    if config.get("split_input_sha256") != file_sha256(input_path):
        raise ValueError("split input SHA-256 mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _verify_selection_report(
        config,
        report,
        model_path=model_path,
        surrogate_path=surrogate_path,
    )
    model = SparseSplineMemoryDPD.load(model_path)
    surrogate = MemoryPolynomialPA.load(surrogate_path)
    _model_contract(config, model, surrogate)

    # This is the only dataset waveform access: desired input, never measured
    # output.  The zero-delay contract means no output is needed for alignment.
    desired_input = load_complex_iq_csv(input_path)
    nperseg = int(config["nperseg"])
    if desired_input.size % nperseg:
        raise ValueError("desired input is not an exact set of complete frames")
    gain = _complex_from_json(config["target_gain"], field="target_gain")
    predistorted_drive = model.predict_segments(desired_input, nperseg)
    no_dpd_output = surrogate.predict_segments(desired_input, nperseg)
    dpd_output = surrogate.predict_segments(predistorted_drive, nperseg)
    ideal_output = gain * desired_input

    # Verify all frozen inputs after inference to close the TOCTOU window.
    _verify_hash_map(artifact_hashes, label="frozen artifact")
    _verify_hash_map(source_hashes, label="source")
    if file_sha256(input_path) != config["split_input_sha256"]:
        raise ValueError("split input changed during replay")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{secrets.token_hex(12)}"
    temporary.mkdir()
    try:
        waveform_path = temporary / "waveforms.npz"
        np.savez_compressed(
            waveform_path,
            schema_version=np.asarray(1, dtype=np.int64),
            desired_input=desired_input,
            predistorted_drive=predistorted_drive,
            no_dpd_output=no_dpd_output,
            dpd_output=dpd_output,
        )
        replay_report = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "legacy_frozen_spline_memory_dpd_replay",
            "claims_scope": {
                "physical_pa_result": False,
                "surrogate_only": True,
                "test_used_for_selection": False,
                "historical_test_access": config["split"] == "test",
            },
            "direction": {
                "deployment_path": (
                    "desired input x -> frozen spline DPD -> frozen PA surrogate"
                ),
                "measured_output_used_as_dpd_input": False,
                "inverse_diagnostic_run": False,
            },
            "dataset": {
                "path": str(dataset),
                "split": config["split"],
                "split_input_sha256": config["split_input_sha256"],
                "dataset_spec_sha256": config["dataset_spec_sha256"],
                "measured_output_opened": False,
            },
            "frozen_decisions": {
                "selected_family": config["selected_family"],
                "model_path": str(model_path),
                "model_sha256": file_sha256(model_path),
                "surrogate_path": str(surrogate_path),
                "surrogate_sha256": file_sha256(surrogate_path),
                "selection_report": str(report_path),
                "selection_report_sha256": file_sha256(report_path),
                "integer_delay_samples": 0,
                "target_gain": _json_complex(gain),
                "nperseg": nperseg,
                "state_reset": "zero history at every nperseg frame",
            },
            "metrics": {
                "without_dpd_vs_ideal": _paired_time_metrics(
                    no_dpd_output,
                    ideal_output,
                    warmup_samples=surrogate.causal_warmup_samples,
                    segment_length=nperseg,
                ),
                "with_dpd_vs_ideal": _paired_time_metrics(
                    dpd_output,
                    ideal_output,
                    warmup_samples=surrogate.causal_warmup_samples,
                    segment_length=nperseg,
                ),
                "predistorted_peak_amplitude": float(
                    np.max(np.abs(predistorted_drive))
                ),
                "predistorted_papr_db": float(
                    10.0
                    * np.log10(
                        np.max(np.abs(predistorted_drive) ** 2)
                        / np.mean(np.abs(predistorted_drive) ** 2)
                    )
                ),
            },
        }
        report_path_out = temporary / "replay_report.json"
        write_json(report_path_out, replay_report)
        spectral_config = _spectral_config(
            config,
            output_archive=output / "waveforms.npz",
            output_archive_sha256=file_sha256(waveform_path),
        )
        spectral_config_path = temporary / "spectral_config.json"
        write_json(spectral_config_path, spectral_config)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "legacy_frozen_spline_memory_dpd_replay_bundle",
            "config_sha256": file_sha256(config_file),
            "input_hashes": {
                "dataset_spec": config["dataset_spec_sha256"],
                "split_input": config["split_input_sha256"],
                "model": file_sha256(model_path),
                "surrogate": file_sha256(surrogate_path),
                "selection_report": file_sha256(report_path),
            },
            "artifacts": {
                "waveforms.npz": file_sha256(waveform_path),
                "replay_report.json": file_sha256(report_path_out),
                "spectral_config.json": file_sha256(spectral_config_path),
            },
            "selection_performed": False,
            "fit_performed": False,
            "measured_output_opened": False,
            "historical_test_access": config["split"] == "test",
            "atomic_publication": True,
        }
        manifest_path = temporary / "completion_manifest.json"
        write_json(manifest_path, manifest)
        os.replace(temporary, output)
        temporary = None  # type: ignore[assignment]
        return replay_report | {
            "artifacts": {
                "waveforms": str(output / "waveforms.npz"),
                "replay_report": str(output / "replay_report.json"),
                "spectral_config": str(output / "spectral_config.json"),
                "completion_manifest": str(output / "completion_manifest.json"),
            }
        }
    finally:
        if temporary is not None and temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--legacy-test-replay", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    replay(
        args.config,
        args.output_dir,
        legacy_test_replay=args.legacy_test_replay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
