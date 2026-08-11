"""One-shot sealed BlackBox evaluation of a preregistered DPD candidate.

The command opens only ``sealed/test_release.json`` and
``sealed/test_input.csv`` after every model, metric, gate and source hash has
been verified against a preregistration file.  It never opens measured test
output: this remains a surrogate-only deployment-direction experiment,

    desired test x -> frozen DPD -> frozen PA evaluator -> compare with g*x.

The output directory is immutable by construction and a second run refuses to
overwrite it.  This is not a physical-PA release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any

import numpy as np

from baseline.spline_memory_dpd import SparseSplineMemoryDPD
from baseline.train_spline import load_complex_iq_csv
from experiments.select_blackbox_dpd import (
    load_frozen_blackbox_dpd_selection,
)
from experiments.select_blackbox_pa import (
    load_frozen_blackbox_pa_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nmse_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    error_power = float(np.mean(np.abs(estimate - reference) ** 2))
    reference_power = float(np.mean(np.abs(reference) ** 2))
    if reference_power <= 0.0:
        raise ValueError("NMSE reference power must be positive")
    if error_power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(error_power / reference_power))


def _signal_summary(signal: np.ndarray) -> dict[str, float]:
    power = float(np.mean(np.abs(signal) ** 2))
    peak = float(np.max(np.abs(signal)))
    return {
        "rms": float(np.sqrt(power)),
        "peak": peak,
        "papr_db": float(10.0 * np.log10(peak * peak / power)),
    }


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


def _load_preregistration(path: Path) -> dict[str, Any]:
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if prereg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported preregistration schema")
    if prereg.get("task") != "blackbox_direct_dpd_one_shot_release":
        raise ValueError("unexpected preregistration task")
    if prereg.get("status") != "frozen_before_test_access":
        raise ValueError("preregistration is not frozen")
    if prereg.get("test_output_opened") is not False:
        raise ValueError("test_output_opened must be false")
    return prereg


def _model_metrics(
    output: np.ndarray,
    ideal: np.ndarray,
    *,
    warmup: int,
    segment_count: int,
) -> dict[str, Any]:
    scored_output = output[warmup:]
    scored_ideal = ideal[warmup:]
    segments = []
    for index, indices in enumerate(
        np.array_split(np.arange(scored_ideal.size), segment_count)
    ):
        segments.append(
            {
                "segment": index,
                "sample_count": int(indices.size),
                "nmse_db": _nmse_db(
                    scored_output[indices], scored_ideal[indices]
                ),
            }
        )
    return {
        "pooled_complex_nmse_db": _nmse_db(scored_output, scored_ideal),
        "warmup_samples": warmup,
        "scored_sample_count": int(scored_ideal.size),
        "segments": segments,
    }


def release(
    preregistration_path: Path,
    output_dir: Path,
    *,
    release_test: bool,
) -> dict[str, Any]:
    if not release_test:
        raise PermissionError("sealed access requires explicit --release-test")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite release: {output_dir}")
    prereg_hash = file_sha256(preregistration_path)
    prereg = _load_preregistration(preregistration_path)

    files = {
        name: PROJECT_ROOT / relative
        for name, relative in prereg["frozen_files"].items()
    }
    for name, path in files.items():
        _verify_hash(
            path,
            prereg["frozen_sha256"][name],
            label=f"frozen {name}",
        )

    # No sealed path is opened before all source/model hashes above pass.
    release_manifest_path = files["sealed_release_manifest"]
    release_manifest = json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    if release_manifest.get("artifact_type") != "blackbox_sealed_test_release":
        raise ValueError("unexpected sealed release manifest")
    test_input_path = release_manifest_path.parent / "test_input.csv"
    _verify_hash(
        test_input_path,
        release_manifest["files_sha256"]["test_input.csv"],
        label="sealed test input",
    )
    test_input_raw = load_complex_iq_csv(test_input_path)
    expected_count = int(release_manifest["split_contract"]["test"]["count"])
    if test_input_raw.size != expected_count:
        raise ValueError("sealed test input count mismatch")

    frozen_pa = load_frozen_blackbox_pa_selection(
        files["pa_selection_manifest"].parent
    )
    original = load_frozen_blackbox_dpd_selection(
        files["ila_dpd_selection_manifest"].parent
    )
    refined = SparseSplineMemoryDPD.load(files["refined_dpd_model"])
    scale = float(frozen_pa.normalization_scale)
    desired = np.asarray(test_input_raw / scale, dtype=np.complex128)
    gain_record = original.manifest["gain"]
    gain = complex(float(gain_record["real"]), float(gain_record["imag"]))
    ideal = gain * desired

    no_dpd_drive = desired
    ila_drive = np.asarray(original.model.predict(desired), dtype=np.complex128)
    refined_drive = np.asarray(refined.predict(desired), dtype=np.complex128)
    no_dpd_output = np.asarray(
        frozen_pa.model.predict(no_dpd_drive), dtype=np.complex128
    )
    ila_output = np.asarray(
        frozen_pa.model.predict(ila_drive), dtype=np.complex128
    )
    refined_output = np.asarray(
        frozen_pa.model.predict(refined_drive), dtype=np.complex128
    )

    warmup = int(prereg["metric_protocol"]["warmup_samples"])
    segment_count = int(prereg["metric_protocol"]["segment_count"])
    metrics = {
        "no_dpd": _model_metrics(
            no_dpd_output, ideal, warmup=warmup, segment_count=segment_count
        ),
        "ila_spline": _model_metrics(
            ila_output, ideal, warmup=warmup, segment_count=segment_count
        ),
        "iterative_direct_spline": _model_metrics(
            refined_output, ideal, warmup=warmup, segment_count=segment_count
        ),
    }
    drive = {
        "ila_spline": _signal_summary(ila_drive[warmup:]),
        "iterative_direct_spline": _signal_summary(refined_drive[warmup:]),
    }
    no_score = metrics["no_dpd"]["pooled_complex_nmse_db"]
    ila_score = metrics["ila_spline"]["pooled_complex_nmse_db"]
    refined_score = metrics["iterative_direct_spline"][
        "pooled_complex_nmse_db"
    ]
    segment_gains_vs_ila = [
        baseline["nmse_db"] - candidate["nmse_db"]
        for baseline, candidate in zip(
            metrics["ila_spline"]["segments"],
            metrics["iterative_direct_spline"]["segments"],
            strict=True,
        )
    ]
    thresholds = prereg["acceptance_gate"]
    checks = {
        "gain_over_ila": bool(
            ila_score - refined_score
            >= float(thresholds["minimum_gain_over_ila_db"])
        ),
        "gain_over_no_dpd": bool(
            no_score - refined_score
            >= float(thresholds["minimum_gain_over_no_dpd_db"])
        ),
        "every_segment_better_than_ila": bool(
            min(segment_gains_vs_ila)
            >= float(thresholds["minimum_each_segment_gain_over_ila_db"])
        ),
        "drive_within_train_pa_support": bool(
            drive["iterative_direct_spline"]["peak"]
            <= float(thresholds["maximum_normalized_drive_amplitude"])
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "blackbox_direct_dpd_one_shot_test_release",
        "scope": "frozen PA surrogate only",
        "physical_pa_result": False,
        "official_spectral_metric_available": False,
        "test_output_opened": False,
        "test_input_access_count": 1,
        "direction": "desired test x -> DPD -> frozen PA -> compare with g*x",
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": prereg_hash,
        },
        "sealed_release_manifest_sha256": file_sha256(release_manifest_path),
        "test_input_sha256": file_sha256(test_input_path),
        "sample_count": int(desired.size),
        "normalization_scale_from_train": scale,
        "target_gain_from_aligned_train": {
            "real": float(gain.real),
            "imag": float(gain.imag),
        },
        "metrics": metrics,
        "drive": drive,
        "derived": {
            "gain_over_ila_db": ila_score - refined_score,
            "gain_over_no_dpd_db": no_score - refined_score,
            "segment_gains_over_ila_db": segment_gains_vs_ila,
        },
        "acceptance_checks": checks,
        "acceptance_pass": bool(all(checks.values())),
        "claims_limit": (
            "untouched waveform evidence through the frozen software PA; "
            "not measured predistorted output from a physical PA"
        ),
        "execution_unix_time": time.time(),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{secrets.token_hex(8)}"
    temporary.mkdir()
    try:
        evaluation_path = temporary / "test_evaluation.json"
        evaluation_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        completion = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "blackbox_direct_dpd_test_release_completion",
            "test_evaluation_sha256": file_sha256(evaluation_path),
            "preregistration_sha256": prereg_hash,
            "atomic_publication": True,
            "rerun_or_overwrite_permitted": False,
        }
        completion_path = temporary / "completion_manifest.json"
        completion_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("release output appeared before publication")
        os.replace(temporary, output_dir)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-test", action="store_true")
    args = parser.parse_args(argv)
    result = release(
        args.preregistration,
        args.output_dir,
        release_test=args.release_test,
    )
    print(json.dumps({
        "acceptance_pass": result["acceptance_pass"],
        "metrics": {
            name: record["pooled_complex_nmse_db"]
            for name, record in result["metrics"].items()
        },
        "derived": result["derived"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
