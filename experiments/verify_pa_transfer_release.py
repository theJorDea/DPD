"""Verify the published APA held-out transfer release.

The held-out pair has already been released by the guarded command.  This
post-publication verifier rehashes the immutable bundle and released dataset
files, reproduces the four scalar time-domain metric records from saved
predictions, and binds the producer/verifier source hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from baseline.train_spline import load_split_pair
from experiments.release_pa_transfer_apa200_to_b import (
    PRIOR_RELEASE_INCIDENT,
    PRIOR_RELEASE_INCIDENT_SHA256,
    load_release_config,
)
from experiments.transfer_pa_apa200_to_b import (
    PROJECT_ROOT,
    _project_path,
    file_sha256,
    metric_summary,
)
from experiments.verify_pa_transfer_bundle import verify_bundle


TOLERANCE_DB = 1e-9


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=TOLERANCE_DB):
        raise RuntimeError(f"{label} mismatch: {actual} != {expected}")


def verify_release(bundle: str | Path) -> dict[str, Any]:
    bundle_path = Path(bundle).resolve()
    manifest_path = bundle_path / "release_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != (
        "held_out_release_completed_after_metric_free_failed_access"
    ):
        raise RuntimeError("release status is unexpected")
    audit = manifest.get("release_access_audit", {})
    if audit.get("current_held_out_access_count") != 2:
        raise RuntimeError("release access count is not two")
    if audit.get("strict_single_open_execution") is not False:
        raise RuntimeError("release incorrectly claims strict single-open execution")
    if audit.get("sha256") != PRIOR_RELEASE_INCIDENT_SHA256:
        raise RuntimeError("release incident hash mismatch")
    incident_path = _project_path(PRIOR_RELEASE_INCIDENT, name="release incident")
    if file_sha256(incident_path) != PRIOR_RELEASE_INCIDENT_SHA256:
        raise RuntimeError("release incident record changed")

    for name, record in manifest["artifacts"].items():
        if file_sha256(bundle_path / record["path"]) != record["sha256"]:
            raise RuntimeError(f"release artifact hash mismatch: {name}")
    evaluations = json.loads(
        (bundle_path / "test_evaluations.json").read_text(encoding="utf-8")
    )
    if evaluations != manifest["reports"]:
        raise RuntimeError("standalone test evaluations differ from manifest reports")

    release_config_path = _project_path(manifest["config"], name="release config")
    if file_sha256(release_config_path) != manifest["config_sha256"]:
        raise RuntimeError("release config hash mismatch")
    release_config = load_release_config(release_config_path)
    pretest_bundle = _project_path(
        release_config["pretest_bundle"],
        name="pretest bundle",
    )
    pretest_verification = verify_bundle(pretest_bundle)
    if file_sha256(pretest_bundle / "transfer_manifest.json") != (
        manifest["pretest_manifest_sha256"]
    ):
        raise RuntimeError("pretest manifest changed after release")

    target_dataset = _project_path(
        release_config["target_dataset"],
        name="target dataset",
    )
    test_input_path = target_dataset / release_config["target_test_files"]["input"]
    test_output_path = target_dataset / release_config["target_test_files"]["output"]
    if file_sha256(test_input_path) != manifest["target_test_hashes"]["input_sha256"]:
        raise RuntimeError("released test input hash mismatch")
    if file_sha256(test_output_path) != manifest["target_test_hashes"]["output_sha256"]:
        raise RuntimeError("released test output hash mismatch")
    test_input, test_output = load_split_pair(target_dataset, "test")
    nperseg = int(release_config["protocol"]["nperseg"])
    if test_input.size != nperseg or test_output.size != nperseg:
        raise RuntimeError("released test shape mismatch")

    with np.load(bundle_path / "test_predictions.npz", allow_pickle=False) as archive:
        predictions = {
            name: np.asarray(archive[name], dtype=np.complex128).copy()
            for name in archive.files
            if name != "schema_version"
        }
    expected_prediction_keys = {
        "causal_gmp__zero_shot_test",
        "causal_gmp__selected_N_test",
        "lag9_sparse_spline_memory__zero_shot_test",
        "lag9_sparse_spline_memory__selected_N_test",
    }
    if set(predictions) != expected_prediction_keys:
        raise RuntimeError("release prediction key set mismatch")

    checked = 0
    for model_record in release_config["frozen_models"]:
        name = model_record["name"]
        warmup = int(release_config["protocol"]["common_warmup_samples"][name])
        modes = (
            (
                "zero_shot",
                f"{name}__zero_shot_test",
            ),
            (
                "coefficient_only_selected_N",
                f"{name}__selected_N_test",
            ),
        )
        for report_key, prediction_key in modes:
            actual = metric_summary(
                predictions[prediction_key],
                test_output,
                (nperseg,),
                warmup=warmup,
            )
            reported = manifest["reports"][name][report_key]["evaluation"]
            _assert_close(
                actual["full_record_nmse_db"],
                reported["full_record_metrics"]["complex_nmse_pooled_db"],
                f"{name}.{report_key}.full",
            )
            _assert_close(
                actual["common_interior_nmse_db"],
                reported["steady_state_metrics"]["complex_nmse_pooled_db"],
                f"{name}.{report_key}.common",
            )
            _assert_close(
                actual["opendpd_compatible_nmse_db"],
                reported["opendpd_compatible_metrics"]["nmse_mean_segment_db"],
                f"{name}.{report_key}.opendpd",
            )
            _assert_close(
                actual["time_domain_rms_evm_db"],
                reported["full_record_metrics"][
                    "time_domain_rms_sample_evm_db"
                ],
                f"{name}.{report_key}.evm",
            )
            checked += 1

    input_integrity = manifest["input_integrity"]
    for key in (
        "pretest_bundle_verified_before_test_load",
        "target_test_hashes_recorded_before_waveform_load",
        "target_test_loaded_after_selected_N_freeze",
    ):
        if input_integrity.get(key) is not True:
            raise RuntimeError(f"release integrity flag failed: {key}")
    for key in (
        "target_test_used_for_selection",
        "target_test_used_for_coefficient_fit",
        "target_test_used_for_delay_gain_or_bin_fit",
    ):
        if input_integrity.get(key) is not False:
            raise RuntimeError(f"release leakage flag failed: {key}")

    producer_path = PROJECT_ROOT / "experiments/release_pa_transfer_apa200_to_b.py"
    verifier_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "bundle": str(bundle_path.relative_to(PROJECT_ROOT)),
        "release_manifest_sha256": file_sha256(manifest_path),
        "release_producer_sha256": file_sha256(producer_path),
        "release_verifier_sha256": file_sha256(verifier_path),
        "checked_metric_records": checked,
        "artifact_hashes_verified": True,
        "pretest_bundle_verified": bool(
            pretest_verification["artifact_hashes_verified"]
        ),
        "target_test_hashes_verified": True,
        "target_test_used_for_selection": False,
        "target_test_used_for_coefficient_fit": False,
        "release_access_count": 2,
        "strict_single_open_execution": False,
        "prior_metric_free_incident_verified": True,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an already-published APA held-out transfer release."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _argument_parser().parse_args()
    result = verify_release(args.bundle)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
