import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA
from experiments import evaluate_fixed_point_pa as runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class FixedPointRunnerTests(unittest.TestCase):
    def test_forbidden_split_is_rejected_before_loader_call(self) -> None:
        with mock.patch.object(runner, "load_split_pair") as loader:
            with self.assertRaisesRegex(RuntimeError, "forbidden split"):
                runner._load_allowed_split(
                    Path("."),
                    "test",
                    allowed=("train", "val"),
                )
            loader.assert_not_called()

    def test_static_coefficient_saturation_is_not_multiplied_by_frame_count(self) -> None:
        from baseline.fixed_point_pa import FixedPointPAStats

        first = FixedPointPAStats(
            sample_count=3,
            input_saturations=1,
            coefficient_saturations=7,
            power_saturations=0,
            scalar_accumulator_saturations=0,
            accumulator_saturations=0,
            output_saturations=0,
            maximum_power_magnitude=4,
            maximum_scalar_accumulator_magnitude=5,
            maximum_accumulator_magnitude=6,
            knot_code_collision_count=2,
            maximum_knot_code_shift=1,
        )
        second = FixedPointPAStats(
            sample_count=4,
            input_saturations=2,
            coefficient_saturations=7,
            power_saturations=1,
            scalar_accumulator_saturations=1,
            accumulator_saturations=1,
            output_saturations=1,
            maximum_power_magnitude=8,
            maximum_scalar_accumulator_magnitude=9,
            maximum_accumulator_magnitude=10,
            knot_code_collision_count=2,
            maximum_knot_code_shift=1,
        )
        aggregate = runner._stats_aggregate([first, second])
        self.assertEqual(aggregate["sample_count"], 7)
        self.assertEqual(aggregate["input_saturations"], 3)
        self.assertEqual(aggregate["coefficient_saturations"], 7)
        self.assertEqual(aggregate["knot_code_collision_count"], 2)
        self.assertEqual(aggregate["maximum_accumulator_magnitude"], 10)

    def test_frozen_coefficient_override_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=runner.PROJECT_ROOT,
            prefix="fixed_point_override_test_",
        ) as temporary:
            root = Path(temporary)
            base = GeneralizedMemoryPolynomialPA(
                GMPConfig(ka=1, la=1),
                np.asarray([1.0 + 0j]),
            )
            model_path = root / "base.npz"
            base.save(model_path)
            coefficients = np.asarray([0.75 - 0.1j])
            archive_path = root / "coefficients.npz"
            np.savez(
                archive_path,
                schema_version=np.asarray(1, dtype=np.int64),
                selected=coefficients,
            )
            manifest_path = root / "manifest.json"
            manifest = {
                "schema_version": 1,
                "status": "pretest_train_validation_only",
                "input_integrity": {
                    "test_never_opened_or_hashed": True,
                    "target_held_out_hash_recorded": False,
                },
                "artifacts": {
                    archive_path.name: {
                        "path": archive_path.name,
                        "sha256": _sha256(archive_path),
                    }
                },
                "source_artifact_hashes": {
                    "toy_gmp/model_path": _sha256(model_path),
                },
                "target_transfer": {
                    "toy_gmp": {
                        "selected_calibration": {
                            "fit": {
                                "coefficient_hash": runner._array_sha256(
                                    coefficients
                                )
                            },
                            "sample_count_per_frame": 4,
                            "status": "feasible",
                            "target_validation_loaded": True,
                            "validation_loaded_after_all_prefix_fits": True,
                        }
                    }
                },
            }
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            record = {
                "type": "causal_gmp",
                "path": str(model_path.relative_to(runner.PROJECT_ROOT)),
                "sha256": _sha256(model_path),
                "coefficient_override": {
                    "archive_path": str(
                        archive_path.relative_to(runner.PROJECT_ROOT)
                    ),
                    "archive_sha256": _sha256(archive_path),
                    "selection_manifest_path": str(
                        manifest_path.relative_to(runner.PROJECT_ROOT)
                    ),
                    "selection_manifest_sha256": _sha256(manifest_path),
                    "key": "selected",
                    "array_sha256": runner._array_sha256(coefficients),
                    "calibration_samples_per_frame": 4,
                    "selection_model_name": "toy_gmp",
                    "required_manifest_status": "pretest_train_validation_only",
                },
            }
            loaded = runner._load_frozen_models(
                {"frozen_models": {"gmp": record}}
            )
            np.testing.assert_array_equal(
                loaded["gmp"]["model"].coefficients,
                coefficients,
            )
            self.assertEqual(
                loaded["gmp"]["coefficient_override"][
                    "calibration_samples_per_frame"
                ],
                4,
            )
            bad = json.loads(json.dumps(record))
            bad["coefficient_override"]["array_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "payload hash mismatch"):
                runner._load_frozen_models(
                    {"frozen_models": {"gmp": bad}}
                )

            bad_manifest = json.loads(json.dumps(manifest))
            bad_manifest["target_transfer"]["toy_gmp"][
                "selected_calibration"
            ]["fit"]["coefficient_hash"] = "0" * 64
            manifest_path.write_text(
                json.dumps(bad_manifest),
                encoding="utf-8",
            )
            bad_selection = json.loads(json.dumps(record))
            bad_selection["coefficient_override"][
                "selection_manifest_sha256"
            ] = _sha256(manifest_path)
            with self.assertRaisesRegex(
                RuntimeError,
                "selected coefficient hash mismatch",
            ):
                runner._load_frozen_models(
                    {"frozen_models": {"gmp": bad_selection}}
                )

    def test_integration_opens_only_train_and_validation(self) -> None:
        rng = np.random.default_rng(123)
        train = (
            rng.normal(size=9) + 1j * rng.normal(size=9)
        ) * 0.15
        train_y = train.copy()
        validation = (
            rng.normal(size=6) + 1j * rng.normal(size=6)
        ) * 0.12
        validation_y = validation.copy()

        with tempfile.TemporaryDirectory(
            dir=runner.PROJECT_ROOT,
            prefix="fixed_point_runner_test_",
        ) as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            files = {
                "spec.json": b"{}",
                "train_input.csv": b"I,Q\n0,0\n",
                "train_output.csv": b"I,Q\n0,0\n",
                "val_input.csv": b"I,Q\n0,0\n",
                "val_output.csv": b"I,Q\n0,0\n",
            }
            for name, payload in files.items():
                (dataset / name).write_bytes(payload)

            model = GeneralizedMemoryPolynomialPA(
                GMPConfig(ka=1, la=1),
                np.asarray([1.0 + 0j]),
            )
            model_path = root / "model.npz"
            model.save(model_path)
            config_path = root / "config.json"
            output_path = root / "report.json"
            config = {
                "schema_version": 1,
                "dataset": str(dataset.relative_to(runner.PROJECT_ROOT)),
                "output_dir": str(root.relative_to(runner.PROJECT_ROOT)),
                "access_policy": {
                    "allowed_splits": ["train", "val"],
                    "test_split_accessed": False,
                },
                "dataset_contract": {
                    "sample_rate_hz": 1.0,
                    "frame_length": 9,
                    "train_frame_lengths": [9],
                    "validation_frame_lengths": [6],
                    "train_sample_count": 9,
                    "validation_sample_count": 6,
                    "alignment_delay_samples": 0,
                    "fractional_delay_applied": False,
                    "required_files_sha256": {
                        name: _sha256(dataset / name)
                        for name in files
                    },
                },
                "frozen_models": {
                    "gmp": {
                        "type": "causal_gmp",
                        "path": str(model_path.relative_to(runner.PROJECT_ROOT)),
                        "sha256": _sha256(model_path),
                    }
                },
                "fixed_point_protocol": {
                    "activation_bits": [12],
                    "scale_guard_ratio": 1.001,
                    "power_bits": 32,
                    "accumulator_bits": 40,
                    "scalar_accumulator_bits": 40,
                    "interpolation_fraction_bits": 8,
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            calls: list[str] = []

            def fake_loader(_dataset: Path, split: str):
                calls.append(split)
                if split == "train":
                    return train, train_y
                if split == "val":
                    return validation, validation_y
                raise AssertionError(f"unexpected split {split}")

            with mock.patch.object(
                runner,
                "load_split_pair",
                side_effect=fake_loader,
            ):
                report = runner.run_from_config(
                    config_path,
                    output_path=output_path,
                    overwrite=True,
                    progress=lambda _message: None,
                )
            self.assertEqual(calls, ["train", "val"])
            self.assertFalse(report["dataset"]["test_split_accessed"])
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                report["models"]["gmp"]["formats"]["12"]["train"][
                    "streaming"
                ]["streaming_chunk_equivalence_passed"],
                True,
            )


if __name__ == "__main__":
    unittest.main()
