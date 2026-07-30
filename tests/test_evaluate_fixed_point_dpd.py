import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from baseline.fixed_point_pa import FixedPointPAStats
from baseline.pa_models import MemoryPolynomialPA
from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
)
from baseline.train_spline import file_sha256
from experiments import evaluate_fixed_point_dpd as runner
from experiments.evaluate_frozen_dpd_spectrum import (
    validate_config as validate_spectral_config,
)


def _write_iq(path: Path, signal: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack((signal.real, signal.imag)),
        delimiter=",",
        header="I,Q",
        comments="",
    )


class FixedPointDPDRunnerTests(unittest.TestCase):
    def _base_config(self) -> dict:
        return {
            "schema_version": 1,
            "task": "frozen_spline_memory_dpd_fixed_point",
            "split": "val",
            "fit_performed": False,
            "selection_performed": False,
            "gain_or_alignment_retuned": False,
            "selected_family": "signal_delay_012",
            "alignment_delay_samples": 0,
            "nperseg": 128,
            "train_sample_count": 256,
            "validation_sample_count": 256,
            "dataset": "dataset",
            "model_path": "model.npz",
            "surrogate_path": "surrogate.npz",
            "selection_report": "selection.json",
            "artifact_sha256": {"model.npz": "0" * 64},
            "source_sha256": {"source.py": "1" * 64},
            "dataset_spec_sha256": "2" * 64,
            "train_input_sha256": "3" * 64,
            "split_input_sha256": "4" * 64,
            "target_gain": {"real": 2.0, "imag": 0.0},
            "expected_model": {
                "branches": [
                    {"signal_delay": 0, "envelope_delay": 0},
                    {"signal_delay": 1, "envelope_delay": 0},
                    {"signal_delay": 2, "envelope_delay": 0},
                ],
                "knot_count": 4,
                "knot_strategy": "quantile",
            },
            "expected_surrogate": {
                "orders": [1],
                "delays": [0],
            },
            "access_policy": {
                "allowed_waveform_files": [
                    "train_input.csv",
                    "val_input.csv",
                ],
                "measured_output_opened": False,
                "test_split_accessed": False,
            },
            "fixed_point_protocol": {
                "activation_bits": [16, 14, 12],
                "scale_guard_ratio": 1.001,
                "power_bits": 48,
                "accumulator_bits": 56,
                "scalar_accumulator_bits": 56,
                "interpolation_fraction_bits": 16,
                "rounding": "nearest_even",
                "overflow": "saturate_and_count",
            },
        }

    def _fixture(self, root: Path) -> tuple[Path, dict]:
        dataset = root / "dataset"
        dataset.mkdir()
        spec = {
            "input_signal_fs": 128.0,
            "bw_main_ch": 20.0,
            "bw_sub_ch": 10.0,
            "n_sub_ch": 2,
            "nperseg": 128,
        }
        spec_path = dataset / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        indices = np.arange(256)
        train = np.exp(2j * np.pi * 5 * indices / 128)
        validation = 0.75 * np.exp(2j * np.pi * 7 * indices / 128)
        train_path = dataset / "train_input.csv"
        validation_path = dataset / "val_input.csv"
        _write_iq(train_path, train)
        _write_iq(validation_path, validation)
        self.assertFalse((dataset / "train_output.csv").exists())
        self.assertFalse((dataset / "val_output.csv").exists())
        self.assertFalse((dataset / "test_input.csv").exists())

        model = SparseSplineMemoryDPD(
            knots=np.asarray([0.0, 0.4, 0.8, 1.2]),
            branches=(
                SplineMemoryBranch(0, 0),
                SplineMemoryBranch(1, 0),
                SplineMemoryBranch(2, 0),
            ),
            coefficients=np.asarray(
                [
                    [1.1 + 0j] * 4,
                    [0j] * 4,
                    [0j] * 4,
                ],
                dtype=np.complex128,
            ),
            knot_strategy="quantile",
        )
        model_path = root / "model.npz"
        model.save(model_path)
        surrogate = MemoryPolynomialPA(
            orders=(1,),
            delays=(0,),
            coefficients=np.asarray([[1.8 + 0j]]),
        )
        surrogate_path = root / "surrogate.npz"
        surrogate.save(surrogate_path)
        selection = {
            "selection": {"split": "validation"},
            "claims_scope": {"test_used_for_selection": False},
            "selected": {
                "signal_delay_012": {
                    "model_sha256": file_sha256(model_path),
                }
            },
            "pa_surrogate": {"sha256": file_sha256(surrogate_path)},
            "alignment": {"frozen_integer_delay_samples": 0},
            "target_gain": {
                "value": {"real": 2.0, "imag": 0.0},
            },
        }
        selection_path = root / "selection.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")

        config = self._base_config()
        config.update(
            {
                "dataset": str(dataset),
                "model_path": str(model_path),
                "surrogate_path": str(surrogate_path),
                "selection_report": str(selection_path),
                "artifact_sha256": {
                    str(model_path): file_sha256(model_path),
                    str(surrogate_path): file_sha256(surrogate_path),
                    str(selection_path): file_sha256(selection_path),
                },
                "source_sha256": {
                    source: file_sha256(runner.PROJECT_ROOT / source)
                    for source in runner.SOURCE_FILES
                },
                "dataset_spec_sha256": file_sha256(spec_path),
                "train_input_sha256": file_sha256(train_path),
                "split_input_sha256": file_sha256(validation_path),
            }
        )
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, config

    def test_config_rejects_test_output_access_and_wrong_precision(self) -> None:
        config = self._base_config()
        config["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation only"):
            runner.validate_config(config)

        config = self._base_config()
        config["access_policy"]["measured_output_opened"] = True
        with self.assertRaisesRegex(ValueError, "measured_output_opened"):
            runner.validate_config(config)

        config = self._base_config()
        config["fixed_point_protocol"]["activation_bits"] = [16, 12]
        with self.assertRaisesRegex(ValueError, "exactly"):
            runner.validate_config(config)

        config = self._base_config()
        config["test_input_sha256"] = "5" * 64
        with self.assertRaisesRegex(ValueError, "unknown"):
            runner.validate_config(config)

    def test_static_statistics_are_not_multiplied_across_frames(self) -> None:
        def stats(samples: int, input_clips: int, maximum: int):
            return FixedPointPAStats(
                sample_count=samples,
                input_saturations=input_clips,
                coefficient_saturations=3,
                power_saturations=1,
                scalar_accumulator_saturations=0,
                accumulator_saturations=0,
                output_saturations=0,
                maximum_power_magnitude=maximum,
                maximum_scalar_accumulator_magnitude=maximum + 1,
                maximum_accumulator_magnitude=maximum + 2,
                interpolation_saturations=0,
                knot_code_collision_count=2,
                maximum_knot_code_shift=1,
            )

        aggregate = runner._stats_aggregate(
            [stats(3, 1, 4), stats(5, 2, 9)]
        )
        self.assertEqual(aggregate["sample_count"], 8)
        self.assertEqual(aggregate["input_saturations"], 3)
        self.assertEqual(aggregate["power_saturations"], 2)
        self.assertEqual(aggregate["coefficient_saturations"], 3)
        self.assertEqual(aggregate["knot_code_collision_count"], 2)
        self.assertEqual(aggregate["maximum_power_magnitude"], 9)

    def test_existing_output_and_hash_tamper_fail_before_waveform_access(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=runner.PROJECT_ROOT,
            prefix="fixed_dpd_guard_test_",
        ) as temporary:
            root = Path(temporary)
            config_path, config = self._fixture(root)
            existing = root / "existing"
            existing.mkdir()
            with mock.patch.object(runner, "load_complex_iq_csv") as loader:
                with self.assertRaises(FileExistsError):
                    runner.evaluate(config_path, existing)
                loader.assert_not_called()

            config["artifact_sha256"][config["model_path"]] = "0" * 64
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "tampered"
            with mock.patch.object(runner, "load_complex_iq_csv") as loader:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    runner.evaluate(config_path, output)
                loader.assert_not_called()
            self.assertFalse(output.exists())

    def test_symlinked_input_is_rejected_before_waveform_access(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=runner.PROJECT_ROOT,
            prefix="fixed_dpd_symlink_test_",
        ) as temporary:
            root = Path(temporary)
            config_path, config = self._fixture(root)
            train_path = Path(config["dataset"]) / "train_input.csv"
            target = root / "real_train.csv"
            train_path.replace(target)
            train_path.symlink_to(target)
            config["train_input_sha256"] = file_sha256(target)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "result"
            with mock.patch.object(runner, "load_complex_iq_csv") as loader:
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "regular file",
                ):
                    runner.evaluate(config_path, output)
                loader.assert_not_called()
            self.assertFalse(output.exists())

    def test_final_recheck_catches_change_during_artifact_materialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=runner.PROJECT_ROOT,
            prefix="fixed_dpd_toctou_test_",
        ) as temporary:
            root = Path(temporary)
            config_path, config = self._fixture(root)
            spec_path = Path(config["dataset"]) / "spec.json"
            original_spectral_config = runner._spectral_config
            changed = False

            def spectral_config(*args, **kwargs):
                nonlocal changed
                result = original_spectral_config(*args, **kwargs)
                if not changed:
                    spec_path.write_text(
                        spec_path.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                    )
                    changed = True
                return result

            output = root / "result"
            with mock.patch.object(
                runner,
                "_spectral_config",
                side_effect=spectral_config,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    runner.evaluate(config_path, output)
            self.assertFalse(output.exists())
            self.assertFalse(
                any(root.glob(f".{output.name}.tmp-*"))
            )

    def test_sealed_integration_freezes_scale_before_validation(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=runner.PROJECT_ROOT,
            prefix="fixed_dpd_integration_test_",
        ) as temporary:
            root = Path(temporary)
            config_path, _ = self._fixture(root)
            output = root / "result"
            events: list[str] = []
            original_loader = runner.load_complex_iq_csv
            original_make_config = runner._make_fixed_config

            def load(path: Path):
                events.append(f"load:{Path(path).name}")
                return original_loader(path)

            def make_config(*args, **kwargs):
                events.append(f"format:{kwargs['bits']}")
                return original_make_config(*args, **kwargs)

            with mock.patch.object(
                runner,
                "load_complex_iq_csv",
                side_effect=load,
            ), mock.patch.object(
                runner,
                "_make_fixed_config",
                side_effect=make_config,
            ):
                report = runner.evaluate(config_path, output)

            self.assertEqual(
                events,
                [
                    "load:train_input.csv",
                    "format:16",
                    "format:14",
                    "format:12",
                    "load:val_input.csv",
                ],
            )
            self.assertTrue(output.is_dir())
            self.assertTrue(
                (output / "completion_manifest.json").is_file()
            )
            self.assertFalse(report["claims_scope"]["physical_pa_result"])
            self.assertFalse(report["claims_scope"]["test_split_accessed"])
            self.assertFalse(report["claims_scope"]["measured_output_opened"])
            self.assertTrue(
                report["claims_scope"][
                    "validation_reused_after_historical_float_model_selection"
                ]
            )
            self.assertFalse(
                report["claims_scope"]["eligible_as_untouched_final_evidence"]
            )
            self.assertEqual(
                report["direction"]["deployment_path"],
                "desired validation x -> frozen fixed-point DPD -> "
                "frozen float PA surrogate",
            )
            self.assertEqual(
                report["protocol"][
                    "cascade_common_warmup_samples_per_frame"
                ],
                2,
            )

            float_operations = report["float_reference"][
                "dpd_operation_count"
            ]
            self.assertIn(
                "complex_nmse_opendpd_scored_interior_db",
                report["float_reference"]["float_dpd_vs_ideal"],
            )
            with np.load(
                output / "waveforms_float.npz",
                allow_pickle=False,
            ) as arrays:
                self.assertEqual(
                    set(arrays.files),
                    {
                        "schema_version",
                        "desired_input",
                        "predistorted_drive",
                        "no_dpd_output",
                        "dpd_output",
                    },
                )
            float_spectral = json.loads(
                (output / "spectral_config_float.json").read_text(
                    encoding="utf-8"
                )
            )
            validate_spectral_config(float_spectral, release_test=False)
            self.assertFalse(
                float_spectral["claim_scope"]["fixed_point_dpd"]
            )
            for bits in (16, 14, 12):
                record = report["formats"][str(bits)]
                self.assertEqual(
                    record["input"]["fractional_bits"],
                    bits - 2,
                )
                self.assertEqual(
                    record["output"]["fractional_bits"],
                    bits - 2,
                )
                self.assertEqual(
                    record["coefficient"]["fractional_bits"],
                    bits - 2,
                )
                self.assertEqual(
                    record["fixed_schedule_operation_count"][
                        "real_multiplications"
                    ],
                    20,
                )
                self.assertEqual(
                    record["fixed_schedule_operation_count"][
                        "real_additions"
                    ],
                    25,
                )
                self.assertNotEqual(
                    record["fixed_schedule_operation_count"],
                    float_operations,
                )
                self.assertEqual(
                    record["coefficient_memory_bytes"],
                    (24 * bits + 7) // 8,
                )
                self.assertEqual(
                    record["constant_memory_bytes"],
                    (4 * 48 + 7) // 8,
                )
                self.assertIn(
                    "not a hardware",
                    record["validation"][
                        "python_integer_reference_timing"
                    ]["scope"],
                )
                self.assertTrue(
                    record["validation"]["streaming"][
                        "streaming_chunk_equivalence_passed"
                    ]
                )
                self.assertTrue(
                    record["validation"]["phase_equivariance"]["bit_exact"]
                )
                stats = record["validation"]["stats"]
                for name in (
                    "input_saturations",
                    "coefficient_saturations",
                    "power_saturations",
                    "scalar_accumulator_saturations",
                    "accumulator_saturations",
                    "output_saturations",
                    "interpolation_saturations",
                    "knot_code_collision_count",
                ):
                    self.assertEqual(stats[name], 0)
                archive = output / f"waveforms_{bits}bit.npz"
                with np.load(archive, allow_pickle=False) as arrays:
                    self.assertEqual(
                        set(arrays.files),
                        {
                            "schema_version",
                            "desired_input",
                            "predistorted_drive",
                            "no_dpd_output",
                            "dpd_output",
                        },
                    )
                    self.assertEqual(arrays["desired_input"].size, 256)
                spectral = json.loads(
                    (
                        output / f"spectral_config_{bits}bit.json"
                    ).read_text(encoding="utf-8")
                )
                validate_spectral_config(spectral, release_test=False)
                self.assertEqual(spectral["split_role"], "validation")
                self.assertTrue(
                    spectral["claim_scope"]["fixed_point_dpd"]
                )
                self.assertEqual(
                    spectral["claim_scope"]["activation_bits"],
                    bits,
                )

            manifest = json.loads(
                (output / "completion_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["atomic_publication"])
            self.assertFalse(manifest["fit_performed"])
            self.assertFalse(manifest["selection_performed"])
            for filename, expected_hash in manifest["artifacts"].items():
                self.assertEqual(
                    file_sha256(output / filename),
                    expected_hash,
                )

    def test_validation_amplitude_cannot_change_frozen_formats(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=runner.PROJECT_ROOT,
            prefix="fixed_dpd_scale_test_",
        ) as temporary:
            root = Path(temporary)
            config_path, config = self._fixture(root)
            baseline = runner.evaluate(config_path, root / "baseline")

            validation_path = Path(config["dataset"]) / "val_input.csv"
            indices = np.arange(256)
            oversized = 3.0 * np.exp(2j * np.pi * 7 * indices / 128)
            _write_iq(validation_path, oversized)
            config["split_input_sha256"] = file_sha256(validation_path)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            stressed = runner.evaluate(config_path, root / "stressed")

            for bits in ("16", "14", "12"):
                for field in ("input", "output", "coefficient", "power"):
                    self.assertEqual(
                        baseline["formats"][bits][field],
                        stressed["formats"][bits][field],
                    )
                self.assertGreater(
                    stressed["formats"][bits]["validation"]["stats"][
                        "input_saturations"
                    ],
                    0,
                )


if __name__ == "__main__":
    unittest.main()
