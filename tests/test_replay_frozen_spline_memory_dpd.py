import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.pa_models import MemoryPolynomialPA
from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
)
from baseline.train_spline import file_sha256
from experiments.replay_frozen_spline_memory_dpd import (
    replay,
    validate_config,
)


class FrozenReplayConfigTests(unittest.TestCase):
    def _base(self) -> dict:
        return {
            "schema_version": 1,
            "task": "legacy_frozen_spline_memory_dpd_replay",
            "split": "val",
            "fit_performed": False,
            "selection_performed": False,
            "gain_or_alignment_retuned": False,
            "selected_family": "signal_delay_012",
            "alignment_delay_samples": 0,
            "nperseg": 128,
            "dataset": "vendor/OpenDPD/datasets/DPA_200MHz",
            "model_path": "experiments/results/spline_memory_dpa200/signal_delay_012.npz",
            "surrogate_path": "experiments/results/spline_memory_dpa200/pa_surrogate.npz",
            "selection_report": "experiments/results/spline_memory_dpa200/memory_ablation_report.json",
            "artifact_sha256": {
                "experiments/results/spline_memory_dpa200/signal_delay_012.npz": "0" * 64,
                "experiments/results/spline_memory_dpa200/pa_surrogate.npz": "1" * 64,
                "experiments/results/spline_memory_dpa200/memory_ablation_report.json": "2" * 64,
            },
            "source_sha256": {
                "baseline/spline_memory_dpd.py": "3" * 64,
            },
            "dataset_spec_sha256": "4" * 64,
            "split_input_sha256": "5" * 64,
            "target_gain": {"real": 1.0, "imag": 0.0},
            "expected_model": {
                "branches": [
                    {"signal_delay": 0, "envelope_delay": 0},
                    {"signal_delay": 1, "envelope_delay": 0},
                    {"signal_delay": 2, "envelope_delay": 0},
                ],
                "knot_count": 24,
                "knot_strategy": "quantile",
            },
            "expected_surrogate": {
                "orders": [1, 3, 5, 7, 9],
                "delays": [0, 1, 2, 3, 4],
            },
        }

    def test_validation_contract_rejects_test_without_explicit_legacy_flag(self):
        config = self._base()
        config["split"] = "test"
        config["historical_test_access"] = True
        with self.assertRaisesRegex(ValueError, "legacy-test-replay"):
            validate_config(config, legacy_test_replay=False)
        validate_config(config, legacy_test_replay=True)

    def test_input_only_contract_rejects_nonzero_delay_and_fit(self):
        config = self._base()
        config["alignment_delay_samples"] = 1
        with self.assertRaisesRegex(ValueError, "zero-delay"):
            validate_config(config, legacy_test_replay=False)
        config["alignment_delay_samples"] = 0
        config["fit_performed"] = True
        with self.assertRaisesRegex(ValueError, "fit_performed"):
            validate_config(config, legacy_test_replay=False)

    def test_config_is_json_object_for_documented_reproducibility(self):
        config = self._base()
        encoded = json.dumps(config, sort_keys=True)
        self.assertIsInstance(encoded, str)

    def test_integration_reads_desired_input_without_measured_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            samples = np.arange(256)
            desired = 0.5 * np.exp(2j * np.pi * 5 * samples / 128)
            input_path = dataset / "val_input.csv"
            np.savetxt(
                input_path,
                np.column_stack((desired.real, desired.imag)),
                delimiter=",",
                header="I,Q",
                comments="",
            )
            self.assertFalse((dataset / "val_output.csv").exists())

            model = SparseSplineMemoryDPD(
                knots=np.asarray([0.0, 1.0]),
                branches=(
                    SplineMemoryBranch(0, 0),
                    SplineMemoryBranch(1, 0),
                    SplineMemoryBranch(2, 0),
                ),
                coefficients=np.asarray(
                    [[1.1 + 0j, 1.1 + 0j], [0j, 0j], [0j, 0j]],
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
            selection_path.write_text(
                json.dumps(selection),
                encoding="utf-8",
            )
            source_paths = [
                Path("baseline/spline_memory_dpd.py"),
                Path("baseline/pa_models.py"),
                Path("baseline/train_spline.py"),
                Path("experiments/replay_frozen_spline_memory_dpd.py"),
            ]
            config = {
                "schema_version": 1,
                "task": "legacy_frozen_spline_memory_dpd_replay",
                "split": "val",
                "fit_performed": False,
                "selection_performed": False,
                "gain_or_alignment_retuned": False,
                "selected_family": "signal_delay_012",
                "alignment_delay_samples": 0,
                "nperseg": 128,
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
                    str(path): file_sha256(path) for path in source_paths
                },
                "dataset_spec_sha256": file_sha256(spec_path),
                "split_input_sha256": file_sha256(input_path),
                "target_gain": {"real": 2.0, "imag": 0.0},
                "expected_model": {
                    "branches": [
                        {"signal_delay": 0, "envelope_delay": 0},
                        {"signal_delay": 1, "envelope_delay": 0},
                        {"signal_delay": 2, "envelope_delay": 0},
                    ],
                    "knot_count": 2,
                    "knot_strategy": "quantile",
                },
                "expected_surrogate": {
                    "orders": [1],
                    "delays": [0],
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "result"
            report = replay(config_path, output)
            self.assertFalse(report["dataset"]["measured_output_opened"])
            self.assertFalse(
                report["direction"]["measured_output_used_as_dpd_input"]
            )
            with np.load(output / "waveforms.npz", allow_pickle=False) as arrays:
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
                np.testing.assert_allclose(
                    arrays["predistorted_drive"],
                    1.1 * desired,
                    rtol=1e-12,
                    atol=1e-12,
                )
            spectral = json.loads(
                (output / "spectral_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(spectral["split_role"], "validation")
            self.assertFalse(spectral["claim_scope"]["physical_pa_measurement"])


if __name__ == "__main__":
    unittest.main()
