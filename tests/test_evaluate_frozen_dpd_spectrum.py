import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_frozen_dpd_spectrum import (
    REQUIRED_WAVEFORM_KEYS,
    evaluate,
    file_sha256,
    validate_config,
)


class FrozenDpdSpectrumRunnerTests(unittest.TestCase):
    def _config(self, root: Path, archive: Path) -> dict:
        source_paths = {
            "baseline/spectral_regions.py": Path(
                "baseline/spectral_regions.py"
            ),
            "baseline/metrics.py": Path("baseline/metrics.py"),
            "experiments/evaluate_frozen_dpd_spectrum.py": Path(
                "experiments/evaluate_frozen_dpd_spectrum.py"
            ),
        }
        return {
            "schema_version": 1,
            "task": "frozen_dpd_spectral_evaluation",
            "split_role": "validation",
            "selection_performed": False,
            "fit_performed": False,
            "gain_or_alignment_retuned": False,
            "claim_scope": {
                "surrogate_only": True,
                "physical_pa_measurement": False,
                "rf_harmonic_claim": False,
                "descriptive_previously_opened_test": False,
            },
            "waveform_archive": str(archive),
            "waveform_archive_sha256": file_sha256(archive),
            "waveform_keys": {key: key for key in REQUIRED_WAVEFORM_KEYS},
            "source_sha256": {
                str(path): file_sha256(path) for path in source_paths.values()
            },
            "fs_hz": 128.0,
            "nperseg": 128,
            "framing": {
                "frame_origin_samples": 0,
                "complete_frames_only": True,
                "state_reset_policy": "reset_before_each_explicit_frame",
                "crop_policy": "archive_contains_exact_complete_frames",
            },
            "main_region": {"name": "main", "low_hz": -10.0, "high_hz": 10.0},
            "regions": [
                {"name": "left", "low_hz": -28.0, "high_hz": -22.0},
                {"name": "right", "low_hz": 22.0, "high_hz": 28.0},
            ],
            "quantiles": [0.0, 0.5, 0.95, 1.0],
        }

    def _archive(self, root: Path) -> Path:
        n = np.arange(256)
        main = np.exp(2j * np.pi * 5 * n / 128)
        no = main + 0.2 * np.exp(2j * np.pi * 25 * n / 128)
        yes = main + 0.02 * np.exp(2j * np.pi * 25 * n / 128)
        def iq_frames(value: np.ndarray) -> np.ndarray:
            frames = value.reshape(2, 128)
            return np.stack((frames.real, frames.imag), axis=-1)
        archive = root / "waveforms.npz"
        np.savez(
            archive,
            schema_version=np.asarray(1, dtype=np.int64),
            desired_input=iq_frames(main),
            predistorted_drive=iq_frames(main * 1.1),
            no_dpd_output=iq_frames(no),
            dpd_output=iq_frames(yes),
        )
        return archive

    def test_sealed_evaluation_writes_finite_summary_and_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            config = self._config(root, archive)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "result"
            summary = evaluate(config_path, output)
            self.assertEqual(summary["frame_count"], 2)
            region = summary["regions"]["right"]
            self.assertEqual(
                region["relative_leakage_improvement_db"]["status"],
                "finite",
            )
            loaded = json.loads((output / "summary.json").read_text())
            self.assertTrue((output / "completion_manifest.json").is_file())
            self.assertEqual(loaded["selection_performed"], False)
            with np.load(output / "spectra.npz", allow_pickle=False) as arrays:
                self.assertIn("frequencies_hz", arrays.files)
                self.assertEqual(arrays["region_0_no_dpd_dbc"].shape, (2,))

    def test_test_requires_release_flag_and_legacy_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            config = self._config(root, archive)
            config["split_role"] = "test"
            with self.assertRaisesRegex(ValueError, "release-test"):
                validate_config(config, release_test=False)
            validate_config(config, release_test=True)
            config["split_role"] = "legacy_test"
            config["claim_scope"]["descriptive_previously_opened_test"] = False
            with self.assertRaisesRegex(ValueError, "descriptive"):
                validate_config(config, release_test=False)

    def test_selection_and_odd_fft_are_rejected_before_waveform_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            config = self._config(root, archive)
            config["selection_performed"] = True
            config["waveform_archive"] = str(root / "missing.npz")
            with self.assertRaisesRegex(ValueError, "selection_performed"):
                validate_config(config, release_test=False)
            config["selection_performed"] = False
            config["nperseg"] = 127
            with self.assertRaisesRegex(ValueError, "even"):
                validate_config(config, release_test=False)

    def test_output_collision_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            config = self._config(root, archive)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "result"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                evaluate(config_path, output)

    def test_waveform_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root)
            link = root / "waveform-link.npz"
            link.symlink_to(archive)
            config = self._config(root, archive)
            config["waveform_archive"] = str(link)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "regular file"):
                evaluate(config_path, root / "result")


if __name__ == "__main__":
    unittest.main()
