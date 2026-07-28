import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.analyze_pa_residuals import (
    analyze_from_config,
    explicit_frame_ids,
)
from experiments.select_pa_mp import select_from_config


class ResidualRunnerTests(unittest.TestCase):
    @staticmethod
    def _write_iq(path: Path, signal: np.ndarray) -> None:
        rows = ["I,Q"]
        rows.extend(f"{value.real:.17g},{value.imag:.17g}" for value in signal)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_explicit_frame_ids_preserve_partial_final_frame(self) -> None:
        np.testing.assert_array_equal(
            explicit_frame_ids(10, 4),
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]),
        )

    def test_oof_and_validation_analysis_never_require_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            selection_output = root / "selection"
            residual_output = root / "residual"
            dataset.mkdir()
            rng = np.random.default_rng(20)
            for split, sample_count in (("train", 24), ("val", 8)):
                x = rng.normal(size=sample_count) + 1j * rng.normal(
                    size=sample_count
                )
                y = (
                    (1.4 - 0.1j) * x
                    + 0.02 * x * np.abs(x) ** 2
                )
                self._write_iq(dataset / f"{split}_input.csv", x)
                self._write_iq(dataset / f"{split}_output.csv", y)
            (dataset / "spec.json").write_text(
                json.dumps(
                    {
                        "input_signal_fs": 8.0,
                        "nperseg": 8,
                        "bw_main_ch": 2.0,
                        "bw_sub_ch": 1.0,
                        "n_sub_ch": 2,
                    }
                ),
                encoding="utf-8",
            )
            selection_config = root / "selection.json"
            selection_config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": str(dataset),
                        "dataset_label": "synthetic",
                        "output_dir": str(selection_output),
                        "alignment_delay": 0,
                        "order_families": [
                            {"name": "odd", "order_sets": [[1, 3]]}
                        ],
                        "delay_counts": [1],
                        "architecture_ridge": 1e-8,
                        "refinement_ridges": [1e-8],
                        "max_real_multiplications_per_sample": 1000,
                    }
                ),
                encoding="utf-8",
            )
            select_from_config(selection_config)
            residual_config = root / "residual.json"
            residual_config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selection_manifest": str(
                            selection_output / "selection_manifest.json"
                        ),
                        "output_dir": str(residual_output),
                        "lag_grid": {"start": -1, "stop": 2, "step": 1},
                        "envelope_lags": [0, 1],
                        "envelope_powers": [1, 2],
                        "slow_time_constants_samples": [1.0, 2.0],
                        "amplitude_quantiles": [0.5, 0.75],
                        "characteristic_bins": 4,
                        "position_bins": 4,
                        "independent_capture_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            manifest = analyze_from_config(residual_config)

            self.assertFalse(manifest["test_split_accessed"])
            self.assertEqual(manifest["accessed_splits"], ["train", "validation"])
            self.assertEqual(manifest["oof_fold_count"], 3)
            self.assertFalse((dataset / "test_input.csv").exists())
            self.assertTrue(
                (residual_output / "train_oof_residual_analysis.json").is_file()
            )
            self.assertTrue(
                (residual_output / "validation_residual_analysis.json").is_file()
            )
            with np.load(
                residual_output / "residual_predictions.npz",
                allow_pickle=False,
            ) as data:
                self.assertEqual(data["train_oof_prediction"].shape, (24,))
                self.assertEqual(data["validation_prediction"].shape, (8,))


if __name__ == "__main__":
    unittest.main()
