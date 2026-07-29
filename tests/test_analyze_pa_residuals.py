import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA
from baseline.pa_models import MemoryPolynomialPA
from baseline.train_spline import file_sha256
from experiments import analyze_pa_residuals as runner
from experiments.select_pa_gmp import select_from_config as select_gmp
from experiments.select_pa_mp import select_from_config as select_mp


PROJECT_ROOT = Path(runner.__file__).resolve().parents[1]


class _SyntheticResidualFixture:
    """Small train/validation fixture that never creates a test split."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".residual-runner-test-",
            dir=PROJECT_ROOT,
        )
        self.root = Path(self._temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self._write_dataset()

    def close(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _write_iq(path: Path, signal: np.ndarray) -> None:
        rows = ["I,Q"]
        rows.extend(
            f"{value.real:.17g},{value.imag:.17g}" for value in signal
        )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _frame_delay(signal: np.ndarray, delay: int) -> np.ndarray:
        result = np.zeros_like(signal)
        nperseg = 64
        for start in range(0, signal.size, nperseg):
            stop = min(start + nperseg, signal.size)
            if delay == 0:
                result[start:stop] = signal[start:stop]
            elif stop - start > delay:
                result[start + delay : stop] = signal[start : stop - delay]
        return result

    def _write_dataset(self) -> None:
        rng = np.random.default_rng(20260729)
        for split, frame_count in (("train", 4), ("val", 2)):
            sample_count = frame_count * 64
            raw = (
                rng.normal(size=sample_count)
                + 1j * rng.normal(size=sample_count)
            )
            x = 0.58 * raw / np.sqrt(np.mean(np.abs(raw) ** 2))
            delayed = self._frame_delay(x, 1)
            y = (
                (1.28 - 0.09j) * x
                + (0.055 + 0.018j) * x * np.abs(x)
                + (0.018 - 0.007j) * delayed
            )
            y += 1e-5 * (
                rng.normal(size=sample_count)
                + 1j * rng.normal(size=sample_count)
            )
            self._write_iq(self.dataset / f"{split}_input.csv", x)
            self._write_iq(self.dataset / f"{split}_output.csv", y)
        self._write_json(
            self.dataset / "spec.json",
            {
                "input_signal_fs": 64.0,
                "nperseg": 64,
                "bw_main_ch": 16.0,
                "bw_sub_ch": 8.0,
                "n_sub_ch": 1,
            },
        )

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()

    def build_selection(
        self,
        kind: str,
    ) -> tuple[Path, Path, dict[str, object]]:
        selection_output = self.root / f"{kind}_selection"
        selection_config = self.root / f"{kind}_selection.json"
        common = {
            "schema_version": 1,
            "dataset": str(self.dataset),
            "dataset_label": f"synthetic_{kind}",
            "output_dir": str(selection_output),
            "alignment_delay": 0,
            "alignment_max_abs_delay": 2,
            "characteristic_bins": 4,
            "max_real_multiplications_per_sample": 1000,
        }
        if kind == "mp":
            selection_value = {
                **common,
                "order_families": [
                    {"name": "odd", "order_sets": [[1, 3]]}
                ],
                "delay_counts": [1],
                "architecture_ridge": 1e-8,
                "refinement_ridges": [1e-8],
            }
            selector = select_mp
        elif kind == "gmp":
            selection_value = {
                **common,
                "ka_values": [2],
                "memory_lengths": [50],
                "topologies": [
                    {
                        "name": "aligned_causal",
                        "kb": 0,
                        "mb": 0,
                        "kc": 0,
                        "mc": 0,
                        "leading_policy": "causal_leading",
                    }
                ],
                "architecture_solver_mode": "truncated_svd",
                "architecture_ridge": 0.0,
                "architecture_svd_rcond": 1e-6,
                "refinement_ridges": [1e-4],
                "selection_metric": "full_record",
            }
            selector = select_gmp
        else:
            raise ValueError(f"unsupported synthetic model kind: {kind}")
        self._write_json(selection_config, selection_value)
        selector(selection_config)
        selection_manifest = selection_output / "selection_manifest.json"
        selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
        return selection_config, selection_manifest, selection

    def write_residual_config(
        self,
        kind: str,
        *,
        selection_config: Path,
        selection_manifest: Path,
        selection: dict[str, object],
        output_name: str | None = None,
    ) -> tuple[Path, Path]:
        residual_output = self.root / (
            output_name if output_name is not None else f"{kind}_residual"
        )
        if kind == "gmp":
            warmup = int(selection["common_warmup_samples_per_frame"])
            cooldown = int(
                selection["common_future_cooldown_samples_per_frame"]
            )
        else:
            # Matched MP/GMP comparison support is deliberately conservative.
            warmup = 49
            cooldown = 0
        residual_config = self.root / (
            f"{output_name or kind}_residual.json"
        )
        self._write_json(
            residual_config,
            {
                "schema_version": 2,
                "dataset": self.relative(self.dataset),
                "selection_manifest": self.relative(selection_manifest),
                "selection_manifest_sha256": file_sha256(selection_manifest),
                "selection_config": self.relative(selection_config),
                "selection_config_sha256": file_sha256(selection_config),
                "output_dir": self.relative(residual_output),
                "expected_common_warmup_samples_per_frame": warmup,
                "expected_common_future_cooldown_samples_per_frame": cooldown,
                "lag_grid": {"start": -1, "stop": 2, "step": 1},
                "envelope_lags": [0, 1],
                "envelope_powers": [1, 2],
                "slow_time_constants_samples": [1.0, 2.0],
                "amplitude_quantiles": [0.5, 0.75],
                "characteristic_bins": 4,
                "position_bins": 4,
                "independent_capture_count": 0,
            },
        )
        return residual_config, residual_output


class ResidualRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _SyntheticResidualFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def _selection_and_config(
        self,
        kind: str,
        *,
        output_name: str | None = None,
    ) -> tuple[Path, Path, Path, dict[str, object]]:
        selection_config, selection_manifest, selection = (
            self.fixture.build_selection(kind)
        )
        residual_config, residual_output = (
            self.fixture.write_residual_config(
                kind,
                selection_config=selection_config,
                selection_manifest=selection_manifest,
                selection=selection,
                output_name=output_name,
            )
        )
        return (
            residual_config,
            residual_output,
            selection_manifest,
            selection,
        )

    def test_explicit_frame_ids_preserve_partial_final_frame(self) -> None:
        np.testing.assert_array_equal(
            runner.explicit_frame_ids(10, 4),
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2]),
        )

    def test_segmented_mask_supports_warmup_and_future_cooldown(self) -> None:
        np.testing.assert_array_equal(
            runner._segmented_interior_mask(
                12,
                nperseg=6,
                warmup_samples=2,
                cooldown_samples=1,
            ),
            np.asarray(
                [
                    False,
                    False,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    False,
                ]
            ),
        )

    def test_mp_regression_uses_only_train_validation_and_frozen_model(
        self,
    ) -> None:
        config, output, _, selection = self._selection_and_config("mp")
        accessed: list[str] = []
        original_loader = runner.load_split_pair

        def tracking_loader(dataset: Path, split: str):
            accessed.append(split)
            return original_loader(dataset, split)

        with mock.patch.object(
            runner,
            "load_split_pair",
            side_effect=tracking_loader,
        ):
            manifest = runner.analyze_from_config(config)

        self.assertEqual(accessed, ["train", "val"])
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["model_class"], "complex_memory_polynomial")
        self.assertFalse(manifest["test_split_accessed"])
        self.assertFalse(manifest["test_file_hashes_recorded"])
        self.assertEqual(manifest["oof_fold_count"], 4)
        self.assertTrue(
            manifest["full_training_refit"]["frozen_npz_reproduction"][
                "passed"
            ]
        )
        self.assertEqual(
            manifest["analysis_common_boundary"][
                "relationship_to_selection_manifest"
            ],
            "conservative_mp_extension",
        )
        self.assertEqual(
            manifest["full_training_refit"]["validation_predictor"],
            "integrity-verified frozen selected NPZ; independent full-"
            "training refit is used only as a reproduction check",
        )
        self.assertIn("selection_source_integrity", manifest)
        self.assertFalse((self.fixture.dataset / "test_input.csv").exists())
        self.assertFalse((self.fixture.dataset / "test_output.csv").exists())

        validation_input = original_loader(
            self.fixture.dataset,
            "val",
        )[0]
        frozen = MemoryPolynomialPA.load(
            Path(selection["selected_model"])
        )
        expected = frozen.predict_segments(validation_input, 64)
        with np.load(
            output / "residual_predictions.npz",
            allow_pickle=False,
        ) as data:
            self.assertEqual(int(data["schema_version"]), 2)
            np.testing.assert_allclose(data["validation_prediction"], expected)
            self.assertEqual(data["train_oof_prediction"].shape, (256,))
            self.assertEqual(data["validation_prediction"].shape, (128,))
            self.assertEqual(
                data["train_selected_model_valid_mask"].shape,
                (256,),
            )
            self.assertEqual(
                data["validation_selected_model_valid_mask"].shape,
                (128,),
            )
            self.assertEqual(
                int(np.count_nonzero(data["train_valid_mask"])),
                4 * (64 - 49),
            )

    def test_gmp_dispatch_preserves_exact_recipe_and_fold_diagnostics(
        self,
    ) -> None:
        config, output, _, selection = self._selection_and_config("gmp")
        manifest = runner.analyze_from_config(config)
        selected = selection["selected_trial"]

        self.assertEqual(
            manifest["model_class"],
            "complex_generalized_memory_polynomial",
        )
        self.assertEqual(
            manifest["selected_recipe"]["gmp_config"],
            selected["gmp_config"],
        )
        self.assertEqual(
            manifest["selected_recipe"]["solver_mode"],
            selected["solver_mode"],
        )
        self.assertEqual(
            manifest["selected_recipe"]["ridge"],
            selected["ridge"],
        )
        self.assertEqual(
            manifest["selected_recipe"]["svd_rcond"],
            selected["svd_rcond"],
        )
        self.assertEqual(
            manifest["analysis_common_boundary"][
                "relationship_to_selection_manifest"
            ],
            "exact_match",
        )
        self.assertLess(
            manifest["operation_count_verification"]["recomputed"][
                "real_multiplications"
            ],
            1000,
        )
        for fold in manifest["oof_folds"]:
            diagnostics = fold["fit_diagnostics"]
            self.assertEqual(
                diagnostics["solver_mode"],
                selected["solver_mode"],
            )
            self.assertEqual(diagnostics["ridge"], selected["ridge"])
            self.assertEqual(
                diagnostics["svd_rcond"],
                selected["svd_rcond"],
            )
            numerical = fold["fit_numerical_diagnostics"]
            self.assertEqual(
                numerical["feature_count"],
                int(selected["fit_diagnostics"]["feature_count"]),
            )
            self.assertGreaterEqual(numerical["solver_rank"], 1)
            self.assertGreaterEqual(numerical["coefficient_l2_norm"], 0.0)
            self.assertIn("held_fraction_above_fit_maximum", fold["input_support"])

        validation_input = runner.load_split_pair(
            self.fixture.dataset,
            "val",
        )[0]
        frozen = GeneralizedMemoryPolynomialPA.load(
            Path(selection["selected_model"])
        )
        expected = frozen.predict_segments(validation_input, 64)
        with np.load(
            output / "residual_predictions.npz",
            allow_pickle=False,
        ) as data:
            np.testing.assert_allclose(data["validation_prediction"], expected)
            self.assertEqual(
                int(np.count_nonzero(data["validation_valid_mask"])),
                2 * (64 - 49),
            )

    def test_selection_manifest_hash_tamper_fails_before_waveform_access(
        self,
    ) -> None:
        config, output, _, _ = self._selection_and_config("mp")
        value = json.loads(config.read_text(encoding="utf-8"))
        value["selection_manifest_sha256"] = "0" * 64
        self.fixture._write_json(config, value)
        with mock.patch.object(runner, "load_split_pair") as loader:
            with self.assertRaisesRegex(ValueError, "selection manifest SHA-256"):
                runner.analyze_from_config(config)
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_gmp_recipe_tamper_fails_before_waveform_access(self) -> None:
        config, output, selection_path, _ = self._selection_and_config("gmp")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["selected_trial"]["gmp_config"]["ka"] = 3
        self.fixture._write_json(selection_path, selection)
        value = json.loads(config.read_text(encoding="utf-8"))
        value["selection_manifest_sha256"] = file_sha256(selection_path)
        self.fixture._write_json(config, value)
        with mock.patch.object(runner, "load_split_pair") as loader:
            with self.assertRaisesRegex(
                ValueError,
                "loaded GMP model gmp_config",
            ):
                runner.analyze_from_config(config)
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_selection_source_tamper_fails_before_waveform_access(
        self,
    ) -> None:
        config, output, selection_path, _ = self._selection_and_config("gmp")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        source_label = next(iter(selection["source_sha256"]))
        selection["source_sha256"][source_label] = "0" * 64
        self.fixture._write_json(selection_path, selection)
        value = json.loads(config.read_text(encoding="utf-8"))
        value["selection_manifest_sha256"] = file_sha256(selection_path)
        self.fixture._write_json(config, value)

        with mock.patch.object(runner, "load_split_pair") as loader:
            with self.assertRaisesRegex(
                ValueError,
                "selection source SHA-256 mismatch",
            ):
                runner.analyze_from_config(config)
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_manifest_with_test_hash_is_rejected_before_waveform_access(
        self,
    ) -> None:
        config, output, selection_path, _ = self._selection_and_config("mp")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["dataset_files_sha256"]["test_input.csv"] = "0" * 64
        self.fixture._write_json(selection_path, selection)
        value = json.loads(config.read_text(encoding="utf-8"))
        value["selection_manifest_sha256"] = file_sha256(selection_path)
        self.fixture._write_json(config, value)
        with mock.patch.object(runner, "load_split_pair") as loader:
            with self.assertRaisesRegex(
                ValueError,
                "must not include test files",
            ):
                runner.analyze_from_config(config)
        loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_existing_bundle_is_immutable_before_waveform_access(self) -> None:
        config, output, _, _ = self._selection_and_config("gmp")
        output.mkdir()
        existing_manifest = output / "residual_manifest.json"
        existing_manifest.write_text('{"sentinel": true}\n', encoding="utf-8")

        with mock.patch.object(runner, "load_split_pair") as loader:
            with self.assertRaisesRegex(
                FileExistsError,
                "immutable residual bundle",
            ):
                runner.analyze_from_config(config)
        loader.assert_not_called()
        self.assertEqual(
            existing_manifest.read_text(encoding="utf-8"),
            '{"sentinel": true}\n',
        )


if __name__ == "__main__":
    unittest.main()
