"""Focused tests for the preregistered A0/A1 sensitivity runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from baseline.train_spline import file_sha256
from experiments import evaluate_fractional_alignment_sensitivity as runner


def _write_iq(path: Path, values: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack((values.real, values.imag)),
        delimiter=",",
        header="I,Q",
        comments="",
    )


def _waveform(times: np.ndarray, phase: float) -> np.ndarray:
    frequencies = np.asarray([-0.091, -0.037, 0.019, 0.073, 0.119])
    amplitudes = np.asarray(
        [
            0.20 + 0.08j,
            -0.16 + 0.11j,
            0.51 - 0.04j,
            -0.12 - 0.18j,
            0.07 + 0.13j,
        ]
    )
    return np.sum(
        amplitudes[:, None]
        * np.exp(
            2j
            * np.pi
            * frequencies[:, None]
            * (times[None, :] + phase)
        ),
        axis=0,
    )


class SyntheticSensitivityCase:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset = root / "dataset"
        self.output = root / "output"
        self.dataset.mkdir()
        self.nperseg = 96
        self.delay = 0.23

        train_frames = self._frames((96, 96, 96, 91), phase_start=0.1)
        validation_frames = self._frames((96, 93), phase_start=1.7)
        train_input = np.concatenate([pair[0] for pair in train_frames])
        train_output = np.concatenate([pair[1] for pair in train_frames])
        validation_input = np.concatenate(
            [pair[0] for pair in validation_frames]
        )
        validation_output = np.concatenate(
            [pair[1] for pair in validation_frames]
        )
        _write_iq(self.dataset / "train_input.csv", train_input)
        _write_iq(self.dataset / "train_output.csv", train_output)
        _write_iq(self.dataset / "val_input.csv", validation_input)
        _write_iq(self.dataset / "val_output.csv", validation_output)
        (self.dataset / "spec.json").write_text(
            json.dumps(
                {
                    "dataset_format": "split_csv",
                    "input_signal_fs": 1.0,
                    "nperseg": self.nperseg,
                    "bw_main_ch": 0.4,
                    "bw_sub_ch": 0.1,
                    "n_sub_ch": 2,
                }
            ),
            encoding="utf-8",
        )

        dataset_hashes = {
            name: file_sha256(self.dataset / name)
            for name in (
                "train_input.csv",
                "train_output.csv",
                "val_input.csv",
                "val_output.csv",
                "spec.json",
            )
        }
        self.selection_path = root / "selection_manifest.json"
        self.selection_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": "forward_pa_identification_model_selection",
                    "model_class": "complex_memory_polynomial",
                    "selection_split": "validation",
                    "test_split_accessed": False,
                    "dataset": str(
                        root / "stale_absolute_manifest_dataset_not_used"
                    ),
                    "dataset_label": "synthetic fractional delay",
                    "dataset_files_sha256": dataset_hashes,
                    "protocol": {
                        "alignment_delay_samples": 0,
                        "fractional_delay_applied": False,
                        "fractional_delay_reliable": True,
                        "fractional_delay_estimate_samples": self.delay,
                        "fractional_delay_offset_samples": self.delay,
                        "fractional_delay_peak_score": 0.99,
                        "nperseg": self.nperseg,
                    },
                    "selected_trial": {
                        "orders": [1, 3],
                        "delays": [0],
                        "ridge": 1e-10,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        self.gmp_leading_policy = "causal_leading"
        self.config_path = root / "sensitivity_config.json"
        self._write_runner_config()

    def _frames(
        self,
        lengths: tuple[int, ...],
        *,
        phase_start: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for index, length in enumerate(lengths):
            times = np.arange(length, dtype=np.float64)
            phase = phase_start + 0.37 * index
            pa_input = _waveform(times, phase)
            delayed_input = _waveform(times - self.delay, phase)
            pa_output = (
                (1.25 - 0.08j) * delayed_input
                + (0.19 + 0.04j)
                * delayed_input
                * np.abs(delayed_input) ** 2
            )
            result.append((pa_input, pa_output))
        return result

    def _fixed_gmp_recipe(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "task": "fixed_causal_gmp_pa_recipe",
            "gmp_config": {
                "ka": 3,
                "la": 1,
                "kb": 0,
                "lb": 0,
                "mb": 0,
                "kc": 0,
                "lc": 0,
                "mc": 0,
                "leading_policy": self.gmp_leading_policy,
            },
            "solver_mode": "ridge_lstsq",
            "ridge": 1e-10,
            "svd_rcond": None,
        }

    def _write_runner_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset": str(self.dataset),
                    "selection_manifest": str(self.selection_path),
                    "selection_manifest_sha256": file_sha256(
                        self.selection_path
                    ),
                    "fixed_gmp_recipe": self._fixed_gmp_recipe(),
                    "output_dir": str(self.output),
                    "alignment_filter": {
                        "tap_count": 17,
                        "kaiser_beta": 8.6,
                    },
                    "decision_rule": {
                        "primary_metric": "common_causal_interior",
                        "gmp_a1_minus_a0_max_db": -0.25,
                        "mp_corroboration_a1_minus_a0_max_db": 0.0,
                        "required_splits": ["train_oof", "validation"],
                        "require_full_record_same_sign": True,
                        "fallback_variant": "a0",
                        "accepted_a1_scope": (
                            "sensitivity_protocol_not_proven_"
                            "feedback_deembedding"
                        ),
                    },
                    "max_real_multiplications_per_sample": 1000,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class FractionalSensitivityRunnerTests(unittest.TestCase):
    def test_opendpd_metric_includes_zero_input_padded_partial_frame(
        self,
    ) -> None:
        references = (
            np.ones(4, dtype=np.complex128),
            2.0 * np.ones(4, dtype=np.complex128),
            np.ones(3, dtype=np.complex128),
        )
        predictions = (
            1.1 * references[0],
            1.2 * references[1],
            9.0 * references[2],
        )
        padded_predictions = (
            predictions[0],
            predictions[1],
            np.pad(predictions[2], (0, 1)),
        )

        metrics = runner._prediction_metrics(
            predictions,
            references,
            common_warmup_samples=1,
            expected_frame_length=4,
            opendpd_padded_prediction_frames=padded_predictions,
        )

        expected_mean_db = float(
            np.mean(
                [
                    20.0 * np.log10(0.1),
                    20.0 * np.log10(0.2),
                    20.0 * np.log10(8.0),
                ]
            )
        )
        compatible = metrics["opendpd_compatible"]
        self.assertAlmostEqual(
            compatible["nmse_mean_segment_db"],
            expected_mean_db,
        )
        self.assertEqual(compatible["complete_frame_count"], 2)
        self.assertEqual(
            compatible["segment_count_including_zero_padded_partial"],
            3,
        )
        self.assertEqual(compatible["scored_sample_count"], 12)
        self.assertEqual(compatible["padded_sample_count"], 12)
        self.assertEqual(
            compatible["actual_nonpadding_scored_sample_count"],
            11,
        )
        self.assertEqual(compatible["zero_padding_sample_count"], 1)
        self.assertEqual(compatible["discarded_partial_tail_samples"], 0)
        self.assertEqual(compatible["partial_frame_count"], 1)

        interior = metrics[
            "opendpd_compatible_common_causal_interior"
        ]
        self.assertAlmostEqual(
            interior["nmse_mean_segment_db"],
            expected_mean_db,
        )
        self.assertEqual(interior["complete_frame_count"], 2)
        self.assertEqual(interior["scored_sample_count"], 9)
        self.assertEqual(interior["padded_sample_count"], 9)
        self.assertEqual(
            interior["actual_nonpadding_scored_sample_count"],
            8,
        )
        self.assertEqual(interior["zero_padding_sample_count"], 1)
        self.assertEqual(
            interior["discarded_warmup_samples_from_actual_frames"],
            3,
        )
        self.assertEqual(interior["discarded_partial_tail_samples"], 0)

    def test_delayed_model_tail_is_scored_after_zero_input_padding(
        self,
    ) -> None:
        model = runner.MemoryPolynomialPA(
            orders=(1,),
            delays=(1,),
            coefficients=np.asarray([[1.0 + 0.0j]]),
        )
        input_frame = np.asarray(
            [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]
        )
        actual_predictions, padded_predictions = runner._predict_frames(
            model,
            (input_frame,),
            expected_frame_length=4,
        )
        self.assertTrue(
            np.array_equal(
                actual_predictions[0],
                model.predict(input_frame),
            )
        )
        reference = actual_predictions[0].copy()

        metrics = runner._prediction_metrics(
            actual_predictions,
            (reference,),
            common_warmup_samples=0,
            expected_frame_length=4,
            opendpd_padded_prediction_frames=padded_predictions,
        )

        self.assertTrue(
            np.array_equal(
                padded_predictions[0],
                np.asarray([0.0, 1.0, 2.0, 3.0]),
            )
        )
        compatible = metrics["opendpd_compatible"]
        expected_db = 10.0 * np.log10(9.0 / 5.0)
        self.assertAlmostEqual(
            compatible["nmse_mean_segment_db"],
            expected_db,
        )
        self.assertEqual(
            compatible["predicted_causal_tail_error_energy"],
            9.0,
        )
        self.assertEqual(
            compatible["predicted_causal_tail_nonzero_sample_count"],
            1,
        )

        with self.assertRaisesRegex(
            ValueError,
            "partial frames require predictions",
        ):
            runner._prediction_metrics(
                actual_predictions,
                (reference,),
                common_warmup_samples=0,
                expected_frame_length=4,
            )

    def test_preregistered_production_gmp_recipe_is_causal_and_832_mul(
        self,
    ) -> None:
        recipe = runner._parse_fixed_gmp_recipe(
            {
                "schema_version": 1,
                "task": "fixed_causal_gmp_pa_recipe",
                "gmp_config": {
                    "ka": 5,
                    "la": 30,
                    "kb": 2,
                    "lb": 30,
                    "mb": 2,
                    "kc": 2,
                    "lc": 30,
                    "mc": 2,
                    "leading_policy": "causal_leading",
                },
                "solver_mode": "truncated_svd",
                "ridge": 0.0,
                "svd_rcond": 1e-4,
            }
        )
        model = runner.GeneralizedMemoryPolynomialPA(
            recipe.config,
            np.zeros(recipe.config.coefficient_count, dtype=np.complex128),
        )

        self.assertEqual(model.operation_count.real_multiplications, 832)
        self.assertEqual(recipe.config.lookahead_samples, 0)
        self.assertEqual(recipe.causal_warmup_samples, 31)
        self.assertEqual(recipe.solver_mode, "truncated_svd")
        self.assertEqual(recipe.svd_rcond, 1e-4)

    def test_end_to_end_is_frame_safe_fixed_and_never_accesses_test(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            accessed: list[str] = []
            original_loader = runner.load_split_pair

            def tracked_loader(
                dataset: str | Path,
                split: str,
            ) -> tuple[np.ndarray, np.ndarray]:
                accessed.append(split)
                return original_loader(dataset, split)

            with mock.patch.object(
                runner,
                "load_split_pair",
                side_effect=tracked_loader,
            ):
                report = runner.evaluate_from_config(case.config_path)

            self.assertEqual(accessed, ["train", "val"])
            self.assertFalse(report["test_split_accessed"])
            self.assertFalse(report["test_file_hashes_recorded"])
            self.assertEqual(
                report["dataset_resolution"][
                    "runner_config_resolved_path"
                ],
                str(case.dataset.resolve()),
            )
            self.assertEqual(
                report["dataset_resolution"][
                    "selection_manifest_legacy_resolved_path"
                ],
                str(
                    (
                        case.root
                        / "stale_absolute_manifest_dataset_not_used"
                    ).resolve()
                ),
            )
            self.assertFalse(
                report["dataset_resolution"]["legacy_path_used_for_io"]
            )
            self.assertEqual(
                set(report["dataset_files_sha256"]),
                {
                    "train_input.csv",
                    "train_output.csv",
                    "val_input.csv",
                    "val_output.csv",
                    "spec.json",
                },
            )
            self.assertEqual(report["framing"]["original_nperseg"], 96)
            self.assertEqual(report["framing"]["effective_nperseg"], 80)
            self.assertEqual(
                report["framing"]["train"]["original_frame_lengths"],
                [96, 96, 96, 91],
            )
            self.assertEqual(
                report["framing"]["train"]["effective_frame_lengths"],
                [80, 80, 80, 75],
            )
            self.assertTrue(
                report["transforms"][
                    "input_support_bit_identical_between_variants"
                ]
            )
            self.assertTrue(
                report["transforms"]["guard_identical_between_variants"]
            )
            self.assertNotEqual(
                report["transforms"]["a0"]["protocol_sha256"],
                report["transforms"]["a1"]["protocol_sha256"],
            )
            self.assertEqual(
                len(report["transforms"]["a1"]["protocol_sha256"]),
                64,
            )

            for variant in ("a0", "a1"):
                for model in ("mp", "gmp"):
                    result = report["results"][variant][model]
                    self.assertFalse(result["architecture_tuning_performed"])
                    self.assertEqual(
                        result["train_inner_original_frame_oof"]["fold_count"],
                        4,
                    )
                    self.assertFalse(
                        result["validation_confirmation"][
                            "fit_or_tuning_on_validation"
                        ]
                    )
                    train_compatible = result[
                        "train_inner_original_frame_oof"
                    ]["metrics"]["opendpd_compatible"]
                    self.assertEqual(
                        train_compatible["complete_frame_count"],
                        3,
                    )
                    self.assertEqual(
                        train_compatible["scored_sample_count"],
                        320,
                    )
                    self.assertEqual(
                        train_compatible[
                            "actual_nonpadding_scored_sample_count"
                        ],
                        315,
                    )
                    self.assertEqual(
                        train_compatible["zero_padding_sample_count"],
                        5,
                    )
                    self.assertEqual(
                        train_compatible[
                            "discarded_partial_tail_samples"
                        ],
                        0,
                    )
                    validation_compatible = result[
                        "validation_confirmation"
                    ]["metrics"]["opendpd_compatible"]
                    self.assertEqual(
                        validation_compatible["complete_frame_count"],
                        1,
                    )
                    self.assertEqual(
                        validation_compatible["scored_sample_count"],
                        160,
                    )
                    self.assertEqual(
                        validation_compatible[
                            "actual_nonpadding_scored_sample_count"
                        ],
                        157,
                    )
                    self.assertEqual(
                        validation_compatible[
                            "zero_padding_sample_count"
                        ],
                        3,
                    )
                    self.assertEqual(
                        validation_compatible[
                            "discarded_partial_tail_samples"
                        ],
                        0,
                    )
                    oof_scope = result[
                        "train_inner_original_frame_oof"
                    ]
                    self.assertEqual(
                        oof_scope["scope"],
                        (
                            "coefficient_fit_oof_conditional_on_full_train_"
                            "frozen_delay"
                        ),
                    )
                    self.assertFalse(
                        oof_scope["delay_estimation_nested_within_oof"]
                    )
                    artifact = result["frozen_full_training_model"]
                    self.assertTrue(Path(artifact["path"]).is_file())
                    self.assertEqual(len(artifact["sha256"]), 64)

            for model in ("mp", "gmp"):
                deltas = report["a1_minus_a0"]["models"][model]
                self.assertLess(
                    deltas[
                        "train_oof_common_interior_a1_minus_a0_db"
                    ],
                    -5.0,
                )
                self.assertLess(
                    deltas[
                        "validation_common_interior_a1_minus_a0_db"
                    ],
                    -5.0,
                )

            persisted = json.loads(
                Path(report["report_path"]).read_text(encoding="utf-8")
            )
            self.assertFalse(persisted["test_split_accessed"])
            self.assertEqual(
                persisted["fixed_model_recipes"]["same_recipe_for_a0_and_a1"],
                True,
            )
            self.assertEqual(
                persisted["fixed_model_recipes"]["gmp_recipe_storage"],
                "inline in hash-recorded sensitivity runner config",
            )
            self.assertEqual(
                persisted["decision_rule"]["primary_metric"],
                "common_causal_interior",
            )
            outcome = persisted["decision_rule_evaluation"]
            self.assertTrue(
                outcome["rule_evaluated_after_all_fixed_results"]
            )
            self.assertFalse(outcome["rule_used_for_fit_or_tuning"])
            self.assertEqual(outcome["predicate_count"], 8)
            self.assertTrue(outcome["all_predicates_passed"])
            self.assertEqual(
                outcome["recommended_protocol_variant"],
                "a1",
            )
            self.assertIn(
                "not proven",
                outcome["accepted_a1_caveat"],
            )
            self.assertIn("--config", persisted["commands"]["reproduce"])
            self.assertNotIn(
                "--overwrite",
                persisted["commands"]["reproduce"],
            )
            self.assertIn(
                "baseline/complexity.py",
                persisted["source_sha256"],
            )
            self.assertTrue(
                persisted["publication"]["report_published_last"]
            )
            lock_metadata = persisted["publication"][
                "atomic_single_writer_lock"
            ]
            self.assertIn("O_CREAT|O_EXCL", lock_metadata["creation"])
            self.assertEqual(len(lock_metadata["owner_payload_sha256"]), 64)
            self.assertFalse(
                (
                    case.output
                    / ".fractional_alignment_sensitivity.lock"
                ).exists()
            )
            self.assertFalse(
                any(
                    path.name.startswith(".")
                    and "publishing" in path.name
                    for path in case.output.iterdir()
                )
            )

    def test_hash_mismatch_fails_before_any_waveform_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            with case.selection_path.open("a", encoding="utf-8") as stream:
                stream.write(" ")

            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_noncausal_fixed_gmp_is_rejected_before_waveform_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            case.gmp_leading_policy = "opendpd_exact"
            case._write_runner_config()

            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(ValueError, "fixed causal GMP"):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_path_only_gmp_indirection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            config = json.loads(case.config_path.read_text(encoding="utf-8"))
            del config["fixed_gmp_recipe"]
            config["fixed_gmp_config"] = "mutable_recipe.json"
            config["fixed_gmp_config_sha256"] = "0" * 64
            case.config_path.write_text(
                json.dumps(config) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(ValueError, "path-only"):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_missing_portable_dataset_or_budget_is_rejected_before_data(
        self,
    ) -> None:
        for missing_key in (
            "dataset",
            "max_real_multiplications_per_sample",
        ):
            with self.subTest(missing_key=missing_key):
                with tempfile.TemporaryDirectory() as temporary:
                    case = SyntheticSensitivityCase(Path(temporary))
                    config = json.loads(
                        case.config_path.read_text(encoding="utf-8")
                    )
                    del config[missing_key]
                    case.config_path.write_text(
                        json.dumps(config) + "\n",
                        encoding="utf-8",
                    )
                    with mock.patch.object(
                        runner,
                        "load_split_pair",
                    ) as loader:
                        with self.assertRaisesRegex(
                            ValueError,
                            "missing keys",
                        ):
                            runner.evaluate_from_config(case.config_path)
                    loader.assert_not_called()

    def test_recipe_at_or_above_mul_budget_fails_before_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            config = json.loads(case.config_path.read_text(encoding="utf-8"))
            config["fixed_gmp_recipe"]["gmp_config"].update(
                {
                    "ka": 100,
                    "la": 30,
                }
            )
            case.config_path.write_text(
                json.dumps(config) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(
                    ValueError,
                    "violates the exclusive <1000",
                ):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_mp_recipe_at_or_above_mul_budget_fails_before_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            selection = json.loads(
                case.selection_path.read_text(encoding="utf-8")
            )
            selection["selected_trial"]["orders"] = list(range(1, 51))
            selection["selected_trial"]["delays"] = list(range(10))
            case.selection_path.write_text(
                json.dumps(selection) + "\n",
                encoding="utf-8",
            )
            config = json.loads(case.config_path.read_text(encoding="utf-8"))
            config["selection_manifest_sha256"] = file_sha256(
                case.selection_path
            )
            case.config_path.write_text(
                json.dumps(config) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(
                    ValueError,
                    "MP recipe requires .*violates the exclusive <1000",
                ):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_exact_publication_temp_collision_fails_before_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            case.output.mkdir()
            temporary_artifact = (
                case.output / ".a0_mp_pa.publishing.npz"
            )
            temporary_artifact.write_bytes(b"owned collision")

            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(
                    FileExistsError,
                    "existing owned lock/final/temp",
                ):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_preexisting_bundle_lock_fails_before_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            case.output.mkdir()
            lock_path = (
                case.output / ".fractional_alignment_sensitivity.lock"
            )
            lock_path.write_text(
                '{"pid":1,"token":"other-run"}\n',
                encoding="utf-8",
            )

            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(
                    FileExistsError,
                    "existing owned lock/final/temp",
                ):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()
            self.assertEqual(
                lock_path.read_text(encoding="utf-8"),
                '{"pid":1,"token":"other-run"}\n',
            )

    def test_config_mutation_during_fit_aborts_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            original_fit = runner._fit_fixed_model
            mutation_done = False

            def mutating_fit(*args, **kwargs):
                nonlocal mutation_done
                if not mutation_done:
                    with case.config_path.open(
                        "a",
                        encoding="utf-8",
                    ) as stream:
                        stream.write(" ")
                    mutation_done = True
                return original_fit(*args, **kwargs)

            with mock.patch.object(
                runner,
                "_fit_fixed_model",
                side_effect=mutating_fit,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "config before model publication SHA-256 mismatch",
                ):
                    runner.evaluate_from_config(case.config_path)

            self.assertTrue(mutation_done)
            lock_path = (
                case.output / ".fractional_alignment_sensitivity.lock"
            )
            self.assertTrue(lock_path.is_file())
            self.assertEqual(list(case.output.iterdir()), [lock_path])

            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(
                    FileExistsError,
                    "existing owned lock/final/temp",
                ):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_bundle_lock_token_tamper_aborts_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            original_fit = runner._fit_fixed_model
            tamper_done = False
            lock_path = (
                case.output / ".fractional_alignment_sensitivity.lock"
            )

            def tampering_fit(*args, **kwargs):
                nonlocal tamper_done
                if not tamper_done:
                    lock_path.write_text(
                        '{"pid":999,"token":"tampered"}\n',
                        encoding="utf-8",
                    )
                    tamper_done = True
                return original_fit(*args, **kwargs)

            with mock.patch.object(
                runner,
                "_fit_fixed_model",
                side_effect=tampering_fit,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "owned bundle lock token/pid changed",
                ):
                    runner.evaluate_from_config(case.config_path)

            self.assertTrue(tamper_done)
            self.assertEqual(
                lock_path.read_text(encoding="utf-8"),
                '{"pid":999,"token":"tampered"}\n',
            )
            self.assertEqual(list(case.output.iterdir()), [lock_path])

    def test_missing_or_modified_decision_rule_is_rejected_before_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            config = json.loads(case.config_path.read_text(encoding="utf-8"))
            del config["decision_rule"]
            case.config_path.write_text(
                json.dumps(config) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(ValueError, "missing keys"):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            config = json.loads(case.config_path.read_text(encoding="utf-8"))
            config["decision_rule"]["gmp_a1_minus_a0_max_db"] = -0.2
            case.config_path.write_text(
                json.dumps(config) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(runner, "load_split_pair") as loader:
                with self.assertRaisesRegex(
                    ValueError,
                    "must equal preregistered",
                ):
                    runner.evaluate_from_config(case.config_path)
            loader.assert_not_called()

    def test_decision_rule_outcome_falls_back_to_a0_on_one_failure(
        self,
    ) -> None:
        metric_names = (
            "train_oof_full_record_a1_minus_a0_db",
            "train_oof_common_interior_a1_minus_a0_db",
            "validation_full_record_a1_minus_a0_db",
            "validation_common_interior_a1_minus_a0_db",
        )
        deltas = {
            "definition": "synthetic",
            "models": {
                model: {name: -1.0 for name in metric_names}
                for model in ("mp", "gmp")
            },
        }
        passing = runner._evaluate_decision_rule(
            deltas,
            dict(runner.DECISION_RULE),
        )
        self.assertTrue(passing["all_predicates_passed"])
        self.assertEqual(passing["recommended_protocol_variant"], "a1")
        self.assertEqual(passing["predicate_count"], 8)

        deltas["models"]["gmp"][
            "validation_common_interior_a1_minus_a0_db"
        ] = -0.249
        failing = runner._evaluate_decision_rule(
            deltas,
            dict(runner.DECISION_RULE),
        )
        self.assertFalse(failing["all_predicates_passed"])
        self.assertEqual(failing["recommended_protocol_variant"], "a0")
        failed_predicate = failing["predicates"][
            "gmp_validation_common_causal_interior"
        ]
        self.assertEqual(failed_predicate["actual_a1_minus_a0_db"], -0.249)
        self.assertEqual(failed_predicate["threshold_db"], -0.25)
        self.assertFalse(failed_predicate["passed"])

    def test_immutable_bundle_preserves_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = SyntheticSensitivityCase(Path(temporary))
            runner.evaluate_from_config(case.config_path)

            with self.assertRaisesRegex(
                FileExistsError,
                "existing owned lock/final/temp",
            ):
                runner.evaluate_from_config(case.config_path)


if __name__ == "__main__":
    unittest.main()
