from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA
from baseline.pa_models import MemoryPolynomialPA
from baseline.train_spline import file_sha256
from experiments.select_blackbox_pa import (
    PROJECT_ROOT,
    SELECTION_FILES,
    _load_config,
    _load_normalized_pairs,
    _select_topology_representatives,
    _verify_selection_view,
    enumerate_candidate_recipes,
    load_frozen_blackbox_pa_selection,
    select_from_config,
)


class BlackBoxPASelectionTests(unittest.TestCase):
    @staticmethod
    def _write_iq(path: Path, signal: np.ndarray) -> None:
        rows = ["I,Q"]
        rows.extend(
            f"{value.real:.17g},{value.imag:.17g}" for value in signal
        )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _write_selection_view(
        self,
        root: Path,
    ) -> tuple[Path, dict[str, np.ndarray], float]:
        selection = root / "prepared" / "selection"
        selection.mkdir(parents=True)
        # A sibling sealed directory is intentionally present.  The selector
        # receives only ``selection`` and must never inspect its parent.
        sealed = root / "prepared" / "sealed"
        sealed.mkdir()
        (sealed / "private_release.bin").write_bytes(b"do not open")

        rng = np.random.default_rng(815)
        truth = GeneralizedMemoryPolynomialPA(
            GMPConfig(ka=2, la=2),
            np.asarray(
                [
                    1.1 - 0.2j,
                    0.12 + 0.04j,
                    0.08 - 0.03j,
                    -0.025 + 0.01j,
                ],
                dtype=np.complex128,
            ),
        )

        def record(count: int) -> tuple[np.ndarray, np.ndarray]:
            source = 6.0 * (
                rng.normal(size=count) + 1j * rng.normal(size=count)
            )
            source /= max(1.0, float(np.max(np.abs(source))) / 8.0)
            undelayed = truth.predict(source)
            measured = np.zeros_like(source)
            measured[2:] = undelayed[:-2]
            measured[2:] += 1e-5 * (
                rng.normal(size=count - 2)
                + 1j * rng.normal(size=count - 2)
            )
            return source, measured

        train_x, train_y = record(96)
        validation_x, validation_y = record(48)
        self._write_iq(selection / "train_input.csv", train_x)
        self._write_iq(selection / "train_output.csv", train_y)
        self._write_iq(selection / "val_input.csv", validation_x)
        self._write_iq(selection / "val_output.csv", validation_y)
        (selection / "spec.json").write_text(
            json.dumps(
                {
                    "sample_rate_status": "unknown",
                    "sequence_policy": (
                        "each split is one independent chronological record"
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        peak = float(np.max(np.abs(train_x)))
        view = {
            "schema_version": 1,
            "artifact_type": "blackbox_selection_view",
            "generator": {
                "project_relative_path": "experiments/prepare_blackbox_data.py",
                "sha256": file_sha256(
                    PROJECT_ROOT / "experiments" / "prepare_blackbox_data.py"
                ),
            },
            "source_filename": "BlackBoxData.mat",
            "source_sha256": "a" * 64,
            "available_splits": ["train", "validation"],
            "test_split_available": False,
            "test_path_or_hash_included": False,
            "split_contract": {
                "indexing": "zero_based_half_open",
                "train": {"start": 5_000, "stop": 5_096, "count": 96},
                "validation": {
                    "start": 5_096,
                    "stop": 5_144,
                    "count": 48,
                },
            },
            "normalization_contract": {
                "csv_values_scaled": False,
                "training_input_peak": peak,
                "recommended_common_scale_for_x_and_y": peak,
                "scale_fitted_from": "train_input_only",
            },
            "semantics": {
                "x": "provisional PA/black-box complex input",
                "y": "provisional corresponding complex output",
            },
            "missing_metadata": ["sample_rate_hz", "frame_boundaries"],
            "files_sha256": {
                name: file_sha256(selection / name)
                for name in SELECTION_FILES
            },
        }
        (selection / "selection_view.json").write_text(
            json.dumps(view, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        arrays = {
            "train_x": train_x,
            "train_y": train_y,
            "validation_x": validation_x,
            "validation_y": validation_y,
        }
        return selection, arrays, peak

    @staticmethod
    def _write_config(
        path: Path,
        *,
        selection: Path,
        output: Path,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selection_dir": str(selection),
                    "output_dir": str(output),
                    "dataset_label": "synthetic BlackBox",
                    "expected_source_sha256": "a" * 64,
                    "expected_selection_view_sha256": file_sha256(
                        selection / "selection_view.json"
                    ),
                    "alignment_max_abs_delay": 4,
                    "practical_ridge_tie_db": 0.1,
                    "ridge_values": [1e-8],
                    "mp_candidates": [
                        {
                            "name": "linear_mp",
                            "orders": [1],
                            "delay_count": 2,
                        }
                    ],
                    "gmp_candidates": [
                        {
                            "name": "nonlinear_gmp",
                            "ka": 2,
                            "la": 2,
                            "leading_policy": "causal_leading",
                        }
                    ],
                    "maximum_fit_count": 2,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_verifies_hashes_and_applies_one_train_only_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, arrays, peak = self._write_selection_view(root)
            verified = _verify_selection_view(selection)
            train_x, train_y, validation_x, validation_y, diagnostic = (
                _load_normalized_pairs(selection, scale=peak)
            )
            np.testing.assert_allclose(train_x, arrays["train_x"] / peak)
            np.testing.assert_allclose(train_y, arrays["train_y"] / peak)
            np.testing.assert_allclose(
                validation_x,
                arrays["validation_x"] / peak,
            )
            np.testing.assert_allclose(
                validation_y,
                arrays["validation_y"] / peak,
            )
            self.assertAlmostEqual(
                diagnostic["normalized_train_input_peak"],
                1.0,
            )
            self.assertEqual(
                verified["verified_file_hashes"],
                verified["view"]["files_sha256"],
            )

    def test_selection_is_deterministic_and_manifest_is_selection_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, arrays, _ = self._write_selection_view(root)
            config_a = root / "config_a.json"
            config_b = root / "config_b.json"
            self._write_config(
                config_a,
                selection=selection,
                output=root / "result_a",
            )
            self._write_config(
                config_b,
                selection=selection,
                output=root / "result_b",
            )
            first = select_from_config(config_a)
            second = select_from_config(config_b)

            first_trial = first["selection"]["selected_trial"]
            second_trial = second["selection"]["selected_trial"]
            self.assertEqual(first_trial["candidate_name"], "nonlinear_gmp")
            self.assertEqual(first_trial["candidate_name"], second_trial["candidate_name"])
            self.assertEqual(first_trial["model_family"], second_trial["model_family"])
            self.assertEqual(first_trial["ridge"], second_trial["ridge"])
            self.assertEqual(first_trial["selection_score_db"], second_trial["selection_score_db"])
            self.assertEqual(first["alignment"]["integer_delay_samples"], 2)
            self.assertFalse(
                first["alignment"]["fractional_delay_applied"]
            )
            self.assertTrue(
                first["selection"]["no_post_fit_alignment_or_gain"]
            )
            self.assertEqual(
                first["selection"]["protocol_revision"],
                "blackbox_pa_v2_stability_aware_ridge",
            )
            self.assertEqual(first["selection"]["practical_ridge_tie_db"], 0.1)
            self.assertEqual(len(first["selection"]["topology_representatives"]), 2)
            self.assertEqual(
                first["selection"]["exact_validation_winner"]["candidate_name"],
                first_trial["candidate_name"],
            )
            self.assertAlmostEqual(
                first["normalization"]["normalized_train_input_peak"],
                1.0,
            )
            self.assertFalse(
                first["candidate_grid"]["pa_deployment_operation_limit_applied"]
            )

            serialized = (
                root / "result_a" / "selection_manifest.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn("test_input.csv", serialized)
            self.assertNotIn("test_output.csv", serialized)
            self.assertNotIn("private_release.bin", serialized)
            self.assertIn("selection_view_sha256", first["data_provenance"])
            self.assertIn("sha256", first["selected_model"])
            self.assertTrue((root / "result_a" / "selected_pa.npz").is_file())
            self.assertTrue(
                (root / "result_a" / "completion_manifest.json").is_file()
            )
            loaded = load_frozen_blackbox_pa_selection(root / "result_a")
            loaded_second = load_frozen_blackbox_pa_selection(root / "result_b")
            self.assertEqual(loaded.integer_delay_samples, 2)
            self.assertAlmostEqual(loaded.normalization_scale, first[
                "normalization"
            ]["common_train_only_scale"])
            restored = loaded.model
            restored_second = loaded_second.model
            self.assertIsInstance(restored, GeneralizedMemoryPolynomialPA)
            self.assertIsInstance(restored_second, GeneralizedMemoryPolynomialPA)
            aligned_x, aligned_y = loaded.align_measured_pair(
                arrays["train_x"],
                arrays["train_y"],
            )
            np.testing.assert_array_equal(aligned_x, arrays["train_x"][:-2])
            np.testing.assert_array_equal(aligned_y, arrays["train_y"][2:])
            np.testing.assert_allclose(
                loaded.predict_aligned_source_units(aligned_x),
                loaded.model.predict(aligned_x / loaded.normalization_scale)
                * loaded.normalization_scale,
            )
            self.assertEqual(
                first["data_provenance"]["semantics"]["x"],
                "provisional PA/black-box complex input",
            )
            self.assertIn(
                "sample_rate_hz",
                first["data_provenance"]["missing_metadata"],
            )
            np.testing.assert_array_equal(
                restored.coefficients,
                restored_second.coefficients,
            )

    def test_invalid_selection_hash_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            with (selection / "train_input.csv").open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write("\n")
            config = root / "config.json"
            output = root / "result"
            self._write_config(config, selection=selection, output=output)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                select_from_config(config)
            self.assertFalse(output.exists())

    def test_config_binds_exact_source_and_selection_view_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            config_path = root / "config.json"
            output = root / "result"
            self._write_config(
                config_path,
                selection=selection,
                output=output,
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["expected_selection_view_sha256"] = "b" * 64
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configured selection_view"):
                select_from_config(config_path)
            self.assertFalse(output.exists())

    def test_refuses_to_overwrite_frozen_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            config = root / "config.json"
            self._write_config(
                config,
                selection=selection,
                output=root / "result",
            )
            select_from_config(config)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                select_from_config(config)

    def test_refuses_even_an_existing_empty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            output = root / "result"
            output.mkdir()
            config = root / "config.json"
            self._write_config(config, selection=selection, output=output)
            with self.assertRaisesRegex(FileExistsError, "output directory"):
                select_from_config(config)

    def test_split_contract_and_csv_counts_are_strictly_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            view_path = selection / "selection_view.json"
            view = json.loads(view_path.read_text(encoding="utf-8"))
            view["split_contract"]["train"]["count"] = 95
            view_path.write_text(json.dumps(view), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count == stop - start"):
                _verify_selection_view(selection)

            view["split_contract"]["train"]["stop"] = 5_095
            view["split_contract"]["validation"]["start"] = 5_095
            view["split_contract"]["validation"]["count"] = 49
            view_path.write_text(json.dumps(view), encoding="utf-8")
            verified = _verify_selection_view(selection)
            with self.assertRaisesRegex(ValueError, "CSV row count"):
                _load_normalized_pairs(
                    selection,
                    scale=float(verified["training_input_peak"]),
                    expected_counts=verified["split_contract"],
                )

    def test_unknown_config_and_candidate_keys_are_rejected(self) -> None:
        config = {
            "schema_version": 1,
            "selection_dir": "unused",
            "output_dir": "unused",
            "dataset_label": "unused",
            "expected_source_sha256": "a" * 64,
            "expected_selection_view_sha256": "b" * 64,
            "alignment_max_abs_delay": 1,
            "practical_ridge_tie_db": 0.1,
            "ridge_values": [1e-8],
            "mp_candidates": [
                {"name": "mp", "orders": [1], "delay_count": 1}
            ],
            "gmp_candidates": [
                {
                    "name": "gmp",
                    "ka": 1,
                    "la": 1,
                    "leading_policy": "causal_leading",
                    "surprise": 1,
                }
            ],
            "maximum_fit_count": 2,
        }
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            enumerate_candidate_recipes(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config["gmp_candidates"][0].pop("surprise")
            config["unknown_top_level"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config has unknown keys"):
                _load_config(path)

    def test_strict_view_schema_and_symlink_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            view_path = selection / "selection_view.json"
            view = json.loads(view_path.read_text(encoding="utf-8"))
            view["unexpected"] = True
            view_path.write_text(json.dumps(view), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level schema mismatch"):
                _verify_selection_view(selection)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            spec = selection / "spec.json"
            outside = root / "outside_spec.json"
            outside.write_text("{}\n", encoding="utf-8")
            spec.unlink()
            spec.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "must not escape"):
                _verify_selection_view(selection)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            linked_selection = root / "selection_link"
            linked_selection.symlink_to(selection, target_is_directory=True)
            config = root / "config.json"
            self._write_config(
                config,
                selection=linked_selection,
                output=root / "result",
            )
            with self.assertRaisesRegex(ValueError, "symlink components"):
                select_from_config(config)

    def test_stability_policy_uses_largest_ridge_inside_topology_tie(self) -> None:
        model = MemoryPolynomialPA(
            (1,),
            (0,),
            np.asarray([[1.0 + 0.0j]]),
        )

        def trial(name: str, ridge: float, score: float) -> dict:
            return {
                "candidate_name": name,
                "model_family": "mp",
                "ridge": ridge,
                "selection_score_db": score,
                "operation_count_per_complex_sample": {
                    "real_multiplications": 4,
                    "real_additions": 2,
                    "nonlinear_operations": 0,
                    "lookups": 0,
                    "real_memory_reads": 2,
                    "real_memory_writes": 0,
                    "stored_real_coefficients": 2,
                },
            }

        exact, representatives, selected = _select_topology_representatives(
            [
                (trial("a", 0.0, -30.10), model),
                (trial("a", 1e-6, -30.02), model),
                (trial("b", 0.0, -30.05), model),
                (trial("b", 1e-6, -29.80), model),
            ],
            practical_ridge_tie_db=0.1,
        )
        self.assertEqual(exact[0]["candidate_name"], "a")
        by_name = {item["candidate_name"]: item for item in representatives}
        self.assertEqual(by_name["a"]["representative_trial"]["ridge"], 1e-6)
        self.assertEqual(by_name["b"]["representative_trial"]["ridge"], 0.0)
        self.assertEqual(selected[0]["candidate_name"], "b")

    def test_completion_loader_detects_model_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            output = root / "result"
            config = root / "config.json"
            self._write_config(config, selection=selection, output=output)
            select_from_config(config)
            with (output / "selected_pa.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "completion hash mismatch"):
                load_frozen_blackbox_pa_selection(output)

    def test_failed_staged_publication_leaves_no_output_or_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, _, _ = self._write_selection_view(root)
            output = root / "result"
            config = root / "config.json"
            self._write_config(config, selection=selection, output=output)
            with mock.patch(
                "experiments.select_blackbox_pa.write_json",
                side_effect=OSError("injected publication failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    select_from_config(config)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".result.staging-*")), [])

    def test_production_grid_contains_the_explicit_189_term_candidate(self) -> None:
        config = json.loads(
            (
                Path(__file__).parents[1]
                / "experiments"
                / "configs"
                / "blackbox_pa_v2.json"
            ).read_text(encoding="utf-8")
        )
        recipes = enumerate_candidate_recipes(config)
        matching = [
            recipe
            for recipe in recipes
            if recipe["name"] == "gmp_189_term_causal"
            and recipe["ridge"] == 1e-8
        ]
        self.assertEqual(len(matching), 1)
        model_config = matching[0]["gmp_config"]
        self.assertEqual(model_config.coefficient_count, 189)
        self.assertEqual(model_config.causal_warmup_samples, 9)
        self.assertEqual(len(recipes), 42)
        self.assertEqual(config["practical_ridge_tie_db"], 0.1)
        self.assertEqual(
            config["output_dir"],
            "experiments/results/blackbox_pa_v2_selection",
        )

    def test_both_model_serializers_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mp = MemoryPolynomialPA(
                (1,),
                (0,),
                np.asarray([[1.0 + 0.0j]]),
            )
            gmp = GeneralizedMemoryPolynomialPA(
                GMPConfig(ka=1, la=1),
                np.asarray([1.0 + 0.0j]),
            )
            mp.save(root / "mp.npz")
            gmp.save(root / "gmp.npz")
            self.assertEqual(MemoryPolynomialPA.load(root / "mp.npz"), mp)
            restored = GeneralizedMemoryPolynomialPA.load(root / "gmp.npz")
            np.testing.assert_array_equal(restored.coefficients, gmp.coefficients)


if __name__ == "__main__":
    unittest.main()
