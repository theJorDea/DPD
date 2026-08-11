from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
import tempfile
import unittest

import numpy as np

from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA
from baseline.complex_spline_dpd import make_knots
from baseline.spline_memory_dpd import (
    SparseSplineMemoryDPD,
    SplineMemoryBranch,
    spline_memory_design_matrix,
)
from baseline.train_spline import file_sha256
from experiments.select_blackbox_dpd import (
    FrozenBlackBoxDPDSelection,
    _evaluate_factorized_trial,
    _evaluator_headroom_diagnostic,
    _improvement_gate,
    _load_config,
    choose_validation_winner,
    enumerate_candidate_recipes,
    factorize_spline_group,
    load_frozen_blackbox_dpd_selection,
    pareto_frontier,
    select_dpd_candidate,
    select_from_config,
    solve_grouped_spline,
)
from experiments.select_blackbox_pa import PROJECT_ROOT, SELECTION_FILES


class BlackBoxDPDSelectionTests(unittest.TestCase):
    @staticmethod
    def _write_iq(path: Path, signal: np.ndarray) -> None:
        rows = ["I,Q"]
        rows.extend(
            f"{value.real:.17g},{value.imag:.17g}" for value in signal
        )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    @staticmethod
    def _pa_truth() -> GeneralizedMemoryPolynomialPA:
        # y = x * (1 + (0.28+0.09j)|x|^2), represented by the aligned
        # q=0,1,2 GMP terms.  It is monotone over the generated amplitudes.
        return GeneralizedMemoryPolynomialPA(
            GMPConfig(ka=3, la=1),
            np.asarray([1.0, 0.0, 0.28 + 0.09j], dtype=np.complex128),
        )

    def _write_selection_and_pa_bundle(
        self,
        root: Path,
    ) -> tuple[Path, Path]:
        prepared = root / "prepared"
        selection = prepared / "selection"
        selection.mkdir(parents=True)
        sealed = prepared / "sealed"
        sealed.mkdir()
        (sealed / "private_release.bin").write_bytes(b"must never be read")

        rng = np.random.default_rng(1408)

        def record(count: int, maximum_radius: float) -> tuple[np.ndarray, np.ndarray]:
            radius = rng.uniform(0.03, maximum_radius, count)
            phase = rng.uniform(-np.pi, np.pi, count)
            normalized_x = radius * np.exp(1j * phase)
            raw_scale = 8.0
            raw_x = raw_scale * normalized_x
            raw_y = raw_scale * self._pa_truth().predict(normalized_x)
            return raw_x, raw_y

        train_x, train_y = record(160, 1.0)
        validation_x, validation_y = record(96, 0.72)
        self._write_iq(selection / "train_input.csv", train_x)
        self._write_iq(selection / "train_output.csv", train_y)
        self._write_iq(selection / "val_input.csv", validation_x)
        self._write_iq(selection / "val_output.csv", validation_y)
        (selection / "spec.json").write_text(
            json.dumps(
                {
                    "sample_rate_status": "unknown",
                    "sequence_policy": "one record per split",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        scale = float(np.max(np.abs(train_x)))
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
            "source_sha256": "b" * 64,
            "available_splits": ["train", "validation"],
            "test_split_available": False,
            "test_path_or_hash_included": False,
            "split_contract": {
                "indexing": "zero_based_half_open",
                "train": {"start": 5000, "stop": 5160, "count": 160},
                "validation": {"start": 5160, "stop": 5256, "count": 96},
            },
            "normalization_contract": {
                "csv_values_scaled": False,
                "training_input_peak": scale,
                "recommended_common_scale_for_x_and_y": scale,
                "scale_fitted_from": "train_input_only",
            },
            "semantics": {
                "x": "provisional PA/black-box complex input",
                "y": "provisional corresponding complex output",
                "status": "must_be_confirmed_by_data_owner",
            },
            "missing_metadata": [
                "sample_rate_hz",
                "occupied_bandwidth_hz",
                "adjacent_or_harmonic_regions_hz",
            ],
            "files_sha256": {
                name: file_sha256(selection / name) for name in SELECTION_FILES
            },
        }
        view_path = selection / "selection_view.json"
        view_path.write_text(
            json.dumps(view, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        pa_bundle = root / "pa_bundle"
        pa_bundle.mkdir()
        model_path = pa_bundle / "selected_pa.npz"
        trials_path = pa_bundle / "validation_trials.json"
        manifest_path = pa_bundle / "selection_manifest.json"
        self._pa_truth().save(model_path)
        trials_path.write_text(
            json.dumps({"schema_version": 1, "trials": []}) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "task": "blackbox_forward_pa_identification_model_selection",
            "data_provenance": {
                "selection_view_sha256": file_sha256(view_path),
                "verified_selection_files_sha256": view["files_sha256"],
                "split_contract": {
                    "train": view["split_contract"]["train"],
                    "validation": view["split_contract"]["validation"],
                },
            },
            "normalization": {"common_train_only_scale": scale},
            "alignment": {"integer_delay_samples": 0},
            "selection": {"selected_trial": {"model_family": "gmp"}},
            "selected_model": {
                "model_family": "gmp",
                "path": str(model_path),
                "sha256": file_sha256(model_path),
            },
            "artifacts": {
                "validation_trials_sha256": file_sha256(trials_path)
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completion = {
            "schema_version": 1,
            "artifact_type": "blackbox_pa_selection_completion",
            "status": "complete",
            "bound_files_sha256": {
                "selection_manifest.json": file_sha256(manifest_path),
                "selected_pa.npz": file_sha256(model_path),
                "validation_trials.json": file_sha256(trials_path),
            },
        }
        (pa_bundle / "completion_manifest.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return selection, pa_bundle

    @staticmethod
    def _config_value(
        selection: Path,
        pa_bundle: Path,
        output: Path,
        *,
        headroom_db: float = 10.0,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "selection_dir": str(selection),
            "expected_source_sha256": "b" * 64,
            "expected_selection_view_sha256": file_sha256(
                selection / "selection_view.json"
            ),
            "pa_bundle_dir": str(pa_bundle),
            "expected_pa_completion_sha256": file_sha256(
                pa_bundle / "completion_manifest.json"
            ),
            "output_dir": str(output),
            "dataset_label": "synthetic BlackBox",
            "knot_strategy": "quantile",
            "knot_counts": [8],
            "ridge_values": [0, 1e-6],
            "branch_topologies": [
                {
                    "name": "memoryless",
                    "branches": [
                        {"signal_delay": 0, "envelope_delay": 0}
                    ],
                }
            ],
            "selection_tolerance_db": 0.1,
            "evaluator_headroom_gate_db": headroom_db,
            "maximum_fit_count": 2,
            "psd_nperseg": 16,
            "psd_noverlap": 8,
        }

    def _write_config(
        self,
        path: Path,
        selection: Path,
        pa_bundle: Path,
        output: Path,
        *,
        headroom_db: float = 10.0,
    ) -> None:
        path.write_text(
            json.dumps(
                self._config_value(
                    selection,
                    pa_bundle,
                    output,
                    headroom_db=headroom_db,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _trial(
        name: str,
        score: float,
        multiplications: int,
        *,
        peak: float = 1.0,
    ) -> dict[str, object]:
        return {
            "topology_name": name,
            "knot_count": 8,
            "ridge": 0.0,
            "hard_valid": True,
            "selection_score_db": score,
            "validation_correct_direction": {
                "perfect_reconstruction": False
            },
            "validation_drive": {"maximum_amplitude": peak},
            "operation_count_per_complex_sample": {
                "real_multiplications": multiplications,
                "real_additions": multiplications,
                "real_divisions": 0,
                "nonlinear_operations": 1,
                "comparisons": 3,
                "lookups": 2,
                "real_memory_reads": 4,
                "real_memory_writes": 0,
                "state_real_values": 0,
                "stored_real_coefficients": 16,
                "stored_real_constants": 15,
            },
        }

    def test_full_declared_grid_has_490_fits_and_expected_costs(self) -> None:
        path = PROJECT_ROOT / "experiments" / "configs" / "blackbox_dpd_v1.json"
        recipes = enumerate_candidate_recipes(_load_config(path))
        self.assertEqual(len(recipes), 490)
        cost_by_topology = {}
        for recipe in recipes:
            if recipe["knot_count"] == 8 and recipe["ridge"] == 0.0:
                cost_by_topology[recipe["topology_name"]] = (
                    recipe["operation_count"].real_multiplications,
                    recipe["operation_count"].real_additions,
                    recipe["operation_count"].nonlinear_operations,
                    recipe["operation_count"].lookups,
                )
        self.assertEqual(cost_by_topology["memoryless"], (9, 8, 1, 2))
        self.assertEqual(cost_by_topology["current_012"], (21, 24, 1, 6))
        self.assertEqual(cost_by_topology["aligned_012"], (27, 28, 3, 6))

    def test_grouped_svd_matches_direct_steady_augmented_lstsq(self) -> None:
        rng = np.random.default_rng(248)
        calibration = (
            rng.normal(size=96) + 1j * rng.normal(size=96)
        ) / 3.0
        branches = (
            SplineMemoryBranch(0, 0),
            SplineMemoryBranch(2, 1),
        )
        warmup = 2
        knots = make_knots(
            calibration,
            7,
            "uniform_amplitude",
        )
        design = spline_memory_design_matrix(
            calibration,
            knots,
            branches,
        )
        true_coefficients = rng.normal(size=design.shape[1]) + 1j * rng.normal(
            size=design.shape[1]
        )
        target = design @ true_coefficients
        target += 1e-4 * (
            rng.normal(size=target.size) + 1j * rng.normal(size=target.size)
        )
        factorization = factorize_spline_group(
            calibration,
            target,
            knots=knots,
            branches=branches,
        )
        self.assertEqual(factorization.causal_warmup_samples, warmup)
        self.assertEqual(factorization.steady_sample_count, target.size - warmup)

        steady_design = design[warmup:]
        steady_target = target[warmup:]
        count = steady_target.size
        normalized_design = steady_design / np.sqrt(float(count))
        normalized_target = steady_target / np.sqrt(float(count))
        for ridge in (0.0, 1e-8, 1e-3):
            model, _ = solve_grouped_spline(
                factorization,
                ridge=ridge,
                knot_strategy="uniform_amplitude",
            )
            if ridge == 0.0:
                augmented_design = normalized_design
                augmented_target = normalized_target
            else:
                augmented_design = np.vstack(
                    (
                        normalized_design,
                        np.sqrt(ridge)
                        * np.eye(design.shape[1], dtype=np.complex128),
                    )
                )
                augmented_target = np.concatenate(
                    (
                        normalized_target,
                        np.zeros(design.shape[1], dtype=np.complex128),
                    )
                )
            direct, _, direct_rank, _ = np.linalg.lstsq(
                augmented_design,
                augmented_target,
                rcond=None,
            )
            np.testing.assert_allclose(
                model.coefficients.reshape(-1),
                direct,
                rtol=2e-10,
                atol=2e-11,
            )
            if ridge == 0.0:
                self.assertEqual(factorization.data_design_rank, direct_rank)

        changed_target = target.copy()
        changed_target[:warmup] += 1e6 + 2e6j
        changed = factorize_spline_group(
            calibration,
            changed_target,
            knots=knots,
            branches=branches,
        )
        np.testing.assert_array_equal(
            factorization.projected_target,
            changed.projected_target,
        )

    def test_improvement_gate_sign_and_all_worse_policy(self) -> None:
        no_dpd = {"mse": 1.0}
        improved = _improvement_gate({"mse": 0.5}, no_dpd)
        worsened = _improvement_gate({"mse": 2.0}, no_dpd)
        self.assertTrue(improved["pass"])
        self.assertGreater(
            improved["improvement_db_10log10_no_dpd_over_candidate"],
            0.0,
        )
        self.assertFalse(worsened["pass"])
        self.assertLess(
            worsened["improvement_db_10log10_no_dpd_over_candidate"],
            0.0,
        )

        first = self._trial("first", -18.0, 9)
        second = self._trial("second", -17.0, 15)
        for trial in (first, second):
            trial["hard_valid"] = False
            trial["eligible_for_diagnostic_selection"] = True
            trial["hard_invalid_reasons"] = ["does_not_improve_no_dpd"]
        selected, policy = select_dpd_candidate(
            [first, second],
            tolerance_db=0.1,
        )
        self.assertEqual(selected["topology_name"], "first")
        self.assertFalse(policy["deployment_recommended"])
        self.assertTrue(policy["no_dpd_is_recommended_when_false"])

    def test_evaluator_headroom_margin_sign_pass_and_fail(self) -> None:
        passed = _evaluator_headroom_diagnostic(
            {"mse": 0.1},
            {"mse": 0.001},
            required_margin_db=10.0,
        )
        failed = _evaluator_headroom_diagnostic(
            {"mse": 0.005},
            {"mse": 0.001},
            required_margin_db=10.0,
        )
        negative = _evaluator_headroom_diagnostic(
            {"mse": 0.0001},
            {"mse": 0.001},
            required_margin_db=10.0,
        )
        self.assertTrue(passed["pass"])
        self.assertAlmostEqual(
            passed["margin_db_10log10_dpd_error_over_pa_model_error"],
            20.0,
        )
        self.assertFalse(failed["pass"])
        self.assertGreater(
            failed["margin_db_10log10_dpd_error_over_pa_model_error"],
            0.0,
        )
        self.assertFalse(negative["pass"])
        self.assertLess(
            negative["margin_db_10log10_dpd_error_over_pa_model_error"],
            0.0,
        )

    def test_rank_and_drive_support_are_hard_invalid(self) -> None:
        pa = GeneralizedMemoryPolynomialPA(
            GMPConfig(ka=1, la=1),
            np.asarray([1.0 + 0.0j]),
        )
        calibration = np.ones(32, dtype=np.complex128) * 0.5
        train_target = calibration * 2.0
        branches = (SplineMemoryBranch(0, 0),)
        knots = np.asarray([0.0, 0.25, 0.5, 0.75])
        factorization = factorize_spline_group(
            calibration,
            train_target,
            knots=knots,
            branches=branches,
        )
        validation = np.ones(16, dtype=np.complex128) * 0.5
        no_dpd_metrics = {
            "mse": 1.0,
            "complex_nmse_pooled_db": 0.0,
        }
        recipe = {
            "topology_name": "rank_and_support",
            "branches": branches,
            "knot_count": 4,
            "ridge": 0.0,
        }
        trial, _ = _evaluate_factorized_trial(
            recipe,
            factorization=factorization,
            calibration_input=calibration,
            train_x=train_target,
            validation_x=validation,
            pa_model=pa,
            gain=1.0 + 0.0j,
            common_warmup_samples=0,
            pa_train_input_peak=0.75,
            knot_strategy="explicit",
            no_dpd_metrics=no_dpd_metrics,
        )
        self.assertFalse(trial["hard_valid"])
        self.assertIn(
            "steady_data_design_rank_deficient",
            trial["structural_invalid_reasons"],
        )
        self.assertIn(
            "predistorted_drive_outside_train_pa_input_support",
            trial["structural_invalid_reasons"],
        )

    def test_tolerance_rule_chooses_cheapest_near_best(self) -> None:
        best = self._trial("best", -30.0, 27)
        cheap = self._trial("cheap", -29.92, 9)
        too_far = self._trial("too_far", -29.89, 5)
        selected = choose_validation_winner(
            [best, cheap, too_far],
            tolerance_db=0.1,
        )
        self.assertEqual(selected["topology_name"], "cheap")

    def test_pareto_frontier_removes_dominated_trial(self) -> None:
        quality = self._trial("quality", -31.0, 20, peak=1.0)
        cheap = self._trial("cheap", -30.0, 9, peak=0.9)
        dominated = self._trial("dominated", -29.0, 25, peak=1.1)
        names = {
            trial["topology_name"]
            for trial in pareto_frontier([quality, cheap, dominated])
        }
        self.assertEqual(names, {"quality", "cheap"})

    def test_correct_direction_selection_is_atomic_and_never_opens_sealed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            output = root / "dpd_bundle"
            config = root / "config.json"
            self._write_config(config, selection, pa_bundle, output)

            original_open = Path.open

            def guarded_open(path: Path, *args: object, **kwargs: object):
                if "sealed" in path.parts:
                    raise AssertionError(f"sealed path was accessed: {path}")
                return original_open(path, *args, **kwargs)

            with (
                mock.patch.object(Path, "open", guarded_open),
                mock.patch(
                    "experiments.select_blackbox_dpd.factorize_spline_group",
                    wraps=factorize_spline_group,
                ) as factorizer,
            ):
                manifest = select_from_config(config)
            self.assertEqual(factorizer.call_count, 1)

            self.assertTrue((output / "completion_manifest.json").is_file())
            loaded = load_frozen_blackbox_dpd_selection(output)
            self.assertEqual(loaded.model.branch_count, 1)
            self.assertTrue(
                manifest["directions"]["measured_validation_y_used_as_dpd_input"]
                is False
            )
            self.assertTrue(
                manifest["evaluator_headroom_gate"]["computed_after_selection"]
            )
            self.assertFalse(
                manifest["evaluator_headroom_gate"]["used_for_ranking"]
            )
            self.assertTrue(
                manifest["evaluator_headroom_gate"]["diagnostic_only"]
            )
            self.assertFalse(
                manifest["evaluator_headroom_gate"]["independent_confirmation"]
            )
            self.assertFalse(
                manifest["spectral_artifact"]["aclr_or_harmonic_metrics_computed"]
            )
            evaluator = manifest["frozen_pa_evaluator"]
            self.assertEqual(
                evaluator["expected_completion_manifest_sha256"],
                evaluator["actual_completion_manifest_sha256"],
            )
            self.assertTrue(
                evaluator["expected_actual_completion_hash_match"]
            )
            self.assertLess(
                manifest["validation_summary"]["selected_dpd"]
                ["complex_nmse_pooled_db"],
                manifest["validation_summary"]
                ["surrogate_no_dpd_reference_used_for_dpd_ranking"]
                ["complex_nmse_pooled_db"],
            )
            self.assertFalse(
                manifest["validation_summary"]
                ["measured_validation_no_dpd_capture"]
                ["used_for_dpd_ranking"]
            )
            frontier = json.loads(
                (output / "pareto_frontier.json").read_text(encoding="utf-8")
            )
            self.assertTrue(frontier["no_dpd_reference_included"])
            self.assertIn(
                "no_dpd",
                {item["topology_name"] for item in frontier["frontier"]},
            )
            self.assertEqual(
                manifest["calibration_timing"]["factorization_count"],
                1,
            )
            self.assertGreater(
                manifest["calibration_timing"]
                ["selection_wall_seconds_before_publication"],
                0.0,
            )
            wrapper = manifest["selected_model"]["source_unit_safety_wrapper"]
            self.assertTrue(
                wrapper[
                    "checks_each_output_or_chunk_against_train_pa_input_support"
                ]
            )
            self.assertFalse(
                wrapper[
                    "overhead_included_in_analytical_optimized_datapath_count"
                ]
            )
            self.assertEqual(
                manifest["candidate_grid"]["operation_count_class"],
                "analytical_optimized_datapath",
            )

            source = np.asarray(
                [0.3 + 0.1j, -0.2 + 0.15j, 0.1 - 0.25j],
                dtype=np.complex128,
            ) * loaded.normalization_scale
            manual = (
                loaded.model.predict(source / loaded.normalization_scale)
                * loaded.normalization_scale
            )
            np.testing.assert_allclose(loaded.predict(source), manual)
            state = loaded.initial_state()
            first, state = loaded.predict_chunk(source[:1], state)
            second, state = loaded.predict_chunk(source[1:], state)
            np.testing.assert_allclose(
                np.concatenate((first, second)),
                loaded.predict(source),
            )
            serialized = (output / "selection_manifest.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("private_release.bin", serialized)
            self.assertNotIn("test_input.csv", serialized)
            self.assertNotIn("test_output.csv", serialized)

    def test_source_unit_wrapper_rejects_output_drive_outside_pa_support(
        self,
    ) -> None:
        model = SparseSplineMemoryDPD(
            knots=np.asarray([0.0, 1.0], dtype=np.float64),
            branches=(SplineMemoryBranch(0, 0),),
            coefficients=np.asarray([[2.0, 2.0]], dtype=np.complex128),
        )
        frozen = FrozenBlackBoxDPDSelection(
            model=model,
            normalization_scale=10.0,
            manifest={
                "selection": {
                    "selected_trial": {
                        "support_checks": {
                            "maximum_train_pa_input_amplitude": 1.0,
                        }
                    }
                }
            },
        )
        np.testing.assert_allclose(
            frozen.predict(np.asarray([4.0 + 0.0j])),
            np.asarray([8.0 + 0.0j]),
        )
        with self.assertRaisesRegex(
            ValueError,
            "output exceeds frozen train PA-input support",
        ):
            frozen.predict(np.asarray([6.0 + 0.0j]))
        with self.assertRaisesRegex(
            ValueError,
            "output exceeds frozen train PA-input support",
        ):
            frozen.predict_chunk(
                np.asarray([6.0 + 0.0j]),
                frozen.initial_state(),
            )

    def test_headroom_threshold_does_not_change_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            config_a = root / "a.json"
            config_b = root / "b.json"
            self._write_config(
                config_a,
                selection,
                pa_bundle,
                root / "result_a",
                headroom_db=0.0,
            )
            self._write_config(
                config_b,
                selection,
                pa_bundle,
                root / "result_b",
                headroom_db=100.0,
            )
            first = select_from_config(config_a)
            second = select_from_config(config_b)
            first_selected = first["selection"]["selected_trial"]
            second_selected = second["selection"]["selected_trial"]
            self.assertEqual(
                (
                    first_selected["topology_name"],
                    first_selected["knot_count"],
                    first_selected["ridge"],
                ),
                (
                    second_selected["topology_name"],
                    second_selected["knot_count"],
                    second_selected["ridge"],
                ),
            )
            self.assertEqual(
                file_sha256(root / "result_a" / "selected_dpd.npz"),
                file_sha256(root / "result_b" / "selected_dpd.npz"),
            )

    def test_tampered_pa_bundle_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            with (pa_bundle / "selected_pa.npz").open("ab") as stream:
                stream.write(b"tamper")
            output = root / "dpd_bundle"
            config = root / "config.json"
            self._write_config(config, selection, pa_bundle, output)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                select_from_config(config)
            self.assertFalse(output.exists())

    def test_unpinned_pa_completion_is_rejected_before_data_or_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            output = root / "dpd_bundle"
            config_value = self._config_value(selection, pa_bundle, output)
            config_value["expected_pa_completion_sha256"] = "0" * 64
            config = root / "config.json"
            config.write_text(json.dumps(config_value), encoding="utf-8")

            with (
                mock.patch(
                    "experiments.select_blackbox_dpd._verify_selection_view"
                ) as verify_view,
                mock.patch(
                    "experiments.select_blackbox_dpd."
                    "load_frozen_blackbox_pa_selection"
                ) as load_pa,
                mock.patch(
                    "experiments.select_blackbox_dpd."
                    "factorize_spline_group"
                ) as fit_dpd,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "completion SHA-256 mismatch",
                ):
                    select_from_config(config)
            verify_view.assert_not_called()
            load_pa.assert_not_called()
            fit_dpd.assert_not_called()
            self.assertFalse(output.exists())

    def test_source_and_selection_view_pins_precede_data_load_and_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            for key, message in (
                ("expected_selection_view_sha256", "selection_view SHA-256"),
                ("expected_source_sha256", "source SHA-256"),
            ):
                output = root / f"output_{key}"
                value = self._config_value(selection, pa_bundle, output)
                value[key] = "0" * 64
                config = root / f"{key}.json"
                config.write_text(json.dumps(value), encoding="utf-8")
                with (
                    mock.patch(
                        "experiments.select_blackbox_dpd._load_normalized_pairs"
                    ) as load_data,
                    mock.patch(
                        "experiments.select_blackbox_dpd.factorize_spline_group"
                    ) as factorize,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        select_from_config(config)
                load_data.assert_not_called()
                factorize.assert_not_called()
                self.assertFalse(output.exists())

    def test_selection_symlink_is_rejected_before_view_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            selection_link = root / "selection_link"
            selection_link.symlink_to(selection, target_is_directory=True)
            output = root / "output"
            value = self._config_value(selection, pa_bundle, output)
            value["selection_dir"] = str(selection_link)
            config = root / "config.json"
            config.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch(
                "experiments.select_blackbox_dpd._verify_selection_view"
            ) as verify_view:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    select_from_config(config)
            verify_view.assert_not_called()
            self.assertFalse(output.exists())

    def test_all_dpd_worse_publishes_diagnostic_not_deployment_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            output = root / "dpd_bundle"
            config = root / "config.json"
            self._write_config(config, selection, pa_bundle, output)

            def forced_failure(
                candidate_metrics: dict[str, object],
                no_dpd_metrics: dict[str, object],
            ) -> dict[str, object]:
                return {
                    "pass": False,
                    "candidate_error_power": candidate_metrics["mse"],
                    "no_dpd_error_power": no_dpd_metrics["mse"],
                    "improvement_db_10log10_no_dpd_over_candidate": -1.0,
                    "policy": (
                        "strictly lower correct-direction validation error power"
                    ),
                }

            with mock.patch(
                "experiments.select_blackbox_dpd._improvement_gate",
                side_effect=forced_failure,
            ):
                manifest = select_from_config(config)
            self.assertFalse(
                manifest["selection"]["deployment_recommended"]
            )
            self.assertTrue(
                manifest["selection"]["no_dpd_is_recommended_when_false"]
            )
            self.assertFalse(
                manifest["selected_model"]["deployment_recommended"]
            )
            self.assertIn(
                "diagnostic",
                manifest["selected_model"]["artifact_role"],
            )

    def test_staged_failure_leaves_no_output_or_temporary_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            output = root / "dpd_bundle"
            config = root / "config.json"
            self._write_config(config, selection, pa_bundle, output)
            with mock.patch(
                "experiments.select_blackbox_dpd.np.savez",
                side_effect=RuntimeError("injected PSD publication failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected PSD"):
                    select_from_config(config)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".dpd_bundle.staging-*")), [])

    def test_unknown_config_key_and_missing_zero_ridge_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            value = self._config_value(
                selection,
                pa_bundle,
                root / "output",
            )
            value["surprise"] = True
            path = root / "unknown.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                _load_config(path)

            value.pop("surprise")
            value["ridge_values"] = [1e-6]
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must include 0"):
                enumerate_candidate_recipes(_load_config(path))

    def test_completed_dpd_bundle_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, pa_bundle = self._write_selection_and_pa_bundle(root)
            output = root / "dpd_bundle"
            config = root / "config.json"
            self._write_config(config, selection, pa_bundle, output)
            select_from_config(config)
            with (output / "pareto_frontier.json").open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_frozen_blackbox_dpd_selection(output)


if __name__ == "__main__":
    unittest.main()
