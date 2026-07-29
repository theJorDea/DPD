import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baseline.spline_hammerstein_pa import SplineHammersteinPA
from baseline.train_spline import load_split_pair
from experiments.run_pa_sph import (
    _acquire_lock,
    _publish_bundle,
    _reference_metrics_or_raise,
)
from experiments.select_pa_sph import (
    SPHOOFProtocol,
    SPHRecipe,
    load_verified_gmp_oof_prediction,
    verify_sph_preregistered_inputs,
)


class SPHRunnerReferenceTests(unittest.TestCase):
    def test_frozen_gmp_oof_metrics_reproduce_from_measured_train(self) -> None:
        verified = verify_sph_preregistered_inputs(
            "experiments/configs/pa_sph_apa200.json"
        )
        prediction = load_verified_gmp_oof_prediction(verified)
        _, measured = load_split_pair(verified["dataset"], "train")
        contract = verified["config"]["dataset_contract"]
        protocol = SPHOOFProtocol(
            segment_length=int(contract["nperseg"]),
            common_warmup_samples=int(
                contract["common_warmup_samples_per_frame"]
            ),
            common_cooldown_samples=int(
                contract["common_future_cooldown_samples_per_frame"]
            ),
        )
        result = _reference_metrics_or_raise(
            prediction,
            measured,
            protocol,
            verified["config"],
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(
            result["metrics"]["full_record_nmse_db"],
            -38.345410298129714,
            places=12,
        )
        self.assertAlmostEqual(
            result["metrics"]["common_interior_nmse_db"],
            -38.750526106525086,
            places=12,
        )


class SPHRunnerPublicationTests(unittest.TestCase):
    def test_bundle_is_atomic_hashed_and_immutable(self) -> None:
        verified = verify_sph_preregistered_inputs(
            "experiments/configs/pa_sph_apa200.json"
        )
        recipe = SPHRecipe("power_uniform", 2, 1, 0.0, 0.0, 0.0)
        model = SplineHammersteinPA(
            knots=np.asarray([0.0, 1.0]),
            control_points=np.asarray([1.0 + 0.0j, 0.8 - 0.1j]),
            fir_tail=np.asarray([], dtype=np.complex128),
            coordinate="power",
            knot_strategy="power_uniform",
        )
        signal = np.asarray([0.1 + 0.2j, -0.3 + 0.1j, 0.5j, 0.7 + 0.2j])
        prediction = model.predict_segments(signal, 2)
        segment_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
        common = np.ones(4, dtype=bool)
        residual = {
            "schema_version": 1,
            "split_role": "synthetic_test_fixture",
            "test_access_permitted": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "immutable_bundle"
            lock, payload = _acquire_lock(output)
            try:
                manifest = _publish_bundle(
                    output,
                    lock=lock,
                    lock_payload=payload,
                    verified=verified,
                    source_hashes={"synthetic": "0" * 64},
                    staged_ledger={
                        "schema_version": 1,
                        "validation_loaded": False,
                        "test_split_accessed": False,
                    },
                    final_recipe=recipe,
                    decision={
                        "classification": "synthetic_fixture",
                        "gate_a_to_b_opened": False,
                    },
                    model=model,
                    fit_diagnostics={"synthetic": True},
                    reference_reproduction={"passed": True},
                    oof_metrics={"full_record_nmse_db": -20.0},
                    train_metrics={"full_record_nmse_db": -21.0},
                    validation_metrics={"full_record_nmse_db": -19.0},
                    train_support={"count_above_fit_maximum": 0},
                    validation_support={"count_above_fit_maximum": 0},
                    stream_checks={"all": True},
                    train_oof_prediction=prediction,
                    reference_gmp_oof_prediction=prediction,
                    train_prediction=prediction,
                    validation_prediction=prediction,
                    train_segment_id=segment_id,
                    validation_segment_id=segment_id,
                    train_common_mask=common,
                    validation_common_mask=common,
                    train_residual_analysis=residual,
                    validation_residual_analysis=residual,
                    execution={
                        "schema_version": 1,
                        "test_split_accessed": False,
                    },
                    input_reverification={"synthetic": True},
                )
                self.assertTrue(output.is_dir())
                self.assertFalse(manifest["test_split_accessed"])
                self.assertFalse(manifest["test_file_hashes_recorded"])
                self.assertTrue(
                    manifest["input_integrity"]["test_never_opened_or_hashed"]
                )
                on_disk = json.loads(
                    (output / "selection_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(on_disk["final_recipe_sha256"], recipe.canonical_sha256)
                for artifact in on_disk["artifacts"].values():
                    self.assertTrue((output / artifact["path"]).is_file())
                restored = SplineHammersteinPA.load(
                    output / "selected_sph_pa.npz"
                )
                np.testing.assert_array_equal(
                    restored.control_points,
                    model.control_points,
                )
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    _acquire_lock(output)
            finally:
                if lock.exists():
                    lock.unlink()


if __name__ == "__main__":
    unittest.main()
