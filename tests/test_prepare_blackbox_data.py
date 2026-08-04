from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import scipy.io
except ImportError:  # pragma: no cover - optional dependency
    scipy = None

from baseline.train_spline import load_complex_iq_csv
from experiments.prepare_blackbox_data import (
    file_sha256,
    load_blackbox_mat,
    prepare_blackbox_data,
)


@unittest.skipIf(scipy is None, "optional scipy dependency is unavailable")
class PrepareBlackBoxDataTests(unittest.TestCase):
    def _write_source(
        self,
        directory: Path,
        *,
        sample_count: int = 20,
        override: dict[str, np.ndarray] | None = None,
    ) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
        index = np.arange(sample_count, dtype=np.float64)
        x = (index + 1j * (100.0 + index))[None, :]
        y = (2.0 * index + 1j * (200.0 + 3.0 * index))[None, :]
        eref = (0.1 * index + 1j * (0.2 + 0.05 * index))[None, :]
        payload = {"x": x, "y": y, "eRef": eref}
        if override:
            payload.update(override)
        source = directory / "BlackBoxData.mat"
        scipy.io.savemat(source, payload)
        return source, x.reshape(-1), y.reshape(-1), eref.reshape(-1)

    def test_exports_exact_chronological_slices_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, x, y, _ = self._write_source(root)
            output = root / "prepared"
            manifest = prepare_blackbox_data(
                source,
                output,
                train_start=2,
                validation_start=10,
                test_start=15,
            )

            np.testing.assert_array_equal(
                load_complex_iq_csv(output / "selection" / "train_input.csv"),
                x[2:10],
            )
            np.testing.assert_array_equal(
                load_complex_iq_csv(output / "selection" / "train_output.csv"),
                y[2:10],
            )
            np.testing.assert_array_equal(
                load_complex_iq_csv(output / "selection" / "val_input.csv"),
                x[10:15],
            )
            np.testing.assert_array_equal(
                load_complex_iq_csv(output / "sealed" / "test_output.csv"),
                y[15:],
            )
            self.assertEqual(manifest["split_contract"]["overlap_samples"], 0)
            self.assertEqual(manifest["split_contract"]["train"]["count"], 8)
            self.assertEqual(manifest["split_contract"]["validation"]["count"], 5)
            self.assertEqual(manifest["split_contract"]["test"]["count"], 5)
            self.assertFalse(
                manifest["semantics"]["eRef_used_as_model_input_or_target"]
            )
            self.assertEqual(manifest["source"]["sha256"], file_sha256(source))
            selection_view = json.loads(
                (output / "selection" / "selection_view.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(selection_view["test_split_available"])
            serialized_selection = json.dumps(selection_view)
            self.assertNotIn("test_input.csv", serialized_selection)
            self.assertNotIn("test_output.csv", serialized_selection)
            self.assertNotIn("test", selection_view["split_contract"])
            saved = json.loads(
                (output / "preparation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved, manifest)

    def test_records_train_only_peak_without_scaling_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, x, _, _ = self._write_source(root)
            manifest = prepare_blackbox_data(
                source,
                root / "prepared",
                train_start=2,
                validation_start=10,
                test_start=15,
            )
            expected_peak = float(np.max(np.abs(x[2:10])))
            contract = manifest["normalization_contract"]
            self.assertEqual(contract["training_input_peak"], expected_peak)
            self.assertFalse(contract["csv_values_scaled"])

    def test_refuses_existing_owned_outputs_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, _, _ = self._write_source(root)
            output = root / "prepared"
            prepare_blackbox_data(
                source,
                output,
                train_start=2,
                validation_start=10,
                test_start=15,
            )
            with self.assertRaises(FileExistsError):
                prepare_blackbox_data(
                    source,
                    output,
                    train_start=2,
                    validation_start=10,
                    test_start=15,
                )

    def test_zero_eref_is_serialized_without_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero = np.zeros((1, 20), dtype=np.complex128)
            source, _, _, _ = self._write_source(root, override={"eRef": zero})
            output = root / "prepared"
            manifest = prepare_blackbox_data(
                source,
                output,
                train_start=2,
                validation_start=10,
                test_start=15,
            )
            self.assertEqual(manifest["eRef_diagnostic"]["power_relative_to_y"], 0.0)
            self.assertIsNone(manifest["eRef_diagnostic"]["power_relative_to_y_db"])
            json.loads((output / "preparation_manifest.json").read_text())

    def test_failed_validation_does_not_publish_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero_y = np.zeros((1, 20), dtype=np.complex128)
            source, _, _, _ = self._write_source(root, override={"y": zero_y})
            output = root / "prepared"
            with self.assertRaisesRegex(ValueError, "non-zero average power"):
                prepare_blackbox_data(
                    source,
                    output,
                    train_start=2,
                    validation_start=10,
                    test_start=15,
                )
            self.assertFalse(output.exists())

    def test_refuses_unprotected_output_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, _, _ = self._write_source(root)
            unsafe = Path(__file__).resolve().parents[1] / "unsafe_capture_export"
            with self.assertRaisesRegex(ValueError, "data/private"):
                prepare_blackbox_data(
                    source,
                    unsafe,
                    train_start=2,
                    validation_start=10,
                    test_start=15,
                )
            self.assertFalse(unsafe.exists())

    def test_rejects_missing_or_invalid_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, _, _ = self._write_source(root)
            scipy.io.savemat(source, {"x": np.ones((1, 20), dtype=np.complex128)})
            with self.assertRaisesRegex(ValueError, "missing variables"):
                load_blackbox_mat(source)

            bad = np.ones((1, 20), dtype=np.complex128)
            bad[0, 4] = np.nan + 1j
            source, _, _, _ = self._write_source(root, override={"eRef": bad})
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                load_blackbox_mat(source)

    def test_rejects_invalid_split_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, _, _ = self._write_source(root)
            with self.assertRaisesRegex(ValueError, "split boundaries"):
                prepare_blackbox_data(
                    source,
                    root / "prepared",
                    train_start=2,
                    validation_start=15,
                    test_start=15,
                )


if __name__ == "__main__":
    unittest.main()
