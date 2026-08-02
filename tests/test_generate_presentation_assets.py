from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.generate_presentation_assets import (
    EXPECTED_OUTPUTS,
    ROOT,
    STATIC_OUTPUTS,
    binned_summary,
    deterministic_indices,
    generate_assets,
    generate_static_assets,
    scored_mask,
    sha256_file,
)


HAS_RENDER_DEPS = (
    importlib.util.find_spec("matplotlib") is not None
    and importlib.util.find_spec("PIL") is not None
)


class PresentationMathTests(unittest.TestCase):
    def test_scored_mask_discards_warmup_at_every_frame(self) -> None:
        mask = scored_mask(10, nperseg=4, warmup=1)
        np.testing.assert_array_equal(
            mask,
            np.asarray([False, True, True, True, False, True, True, True, False, True]),
        )

    def test_deterministic_indices_keep_endpoints(self) -> None:
        indices = deterministic_indices(101, 5)
        np.testing.assert_array_equal(indices, np.asarray([0, 25, 50, 75, 100]))

    def test_binned_summary_is_not_random(self) -> None:
        x = np.repeat(np.asarray([0.1, 0.3, 0.6, 0.9]), 10)
        y = 2.0 * x
        first = binned_summary(x, y, bins=4, minimum_count=8)
        second = binned_summary(x, y, bins=4, minimum_count=8)
        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(first[1], np.asarray([0.2, 0.6, 1.2, 1.8]))


@unittest.skipUnless(HAS_RENDER_DEPS, "presentation rendering dependencies are optional")
class PresentationRenderingTests(unittest.TestCase):
    def test_fresh_generation_is_sealed_and_surrogate_labelled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dpd_presentation_test_") as temporary:
            output = Path(temporary) / "assets"
            manifest = generate_assets(ROOT, output)
            self.assertEqual(set(output.iterdir()), {output / name for name in EXPECTED_OUTPUTS})
            self.assertTrue(manifest["claims_scope"]["surrogate_only"])
            self.assertFalse(manifest["claims_scope"]["physical_pa_measurement"])
            self.assertFalse(manifest["claims_scope"]["rf_harmonic_claim"])
            self.assertFalse(manifest["rendering_contract"]["training_epochs_claimed"])
            self.assertTrue(manifest["rendering_contract"]["animation_uses_only_saved_states"])
            self.assertEqual(
                manifest["generator_sha256"],
                sha256_file(ROOT / manifest["generator"]),
            )
            self.assertEqual(manifest["renderer_environment"]["numpy"], np.__version__)
            stored = json.loads((output / "presentation_manifest.json").read_text())
            for name, digest in stored["outputs"].items():
                self.assertEqual(sha256_file(output / name), digest)
                self.assertGreater((output / name).stat().st_size, 10_000)
            from PIL import Image

            with Image.open(output / "dpd_overview.gif") as animation:
                self.assertEqual(animation.n_frames, 6)
                self.assertGreaterEqual(animation.width, 1000)

    def test_nonempty_output_requires_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dpd_presentation_guard_") as temporary:
            output = Path(temporary) / "assets"
            generate_static_assets(ROOT, output)
            self.assertEqual(
                set(output.iterdir()),
                {output / name for name in STATIC_OUTPUTS | {"presentation_manifest.json"}},
            )
            with self.assertRaises(FileExistsError):
                generate_static_assets(ROOT, output)

    def test_unknown_output_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dpd_presentation_unknown_") as temporary:
            output = Path(temporary) / "assets"
            output.mkdir()
            (output / "user-owned.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_static_assets(ROOT, output, force=True)
            self.assertEqual((output / "user-owned.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
