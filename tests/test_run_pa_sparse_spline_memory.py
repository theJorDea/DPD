import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_pa_sparse_spline_memory import run_from_config


PRODUCTION_CONFIG = Path(
    "experiments/configs/pa_sparse_spline_memory_apa200.json"
)


class SparsePARunnerIntegrityTests(unittest.TestCase):
    def test_evidence_tamper_fails_before_waveform_loader(self) -> None:
        config = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
        config["evidence"]["gmp_residual_predictions"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch(
                "experiments.run_pa_sparse_spline_memory.load_split_pair"
            ) as loader:
                with self.assertRaisesRegex(ValueError, "evidence hash mismatch"):
                    run_from_config(path, progress=lambda _: None)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
