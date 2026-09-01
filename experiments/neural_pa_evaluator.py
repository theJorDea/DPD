"""Evaluator wrapper for OpenDPD neural PA surrogate checkpoints.

Loads a checkpoint produced by ``train_opendpd_neural_surrogate`` and
exposes the frame-wise offline evaluation convention used for training:

    predict(signal) = concat over 200-sample frames of model(frame),
    each frame processed with a fresh recurrent state.

This is the surrogate's own training convention (frame resets, no warmup
discard), so its numbers are comparable to the training/validation NMSE
reports, NOT to the causal full-record convention of the polynomial
evaluators.  Use it as an independent third judge for transferability
checks, never as a deployed causal model.

Requires torch at runtime (CPU build is sufficient for evaluation).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "OpenDPD"


class NeuralPASurrogateEvaluator:
    """Frame-wise offline evaluator around a trained OpenDPD backbone."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        checkpoint_path = Path(checkpoint_path)
        if not VENDOR_ROOT.is_dir():
            raise FileNotFoundError("vendor/OpenDPD is not checked out")
        sys.path.insert(0, str(VENDOR_ROOT))
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        self.input_scale = float(checkpoint["input_scale"])
        self.frame_length = int(checkpoint["frame_length"])
        spec = checkpoint["model_spec"]
        model_name = str(spec["type"])
        if model_name == "tres_gru":
            from backbones.tres_gru import TResGRU

            model = TResGRU(
                input_size=2,
                hidden_size=int(spec["hidden_size"]),
                output_size=2,
                num_layers=int(spec.get("num_layers", 1)),
            )
        elif model_name == "tres_deltagru":
            from backbones.tres_deltagru import TResDeltaGRU

            model = TResDeltaGRU(
                input_size=2,
                hidden_size=int(spec["hidden_size"]),
                output_size=2,
                num_layers=int(spec.get("num_layers", 1)),
            )
        else:
            raise ValueError(f"unsupported backbone: {model_name}")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self._model = model.to(device)
        self._device = device

    def predict(self, signal: np.ndarray) -> np.ndarray:
        values = np.asarray(signal, dtype=np.complex128)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("signal must be a non-empty 1-D sequence")
        normalized = values / self.input_scale
        outputs = np.empty(values.size, dtype=np.complex128)
        frame_length = self.frame_length
        with torch.no_grad():
            for start in range(0, values.size, frame_length):
                stop = min(start + frame_length, values.size)
                frame = normalized[start:stop]
                inputs = torch.from_numpy(
                    np.stack((frame.real, frame.imag), axis=-1).astype(
                        np.float32
                    )
                ).unsqueeze(0).to(self._device)
                prediction = self._model(inputs)[0].cpu().numpy()
                outputs[start:stop] = (
                    prediction[:, 0] + 1j * prediction[:, 1]
                ) * self.input_scale
        return outputs
