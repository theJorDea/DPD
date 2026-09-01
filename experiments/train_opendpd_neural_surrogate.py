"""Train an OpenDPD neural PA surrogate (Stage 5, requires torch/CUDA).

This runner wraps the upstream OpenDPD backbones (``TResGRU``,
``TResDeltaGRU``) to train a high-fidelity PA surrogate on the repository
dataset splits, following the published recipe (AdamW, lr 5e-3, batch 64,
frame 200, ~300 epochs, H=27).  A neural surrogate is the intended Stage-5
evaluator: its fidelity (typically several dB beyond polynomial models on
these captures) is what makes the deeper cascade gates meaningful.

Honest scope notes:

* this file was syntax-checked but never executed on this machine
  (CPU-only Windows box, torch not installed); run it on the GPU host
  after ``pip install torch``;
* the upstream backbone looks ahead inside a frame (TCN dilation 16, and
  the feature extractor rolls by -1), so the trained surrogate is an
  *offline* evaluator whose NMSE convention is frame-wise with resets,
  not the causal full-record convention of the polynomial evaluators;
  the companion evaluator wrapper must state this explicitly;
* only train/val splits are read here; test stays sealed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "OpenDPD"


def _load_split(dataset: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from baseline.train_spline import load_split_pair

    return load_split_pair(dataset, split)


def _frames(values: np.ndarray, frame_length: int, stride: int = 0):
    step = frame_length if stride <= 0 else min(stride, frame_length)
    for start in range(0, values.size - frame_length + 1, step):
        yield values[start : start + frame_length]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required for neural surrogate training. Install it "
            "on the GPU host (pip install torch) and rerun. This machine "
            "is CPU-only by policy."
        ) from error
    if not VENDOR_ROOT.is_dir():
        raise SystemExit("vendor/OpenDPD is not checked out")
    sys.path.insert(0, str(VENDOR_ROOT))

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("task") != "opendpd_neural_pa_surrogate_stage5":
        raise ValueError("unexpected task")
    model_spec = config["model"]
    train_spec = config["training"]

    torch.manual_seed(int(config.get("seed", 0)))
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = PROJECT_ROOT / config["dataset"]
    train_x, train_y = _load_split(dataset, "train")
    val_x, val_y = _load_split(dataset, "val")

    # Frozen normalization from train only; stored for the evaluator wrap.
    scale = float(np.max(np.abs(train_x)))
    train_xn = train_x / scale
    train_yn = train_y / scale
    val_xn = val_x / scale
    val_yn = val_y / scale

    frame_length = int(train_spec["frame_length"])
    frame_stride = int(train_spec.get("frame_stride", 0))
    model_name = str(model_spec["type"])
    if model_name == "tres_gru":
        from backbones.tres_gru import TResGRU

        model = TResGRU(
            input_size=2,
            hidden_size=int(model_spec["hidden_size"]),
            output_size=2,
            num_layers=int(model_spec.get("num_layers", 1)),
        )
    elif model_name == "tres_deltagru":
        from backbones.tres_deltagru import TResDeltaGRU

        model = TResDeltaGRU(
            input_size=2,
            hidden_size=int(model_spec["hidden_size"]),
            output_size=2,
            num_layers=int(model_spec.get("num_layers", 1)),
        )
    else:
        raise ValueError(f"unsupported backbone: {model_name}")
    model = model.to(device)
    model.reset_parameters()

    resume_path = config.get("resume_from")
    if resume_path:
        resume = torch.load(
            PROJECT_ROOT / resume_path, map_location=device, weights_only=False
        )
        model.load_state_dict(resume["model_state"])
        print(f"resumed weights from {resume_path}")

    frame_tensors = [
        torch.from_numpy(
            np.stack(
                [
                    np.stack(
                        (frame.real, frame.imag), axis=-1
                    ).astype(np.float32)
                    for frame in _frames(
                        train_xn, frame_length, frame_stride
                    )
                ]
            )
        ),
        torch.from_numpy(
            np.stack(
                [
                    np.stack(
                        (frame.real, frame.imag), axis=-1
                    ).astype(np.float32)
                    for frame in _frames(
                        train_yn, frame_length, frame_stride
                    )
                ]
            )
        ),
    ]
    loader = DataLoader(
        TensorDataset(frame_tensors[0], frame_tensors[1]),
        batch_size=int(train_spec["batch_size"]),
        shuffle=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_spec["lr"]),
    )
    if train_spec.get("lr_schedule") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(train_spec["epochs"])
        )
    else:
        scheduler = None
    epochs = int(train_spec["epochs"])
    started = time.perf_counter()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = torch.mean((predictions - targets) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        if scheduler is not None:
            scheduler.step()
        if epoch % int(train_spec.get("val_every", 10)) == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                errors = []
                powers = []
                for frame_x, frame_y in zip(
                    _frames(val_xn, frame_length),
                    _frames(val_yn, frame_length),
                ):
                    inputs = torch.from_numpy(
                        np.stack(
                            (frame_x.real, frame_x.imag), axis=-1
                        ).astype(np.float32)
                    ).unsqueeze(0).to(device)
                    targets = torch.from_numpy(
                        np.stack(
                            (frame_y.real, frame_y.imag), axis=-1
                        ).astype(np.float32)
                    ).unsqueeze(0).to(device)
                    prediction = model(inputs)
                    errors.append(
                        torch.sum((prediction - targets) ** 2).item()
                    )
                    powers.append(torch.sum(targets**2).item())
                val_nmse_db = 10.0 * np.log10(
                    sum(errors) / sum(powers)
                )
            history.append(
                {"epoch": epoch, "train_loss": epoch_loss / max(batches, 1), "val_nmse_db": val_nmse_db}
            )
            print(f"epoch {epoch}: val NMSE {val_nmse_db:.4f} dB")

    output_dir = PROJECT_ROOT / config["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_spec": model_spec,
            "input_scale": scale,
            "frame_length": frame_length,
        },
        output_dir / "neural_surrogate.pt",
    )
    write_json = output_dir / "training_report.json"
    write_json.write_text(
        json.dumps(
            {
                "task": "opendpd_neural_pa_surrogate_stage5",
                "model": model_spec,
                "training": train_spec,
                "input_scale": scale,
                "final_val_nmse_db": history[-1]["val_nmse_db"] if history else None,
                "history": history,
                "device": str(device),
                "elapsed_seconds": time.perf_counter() - started,
                "claims_scope": {
                    "train_val_only": True,
                    "test_split_accessed": False,
                    "offline_frame_evaluator": True,
                    "causal_deployed_evaluator": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("surrogate checkpoint:", output_dir / "neural_surrogate.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
