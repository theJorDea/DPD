"""Model-independent unpredictability floor via kNN twin histories.

For each query sample, find k nearest neighbours in the phase-rotated
input-history space, fit a local linear model of the measured output on
the neighbours' features, and measure the remaining spread.  A function
of the input history cannot beat this spread on average: it is a
model-free upper bound on the unpredictable floor of y|x.

Saturation over L (history length) is the answer: if kNN saturates at
-36 dB, judges at -35.3 are already at the floor; if at -42, there is
6-7 dB of unmodelled structure to hunt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.train_spline import load_split_pair  # noqa: E402


def knn_floor(
    x: np.ndarray,
    y: np.ndarray,
    L: int,
    k: int = 32,
    queries: int = 12000,
    block: int = 2000,
    seed: int = 0,
) -> float:
    N = len(x)
    rng = np.random.default_rng(seed)
    idx = np.arange(L, N)
    history = np.stack([x[idx - m] for m in range(L)], axis=1)
    rot = np.exp(-1j * np.angle(history[:, 0]))[:, None]
    z_rot = history * rot
    y_rot = y[idx] * rot[:, 0]
    features = np.concatenate(
        [z_rot.real, z_rot.imag[:, 1:]], axis=1
    ).astype(np.float32)
    mean = features.mean(0)
    std = features.std(0) + 1e-12
    features = (features - mean) / std
    query_idx = rng.choice(features.shape[0], size=queries, replace=False)
    p = features.shape[1] + 1
    guard = L + 4  # exclude overlapping-window neighbours
    f_norms = np.einsum("ij,ij->i", features, features)
    squares = np.empty(queries, dtype=np.float64)
    for start in range(0, queries, block):
        q_idx = query_idx[start : start + block]
        dots = features[q_idx] @ features.T
        dist = f_norms[q_idx][:, None] + f_norms[None, :] - 2.0 * dots
        # Exclude self and temporally overlapping histories.
        gap = np.abs(idx[None, :] - idx[q_idx][:, None])
        dist[gap < guard] = np.inf
        best = np.argpartition(dist, k, axis=1)[:, :k]
        for row, neighbours in enumerate(best):
            A = np.concatenate(
                [np.ones((k, 1), dtype=np.float32), features[neighbours]],
                axis=1,
            )
            target = y_rot[neighbours]
            coeffs, *_ = np.linalg.lstsq(A.astype(np.float64), target, rcond=None)
            residual = target - A.astype(np.float64) @ coeffs
            squares[start + row] = np.mean(np.abs(residual) ** 2)
    corrected = squares * k / max(k - p, 1)
    floor = float(np.mean(corrected) / np.mean(np.abs(y[idx]) ** 2))
    return float(10 * np.log10(floor))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--L", type=int, nargs="+", default=(1, 2, 3, 5, 8))
    args = parser.parse_args()

    x, y = load_split_pair(Path(args.dataset).resolve(), "train")
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    report: dict[str, object] = {"label": args.label, "samples": int(x.size)}
    for L in args.L:
        value = knn_floor(x, y, L)
        report[f"L_{L}"] = value
        print(f"{args.label} L={L}: kNN floor {value:.2f} dB", flush=True)
    Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
