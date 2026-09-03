"""Long oscillating envelope-memory test (PDN / bias-network hypothesis).

All previously killed classes were short or monotonic in memory.  This
test covers 100 kHz - 40 MHz resonant dynamics on |x|^2 (power-draw
resonance in the supply network) and asymmetric trap kernels (fast
capture, slow release).  Features enter additively (LF leakage into the
output) and multiplicatively (gain modulation x * s[n]).

Read: multiplicative bank >= 0.5 dB with 1-2 leading resonances ->
class found; one biquad per resonance + one complex multiply in deploy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA  # noqa: E402
from baseline.train_spline import load_split_pair  # noqa: E402

WARM = 256


def envelope_bank(x: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    a2 = np.abs(x) ** 2
    a2 = a2 - a2.mean()
    feats: list[np.ndarray] = []
    names: list[str] = []
    for f in (0.3e6, 1e6, 2e6, 5e6, 10e6, 20e6, 40e6):
        for q in (3, 10):
            w = 2 * np.pi * f / fs
            r = float(np.exp(-w / (2 * q)))
            b = [1 - r]
            a = [1.0, -2 * r * np.cos(w), r * r]
            n = np.arange(len(a2))
            z = lfilter(b, a, a2) * np.exp(-1j * w * n)
            feats.append(z)
            names.append(f"res f={f / 1e6:g}MHz Q={q}")
    for tau_rel in (64, 512, 4096):
        e = np.empty_like(a2)
        acc = 0.0
        d = 1 - 1 / tau_rel
        for i, v in enumerate(a2):
            acc = max(v, acc * d)
            e[i] = acc
        feats.append(e - e.mean())
        names.append(f"trap rel={tau_rel}")
    return np.stack(feats, axis=1), names


def ls_report(
    r: np.ndarray, G: np.ndarray, names: list[str], warm: int = WARM
) -> dict[str, object]:
    body = slice(warm, len(r) - warm)
    Gr, rr = G[body], r[body]
    norms = np.linalg.norm(Gr, axis=0)
    norms[norms == 0] = 1.0
    Gn = Gr / norms
    coeffs, *_ = np.linalg.lstsq(Gn, rr, rcond=None)
    e = rr - Gn @ coeffs
    explained_db = float(
        10
        * np.log10(
            np.mean(np.abs(rr) ** 2) / max(np.mean(np.abs(e) ** 2), 1e-300)
        )
    )
    top = np.argsort(-np.abs(coeffs))[:4]
    return {
        "explained_db": explained_db,
        "columns": int(G.shape[1]),
        "top": [
            {"feature": names[i], "coeff": float(abs(coeffs[i]))}
            for i in top
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fs", type=float, default=800e6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pa_a = GeneralizedMemoryPolynomialPA.load(Path(args.pa).resolve())
    pa_b = GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())
    train_x, train_y = load_split_pair(Path(args.dataset).resolve(), "train")
    train_x = np.asarray(train_x).reshape(-1)
    train_y = np.asarray(train_y).reshape(-1)

    warm = max(pa_a.config.causal_warmup_samples, WARM)
    F, names = envelope_bank(train_x, args.fs)
    F = F[warm : len(F) - warm]
    Fw = np.concatenate([F.real, F.imag], axis=1)
    names_w = [f"Re({n})" for n in names] + [f"Im({n})" for n in names]

    report: dict[str, object] = {"label": args.label}
    for tag, pa in (("a", pa_a), ("b", pa_b)):
        r = train_y - pa.predict(train_x)
        r_w = r[warm : len(r) - warm]
        report[f"judge_{tag}_additive"] = ls_report(
            r_w, Fw, names_w, warm=0
        )
        x_shift = train_x[warm : len(train_x) - warm]
        G_mult = np.concatenate([x_shift[:, None] * Fw], axis=1)
        report[f"judge_{tag}_multiplicative_x_s"] = ls_report(
            r_w, G_mult, names_w, warm=0
        )
        print(
            f"judge {tag}: additive "
            f"{report[f'judge_{tag}_additive']['explained_db']:+.3f} dB | "
            f"multiplicative "
            f"{report[f'judge_{tag}_multiplicative_x_s']['explained_db']:+.3f} dB",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
