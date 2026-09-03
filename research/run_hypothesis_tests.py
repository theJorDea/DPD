"""Hypothesis tests H1-H5 for frozen-judge residuals + amplitude-bin split.

H1 LO phase/amplitude noise: s = r*conj(yhat); Im/Re power, LF PSD slope.
H2 even-order RX products: LS of r on {|x[n-d]|^2, |x[n-d]|^4, 1}, d in [-8, 8]
   (no x-factor columns - invisible to GMP/WL by construction).
H3 long envelope memory: leaky-integrator bank features x*e_tau etc.
H4 cross-lag 3rd-order Volterra x[n-m1]x[n-m2]conj(x[n-m3]), m1 != m2.
H5 sampling jitter: residual/signal PSD ratio profile center vs edges.

Also: per-amplitude-bin cascade error breakdown for the current final DPD
models through both judges (attribution of the judge-divergence gap).
Writes one JSON report per dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA  # noqa: E402
from baseline.spline_memory_dpd import SparseSplineMemoryDPD  # noqa: E402
from baseline.train_spline import load_split_pair  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402

WARM = 64


def _shift(v: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros_like(v)
    if d >= 0:
        out[d:] = v[: len(v) - d] if d else v
    else:
        out[: d] = v[-d:]
    return out


def ls_explain(r: np.ndarray, F: np.ndarray, warm: int = WARM) -> dict:
    """Fit r ~ F@c on train-like arrays; return residual reduction in dB."""
    body = slice(warm, len(r) - warm)
    Fr, rr = F[body], r[body]
    coeffs, *_ = np.linalg.lstsq(Fr, rr, rcond=None)
    e = rr - Fr @ coeffs
    return {
        "residual_fraction_db": float(
            10 * np.log10(np.mean(np.abs(e) ** 2) / np.mean(np.abs(rr) ** 2))
        ),
        "explained_db": float(
            10
            * np.log10(
                np.mean(np.abs(rr) ** 2) / max(np.mean(np.abs(e) ** 2), 1e-300)
            )
        ),
        "columns": int(F.shape[1]),
    }


def h1_phase_noise(
    yhat: np.ndarray, r: np.ndarray, nfft: int = 1 << 15
) -> dict:
    s = r * np.conj(yhat)
    p_re = float(np.mean(s.real**2))
    p_im = float(np.mean(s.imag**2))
    result: dict[str, object] = {
        "im_over_re_power_db": float(10 * np.log10(p_im / max(p_re, 1e-300)))
    }
    nfft = min(nfft, len(s))
    slopes = {}
    for name, v in (("re", s.real), ("im", s.imag)):
        v0 = v[:nfft] - np.mean(v[:nfft])
        psd = np.abs(np.fft.rfft(v0)) ** 2
        f = np.fft.rfftfreq(nfft)
        lo = float(psd[(f > 3e-5) & (f < 3e-4)].mean())
        hi = float(psd[(f > 5e-3) & (f < 5e-2)].mean())
        slopes[name] = float(10 * np.log10(lo / hi))
    result["psd_slope_lo_over_hi_db"] = slopes
    return result


def h2_even_order(x: np.ndarray, r: np.ndarray, d_max: int = 8) -> dict:
    a2 = np.abs(x) ** 2
    columns = [np.ones_like(x)]
    for d in range(-d_max, d_max + 1):
        columns.append(_shift(a2, d))
    for d in range(-4, 5):
        columns.append(_shift(a2**2, d))
    F = np.stack(columns, axis=1).astype(np.complex128)
    return ls_explain(r, F)


def h3_long_memory(x: np.ndarray, r: np.ndarray) -> dict:
    try:
        from scipy.signal import lfilter
    except ImportError:
        lfilter = None
    a2 = np.abs(x) ** 2
    # Keep the fitting window non-empty on short records.
    taus = tuple(t for t in (16, 64, 256, 1024, 4096, 16384) if 3 * t < len(x))
    if not taus:
        taus = (16,)
    columns = []
    for tau in taus:
        a = 1.0 / tau
        if lfilter is not None:
            e = lfilter([a], [1.0, -(1.0 - a)], a2)
        else:
            e = np.empty_like(a2)
            acc = a2[0]
            for i in range(len(a2)):
                acc += a * (a2[i] - acc)
                e[i] = acc
        columns += [x * e, x * e * a2, x * e**2]
    F = np.stack(columns, axis=1)
    return ls_explain(r, F, warm=max(taus))


def h4_volterra_cross(x: np.ndarray, r: np.ndarray, lag: int = 4) -> dict:
    columns = []
    for m1 in range(lag):
        for m2 in range(m1 + 1, lag):
            for m3 in range(lag):
                columns.append(
                    _shift(x, m1) * _shift(x, m2) * np.conj(_shift(x, m3))
                )
    F = np.stack(columns, axis=1)
    return ls_explain(r, F)


def h5_jitter(x: np.ndarray, r: np.ndarray, nfft: int = 1 << 15) -> dict:
    nfft = min(nfft, len(x))
    X = np.abs(np.fft.fftshift(np.fft.fft(x[:nfft]))) ** 2
    R = np.abs(np.fft.fftshift(np.fft.fft(r[:nfft]))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(nfft))
    mask = X > X.max() * 1e-2
    ratio = R[mask] / np.maximum(X[mask], 1e-300)
    ff = f[mask]
    center = float(ratio[np.abs(ff) < 0.02].mean())
    edge = float(ratio[np.abs(ff) > 0.7 * np.abs(ff).max()].mean())
    mean_ratio = float(ratio.mean())
    return {
        "center_over_mean": center / mean_ratio,
        "edge_over_mean": edge / mean_ratio,
    }


def amplitude_bins(
    u: np.ndarray,
    desired: np.ndarray,
    gain: complex,
    pa_a: GeneralizedMemoryPolynomialPA,
    pa_b: GeneralizedMemoryPolynomialPA,
    bins: int = 24,
) -> dict:
    e_a = pa_a.predict(u) - gain * desired
    e_b = pa_b.predict(u) - gain * desired
    d_ab = pa_a.predict(u) - pa_b.predict(u)
    ref = float(np.mean(np.abs(gain * desired) ** 2))
    edges = np.quantile(np.abs(u), np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(np.abs(u), edges) - 1, 0, bins - 1)
    per_bin = []
    bin_powers = np.zeros(bins)
    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            continue
        power = float(np.sum(np.abs(d_ab[mask]) ** 2))
        bin_powers[b] = power
        per_bin.append(
            {
                "amp_low": float(edges[b]),
                "amp_high": float(edges[b + 1]),
                "count": int(mask.sum()),
                "err_a_nmse_db": float(
                    10 * np.log10(np.mean(np.abs(e_a[mask]) ** 2) / ref)
                ),
                "err_b_nmse_db": float(
                    10 * np.log10(np.mean(np.abs(e_b[mask]) ** 2) / ref)
                ),
                "judge_divergence_nmse_db": float(
                    10 * np.log10(np.mean(np.abs(d_ab[mask]) ** 2) / ref)
                ),
            }
        )
    total_power = float(np.sum(np.abs(d_ab) ** 2))
    order = np.argsort(bin_powers)[::-1]
    top2_share = float(bin_powers[order[:2]].sum() / max(total_power, 1e-300))
    overall = {
        "err_a_nmse_db": float(10 * np.log10(np.mean(np.abs(e_a) ** 2) / ref)),
        "err_b_nmse_db": float(10 * np.log10(np.mean(np.abs(e_b) ** 2) / ref)),
        "judge_divergence_nmse_db": float(
            10 * np.log10(np.mean(np.abs(d_ab) ** 2) / ref)
        ),
        "top2_amplitude_bins_share_of_divergence": top2_share,
    }
    return {"overall": overall, "per_bin": per_bin}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", required=True)
    parser.add_argument("--dpd", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    pa_a = GeneralizedMemoryPolynomialPA.load(Path(args.pa).resolve())
    pa_b = GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())
    dpd = SparseSplineMemoryDPD.load(Path(args.dpd).resolve())

    train_x, train_y = load_split_pair(dataset, "train")
    val_x, val_y = load_split_pair(dataset, "val")
    train_x = np.asarray(train_x).reshape(-1)
    train_y = np.asarray(train_y).reshape(-1)
    val_x = np.asarray(val_x).reshape(-1)
    val_y = np.asarray(val_y).reshape(-1)

    warm = max(pa_a.config.causal_warmup_samples, WARM)
    gain = complex(np.vdot(train_x, pa_a.predict(train_x)) / np.vdot(train_x, train_x))

    yhat = pa_a.predict(train_x)
    r = train_y - yhat

    report: dict[str, object] = {
        "label": args.label,
        "residual_train_nmse_db": float(nmse_db(yhat, train_y, warm)),
    }
    report["H1_phase_noise"] = h1_phase_noise(yhat, r)
    report["H2_even_order"] = h2_even_order(train_x, r)
    report["H3_long_memory"] = h3_long_memory(train_x, r)
    report["H4_volterra_cross"] = h4_volterra_cross(train_x, r)
    report["H5_jitter"] = h5_jitter(train_x, r)

    # Transfer check: coefficients fitted on train, applied to val residual.
    body = slice(warm, len(train_x) - warm)
    a2 = np.abs(train_x) ** 2
    columns = [np.ones_like(train_x)] + [
        _shift(a2, d) for d in range(-8, 9)
    ] + [_shift(a2**2, d) for d in range(-4, 5)]
    F = np.stack(columns, axis=1).astype(np.complex128)
    coeffs, *_ = np.linalg.lstsq(F[body], r[body], rcond=None)
    val_yhat = pa_a.predict(val_x)
    val_r = val_y - val_yhat
    a2v = np.abs(val_x) ** 2
    columns_v = [np.ones_like(val_x)] + [
        _shift(a2v, d) for d in range(-8, 9)
    ] + [_shift(a2v**2, d) for d in range(-4, 5)]
    Fv = np.stack(columns_v, axis=1).astype(np.complex128)
    explained = float(
        10
        * np.log10(
            np.mean(np.abs(val_r) ** 2)
            / max(np.mean(np.abs(val_r - Fv @ coeffs) ** 2), 1e-300)
        )
    )
    report["H2_val_transfer_explained_db"] = explained

    # Amplitude-bin attribution on the validation block.
    u = dpd.predict(val_x)
    report["amplitude_bins_val"] = amplitude_bins(u, val_x, gain, pa_a, pa_b)

    Path(args.output).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
