"""Residual diagnostics for frozen GMP PA evaluators (D1-D8 research program).

Answers the "structure vs noise" question for the minus-50 campaign:
runs whiteness, DC, fractional-delay, gain-drift, extended-dictionary
(conjugate/acausal/extra-lagging), frame-repeat, and half-split
refit-agreement tests on the residuals of the frozen GMP evaluators.

Read-only with respect to frozen artifacts; writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.gmp_pa import GeneralizedMemoryPolynomialPA, fit_gmp_pa  # noqa: E402
from baseline.train_spline import (  # noqa: E402
    load_split_pair,
    load_dataset_spec,
)
from baseline.direct_learning import nmse_db  # noqa: E402


def interior_slice(n: int, pad: int) -> slice:
    return slice(pad, n - pad)


def whiteness_psd_ratio(r: np.ndarray, segment: int = 2560) -> dict:
    """Welch-style PSD of the residual; flatness = max/median ratio."""
    hops = (len(r) - segment) // (segment // 2)
    psd = np.zeros(segment)
    count = 0
    window = np.hanning(segment)
    for hop in range(hops):
        start = hop * (segment // 2)
        frame = r[start:start + segment] * window
        psd += np.abs(np.fft.fft(frame)) ** 2
        count += 1
    psd /= max(count, 1)
    freqs = np.fft.fftshift(np.fft.fftfreq(segment))
    psd_shift = np.fft.fftshift(psd)
    # Coarse PSD shape (64 log-spaced mean bins) and band energy shares.
    edges = np.linspace(0, len(psd_shift), 65, dtype=int)
    shape = [
        float(np.mean(psd_shift[edges[i]:edges[i + 1]]))
        for i in range(64)
    ]
    shape = np.asarray(shape)
    shape_db = np.round(10 * np.log10(shape / shape.max() + 1e-30), 1).tolist()
    abs_f = np.abs(freqs)
    total = float(np.mean(psd_shift))
    return {
        "psd_max_over_median_db": float(10 * np.log10(psd.max() / np.median(psd))),
        "psd_dc_over_median_db": float(10 * np.log10(psd[0] / np.median(psd))),
        "psd_shape_64bins_db": shape_db,
        "energy_share_dc_2MHz": float(
            np.mean(psd_shift[abs_f < 0.0025]) / total
        ),
        "energy_share_inband": float(
            np.mean(psd_shift[abs_f < 0.13]) / total
        ),
        "energy_share_guard_band": float(
            np.mean(psd_shift[abs_f >= 0.13]) / total
        ),
    }


def fractional_delay_blocks(y: np.ndarray, yhat: np.ndarray, block: int = 8192) -> dict:
    """D4: per-block fractional delay between measurement and model output."""
    block = min(block, max(len(y) // 2, 256))
    taus = []
    for start in range(0, len(y) - block, block):
        Y = np.fft.fft(y[start:start + block])
        H = np.fft.fft(yhat[start:start + block])
        mask = np.abs(H) > 0.05 * np.abs(H).max()
        if mask.sum() < 16:
            continue
        phase = np.unwrap(np.angle(Y[mask] * np.conj(H[mask])))
        freqs = np.fft.fftfreq(block)[mask]
        slope = np.polyfit(2 * np.pi * freqs, phase, 1)[0]
        taus.append(-slope)
    taus = np.asarray(taus)
    mean_tau = float(taus.mean())
    # Empirical NMSE gain from applying the mean fractional correction.
    n = len(yhat)
    freqs = np.fft.fftfreq(n)
    corrected = np.fft.ifft(np.fft.fft(yhat) * np.exp(2j * np.pi * freqs * mean_tau))
    return {
        "mean_tau_samples": mean_tau,
        "tau_drift_last_minus_first": float(taus[-1] - taus[0]),
        "tau_std_samples": float(taus.std()),
        "nmse_before_db": float(nmse_db(yhat, y, 64)),
        "nmse_after_phase_correction_db": float(nmse_db(corrected, y, 64)),
    }


def gain_drift(y: np.ndarray, yhat: np.ndarray, window: int = 8192) -> dict:
    """D5: windowed complex gain between model output and measurement."""
    gains = []
    for start in range(0, len(y) - window, window):
        h = yhat[start:start + window]
        gains.append(np.vdot(h, y[start:start + window]) / np.vdot(h, h))
    g = np.asarray(gains)
    g0 = g.mean()
    drift_nmse = float(10 * np.log10(np.mean(np.abs(g - g0) ** 2) / abs(g0) ** 2))
    return {
        "gain_mag_std_percent": float(100 * np.abs(g).std() / abs(g0)),
        "gain_phase_std_deg": float(np.degrees(np.angle(g / g0).std())),
        "drift_nmse_equivalent_db": drift_nmse,
    }


def fit_extensions(
    x_fit: np.ndarray,
    r_fit: np.ndarray,
    groups: dict[str, list[np.ndarray]],
    ridge: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Fit ridge-LS coefficients for each dictionary group against residual."""
    fitted = {}
    pad = 64
    n = len(r_fit)
    body = interior_slice(n, pad)
    for name, columns in groups.items():
        phi = np.stack([c[body] for c in columns], axis=1)
        target = r_fit[body]
        phi_h_phi = phi.conj().T @ phi
        theta = np.linalg.solve(
            phi_h_phi + ridge * len(target) * np.eye(phi.shape[1]),
            phi.conj().T @ target,
        )
        fitted[name] = theta
    return fitted


def make_groups(x: np.ndarray, lag_max: int) -> dict[str, list[np.ndarray]]:
    """Build extended-dictionary column groups from the drive signal."""
    n = len(x)
    power2 = np.abs(x) ** 2
    power4 = power2 ** 2

    def shift(v: np.ndarray, lag: int) -> np.ndarray:
        if lag >= 0:
            out = np.zeros_like(v)
            out[lag:] = v[: n - lag]
        else:
            out = np.zeros_like(v)
            out[: n + lag] = v[-lag:]
        return out

    conj = np.conj(x)
    groups: dict[str, list[np.ndarray]] = {
        "wl_conjugate": [],
        "acausal": [],
        "extra_lagging": [],
        "two_envelope": [],
    }
    for k_exp, pw in ((1, power2), (2, power4)):
        for m in range(0, 6):
            for d in range(0, 4):
                env = shift(pw, d) if d else pw
                groups["wl_conjugate"].append(shift(conj, m) * env ** k_exp)
        for m in range(-8, 0):
            groups["acausal"].append(shift(x, m) * pw ** k_exp)
        for m in range(lag_max, lag_max + 8):
            groups["extra_lagging"].append(shift(x, m) * pw ** k_exp)
    groups["two_envelope"].append(x * power2 * shift(power2, 1))
    groups["two_envelope"].append(x * power4 * shift(power2, 2))
    return groups


def evaluate_extensions(
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    base_hat: np.ndarray,
    groups: dict[str, list[np.ndarray]],
    fitted: dict[str, dict[str, np.ndarray]],
    warmup: int = 64,
) -> dict[str, dict[str, float]]:
    """Return per-group val NMSE with and without each extension."""
    base = float(nmse_db(base_hat, y_eval, warmup))
    result = {"base": base}
    for fit_name, per_group in fitted.items():
        for group_name, theta in per_group.items():
            columns = groups[group_name]
            extra = sum(c * t for c, t in zip(columns, theta, strict=True))
            key = f"{fit_name}+{group_name}"
            result[key] = float(nmse_db(base_hat + extra, y_eval, warmup))
    return result


def frame_repeat_test(x: np.ndarray, min_lag: int = 4096, top: int = 5) -> dict:
    """D8: long-lag autocorrelation of the drive (repeated frames?)."""
    n = len(x)
    spectrum = np.fft.fft(x, 2 * n)
    acf = np.fft.ifft(spectrum * np.conj(spectrum)).real
    acf /= acf[0]
    lags = np.arange(min_lag, n - min_lag)
    values = np.abs(acf[min_lag : n - min_lag])
    order = np.argsort(values)[::-1][:top]
    return {
        "max_abs_acf": float(values.max()),
        "top_lags": [int(lags[i]) for i in order],
        "top_values": [float(values[i]) for i in order],
    }


def half_split_agreement(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config,
    segment_length: int,
    ridge: float,
) -> dict:
    """Fit the same GMP architecture on two train halves; compare on val."""
    mid = len(x_train) // 2
    predictions = []
    fidelities = []
    for lo, hi in ((0, mid), (mid, len(x_train))):
        model, _ = fit_gmp_pa(
            x_train[lo:hi],
            y_train[lo:hi],
            config=config,
            ridge=ridge,
            segment_length=segment_length,
        )
        pred = model.predict(x_val)
        warm = max(model.config.causal_warmup_samples, 64)
        predictions.append(pred)
        fidelities.append(float(nmse_db(pred, y_val, warm)))
    warm = max(config.causal_warmup_samples, 64)
    agreement = float(nmse_db(predictions[0], predictions[1], warm))
    return {
        "half_fidelity_val_db": fidelities,
        "prediction_agreement_val_db": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    spec = load_dataset_spec(dataset)
    segment_length = int(spec["nperseg"])

    pa = GeneralizedMemoryPolynomialPA.load(Path(args.pa).resolve())
    pa_b = (
        GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())
        if args.pa_b
        else None
    )

    train_x, train_y = load_split_pair(dataset, "train")
    val_x, val_y = load_split_pair(dataset, "val")
    # load_split_pair may return frame-blocked (frames × nperseg) arrays;
    # every diagnostic below works on flat sample streams.
    train_x = np.asarray(train_x).reshape(-1)
    train_y = np.asarray(train_y).reshape(-1)
    val_x = np.asarray(val_x).reshape(-1)
    val_y = np.asarray(val_y).reshape(-1)
    warm = max(pa.config.causal_warmup_samples, 64)

    report: dict = {"label": args.label, "gmp_config": str(pa.config)}

    # Fidelity control against the frozen numbers.
    for name, xx, yy in (
        ("train", train_x, train_y),
        ("val", val_x, val_y),
    ):
        report[f"fidelity_a_{name}_db"] = float(nmse_db(pa.predict(xx), yy, warm))
        if pa_b is not None:
            report[f"fidelity_b_{name}_db"] = float(
                nmse_db(pa_b.predict(xx), yy, warm)
            )

    # Residual diagnostics on the frozen evaluator A.
    yhat_train = pa.predict(train_x)
    r_train = train_y - yhat_train
    yhat_val = pa.predict(val_x)
    r_val = val_y - yhat_val

    report["whiteness_train"] = whiteness_psd_ratio(r_train)
    report["whiteness_val"] = whiteness_psd_ratio(r_val)
    report["dc_share_train_db"] = float(
        10 * np.log10(abs(r_train.mean()) ** 2 / np.mean(np.abs(r_train) ** 2))
    )
    report["dc_share_val_db"] = float(
        10 * np.log10(abs(r_val.mean()) ** 2 / np.mean(np.abs(r_val) ** 2))
    )
    report["fractional_delay_val"] = fractional_delay_blocks(val_y, yhat_val)
    report["gain_drift_val"] = gain_drift(val_y, yhat_val)
    report["gain_drift_train"] = gain_drift(train_y, yhat_train)
    report["frame_repeats_train"] = frame_repeat_test(train_x)

    # Extended-dictionary groups: fit on train, transfer to val.
    groups_train = make_groups(train_x, lag_max=pa.config.la)
    groups_val = make_groups(val_x, lag_max=pa.config.la)
    fitted = {"extensions": fit_extensions(train_x, r_train, groups_train)}
    report["extension_nmse_val_db"] = evaluate_extensions(
        val_x, val_y, yhat_val, groups_val, fitted, warmup=warm
    )
    report["extension_nmse_train_db"] = evaluate_extensions(
        train_x, train_y, yhat_train, groups_train, fitted, warmup=warm
    )

    # Half-split refit agreement for the same architecture.
    report["half_split"] = half_split_agreement(
        train_x,
        train_y,
        val_x,
        val_y,
        pa.config,
        segment_length=segment_length,
        ridge=1e-5,
    )

    Path(args.output).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
