"""Oracle-DPD attribution: optimize the drive u as free variables through
judge A (torch GMP), then evaluate through judge B.

Diagnoses the cascade gap:
  - oracle ~ judge fidelities on both  -> DPD capacity is the limit
  - oracle good on A, bad on B         -> judges diverge where the drive goes
  - oracle stuck well above fidelities -> judge A is hard to invert

Writes a JSON report with amplitude-bin breakdown of the oracle drive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA, gmp_terms  # noqa: E402
from baseline.train_spline import load_split_pair  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402


class TorchGMP(torch.nn.Module):
    """Differentiable mirror of GeneralizedMemoryPolynomialPA.predict."""

    def __init__(self, npz_path: Path) -> None:
        super().__init__()
        pa = GeneralizedMemoryPolynomialPA.load(npz_path)
        self.config = pa.config
        self.register_buffer(
            "coefficients",
            torch.as_tensor(pa.coefficients, dtype=torch.complex128),
        )
        self.maximum_exponent = max(
            self.config.ka - 1, self.config.kb, self.config.kc
        )

    @staticmethod
    def _shift(v: torch.Tensor, d: int) -> torch.Tensor:
        """Mirror gmp_pa._delay: values[n-d], zeros outside the record."""
        if d == 0:
            return v
        out = torch.zeros_like(v)
        n = v.numel()
        if d > 0:
            out[d:] = v[: n - d]
        else:
            out[: n + d] = v[-d:]
        return out

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        n = u.numel()
        magnitude = u.abs()
        powers = {1: magnitude, 2: magnitude**2}
        for exponent in range(3, self.maximum_exponent + 1):
            powers[exponent] = powers[exponent - 2] * powers[2]
        weights = [
            torch.zeros(n, dtype=torch.complex128) for _ in range(self.config.base_delay_count)
        ]
        for coefficient, term in zip(
            self.coefficients, gmp_terms(self.config), strict=True
        ):
            if term.exponent == 0:
                weights[term.signal_delay] = weights[term.signal_delay] + coefficient
            else:
                delayed = self._shift(powers[term.exponent], term.envelope_delay)
                weights[term.signal_delay] = (
                    weights[term.signal_delay] + coefficient * delayed
                )
        output = torch.zeros(n, dtype=torch.complex128)
        for signal_delay, weight in enumerate(weights):
            output = output + self._shift(u, signal_delay) * weight
        return output


def amplitude_bins(
    u: np.ndarray,
    desired: np.ndarray,
    gain: complex,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    bins: int = 24,
) -> dict:
    e_a = pred_a - gain * desired
    e_b = pred_b - gain * desired
    d_ab = pred_a - pred_b
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
    total = float(np.sum(np.abs(d_ab) ** 2))
    order = np.argsort(bin_powers)[::-1]
    return {
        "overall": {
            "err_a_nmse_db": float(10 * np.log10(np.mean(np.abs(e_a) ** 2) / ref)),
            "err_b_nmse_db": float(10 * np.log10(np.mean(np.abs(e_b) ** 2) / ref)),
            "judge_divergence_nmse_db": float(
                10 * np.log10(np.mean(np.abs(d_ab) ** 2) / ref)
            ),
            "top2_share": float(bin_powers[order[:2]].sum() / max(total, 1e-300)),
        },
        "per_bin": per_bin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pa", required=True)
    parser.add_argument("--pa-b", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="adam", choices=("adam", "lbfgs"))
    parser.add_argument("--joint", action="store_true", help="optimize through judges A and B stacked")
    parser.add_argument("--block", default="val", choices=("val", "fit", "selection"))
    parser.add_argument("--subblock", default=None, help="lo,hi within the chosen block")
    parser.add_argument("--save-drive", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    pa_a_npz = Path(args.pa).resolve()
    pa_a = GeneralizedMemoryPolynomialPA.load(pa_a_npz)
    pa_b = GeneralizedMemoryPolynomialPA.load(Path(args.pa_b).resolve())

    train_x, train_y = load_split_pair(dataset, "train")
    val_x, val_y = load_split_pair(dataset, "val")
    train_x = np.asarray(train_x).reshape(-1)
    val_x = np.asarray(val_x).reshape(-1)
    val_y = np.asarray(val_y).reshape(-1)

    warm = max(pa_a.config.causal_warmup_samples, 16)
    gain = complex(np.vdot(train_x, pa_a.predict(train_x)) / np.vdot(train_x, train_x))

    if args.block == "val":
        block_x = val_x
    else:
        block_x = train_x
    if args.subblock:
        lo, hi = (int(v) for v in args.subblock.split(","))
        block_x = block_x[lo:hi]

    model = TorchGMP(pa_a_npz)
    x = torch.as_tensor(val_x, dtype=torch.complex128)
    # Cascade target: ideal PA output for the block INPUT signal.
    target = torch.as_tensor(gain * block_x, dtype=torch.complex128)
    ref_power = float(torch.mean(torch.abs(target) ** 2).item())
    cap = float(np.max(np.abs(block_x))) * 1.15

    model_b_torch = TorchGMP(Path(args.pa_b).resolve())

    results: dict[str, object] = {"label": args.label, "gain": [gain.real, gain.imag]}
    if args.joint:
        results["joint"] = True

    for headroom, tag in ((0.15, "oracle_headroom_15"), (0.0, "oracle_no_headroom")):
        cap = float(np.max(np.abs(block_x))) * (1.0 + headroom)
        u = torch.as_tensor(block_x, dtype=torch.complex128).clone().requires_grad_(True)
        if args.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [u], lr=1.0, max_iter=args.iterations, history_size=60,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                opt.zero_grad()
                res_a = (model(u) - target)[warm:]
                loss = torch.mean(torch.abs(res_a) ** 2) / ref_power
                if args.joint:
                    res_b = (model_b_torch(u) - target)[warm:]
                    loss = loss + torch.mean(torch.abs(res_b) ** 2) / ref_power
                over = torch.relu(u.abs() - cap)
                loss = loss + 10.0 * torch.mean(over**2) / ref_power
                loss.backward()
                return loss

            opt.step(closure)
            iterations_run = args.iterations
        else:
            opt = torch.optim.Adam([u], lr=args.lr)
            penalty = 10.0
            iterations_run = args.iterations
            for iteration in range(args.iterations):
                opt.zero_grad()
                y_a = model(u)
                residual = (y_a - target)[warm:]
                loss = torch.mean(torch.abs(residual) ** 2) / ref_power
                if args.joint:
                    res_b = (model_b_torch(u) - target)[warm:]
                    loss = loss + torch.mean(torch.abs(res_b) ** 2) / ref_power
                over = torch.relu(u.abs() - cap)
                loss = loss + penalty * torch.mean(over**2) / ref_power
                loss.backward()
                opt.step()
                if iteration in (999, 2999):
                    with torch.no_grad():
                        current = float(
                            10
                            * np.log10(
                                torch.mean(torch.abs(residual) ** 2).item() / ref_power
                            )
                        )
                    print(f"{tag} iter {iteration}: NMSE_A {current:.3f}", flush=True)
        with torch.no_grad():
            u_np = u.detach().numpy()
            pred_a = model(u).numpy()
            pred_b = pa_b.predict(u_np)
            target_np = target.numpy()
            nmse_a = float(nmse_db(pred_a, target_np, warm))
            nmse_b = float(nmse_db(pred_b, target_np, warm))
            peak_growth = float(np.max(np.abs(u_np)) / np.max(np.abs(block_x)))
        print(
            f"{tag}: A {nmse_a:.3f} | B {nmse_b:.3f} | peak x{peak_growth:.3f}",
            flush=True,
        )
        if args.save_drive and (args.joint or tag == "oracle_no_headroom"):
            np.savez(
                args.save_drive,
                block_input=block_x,
                oracle_drive=u_np,
                gain=np.asarray([gain.real, gain.imag]),
                block=np.asarray(args.block),
            )
        results[tag] = {
            "nmse_through_a_db": nmse_a,
            "nmse_through_b_db": nmse_b,
            "peak_growth": peak_growth,
            "optimizer": args.optimizer,
            "amplitude_bins": amplitude_bins(
                u_np, block_x, gain, pred_a, pred_b
            ),
        }

    fidelity_a = float(
        nmse_db(pa_a.predict(val_x), gain * val_x, warm)
    )
    fidelity_b = float(
        nmse_db(pa_b.predict(val_x), gain * val_x, warm)
    )
    results["judge_fidelity_val_db"] = {"a": fidelity_a, "b": fidelity_b}
    results["block"] = args.block
    results["peak_growth_note"] = (
        "peak growth on no_headroom comes from the soft support penalty"
    )
    Path(args.output).write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
