"""Capacity-judge research: larger GMP topologies for the frozen-evaluator
ceiling, fitted on train and scored on val (protocol identical to
select_pa_gmp; test split never accessed). Writes one winner package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.gmp_pa import GMPConfig, GeneralizedMemoryPolynomialPA, fit_gmp_pa  # noqa: E402
from baseline.train_spline import load_split_pair, load_dataset_spec  # noqa: E402
from baseline.direct_learning import nmse_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    dataset = Path(cfg["dataset"]).resolve()
    spec = load_dataset_spec(dataset)
    segment_length = int(cfg.get("segment_length", spec["nperseg"]))
    ridge = float(cfg.get("ridge", 1e-5))

    train_x, train_y = load_split_pair(dataset, "train")
    val_x, val_y = load_split_pair(dataset, "val")
    train_x = np.asarray(train_x).reshape(-1)
    train_y = np.asarray(train_y).reshape(-1)
    val_x = np.asarray(val_x).reshape(-1)
    val_y = np.asarray(val_y).reshape(-1)

    out_dir = Path(cfg["output_dir"]).resolve()
    if any(out_dir.iterdir()) if out_dir.exists() else False:
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    for top in cfg["topologies"]:
        config = GMPConfig(
            ka=int(top["ka"]),
            la=int(top["la"]),
            kb=int(top["kb"]),
            lb=int(top["lb"]),
            mb=int(top["mb"]),
            kc=int(top["kc"]),
            lc=int(top["lc"]),
            mc=int(top["mc"]),
        )
        model, diag = fit_gmp_pa(
            train_x,
            train_y,
            config=config,
            ridge=ridge,
            segment_length=segment_length,
        )
        warm = max(config.causal_warmup_samples, 64)
        val_pred = model.predict(val_x)
        val_db = float(nmse_db(val_pred, val_y, warm))
        train_db = float(nmse_db(model.predict(train_x), train_y, warm))
        trials.append(
            {
                "topology": top,
                "coefficients": int(config.coefficient_count),
                "train_nmse_db": train_db,
                "val_nmse_db": val_db,
            }
        )
        print(
            f"ka={config.ka} la={config.la} p={config.coefficient_count}: "
            f"train {train_db:.3f} | val {val_db:.3f}",
            flush=True,
        )

    winner = min(trials, key=lambda t: t["val_nmse_db"])
    print(
        f"WINNER val {winner['val_nmse_db']:.3f} (baseline {cfg.get('baseline_fidelity_val_db')})",
        flush=True,
    )

    # Refit the winning topology on the full train split and save.
    top = winner["topology"]
    config = GMPConfig(
        ka=int(top["ka"]),
        la=int(top["la"]),
        kb=int(top["kb"]),
        lb=int(top["lb"]),
        mb=int(top["mb"]),
        kc=int(top["kc"]),
        lc=int(top["lc"]),
        mc=int(top["mc"]),
    )
    model, diag = fit_gmp_pa(
        train_x,
        train_y,
        config=config,
        ridge=ridge,
        segment_length=segment_length,
    )
    model.save(out_dir / "selected_gmp_pa.npz")
    report = {
        "task": cfg.get("task"),
        "dataset_label": str(dataset),
        "ridge": ridge,
        "trials": trials,
        "winner": winner,
        "refit_train_nmse_db": float(nmse_db(model.predict(train_x), train_y, 64)),
    }
    (out_dir / "capacity_judge_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["winner"], indent=1), flush=True)


if __name__ == "__main__":
    main()
