"""Re-run the numerical paths in ``MY_PA_DPD.ipynb`` without plotting.

The script is intentionally a reproduction aid, not a corrected claim:

* the notebook has no validation split;
* ``circular`` is ``y_test/g -> inverse -> PA_hat``;
* ``correct_direction`` is ``x_test -> inverse -> PA_hat -> g*x_test``.

The local ``chaotic_library`` source is imported by path so its invalid optional
PEP-508 dependency does not require changing the upstream checkout.  Run from
the workspace with the optional audit environment:

    .venv/bin/python experiments/reproduce_egor.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHAOTIC_SOURCE = ROOT / "vendor" / "chaotic_library" / "src"
if str(CHAOTIC_SOURCE) not in sys.path:
    sys.path.insert(0, str(CHAOTIC_SOURCE))

from chaotic_library import EnhancedESN_FAN  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402


def load_iq(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected I,Q CSV: {path}")
    return values[:, 0] + 1j * values[:, 1]


def iq_matrix(signal: np.ndarray) -> np.ndarray:
    return np.column_stack((signal.real, signal.imag))


def complex_nmse_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        10.0
        * np.log10(
            np.sum(np.abs(estimate - reference) ** 2)
            / np.sum(np.abs(reference) ** 2)
        )
    )


def component_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    return {
        "mse_i": float(mean_squared_error(reference.real, estimate.real)),
        "mse_q": float(mean_squared_error(reference.imag, estimate.imag)),
        "r2_i": float(r2_score(reference.real, estimate.real)),
        "r2_q": float(r2_score(reference.imag, estimate.imag)),
        "complex_nmse_pooled_db": complex_nmse_db(estimate, reference),
    }


def make_model(
    *,
    reservoir_size: int,
    random_state: int,
    spectral_radius: float,
) -> EnhancedESN_FAN:
    return EnhancedESN_FAN(
        input_dim=2,
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        sparsity=0.1,
        ridge_alpha=1e-2,
        leaking_rate=0.3,
        poly_order=2,
        fan_terms=8,
        clip_value=3.0,
        random_state=random_state,
    )


def run(data_directory: Path) -> dict[str, Any]:
    train_input = load_iq(data_directory / "train_input.csv")
    train_output = load_iq(data_directory / "train_output.csv")
    test_input = load_iq(data_directory / "test_input.csv")
    test_output = load_iq(data_directory / "test_output.csv")
    coefficient = (
        np.mean(np.abs(train_output.real))
        + np.mean(np.abs(train_output.imag))
    ) / (
        np.mean(np.abs(train_input.real))
        + np.mean(np.abs(train_input.imag))
    )

    started = time.perf_counter()
    pa_i = make_model(reservoir_size=800, random_state=42, spectral_radius=0.95)
    pa_q = make_model(reservoir_size=800, random_state=43, spectral_radius=0.95)
    pa_i.fit(iq_matrix(train_input), train_output.real)
    pa_q.fit(iq_matrix(train_input), train_output.imag)
    pa_elapsed = time.perf_counter() - started
    pa_prediction = pa_i.predict(iq_matrix(test_input)) + 1j * pa_q.predict(
        iq_matrix(test_input)
    )

    normalized_train_output = train_output / coefficient
    normalized_test_output = test_output / coefficient
    dpd_i = make_model(reservoir_size=600, random_state=100, spectral_radius=0.90)
    dpd_q = make_model(reservoir_size=600, random_state=101, spectral_radius=0.90)
    dpd_started = time.perf_counter()
    dpd_i.fit(iq_matrix(normalized_train_output), train_input.real)
    dpd_q.fit(iq_matrix(normalized_train_output), train_input.imag)
    dpd_elapsed = time.perf_counter() - dpd_started

    # Circular diagnostic path from notebook cells 10/14.
    circular_dpd = dpd_i.predict(iq_matrix(normalized_test_output)) + 1j * dpd_q.predict(
        iq_matrix(normalized_test_output)
    )
    circular_output = pa_i.predict(iq_matrix(circular_dpd)) + 1j * pa_q.predict(
        iq_matrix(circular_dpd)
    )

    # Correct direction from notebook cell 11.
    correct_dpd = dpd_i.predict(iq_matrix(test_input)) + 1j * dpd_q.predict(
        iq_matrix(test_input)
    )
    correct_output = pa_i.predict(iq_matrix(correct_dpd)) + 1j * pa_q.predict(
        iq_matrix(correct_dpd)
    )
    ideal_output = coefficient * test_input

    return {
        "schema_version": 1,
        "artifact_type": "egor_notebook_cpu_reproduction",
        "claims_scope": {
            "physical_pa_result": False,
            "correct_direction_result": "frozen PA surrogate only",
            "circular_result": "inverse-forward consistency diagnostic only",
            "validation_used": False,
        },
        "source": {
            "notebook": str(
                ROOT / "vendor" / "DPD_for_PA" / "MY_PA_DPD.ipynb"
            ),
            "chaotic_library_source": str(CHAOTIC_SOURCE),
            "data_directory": str(data_directory.resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "train_samples": int(train_input.size),
            "test_samples": int(test_input.size),
            "validation_files_present": False,
        },
        "notebook_parameters": {
            "gain_definition": "mean absolute Cartesian I/Q ratio",
            "gain": float(coefficient),
            "pa_reservoir_size": 800,
            "dpd_reservoir_size": 600,
            "pa_seeds": [42, 43],
            "dpd_seeds": [100, 101],
            "sparsity_argument": 0.1,
            "ridge_alpha": 1e-2,
            "fan_terms": 8,
            "polynomial_order": 2,
        },
        "timing_seconds": {
            "pa_pair_fit": float(pa_elapsed),
            "dpd_pair_fit": float(dpd_elapsed),
            "total": float(time.perf_counter() - started),
        },
        "pa_test": component_metrics(pa_prediction, test_output),
        "circular_inverse_forward_test": {
            **component_metrics(circular_output, test_output),
            "dpd_input": "test_output / gain",
        },
        "correct_direction_surrogate_test": {
            **component_metrics(correct_output, ideal_output),
            "dpd_input": "test_input desired signal",
            "target": "gain * test_input",
            "predistorted_peak_amplitude": float(np.max(np.abs(correct_dpd))),
            "desired_peak_amplitude": float(np.max(np.abs(test_input))),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=ROOT / "vendor" / "DPD_for_PA" / "data1",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "experiments" / "results" / "egor_reproduction_dpa200.json",
    )
    args = parser.parse_args()
    report = run(args.data_directory.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
