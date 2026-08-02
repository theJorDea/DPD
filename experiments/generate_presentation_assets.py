"""Render deterministic presentation figures from frozen DPD evidence.

This module is deliberately a read-only visualizer.  It does not fit, select,
retune, or evaluate a model.  Every plotted waveform and spectrum comes from a
completed validation-only bundle whose hashes are checked before it is opened.

The figures resemble the useful four-panel OpenDPD overview, but they do not
pretend that the closed-form spline fit has a neural epoch history.  The fourth
panel is a time-domain comparison rather than a constellation: a sealed,
validation-only demodulation contract is not available for these captures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATIC_OUTPUTS = {
    "overview_dpa200.png",
    "overview_apa200.png",
    "complexity_proxy.png",
    "fixed_point_stability.png",
}
ANIMATION_OUTPUTS = {"dpd_overview.gif"}
EXPECTED_OUTPUTS = STATIC_OUTPUTS | ANIMATION_OUTPUTS | {
    "presentation_manifest.json",
}

COLOR_NO_DPD = "#C44E52"
COLOR_FLOAT = "#4C72B0"
COLOR_FIXED = "#55A868"
COLOR_TARGET = "#30343B"
COLOR_BAND = "#8172B3"


@dataclass(frozen=True)
class CasePaths:
    key: str
    label: str
    fixed_dir: Path
    spectrum_float_dir: Path
    spectrum_12_dir: Path
    config_path: Path


@dataclass
class CaseData:
    key: str
    label: str
    gain: complex
    nperseg: int
    warmup: int
    float_waveforms: dict[str, np.ndarray]
    fixed12_waveforms: dict[str, np.ndarray]
    float_spectrum: dict[str, np.ndarray]
    fixed12_spectrum: dict[str, np.ndarray]
    float_spectral_summary: dict[str, Any]
    fixed12_spectral_summary: dict[str, Any]
    fixed_report: dict[str, Any]
    input_paths: tuple[Path, ...]


def _case_paths(root: Path) -> tuple[CasePaths, ...]:
    results = root / "experiments" / "results"
    configs = root / "experiments" / "configs"
    return (
        CasePaths(
            key="dpa200",
            label="DPA_200MHz",
            fixed_dir=results / "dpd_fixed_point_dpa200_validation",
            spectrum_float_dir=results
            / "dpd_fixed_point_dpa200_spectrum_float_validation",
            spectrum_12_dir=results
            / "dpd_fixed_point_dpa200_spectrum_12bit_validation",
            config_path=configs / "dpd_fixed_point_dpa200_validation.json",
        ),
        CasePaths(
            key="apa200",
            label="APA_200MHz",
            fixed_dir=results / "dpd_fixed_point_apa200_validation",
            spectrum_float_dir=results
            / "dpd_fixed_point_apa200_spectrum_float_validation",
            spectrum_12_dir=results
            / "dpd_fixed_point_apa200_spectrum_12bit_validation",
            config_path=configs / "dpd_fixed_point_apa200_validation.json",
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verify_manifest_artifact(manifest_path: Path, artifact_name: str) -> Path:
    manifest = _read_json(manifest_path)
    if manifest.get("atomic_publication") is not True:
        raise ValueError(f"bundle is not atomically completed: {manifest_path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or artifact_name not in artifacts:
        raise ValueError(f"artifact {artifact_name!r} is not sealed by {manifest_path}")
    artifact_path = manifest_path.parent / artifact_name
    expected = artifacts[artifact_name]
    actual = sha256_file(artifact_path)
    if actual != expected:
        raise ValueError(
            f"hash mismatch for {artifact_path}: expected {expected}, got {actual}"
        )
    return artifact_path


def _verify_config(fixed_manifest_path: Path, config_path: Path) -> None:
    manifest = _read_json(fixed_manifest_path)
    expected = manifest.get("config_sha256")
    actual = sha256_file(config_path)
    if expected != actual:
        raise ValueError(
            f"config hash mismatch for {config_path}: expected {expected}, got {actual}"
        )


def _load_npz(path: Path, required: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"missing {sorted(missing)} in {path}")
        result = {name: np.asarray(archive[name]) for name in required}
    for name, value in result.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite values in {path}:{name}")
    return result


def load_case(paths: CasePaths) -> CaseData:
    fixed_manifest = paths.fixed_dir / "completion_manifest.json"
    float_waveform_path = _verify_manifest_artifact(
        fixed_manifest, "waveforms_float.npz"
    )
    fixed12_waveform_path = _verify_manifest_artifact(
        fixed_manifest, "waveforms_12bit.npz"
    )
    fixed_report_path = _verify_manifest_artifact(
        fixed_manifest, "fixed_point_report.json"
    )
    _verify_config(fixed_manifest, paths.config_path)

    float_spectrum_manifest = paths.spectrum_float_dir / "completion_manifest.json"
    float_spectrum_path = _verify_manifest_artifact(
        float_spectrum_manifest, "spectra.npz"
    )
    float_summary_path = _verify_manifest_artifact(
        float_spectrum_manifest, "summary.json"
    )
    fixed12_spectrum_manifest = paths.spectrum_12_dir / "completion_manifest.json"
    fixed12_spectrum_path = _verify_manifest_artifact(
        fixed12_spectrum_manifest, "spectra.npz"
    )
    fixed12_summary_path = _verify_manifest_artifact(
        fixed12_spectrum_manifest, "summary.json"
    )

    waveform_keys = (
        "desired_input",
        "predistorted_drive",
        "no_dpd_output",
        "dpd_output",
    )
    spectrum_keys = (
        "frequencies_hz",
        "no_dpd_average_power_spectrum",
        "dpd_average_power_spectrum",
    )
    float_waveforms = _load_npz(float_waveform_path, waveform_keys)
    fixed12_waveforms = _load_npz(fixed12_waveform_path, waveform_keys)
    float_spectrum = _load_npz(float_spectrum_path, spectrum_keys)
    fixed12_spectrum = _load_npz(fixed12_spectrum_path, spectrum_keys)
    for name in waveform_keys:
        if float_waveforms[name].shape != fixed12_waveforms[name].shape:
            raise ValueError(f"float/fixed waveform shape mismatch for {paths.label}:{name}")

    config = _read_json(paths.config_path)
    gain_record = config.get("target_gain", {})
    gain = complex(float(gain_record["real"]), float(gain_record["imag"]))
    fixed_report = _read_json(fixed_report_path)
    float_summary = _read_json(float_summary_path)
    fixed12_summary = _read_json(fixed12_summary_path)

    for summary_path, summary in (
        (float_summary_path, float_summary),
        (fixed12_summary_path, fixed12_summary),
    ):
        scope = summary.get("claims_scope", {})
        if scope.get("surrogate_only") is not True:
            raise ValueError(f"presentation input is not surrogate-labelled: {summary_path}")
        if scope.get("physical_pa_measurement") is not False:
            raise ValueError(f"unexpected physical-PA claim in {summary_path}")
        if scope.get("rf_harmonic_claim") is not False:
            raise ValueError(f"unexpected RF-harmonic claim in {summary_path}")

    direction = fixed_report.get("direction", {})
    if direction.get("measured_output_used_as_dpd_input") is not False:
        raise ValueError(f"unsafe DPD direction in {fixed_report_path}")
    if fixed_report.get("dataset", {}).get("test_split_accessed") is not False:
        raise ValueError(f"test split was accessed in {fixed_report_path}")

    metrics = fixed_report["float_reference"]["float_dpd_vs_ideal"]
    return CaseData(
        key=paths.key,
        label=paths.label,
        gain=gain,
        nperseg=int(config["nperseg"]),
        warmup=int(metrics["causal_warmup_samples_per_frame"]),
        float_waveforms=float_waveforms,
        fixed12_waveforms=fixed12_waveforms,
        float_spectrum=float_spectrum,
        fixed12_spectrum=fixed12_spectrum,
        float_spectral_summary=float_summary,
        fixed12_spectral_summary=fixed12_summary,
        fixed_report=fixed_report,
        input_paths=(
            fixed_manifest,
            float_waveform_path,
            fixed12_waveform_path,
            fixed_report_path,
            paths.config_path,
            float_spectrum_manifest,
            float_spectrum_path,
            float_summary_path,
            fixed12_spectrum_manifest,
            fixed12_spectrum_path,
            fixed12_summary_path,
        ),
    )


def scored_mask(length: int, nperseg: int, warmup: int) -> np.ndarray:
    if length <= 0 or nperseg <= 0 or warmup < 0 or warmup >= nperseg:
        raise ValueError("invalid scored-mask dimensions")
    mask = np.ones(length, dtype=bool)
    for start in range(0, length, nperseg):
        mask[start : min(start + warmup, length)] = False
    return mask


def deterministic_indices(length: int, maximum: int) -> np.ndarray:
    if length <= 0 or maximum <= 0:
        raise ValueError("length and maximum must be positive")
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 44,
    minimum_count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays with equal shape")
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers: list[float] = []
    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for index in range(bins):
        upper_inclusive = index == bins - 1
        selected = (x >= edges[index]) & (
            (x <= edges[index + 1]) if upper_inclusive else (x < edges[index + 1])
        )
        values = y[selected]
        if values.size < minimum_count:
            continue
        centers.append(0.5 * (edges[index] + edges[index + 1]))
        lows.append(float(np.quantile(values, 0.1)))
        medians.append(float(np.median(values)))
        highs.append(float(np.quantile(values, 0.9)))
    return tuple(np.asarray(v, dtype=np.float64) for v in (centers, medians, lows, highs))


def _series(case: CaseData, output: np.ndarray) -> dict[str, np.ndarray]:
    desired = case.float_waveforms["desired_input"]
    mask = scored_mask(desired.size, case.nperseg, case.warmup)
    desired = desired[mask]
    output = output[mask]
    peak = float(np.max(np.abs(desired)))
    amplitude = np.abs(desired) / peak
    normalized_output = output / case.gain
    output_amplitude = np.abs(normalized_output) / peak
    phase = np.rad2deg(np.angle(output * np.conj(case.gain * desired)))
    phase_valid = amplitude >= 0.04
    return {
        "desired": desired,
        "output": normalized_output,
        "amplitude": amplitude,
        "output_amplitude": output_amplitude,
        "phase": phase,
        "phase_valid": phase_valid,
    }


def _metric_value(record: dict[str, Any]) -> float:
    value = record.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"expected finite metric record, got {record!r}")
    return float(value)


def _style_matplotlib() -> tuple[Any, Any]:
    cache_dir = Path(tempfile.gettempdir()) / "dpd-presentation-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError(
            "presentation rendering needs requirements-presentation.txt"
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return matplotlib, plt


def _plot_characteristic(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str,
    scatter: bool = True,
) -> None:
    if scatter:
        indices = deterministic_indices(x.size, 2400)
        ax.scatter(
            x[indices],
            y[indices],
            s=4,
            alpha=0.055,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
    centers, median, low, high = binned_summary(x, y)
    ax.fill_between(centers, low, high, color=color, alpha=0.12, linewidth=0)
    ax.plot(centers, median, color=color, linewidth=2.1, label=label)


def _spectrum_db(case: CaseData, stage: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stage not in {"no_dpd", "float", "fixed12"}:
        raise ValueError(f"unsupported presentation stage: {stage}")
    spectra = case.fixed12_spectrum if stage == "fixed12" else case.float_spectrum
    frequency = spectra["frequencies_hz"] / 1e6
    no_power = spectra["no_dpd_average_power_spectrum"]
    compared_power = (
        no_power if stage == "no_dpd" else spectra["dpd_average_power_spectrum"]
    )
    reference = max(float(np.max(no_power)), np.finfo(np.float64).tiny)
    floor = np.finfo(np.float64).tiny
    no_db = 10.0 * np.log10(np.maximum(no_power, floor) / reference)
    compared_db = 10.0 * np.log10(np.maximum(compared_power, floor) / reference)
    order = np.argsort(frequency)
    return frequency[order], no_db[order], compared_db[order]


def render_overview(
    case: CaseData,
    output_path: Path,
    *,
    stage: str = "float",
    dpi: int = 150,
) -> None:
    if stage not in {"no_dpd", "float", "fixed12"}:
        raise ValueError(f"unsupported presentation stage: {stage}")
    _, plt = _style_matplotlib()
    no_series = _series(case, case.float_waveforms["no_dpd_output"])
    if stage == "fixed12":
        compared_output = case.fixed12_waveforms["dpd_output"]
        compared_color = COLOR_FIXED
        compared_label = "12-bit DPD + PA"
        stage_title = "12-bit fixed-point spline DPD"
        summary = case.fixed12_spectral_summary
        compared_nmse = case.fixed_report["formats"]["12"]["validation"][
            "fixed_cascade_vs_ideal"
        ]["complex_nmse_pooled_db"]
    elif stage == "float":
        compared_output = case.float_waveforms["dpd_output"]
        compared_color = COLOR_FLOAT
        compared_label = "Float spline DPD + PA"
        stage_title = "floating-point spline DPD"
        summary = case.float_spectral_summary
        compared_nmse = case.fixed_report["float_reference"]["float_dpd_vs_ideal"][
            "complex_nmse_pooled_db"
        ]
    else:
        compared_output = None
        compared_color = COLOR_NO_DPD
        compared_label = "PA only (no DPD)"
        stage_title = "PA only — no DPD"
        summary = case.float_spectral_summary
        compared_nmse = None
    compared_series = None if compared_output is None else _series(case, compared_output)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    figure.suptitle(
        f"Spline-memory DPD overview — {case.label} — {stage_title}\n"
        "frozen validation · legacy PA surrogate · not a physical-PA result",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "--", color=COLOR_TARGET, linewidth=1.3, label="Ideal")
    _plot_characteristic(
        ax,
        no_series["amplitude"],
        no_series["output_amplitude"],
        color=COLOR_NO_DPD,
        label="PA only (no DPD)",
    )
    if compared_series is not None:
        _plot_characteristic(
            ax,
            compared_series["amplitude"],
            compared_series["output_amplitude"],
            color=compared_color,
            label=compared_label,
        )
    ax.set(xlabel="Normalized desired amplitude", ylabel="Normalized output amplitude")
    ax.set_xlim(0, 1.01)
    upper = max(
        1.05,
        float(np.quantile(no_series["output_amplitude"], 0.998)) * 1.05,
        float(
            np.quantile(
                no_series["output_amplitude"]
                if compared_series is None
                else compared_series["output_amplitude"],
                0.998,
            )
        )
        * 1.05,
    )
    ax.set_ylim(0, min(upper, 1.45))
    ax.set_title("AM/AM characteristic", loc="left")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[0, 1]
    ax.axhline(0.0, linestyle="--", color=COLOR_TARGET, linewidth=1.3, label="Ideal")
    phase_series = [(no_series, COLOR_NO_DPD, "PA only (no DPD)")]
    if compared_series is not None:
        phase_series.append((compared_series, compared_color, compared_label))
    for series, color, label in phase_series:
        valid = series["phase_valid"]
        _plot_characteristic(
            ax,
            series["amplitude"][valid],
            series["phase"][valid],
            color=color,
            label=label,
        )
    phase_values = np.concatenate(
        tuple(series["phase"][series["phase_valid"]] for series, _, _ in phase_series)
    )
    phase_limit = max(10.0, float(np.quantile(np.abs(phase_values), 0.985)) * 1.1)
    ax.set_ylim(-min(phase_limit, 90.0), min(phase_limit, 90.0))
    ax.set_xlim(0, 1.01)
    ax.set(xlabel="Normalized desired amplitude", ylabel="Phase error (degrees)")
    ax.set_title("AM/PM characteristic", loc="left")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1, 0]
    frequency, no_db, compared_db = _spectrum_db(case, stage)
    ax.plot(frequency, no_db, color=COLOR_NO_DPD, linewidth=1.0, label="PA only")
    if stage != "no_dpd":
        ax.plot(
            frequency,
            compared_db,
            color=compared_color,
            linewidth=1.1,
            label=compared_label,
        )
    main = summary["main_region"]
    ax.axvspan(main["low_hz"] / 1e6, main["high_hz"] / 1e6, color="#DDDDDD", alpha=0.18)
    for region in (summary["regions"]["left_adjacent"], summary["regions"]["right_adjacent"]):
        ax.axvspan(
            region["low_hz"] / 1e6,
            region["high_hz"] / 1e6,
            color=COLOR_BAND,
            alpha=0.11,
        )
    left = summary["regions"]["left_adjacent"]
    right = summary["regions"]["right_adjacent"]
    if stage == "no_dpd":
        text = "Configured adjacent regions\nno-DPD reference"
    else:
        text = (
            "Configured adjacent improvement\n"
            f"left  {_metric_value(left['relative_leakage_improvement_db']):+.2f} dB\n"
            f"right {_metric_value(right['relative_leakage_improvement_db']):+.2f} dB"
        )
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
    )
    all_spectra = no_db if stage == "no_dpd" else np.concatenate((no_db, compared_db))
    lower = max(-120.0, float(np.nanpercentile(all_spectra, 0.5)))
    ax.set_ylim(lower, 3.0)
    ax.set(xlabel="Complex-baseband frequency (MHz)", ylabel="PSD (dB, common reference)")
    ax.set_title("Power spectral density", loc="left")
    ax.legend(loc="lower center", ncol=2, fontsize=9)

    ax = axes[1, 1]
    scored = np.flatnonzero(scored_mask(
        case.float_waveforms["desired_input"].size, case.nperseg, case.warmup
    ))
    window = scored[: min(180, scored.size)]
    target = case.float_waveforms["desired_input"][window]
    no_output = case.float_waveforms["no_dpd_output"][window] / case.gain
    selected_output = None if compared_output is None else compared_output[window] / case.gain
    samples = np.arange(window.size)
    ax.plot(samples, np.abs(target), color=COLOR_TARGET, linewidth=1.4, label="Desired |x|")
    ax.plot(samples, np.abs(no_output), color=COLOR_NO_DPD, linewidth=1.0, alpha=0.85, label="PA only / g")
    if selected_output is not None:
        ax.plot(
            samples,
            np.abs(selected_output),
            color=compared_color,
            linewidth=1.1,
            label=f"{compared_label} / g",
        )
    no_nmse = case.fixed_report["float_reference"]["no_dpd_vs_ideal"]["complex_nmse_pooled_db"]
    metric_text = (
        f"No-DPD pooled NMSE\n{no_nmse:.2f} dB"
        if compared_nmse is None
        else f"Pooled NMSE\n{no_nmse:.2f} → {compared_nmse:.2f} dB"
    )
    ax.text(
        0.98,
        0.96,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
    )
    ax.set(xlabel="Scored sample index", ylabel="Amplitude")
    ax.set_title("Time-domain target tracking", loc="left")
    ax.legend(loc="lower right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "theJorDea/DPD presentation generator"},
    )
    plt.close(figure)


def render_complexity_proxy(output_path: Path) -> None:
    _, plt = _style_matplotlib()
    labels = (
        "This work\nspline-memory",
        "OpenDPD\nGRU-H16",
        "OpenDPD\nTRes-GRU-H15",
        "Egor\nEnhancedESN-FAN",
    )
    values = np.asarray((21.0, 944.0, 1058.0, 728_622.0))
    colors = (COLOR_FIXED, COLOR_FLOAT, COLOR_BAND, "#DD8452")
    figure, ax = plt.subplots(figsize=(10.8, 5.5), constrained_layout=True)
    positions = np.arange(values.size)
    bars = ax.barh(positions, values, color=colors, alpha=0.9)
    ax.set_xscale("log")
    ax.axvline(1000.0, color=COLOR_NO_DPD, linestyle="--", linewidth=1.5)
    ax.text(1000.0, 3.55, "1000-MUL reference", color=COLOR_NO_DPD, ha="center", va="bottom", fontsize=9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Analytical real multiplications / complex sample (log scale)")
    ax.set_title("Inference arithmetic proxy — not a target timing measurement", loc="left")
    for bar, value in zip(bars, values, strict=True):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontsize=10)
    ax.text(
        0.0,
        -0.19,
        "Counts use audited analytical conventions. Nonlinear operations, memory traffic, parallelism and target latency are separate.",
        transform=ax.transAxes,
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight", metadata={"Software": "theJorDea/DPD presentation generator"})
    plt.close(figure)


def render_fixed_point_stability(cases: tuple[CaseData, ...], output_path: Path) -> None:
    _, plt = _style_matplotlib()
    formats = ("float", "16", "14", "12")
    x = np.arange(len(formats))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)
    for case, color, marker in zip(cases, (COLOR_FLOAT, COLOR_FIXED), ("o", "s"), strict=True):
        report = case.fixed_report
        nmse = [report["float_reference"]["float_dpd_vs_ideal"]["complex_nmse_pooled_db"]]
        peak = [report["float_reference"]["predistorted_drive"]["maximum_amplitude"]]
        for bits in (16, 14, 12):
            record = report["formats"][str(bits)]["validation"]
            nmse.append(record["fixed_cascade_vs_ideal"]["complex_nmse_pooled_db"])
            peak.append(record["fixed_drive"]["maximum_amplitude"])
        axes[0].plot(x, nmse, marker=marker, linewidth=2, color=color, label=case.label)
        axes[1].plot(x, peak, marker=marker, linewidth=2, color=color, label=case.label)
    axes[0].set_xticks(x, formats)
    axes[0].set(xlabel="Arithmetic format", ylabel="Cascade pooled NMSE (dB)")
    axes[0].set_title("Fixed-point quality preservation", loc="left")
    axes[0].legend(fontsize=9)
    axes[1].set_xticks(x, formats)
    axes[1].set(xlabel="Arithmetic format", ylabel="Maximum predistorted amplitude")
    axes[1].set_title("Peak-drive stability", loc="left")
    axes[1].legend(fontsize=9)
    figure.suptitle(
        "Software bit-accurate replay · frozen validation · no physical-PA or RTL claim",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight", metadata={"Software": "theJorDea/DPD presentation generator"})
    plt.close(figure)


def render_overview_animation(cases: tuple[CaseData, ...], output_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError(
            "GIF rendering needs requirements-presentation.txt"
        ) from exc

    stages = (
        (cases[0], "no_dpd", 1400),
        (cases[0], "float", 1900),
        (cases[0], "fixed12", 2100),
        (cases[1], "no_dpd", 1400),
        (cases[1], "float", 1900),
        (cases[1], "fixed12", 2600),
    )
    frames: list[Any] = []
    durations: list[int] = []
    with tempfile.TemporaryDirectory(prefix="dpd_presentation_frames_") as temporary:
        frame_dir = Path(temporary)
        for index, (case, stage, duration) in enumerate(stages):
            frame_path = frame_dir / f"frame_{index:02d}.png"
            render_overview(case, frame_path, stage=stage, dpi=100)
            with Image.open(frame_path) as source:
                frame = source.convert("RGB")
                if frame.width > 1320:
                    height = int(round(frame.height * 1320 / frame.width))
                    frame = frame.resize((1320, height), Image.Resampling.LANCZOS)
                quantized = frame.quantize(
                    colors=128,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE,
                )
                frames.append(quantized.copy())
            durations.append(duration)
    if len(frames) != 6:
        raise RuntimeError(f"expected six presentation frames, got {len(frames)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
        comment=b"validation-only surrogate presentation; no physical-PA claim",
    )


def _prepare_output(
    output_dir: Path,
    force: bool,
    allowed_outputs: set[str],
) -> None:
    if output_dir.exists():
        unknown = {entry.name for entry in output_dir.iterdir()} - allowed_outputs
        if unknown:
            raise FileExistsError(
                f"refusing to touch output directory with unknown entries: {sorted(unknown)}"
            )
        if any(output_dir.iterdir()) and not force:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def generate_assets(
    root: Path,
    output_dir: Path,
    *,
    force: bool = False,
    include_animation: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    generated_outputs = set(STATIC_OUTPUTS)
    if include_animation:
        generated_outputs.update(ANIMATION_OUTPUTS)
    allowed_outputs = generated_outputs | {"presentation_manifest.json"}
    _prepare_output(output_dir, force, allowed_outputs)
    cases = tuple(load_case(paths) for paths in _case_paths(root))

    render_overview(cases[0], output_dir / "overview_dpa200.png")
    render_overview(cases[1], output_dir / "overview_apa200.png")
    render_complexity_proxy(output_dir / "complexity_proxy.png")
    render_fixed_point_stability(cases, output_dir / "fixed_point_stability.png")
    if include_animation:
        render_overview_animation(cases, output_dir / "dpd_overview.gif")

    generated = sorted(generated_outputs)
    inputs = sorted({path.resolve() for case in cases for path in case.input_paths})
    manifest = {
        "schema_version": 1,
        "artifact_type": "readme_presentation_assets",
        "generator": "experiments/generate_presentation_assets.py",
        "generator_sha256": sha256_file(
            root / "experiments" / "generate_presentation_assets.py"
        ),
        "renderer_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "pillow": __import__("PIL").__version__,
        },
        "claims_scope": {
            "surrogate_only": True,
            "validation_reused_after_historical_selection": True,
            "physical_pa_measurement": False,
            "rf_harmonic_claim": False,
            "target_timing_claim": False,
        },
        "rendering_contract": {
            "common_psd_reference": "maximum no-DPD spectrum power per dataset",
            "amam_ampm": "deterministic scatter plus per-amplitude-bin median and 10/90 percentiles",
            "ampm_minimum_normalized_amplitude": 0.04,
            "constellation_omitted": "no sealed validation-only demodulation contract",
            "training_epochs_claimed": False,
            "animation_storyboard": (
                "DPA no-DPD -> DPA float -> DPA 12-bit -> "
                "APA no-DPD -> APA float -> APA 12-bit"
                if include_animation
                else None
            ),
            "animation_uses_only_saved_states": include_animation,
        },
        "inputs": {str(path.relative_to(root)): sha256_file(path) for path in inputs},
        "outputs": {name: sha256_file(output_dir / name) for name in generated},
    }
    manifest_path = output_dir / "presentation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_static_assets(
    root: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Backward-compatible static-only entry point used by focused tests."""

    return generate_assets(
        root,
        output_dir,
        force=force,
        include_animation=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "assets" / "presentation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the known generated presentation files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = generate_assets(args.root, args.output_dir, force=args.force)
    print(f"PASS: generated {len(manifest['outputs'])} presentation assets")
    print("scope: validation-only surrogate evidence; no physical-PA/Huawei claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
