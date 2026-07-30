"""Frozen-definition spectral regions for no-DPD versus DPD evaluation.

This module deliberately sits outside :mod:`baseline.metrics`: historical
benchmark manifests bind that module byte-for-byte.  The spectrum primitive is
reused without changing those sealed artifacts.

Configured complex-baseband regions are not automatically RF harmonics.
Claims about harmonics around ``2fc`` or ``3fc`` require an RF capture whose
frequency span contains them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from baseline.metrics import as_complex, opendpd_power_spectrum


@dataclass(frozen=True)
class SpectralRegion:
    """One half-open configured complex-baseband region ``[low, high)``."""

    name: str
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("spectral region name must be non-empty")
        if not np.isfinite(self.low_hz) or not np.isfinite(self.high_hz):
            raise ValueError("spectral region bounds must be finite")
        if float(self.low_hz) >= float(self.high_hz):
            raise ValueError("spectral region low_hz must be below high_hz")
        object.__setattr__(self, "low_hz", float(self.low_hz))
        object.__setattr__(self, "high_hz", float(self.high_hz))


def _explicit_segments(
    signal: np.ndarray,
    nperseg: int,
    *,
    name: str,
) -> np.ndarray:
    """Return explicit records for segment-level spectral statistics."""

    if not isinstance(nperseg, (int, np.integer)) or int(nperseg) < 2:
        raise ValueError("nperseg must be an integer >= 2")
    nperseg = int(nperseg)
    array = as_complex(signal, name=name)
    if array.ndim == 1:
        if array.size < nperseg:
            raise ValueError(f"{name} must contain at least nperseg samples")
        if array.size % nperseg:
            raise ValueError(
                f"{name} length must be an exact multiple of nperseg"
            )
        return array.reshape(-1, nperseg)
    if array.shape[-1] != nperseg:
        raise ValueError(f"last axis of {name} must equal nperseg")
    return array.reshape(-1, nperseg)


def _segment_power_spectra(
    signal: np.ndarray,
    *,
    fs: float,
    nperseg: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen OpenDPD-compatible spectrum to each explicit record."""

    segments = _explicit_segments(signal, nperseg, name=name)
    frequency_grid: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    for segment in segments:
        frequencies, power = opendpd_power_spectrum(
            segment,
            fs=fs,
            nperseg=nperseg,
        )
        if frequency_grid is None:
            frequency_grid = frequencies
        elif not np.array_equal(frequency_grid, frequencies):
            raise RuntimeError("internal spectrum frequency grids differ")
        spectra.append(power)
    if frequency_grid is None:
        raise RuntimeError("internal error: no spectral segments")
    return frequency_grid, np.stack(spectra, axis=0)


def _power_ratio_db(numerator: float, denominator: float) -> float:
    if numerator < 0.0 or denominator < 0.0:
        raise ValueError("spectral powers cannot be negative")
    if denominator == 0.0:
        raise ValueError("reference spectral power must be positive")
    if numerator == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(numerator / denominator))


def _suppression_db(no_dpd_power: float, dpd_power: float) -> float:
    """Positive dB means that DPD reduced absolute power in a region."""

    if no_dpd_power < 0.0 or dpd_power < 0.0:
        raise ValueError("spectral powers cannot be negative")
    if no_dpd_power == 0.0 and dpd_power == 0.0:
        raise ValueError("suppression is undefined when both powers are zero")
    if dpd_power == 0.0:
        return float("inf")
    if no_dpd_power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(no_dpd_power / dpd_power))


def configured_spectral_region_report(
    no_dpd_signal: np.ndarray,
    dpd_signal: np.ndarray,
    *,
    fs: float,
    nperseg: int,
    main_region: SpectralRegion,
    regions: Sequence[SpectralRegion],
    quantiles: Sequence[float] = (0.0, 0.5, 0.95, 1.0),
) -> dict[str, object]:
    """Measure region levels and positive no-DPD-to-DPD suppression.

    Absolute integrated region powers are used for
    ``suppression_db = 10*log10(P_no_dpd/P_dpd)``.  Region levels in dBc are
    normalized to the corresponding signal's own integrated main-region
    power.  Results include pooled spectra, per-record values and quantiles.
    """

    if not isinstance(main_region, SpectralRegion):
        raise TypeError("main_region must be a SpectralRegion")
    region_list = tuple(regions)
    if not region_list:
        raise ValueError("at least one spectral region is required")
    if any(not isinstance(region, SpectralRegion) for region in region_list):
        raise TypeError("every configured region must be a SpectralRegion")
    names = [region.name for region in region_list]
    if len(names) != len(set(names)):
        raise ValueError("spectral region names must be unique")
    quantile_values = np.asarray(tuple(quantiles), dtype=float)
    if (
        quantile_values.ndim != 1
        or quantile_values.size == 0
        or not np.all(np.isfinite(quantile_values))
        or np.any(quantile_values < 0.0)
        or np.any(quantile_values > 1.0)
        or np.any(np.diff(quantile_values) < 0.0)
    ):
        raise ValueError("quantiles must be finite, sorted, and within [0, 1]")

    no_dpd = as_complex(no_dpd_signal, name="no_dpd_signal")
    with_dpd = as_complex(dpd_signal, name="dpd_signal")
    if no_dpd.shape != with_dpd.shape:
        raise ValueError("no-DPD and DPD signals must have identical shapes")
    frequencies, no_dpd_spectra = _segment_power_spectra(
        no_dpd,
        fs=fs,
        nperseg=nperseg,
        name="no_dpd_signal",
    )
    frequencies_dpd, dpd_spectra = _segment_power_spectra(
        with_dpd,
        fs=fs,
        nperseg=nperseg,
        name="dpd_signal",
    )
    if not np.array_equal(frequencies, frequencies_dpd):
        raise RuntimeError("internal spectrum frequency grids differ")

    bin_width = float(fs) / int(nperseg)
    nyquist_low = float(np.min(frequencies))
    nyquist_high_exclusive = float(np.max(frequencies) + bin_width)

    def mask_for(region: SpectralRegion) -> np.ndarray:
        if (
            region.low_hz < nyquist_low
            or region.high_hz > nyquist_high_exclusive
        ):
            raise ValueError(
                f"spectral region {region.name!r} exceeds the Nyquist range"
            )
        mask = (
            (frequencies >= region.low_hz)
            & (frequencies < region.high_hz)
        )
        if not np.any(mask):
            raise ValueError(
                f"spectral region {region.name!r} contains no FFT bins"
            )
        return mask

    main_mask = mask_for(main_region)
    no_main_per_frame = np.sum(no_dpd_spectra[:, main_mask], axis=-1)
    dpd_main_per_frame = np.sum(dpd_spectra[:, main_mask], axis=-1)
    if np.any(no_main_per_frame <= 0.0) or np.any(dpd_main_per_frame <= 0.0):
        raise ValueError("every frame must have positive main-region power")

    no_average_spectrum = np.mean(no_dpd_spectra, axis=0)
    dpd_average_spectrum = np.mean(dpd_spectra, axis=0)
    no_main = float(np.sum(no_average_spectrum[main_mask]))
    dpd_main = float(np.sum(dpd_average_spectrum[main_mask]))
    rows: dict[str, object] = {}
    for region in region_list:
        mask = mask_for(region)
        no_per_frame = np.sum(no_dpd_spectra[:, mask], axis=-1)
        dpd_per_frame = np.sum(dpd_spectra[:, mask], axis=-1)
        if np.any((no_per_frame == 0.0) & (dpd_per_frame == 0.0)):
            raise ValueError(
                f"suppression for region {region.name!r} has zero/zero frame"
            )
        no_dbc_per_frame = np.asarray(
            [
                _power_ratio_db(float(power), float(reference))
                for power, reference in zip(no_per_frame, no_main_per_frame)
            ]
        )
        dpd_dbc_per_frame = np.asarray(
            [
                _power_ratio_db(float(power), float(reference))
                for power, reference in zip(dpd_per_frame, dpd_main_per_frame)
            ]
        )
        suppression_per_frame = np.asarray(
            [
                _suppression_db(float(before), float(after))
                for before, after in zip(no_per_frame, dpd_per_frame)
            ]
        )
        no_power = float(np.sum(no_average_spectrum[mask]))
        dpd_power = float(np.sum(dpd_average_spectrum[mask]))
        rows[region.name] = {
            "low_hz": region.low_hz,
            "high_hz": region.high_hz,
            "no_dpd_integrated_power": no_power,
            "dpd_integrated_power": dpd_power,
            "no_dpd_dbc": _power_ratio_db(no_power, no_main),
            "dpd_dbc": _power_ratio_db(dpd_power, dpd_main),
            "suppression_db": _suppression_db(no_power, dpd_power),
            "per_frame": {
                "no_dpd_dbc": no_dbc_per_frame,
                "dpd_dbc": dpd_dbc_per_frame,
                "suppression_db": suppression_per_frame,
            },
            "quantiles": {
                "probabilities": quantile_values.copy(),
                "no_dpd_dbc": np.quantile(no_dbc_per_frame, quantile_values),
                "dpd_dbc": np.quantile(dpd_dbc_per_frame, quantile_values),
                "suppression_db": np.quantile(
                    suppression_per_frame,
                    quantile_values,
                ),
            },
        }

    return {
        "definition": {
            "domain": "configured_complex_baseband_regions",
            "rf_harmonic_claim": False,
            "interval_convention": "[low_hz, high_hz)",
            "suppression_sign": (
                "positive means lower absolute region power with DPD"
            ),
            "window": "periodic_hann",
            "detrend": "constant",
            "segment_overlap_samples": 0,
            "spectrum_scaling": "spectrum",
            "threshold_applied": False,
        },
        "fs_hz": float(fs),
        "nperseg": int(nperseg),
        "frame_count": int(no_dpd_spectra.shape[0]),
        "main_region": {
            "name": main_region.name,
            "low_hz": main_region.low_hz,
            "high_hz": main_region.high_hz,
            "no_dpd_integrated_power": no_main,
            "dpd_integrated_power": dpd_main,
        },
        "frequencies_hz": frequencies,
        "no_dpd_average_power_spectrum": no_average_spectrum,
        "dpd_average_power_spectrum": dpd_average_spectrum,
        "regions": rows,
    }
