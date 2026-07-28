"""Pure-NumPy metrics for complex-baseband PA/DPD experiments.

The module deliberately keeps similarly named metrics separate:

``nmse_pooled_db``
    One error-energy/reference-energy ratio pooled over every sample.

``nmse_opendpd_db``
    The OpenDPD repository convention: form one ratio per supplied segment,
    convert every ratio to dB, then take the arithmetic mean in dB.

``time_domain_rms_evm_db``
    RMS sample-error EVM in the time domain.  This is not constellation EVM.

``opendpd_spectral_evm_db``
    OpenDPD's repository-specific mean-absolute FFT-domain error.  Despite its
    historical name, it is neither RMS time-domain nor demodulated
    constellation EVM.

``opendpd_aclr_db``
    OpenDPD's adjacent-subchannel power divided by the strongest configured
    in-band subchannel.

``standard_aclr_db``
    Integrated adjacent-band power divided by integrated total main-band
    power.  Results use the leakage convention (negative dBc is better).

No metric silently estimates delay or complex gain.  Align signals first and
record whether a gain was fitted, fixed, or intentionally left uncompensated.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ACLRResult(NamedTuple):
    """Left, right, and arithmetic-mean leakage ratios in dB."""

    left_db: float
    right_db: float
    average_db: float


def as_complex(signal: np.ndarray, *, name: str = "signal") -> np.ndarray:
    """Convert complex data or an ``[..., 2]`` real I/Q array to complex128."""

    array = np.asarray(signal)
    if array.ndim == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.iscomplexobj(array) and array.ndim >= 2 and array.shape[-1] == 2:
        array = array[..., 0] + 1j * array[..., 1]
    array = np.asarray(array, dtype=np.complex128)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _paired_complex(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    estimate_array = as_complex(estimate, name="estimate")
    reference_array = as_complex(reference, name="reference")
    if estimate_array.shape != reference_array.shape:
        raise ValueError("estimate and reference must have the same shape")
    return estimate_array, reference_array


def _power_ratio_db(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("metric reference power must be positive")
    if numerator < 0.0:
        raise ValueError("metric numerator power cannot be negative")
    if numerator == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(numerator / denominator))


def _amplitude_ratio_db(ratio: float) -> float:
    if ratio < 0.0:
        raise ValueError("amplitude ratio cannot be negative")
    if ratio == 0.0:
        return float("-inf")
    return float(20.0 * np.log10(ratio))


def nmse_pooled_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Complex NMSE in dB using one pooled error-power ratio.

    Formula:

    ``10*log10(sum(abs(estimate-reference)**2) / sum(abs(reference)**2))``.
    """

    estimate_array, reference_array = _paired_complex(estimate, reference)
    error_power = float(np.sum(np.abs(estimate_array - reference_array) ** 2))
    reference_power = float(np.sum(np.abs(reference_array) ** 2))
    return _power_ratio_db(error_power, reference_power)


def nmse_opendpd_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """OpenDPD mean-per-segment NMSE in dB.

    The last axis is time.  Every leading-index item is one segment.  A 1-D
    input is treated as one segment.  This reproduces

    ``mean(10*log10(mean_t(|e|²) / mean_t(|reference|²)))``

    and generally differs from pooled NMSE when segment powers differ.
    """

    estimate_array, reference_array = _paired_complex(estimate, reference)
    if estimate_array.ndim == 1:
        estimate_array = estimate_array[np.newaxis, :]
        reference_array = reference_array[np.newaxis, :]
    else:
        estimate_array = estimate_array.reshape(-1, estimate_array.shape[-1])
        reference_array = reference_array.reshape(-1, reference_array.shape[-1])

    error_power = np.mean(np.abs(estimate_array - reference_array) ** 2, axis=-1)
    reference_power = np.mean(np.abs(reference_array) ** 2, axis=-1)
    if np.any(reference_power <= 0.0):
        raise ValueError("every OpenDPD NMSE segment must have positive reference power")
    with np.errstate(divide="ignore"):
        per_segment_db = 10.0 * np.log10(error_power / reference_power)
    return float(np.mean(per_segment_db))


def time_domain_rms_evm(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Return pooled RMS time-domain EVM as a linear amplitude ratio.

    ``sqrt(sum(abs(error)**2) / sum(abs(reference)**2))``.

    This is a sample-domain error measure, not demodulated constellation EVM.
    """

    estimate_array, reference_array = _paired_complex(estimate, reference)
    error_power = float(np.sum(np.abs(estimate_array - reference_array) ** 2))
    reference_power = float(np.sum(np.abs(reference_array) ** 2))
    if reference_power <= 0.0:
        raise ValueError("RMS EVM is undefined for a zero-energy reference")
    return float(np.sqrt(error_power / reference_power))


def time_domain_rms_evm_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Return pooled RMS time-domain EVM in dB (``20*log10(EVM_rms)``)."""

    return _amplitude_ratio_db(time_domain_rms_evm(estimate, reference))


def _fixed_length_segments(
    signal: np.ndarray,
    nperseg: int,
    *,
    name: str,
) -> np.ndarray:
    if not isinstance(nperseg, (int, np.integer)) or int(nperseg) <= 1:
        raise ValueError("nperseg must be an integer greater than one")
    nperseg = int(nperseg)
    array = as_complex(signal, name=name)
    if array.ndim == 1:
        if array.size % nperseg != 0:
            raise ValueError(
                f"{name} length must be an exact multiple of nperseg; "
                "segment or crop it explicitly"
            )
        return array.reshape(-1, nperseg)
    if array.shape[-1] != nperseg:
        raise ValueError(f"{name} last axis must equal nperseg")
    return array.reshape(-1, nperseg)


def periodic_hann(length: int) -> np.ndarray:
    """Return SciPy-compatible FFT-bin (periodic) Hann samples."""

    if not isinstance(length, (int, np.integer)) or int(length) <= 0:
        raise ValueError("window length must be a positive integer")
    length = int(length)
    if length == 1:
        return np.ones(1, dtype=float)
    samples = np.arange(length, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * samples / length)


def welch_numpy(
    signal: np.ndarray,
    *,
    fs: float,
    nperseg: int,
    noverlap: int | None = None,
    scaling: str = "spectrum",
    detrend: str | None = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy two-sided Welch spectrum matching OpenDPD/SciPy defaults.

    Explicit behavior:

    - periodic Hann window;
    - ``nfft == nperseg``;
    - default ``noverlap == nperseg//2``;
    - no boundary extension or padding;
    - mean average over Welch windows;
    - two-sided spectrum for complex baseband;
    - ``detrend='constant'`` subtracts each window mean;
    - ``scaling='spectrum'`` divides by ``abs(sum(window))**2``;
    - ``scaling='density'`` divides by ``fs*sum(window**2)``.

    Frequencies are returned in unshifted FFT order, as from
    ``scipy.signal.welch(..., return_onesided=False)``.  Leading input axes are
    preserved; only internal Welch windows are averaged.
    """

    array = as_complex(signal)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be positive and finite")
    if not isinstance(nperseg, (int, np.integer)) or int(nperseg) <= 1:
        raise ValueError("nperseg must be an integer greater than one")
    nperseg = int(nperseg)
    if array.shape[-1] < nperseg:
        raise ValueError("signal is shorter than nperseg")

    if noverlap is None:
        noverlap = nperseg // 2
    if not isinstance(noverlap, (int, np.integer)):
        raise TypeError("noverlap must be an integer")
    noverlap = int(noverlap)
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")

    if detrend not in ("constant", None):
        raise ValueError("detrend must be 'constant' or None")
    if scaling not in ("spectrum", "density"):
        raise ValueError("scaling must be 'spectrum' or 'density'")

    step = nperseg - noverlap
    windows = np.lib.stride_tricks.sliding_window_view(
        array,
        window_shape=nperseg,
        axis=-1,
    )[..., ::step, :]
    if detrend == "constant":
        windows = windows - np.mean(windows, axis=-1, keepdims=True)

    window = periodic_hann(nperseg)
    windowed = windows * window
    spectrum = np.fft.fft(windowed, n=nperseg, axis=-1)
    if scaling == "spectrum":
        scale = float(abs(np.sum(window)) ** 2)
    else:
        scale = float(fs * np.sum(window ** 2))
    power = np.abs(spectrum) ** 2 / scale
    power = np.mean(power, axis=-2)
    frequencies = np.fft.fftfreq(nperseg, d=1.0 / float(fs))
    return frequencies, power


def opendpd_power_spectrum(
    signal: np.ndarray,
    *,
    fs: float = 800e6,
    nperseg: int = 2560,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce OpenDPD's shifted, segment-averaged Welch power spectrum.

    Input is explicitly divided into non-overlapping ``nperseg`` records
    first.  Welch is then called on each record; because each record length is
    exactly ``nperseg``, it contributes one periodic-Hann, constant-detrended
    spectrum.  Spectra are averaged over records as in
    ``vendor/OpenDPD/utils/metrics.py:154-187``.
    """

    segments = _fixed_length_segments(signal, nperseg, name="signal")
    frequencies, power = welch_numpy(
        segments,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="spectrum",
        detrend="constant",
    )
    half_nfft = nperseg // 2
    frequencies = np.concatenate(
        (frequencies[half_nfft:], frequencies[:half_nfft])
    )
    power = np.concatenate(
        (power[..., half_nfft:], power[..., :half_nfft]),
        axis=-1,
    )
    power = np.mean(power, axis=0)
    return frequencies, power


def _opendpd_channel_layout(
    frequencies: np.ndarray,
    bandwidth_main: float,
    n_subchannels: int,
) -> tuple[int, int, int]:
    if not np.isfinite(bandwidth_main) or bandwidth_main <= 0.0:
        raise ValueError("bandwidth_main must be positive and finite")
    if not isinstance(n_subchannels, (int, np.integer)) or int(n_subchannels) <= 0:
        raise ValueError("n_subchannels must be a positive integer")
    n_subchannels = int(n_subchannels)

    left_candidates = np.flatnonzero(frequencies >= -bandwidth_main / 2.0)
    right_candidates = np.flatnonzero(frequencies <= bandwidth_main / 2.0)
    if left_candidates.size == 0 or right_candidates.size == 0:
        raise ValueError("main channel lies outside the sampled frequency range")
    index_left = int(np.min(left_candidates))
    index_right = int(np.max(right_candidates))
    channel_length = int((index_right - index_left) / n_subchannels)
    if channel_length <= 0:
        raise ValueError("frequency resolution is too low for configured subchannels")
    return index_left, index_right, channel_length


def opendpd_spectral_evm_db(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    fs: float = 800e6,
    bandwidth_main: float = 200e6,
    n_subchannels: int = 10,
    nperseg: int = 2560,
) -> float:
    """Exact repository-style OpenDPD FFT-domain “EVM” in dB.

    For every record and configured main-channel subchannel, this computes

    ``mean(abs(FFT(estimate)-FFT(reference))) / mean(abs(FFT(reference)))``.

    Ratios are averaged first over subchannels and records, then converted by
    ``20*log10``.  No Welch window is used here; it is a rectangular
    ``nperseg`` FFT, matching ``vendor/OpenDPD/utils/metrics.py:55-108``.
    """

    estimate_segments = _fixed_length_segments(estimate, nperseg, name="estimate")
    reference_segments = _fixed_length_segments(reference, nperseg, name="reference")
    if estimate_segments.shape != reference_segments.shape:
        raise ValueError("estimate and reference must contain the same segments")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be positive and finite")

    estimate_spectrum = np.fft.fftshift(
        np.fft.fft(estimate_segments, n=nperseg, axis=-1),
        axes=-1,
    )
    reference_spectrum = np.fft.fftshift(
        np.fft.fft(reference_segments, n=nperseg, axis=-1),
        axes=-1,
    )
    frequencies = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / float(fs)))
    index_left, _, channel_length = _opendpd_channel_layout(
        frequencies,
        bandwidth_main,
        n_subchannels,
    )

    channel_errors = np.empty(
        (estimate_segments.shape[0], int(n_subchannels)),
        dtype=float,
    )
    for channel in range(int(n_subchannels)):
        start = index_left + channel * channel_length
        stop = start + channel_length
        numerator = np.mean(
            np.abs(estimate_spectrum[:, start:stop] - reference_spectrum[:, start:stop]),
            axis=-1,
        )
        denominator = np.mean(np.abs(reference_spectrum[:, start:stop]), axis=-1)
        if np.any(denominator <= 0.0):
            raise ValueError("reference spectrum has a zero-magnitude subchannel")
        channel_errors[:, channel] = numerator / denominator

    return _amplitude_ratio_db(float(np.mean(channel_errors)))


def opendpd_aclr_db(
    signal: np.ndarray,
    *,
    fs: float = 800e6,
    nperseg: int = 2560,
    bandwidth_main: float = 200e6,
    n_subchannels: int = 10,
) -> ACLRResult:
    """Exact OpenDPD repository-style adjacent-subchannel leakage in dB.

    This is not conventional total-main-channel ACLR.  Each adjacent band has
    one configured subchannel width, and its integrated Welch power is divided
    by the *strongest* in-band subchannel power.  Welch behavior is documented
    by :func:`welch_numpy`.
    """

    frequencies, power = opendpd_power_spectrum(
        signal,
        fs=fs,
        nperseg=nperseg,
    )
    index_left, index_right, channel_length = _opendpd_channel_layout(
        frequencies,
        bandwidth_main,
        n_subchannels,
    )
    if index_left - channel_length < 0:
        raise ValueError("left adjacent subchannel lies outside Nyquist range")
    if index_right + channel_length > power.size:
        raise ValueError("right adjacent subchannel lies outside Nyquist range")

    inband_powers = np.empty(int(n_subchannels), dtype=float)
    for channel in range(int(n_subchannels)):
        start = index_left + channel * channel_length
        stop = start + channel_length
        inband_powers[channel] = float(np.sum(power[start:stop]))
    strongest_inband = float(np.max(inband_powers))

    left_power = float(
        np.sum(power[index_left - channel_length:index_left])
    )
    right_power = float(
        np.sum(power[index_right:index_right + channel_length])
    )
    left_db = _power_ratio_db(left_power, strongest_inband)
    right_db = _power_ratio_db(right_power, strongest_inband)
    return ACLRResult(left_db, right_db, (left_db + right_db) / 2.0)


def _integrated_band_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    low: float,
    high: float,
    name: str,
) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if not np.any(mask):
        raise ValueError(f"{name} contains no FFT bins")
    return float(np.sum(power[mask]))


def standard_aclr_db(
    signal: np.ndarray,
    *,
    fs: float,
    nperseg: int,
    bandwidth_main: float,
    bandwidth_adjacent: float | None = None,
    guard_bandwidth: float = 0.0,
    center_frequency: float = 0.0,
) -> ACLRResult:
    """Return conventional integrated-band ACLR using negative-dBc leakage.

    ``ACLR_left = 10*log10(P_left_adjacent / P_total_main)`` and likewise
    on the right.  The main band is centered on ``center_frequency``.
    Adjacent bands begin after ``guard_bandwidth`` and default to the same
    width as the main band.  Half-open frequency intervals ``[low, high)`` are
    used to avoid double-counting boundary bins.
    """

    if not np.isfinite(bandwidth_main) or bandwidth_main <= 0.0:
        raise ValueError("bandwidth_main must be positive and finite")
    if bandwidth_adjacent is None:
        bandwidth_adjacent = bandwidth_main
    if not np.isfinite(bandwidth_adjacent) or bandwidth_adjacent <= 0.0:
        raise ValueError("bandwidth_adjacent must be positive and finite")
    if not np.isfinite(guard_bandwidth) or guard_bandwidth < 0.0:
        raise ValueError("guard_bandwidth must be finite and non-negative")
    if not np.isfinite(center_frequency):
        raise ValueError("center_frequency must be finite")

    frequencies, power = opendpd_power_spectrum(
        signal,
        fs=fs,
        nperseg=nperseg,
    )
    main_low = center_frequency - bandwidth_main / 2.0
    main_high = center_frequency + bandwidth_main / 2.0
    left_high = main_low - guard_bandwidth
    left_low = left_high - bandwidth_adjacent
    right_low = main_high + guard_bandwidth
    right_high = right_low + bandwidth_adjacent

    nyquist_low = float(np.min(frequencies))
    bin_width = float(fs) / int(nperseg)
    nyquist_high_exclusive = float(np.max(frequencies) + bin_width)
    if left_low < nyquist_low or right_high > nyquist_high_exclusive:
        raise ValueError("configured main/adjacent bands exceed the Nyquist range")

    main_power = _integrated_band_power(
        frequencies,
        power,
        main_low,
        main_high,
        "main band",
    )
    left_power = _integrated_band_power(
        frequencies,
        power,
        left_low,
        left_high,
        "left adjacent band",
    )
    right_power = _integrated_band_power(
        frequencies,
        power,
        right_low,
        right_high,
        "right adjacent band",
    )
    left_db = _power_ratio_db(left_power, main_power)
    right_db = _power_ratio_db(right_power, main_power)
    return ACLRResult(left_db, right_db, (left_db + right_db) / 2.0)


def peak_amplitude(signal: np.ndarray) -> float:
    """Return ``max(abs(signal))``."""

    return float(np.max(np.abs(as_complex(signal))))


def papr_db(signal: np.ndarray) -> float:
    """Return PAPR as ``10*log10(max(|x|²) / mean(|x|²))``."""

    array = as_complex(signal)
    power = np.abs(array) ** 2
    mean_power = float(np.mean(power))
    if mean_power <= 0.0:
        raise ValueError("PAPR is undefined for an all-zero signal")
    return _power_ratio_db(float(np.max(power)), mean_power)


def bin_am_am_am_pm(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    *,
    bins: int | np.ndarray = 32,
    amplitude_range: tuple[float, float] | None = None,
    amplitude_floor: float = 0.0,
    normalize_output_gain: complex | None = None,
) -> dict[str, np.ndarray]:
    """Bin AM/AM and AM/PM against input amplitude.

    AM/PM uses a circular mean of
    ``angle(output * conj(input))``.  Samples with input amplitude not greater
    than ``amplitude_floor`` are excluded because their phase is undefined.
    If ``normalize_output_gain`` is supplied, output is divided by that
    non-zero complex gain before both AM/AM and AM/PM are computed.

    Empty bins contain NaN values.  The returned dictionary contains:

    ``bin_edges``, ``bin_centers``, ``count``, ``input_amplitude_mean``,
    ``output_amplitude_mean``, ``am_am_gain``, ``am_pm_rad``,
    ``am_pm_deg``, and ``phase_concentration``.
    """

    input_array, output_array = _paired_complex(input_signal, output_signal)
    input_array = input_array.reshape(-1)
    output_array = output_array.reshape(-1)
    if not np.isfinite(amplitude_floor) or amplitude_floor < 0.0:
        raise ValueError("amplitude_floor must be finite and non-negative")
    if normalize_output_gain is not None:
        normalize_output_gain = complex(normalize_output_gain)
        if not np.isfinite(normalize_output_gain) or abs(normalize_output_gain) == 0.0:
            raise ValueError("normalize_output_gain must be finite and non-zero")
        output_array = output_array / normalize_output_gain

    input_amplitude = np.abs(input_array)
    valid_phase = input_amplitude > amplitude_floor
    if not np.any(valid_phase):
        raise ValueError("no samples remain above amplitude_floor")

    if np.isscalar(bins):
        if not isinstance(bins, (int, np.integer)) or int(bins) <= 0:
            raise ValueError("bins must be a positive integer or explicit edges")
        if amplitude_range is None:
            low = 0.0
            high = float(np.max(input_amplitude[valid_phase]))
        else:
            low, high = map(float, amplitude_range)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("amplitude_range must be finite with high > low")
        bin_edges = np.linspace(low, high, int(bins) + 1)
    else:
        if amplitude_range is not None:
            raise ValueError("amplitude_range cannot be used with explicit bin edges")
        bin_edges = np.asarray(bins, dtype=float)
        if (
            bin_edges.ndim != 1
            or bin_edges.size < 2
            or not np.all(np.isfinite(bin_edges))
            or not np.all(np.diff(bin_edges) > 0.0)
        ):
            raise ValueError("explicit bin edges must be finite and strictly increasing")

    n_bins = bin_edges.size - 1
    bin_index = np.searchsorted(bin_edges, input_amplitude, side="right") - 1
    bin_index[input_amplitude == bin_edges[-1]] = n_bins - 1
    in_range = (
        valid_phase
        & (bin_index >= 0)
        & (bin_index < n_bins)
    )

    count = np.zeros(n_bins, dtype=int)
    input_mean = np.full(n_bins, np.nan, dtype=float)
    output_mean = np.full(n_bins, np.nan, dtype=float)
    am_am_gain = np.full(n_bins, np.nan, dtype=float)
    am_pm_rad = np.full(n_bins, np.nan, dtype=float)
    concentration = np.full(n_bins, np.nan, dtype=float)
    output_amplitude = np.abs(output_array)
    phase_difference = np.angle(output_array * np.conj(input_array))

    for index in range(n_bins):
        selected = in_range & (bin_index == index)
        count[index] = int(np.count_nonzero(selected))
        if count[index] == 0:
            continue
        input_mean[index] = float(np.mean(input_amplitude[selected]))
        output_mean[index] = float(np.mean(output_amplitude[selected]))
        am_am_gain[index] = output_mean[index] / input_mean[index]
        phasor_mean = np.mean(np.exp(1j * phase_difference[selected]))
        am_pm_rad[index] = float(np.angle(phasor_mean))
        concentration[index] = float(abs(phasor_mean))

    return {
        "bin_edges": bin_edges,
        "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "count": count,
        "input_amplitude_mean": input_mean,
        "output_amplitude_mean": output_mean,
        "am_am_gain": am_am_gain,
        "am_pm_rad": am_pm_rad,
        "am_pm_deg": np.rad2deg(am_pm_rad),
        "phase_concentration": concentration,
    }
