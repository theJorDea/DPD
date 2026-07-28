"""Time and complex-gain alignment for complex baseband signals.

Delay sign convention
---------------------
Every function in this module defines an integer delay ``d`` by

``observed[n] ~= gain * reference[n - d]``.

Consequently, a positive delay means that ``observed`` lags ``reference``.
For ``d >= 0`` the aligned arrays are ``reference[:-d]`` and
``observed[d:]``.  A negative delay means that ``observed`` leads the
reference.

The fractional-delay result is deliberately a *diagnostic*.  It fits a
parabola to three normalized correlation-power samples around the best
integer lag.  It neither resamples nor modifies either signal and should not
be mistaken for a calibrated fractional-delay filter.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class FractionalDelayDiagnostic(NamedTuple):
    """Sub-sample peak diagnostic based on a three-point parabola.

    ``fractional_offset`` is relative to ``integer_delay`` and lies in
    ``[-0.5, 0.5]`` when ``reliable`` is true.  It is NaN when the integer
    peak lies at the search boundary or has no concave local parabola.
    """

    integer_delay: int
    fractional_offset: float
    estimated_delay: float
    peak_score: float
    curvature: float
    reliable: bool


def _as_complex_1d(signal: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(signal)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional complex sequence")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def complex_ls_gain(reference: np.ndarray, observed: np.ndarray) -> complex:
    """Return the least-squares complex gain from ``reference`` to ``observed``.

    The returned value minimizes

    ``sum(abs(observed - gain * reference) ** 2)``

    and is therefore

    ``gain = sum(conj(reference) * observed) / sum(abs(reference) ** 2)``.

    No delay search, DC removal, or gain application is performed implicitly.
    """

    reference_array = _as_complex_1d(reference, "reference")
    observed_array = _as_complex_1d(observed, "observed")
    if reference_array.shape != observed_array.shape:
        raise ValueError("reference and observed must have the same shape")

    reference_energy = float(np.vdot(reference_array, reference_array).real)
    if reference_energy <= 0.0:
        raise ValueError("complex gain is undefined for a zero-energy reference")
    return complex(np.vdot(reference_array, observed_array) / reference_energy)


def overlap_for_delay(
    reference: np.ndarray,
    observed: np.ndarray,
    delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop signals to the overlap implied by the documented delay convention."""

    reference_array = _as_complex_1d(reference, "reference")
    observed_array = _as_complex_1d(observed, "observed")
    if not isinstance(delay, (int, np.integer)):
        raise TypeError("delay must be an integer")
    delay = int(delay)

    reference_start = max(0, -delay)
    reference_stop = min(reference_array.size, observed_array.size - delay)
    if reference_stop <= reference_start:
        raise ValueError("delay leaves no overlap between reference and observed")

    observed_start = reference_start + delay
    observed_stop = reference_stop + delay
    return (
        reference_array[reference_start:reference_stop],
        observed_array[observed_start:observed_stop],
    )


def delay_correlation_scores(
    reference: np.ndarray,
    observed: np.ndarray,
    max_abs_delay: int,
    *,
    min_overlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gain-invariant normalized correlation power for candidate delays.

    For each delay, the score is

    ``abs(sum(conj(x) * y))**2 / (sum(abs(x)**2) * sum(abs(y)**2))``.

    It lies in ``[0, 1]`` up to floating-point roundoff.  The normalization
    makes the search insensitive to a constant complex gain.
    """

    reference_array = _as_complex_1d(reference, "reference")
    observed_array = _as_complex_1d(observed, "observed")
    if not isinstance(max_abs_delay, (int, np.integer)):
        raise TypeError("max_abs_delay must be an integer")
    max_abs_delay = int(max_abs_delay)
    if max_abs_delay < 0:
        raise ValueError("max_abs_delay must be non-negative")

    if min_overlap is None:
        min_overlap = max(2, min(reference_array.size, observed_array.size) // 2)
    if not isinstance(min_overlap, (int, np.integer)) or int(min_overlap) < 2:
        raise ValueError("min_overlap must be an integer of at least two")
    min_overlap = int(min_overlap)

    delays = np.arange(-max_abs_delay, max_abs_delay + 1, dtype=int)
    scores = np.full(delays.shape, np.nan, dtype=float)
    for index, delay in enumerate(delays):
        try:
            reference_overlap, observed_overlap = overlap_for_delay(
                reference_array,
                observed_array,
                int(delay),
            )
        except ValueError:
            continue
        if reference_overlap.size < min_overlap:
            continue

        reference_energy = float(np.vdot(reference_overlap, reference_overlap).real)
        observed_energy = float(np.vdot(observed_overlap, observed_overlap).real)
        denominator = reference_energy * observed_energy
        if denominator <= 0.0:
            continue
        cross_power = float(abs(np.vdot(reference_overlap, observed_overlap)) ** 2)
        scores[index] = min(1.0, max(0.0, cross_power / denominator))

    if not np.any(np.isfinite(scores)):
        raise ValueError("no candidate delay has enough non-zero-energy overlap")
    return delays, scores


def estimate_integer_delay(
    reference: np.ndarray,
    observed: np.ndarray,
    max_abs_delay: int,
    *,
    min_overlap: int | None = None,
) -> int:
    """Estimate the integer delay at maximum normalized correlation power."""

    delays, scores = delay_correlation_scores(
        reference,
        observed,
        max_abs_delay,
        min_overlap=min_overlap,
    )
    finite = np.isfinite(scores)
    best_score = np.max(scores[finite])
    candidates = delays[finite & np.isclose(scores, best_score, rtol=1e-13, atol=1e-15)]
    # Prefer the smallest absolute delay for a numerically exact tie, then the
    # smaller signed value.  This avoids a systematic edge preference.
    order = np.lexsort((candidates, np.abs(candidates)))
    return int(candidates[order[0]])


def fractional_delay_diagnostic(
    reference: np.ndarray,
    observed: np.ndarray,
    max_abs_delay: int,
    *,
    min_overlap: int | None = None,
) -> FractionalDelayDiagnostic:
    """Estimate a local sub-sample correlation peak without resampling.

    A reliable result only means that the sampled correlation-power peak has
    a concave three-point parabolic fit whose vertex lies within half a sample
    of the integer maximum.  Band-limited interpolation and a residual phase
    check are still required before applying a fractional-delay correction.
    """

    delays, scores = delay_correlation_scores(
        reference,
        observed,
        max_abs_delay,
        min_overlap=min_overlap,
    )
    finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
    peak_index = int(np.argmax(finite_scores))
    integer_delay = int(delays[peak_index])
    peak_score = float(scores[peak_index])

    if (
        peak_index == 0
        or peak_index == scores.size - 1
        or not np.all(np.isfinite(scores[peak_index - 1:peak_index + 2]))
    ):
        return FractionalDelayDiagnostic(
            integer_delay,
            float("nan"),
            float(integer_delay),
            peak_score,
            float("nan"),
            False,
        )

    score_left, score_center, score_right = scores[peak_index - 1:peak_index + 2]
    curvature = float(score_left - 2.0 * score_center + score_right)
    if curvature >= 0.0 or abs(curvature) <= np.finfo(float).eps:
        return FractionalDelayDiagnostic(
            integer_delay,
            float("nan"),
            float(integer_delay),
            peak_score,
            curvature,
            False,
        )

    fractional_offset = float(0.5 * (score_left - score_right) / curvature)
    reliable = bool(abs(fractional_offset) <= 0.5)
    if not reliable:
        fractional_offset = float("nan")
        estimated_delay = float(integer_delay)
    else:
        estimated_delay = float(integer_delay + fractional_offset)
    return FractionalDelayDiagnostic(
        integer_delay,
        fractional_offset,
        estimated_delay,
        peak_score,
        curvature,
        reliable,
    )


def align_and_estimate_gain(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    delay: int | None = None,
    max_abs_delay: int | None = None,
    min_overlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, complex]:
    """Align a pair by integer delay and estimate gain on the cropped overlap.

    Exactly one of the following is required:

    - supply ``delay`` to use a pre-established delay;
    - omit ``delay`` and supply ``max_abs_delay`` to estimate it.

    The returned observed array is *not* gain-corrected.  This makes the
    operation explicit: callers that need a unity-gain comparison should use
    ``observed_aligned / gain``.
    """

    if delay is None:
        if max_abs_delay is None:
            raise ValueError("max_abs_delay is required when delay is not supplied")
        delay = estimate_integer_delay(
            reference,
            observed,
            max_abs_delay,
            min_overlap=min_overlap,
        )
    elif max_abs_delay is not None:
        raise ValueError("supply delay or max_abs_delay, not both")

    reference_aligned, observed_aligned = overlap_for_delay(reference, observed, delay)
    gain = complex_ls_gain(reference_aligned, observed_aligned)
    return reference_aligned, observed_aligned, int(delay), gain
