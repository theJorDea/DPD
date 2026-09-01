"""Bounded iterative direct learning for linear-in-parameter DPD models.

The deployment problem is

    desired x -> DPD D_theta -> frozen PA evaluator P -> compare with g*x.

This module improves a frozen DPD by *direct* updates through the evaluator:

1. ``bounded_direct_update`` — one damped Gauss-Newton step on the DPD
   coefficients.  The Jacobian of the cascade output with respect to the
   real/imaginary parts of every coefficient is assembled with forward
   differences through the frozen evaluator, and the ridge/step candidate
   that minimises the cascade NMSE on a *disjoint train-only advisor slice*
   is selected.  This reproduces the arithmetic of the audited
   ``evaluate_blackbox_dpd_hypotheses`` refinement (+4.3 dB on BlackBox).
2. ``iterative_direct_schedule`` — repeated bounded updates over a schedule
   of disjoint train slices with an explicit stop rule and per-iteration
   records.  Nothing in the loop ever touches a validation or test split.
3. ``ilc_waveform_refinement`` — gradient-free iterative-learning-control on
   the drive waveform ``u <- u + beta * (g*x - P(u))`` followed (by the
   caller) with a plain causal least-squares refit of the DPD on
   ``(x, u_final)`` pairs.  This is the offline calibration path; the
   deployed DPD stays causal and linear-in-parameters.

Discipline preserved from the repository protocol:

* only explicitly supplied train slices are read;
* the advisor slice used for candidate selection is disjoint from the fit
  slice used for the Jacobian;
* drive support is checked against the evaluator's training maximum, so a
  candidate that would push the PA outside its identified range is rejected;
* phase equivariance is preserved exactly: knots and branches never change,
  so ``D(x e^{j phi}) = e^{j phi} D(x)`` holds for every update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .spline_memory_dpd import (
    SparseSplineMemoryDPD,
    spline_memory_design_matrix,
)

__all__ = [
    "DirectLearningConfig",
    "DirectUpdateResult",
    "bounded_direct_update",
    "bounded_direct_update_core",
    "iterative_direct_schedule",
    "iterative_direct_schedule_core",
    "model_with_delta",
    "nmse_db",
    "signal_summary",
    "ilc_waveform_refinement",
]


def _as_complex_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def nmse_db(estimate: np.ndarray, reference: np.ndarray, warmup: int) -> float:
    """Pooled complex NMSE in dB after discarding ``warmup`` samples."""

    estimate = _as_complex_vector(estimate, name="estimate")
    reference = _as_complex_vector(reference, name="reference")
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have equal shape")
    if not isinstance(warmup, (int, np.integer)) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    warmup = int(warmup)
    if reference.size <= warmup:
        raise ValueError("warmup must leave at least one scored sample")
    error = estimate[warmup:] - reference[warmup:]
    error_power = float(np.mean(np.abs(error) ** 2))
    reference_power = float(np.mean(np.abs(reference[warmup:]) ** 2))
    if reference_power <= 0.0:
        raise ValueError("NMSE reference must have positive power")
    if error_power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(error_power / reference_power))


def signal_summary(signal: np.ndarray, warmup: int = 0) -> dict[str, float]:
    """RMS, peak and PAPR of the scored part of a signal."""

    values = _as_complex_vector(signal, name="signal")[warmup:]
    power = float(np.mean(np.abs(values) ** 2))
    peak = float(np.max(np.abs(values)))
    if power <= 0.0:
        raise ValueError("signal must have positive power")
    return {
        "rms": float(np.sqrt(power)),
        "peak": peak,
        "papr_db": float(10.0 * np.log10(peak * peak / power)),
    }


@dataclass(frozen=True)
class DirectLearningConfig:
    """Candidate grids and guards for one bounded direct update."""

    ridge_values: tuple[float, ...] = (
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
    )
    step_values: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
    epsilon: float = 1e-4
    support_headroom: float = 64.0 * np.finfo(float).eps

    def __post_init__(self) -> None:
        if not self.ridge_values or not self.step_values:
            raise ValueError("ridge and step grids must not be empty")
        for ridge in self.ridge_values:
            if not np.isfinite(ridge) or ridge < 0.0:
                raise ValueError("ridge values must be finite and non-negative")
        for step in self.step_values:
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("step values must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.support_headroom < 0.0:
            raise ValueError("support_headroom must be non-negative")


@dataclass(frozen=True)
class DirectUpdateResult:
    """Outcome of one bounded direct update."""

    delta: np.ndarray
    selected_ridge: float
    selected_step: float
    advisor_nmse_db: float
    advisor_no_dpd_nmse_db: float
    advisor_improvement_db: float
    maximum_drive: float
    delta_relative_norm: float
    support_valid: bool
    candidate_count: int
    fit_sample_count: int
    coefficient_count: int
    elapsed_seconds: float
    selected_primary_nmse_db: float | None = None
    selected_secondary_nmse_db: float | None = None
    candidates: list[dict] = field(default_factory=list)


def _validate_evaluator(pa: object) -> None:
    if not hasattr(pa, "predict") or not callable(pa.predict):
        raise TypeError("pa evaluator must provide predict(drive)")


def cascade_output(
    pa: object,
    drive: np.ndarray,
) -> np.ndarray:
    """Evaluate the frozen PA on a drive waveform."""

    _validate_evaluator(pa)
    return np.asarray(pa.predict(_as_complex_vector(drive, name="drive")))


def design_for_model(
    model: SparseSplineMemoryDPD,
    signal: np.ndarray,
) -> np.ndarray:
    """Complex calibration dictionary of a spline-memory DPD."""

    return spline_memory_design_matrix(
        signal,
        model.knots,
        model.branches,
    )


def model_with_delta(
    model: SparseSplineMemoryDPD,
    delta: np.ndarray,
    step: float,
) -> SparseSplineMemoryDPD:
    """Return the model moved by ``step * delta`` along its coefficient vector."""

    delta = _as_complex_vector(delta, name="delta")
    if delta.size != model.coefficients.size:
        raise ValueError("delta length must equal the coefficient count")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    complex_delta = delta.reshape(model.coefficients.shape)
    return SparseSplineMemoryDPD(
        knots=model.knots,
        branches=model.branches,
        coefficients=model.coefficients + step * complex_delta,
        knot_strategy=model.knot_strategy,
    )


def cascade_jacobian_forward_difference(
    design: np.ndarray,
    flat_coefficients: np.ndarray,
    pa: object,
    *,
    epsilon: float,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real Jacobian of the cascade output w.r.t. coefficient parts.

    ``design`` is the (N, M) complex dictionary of the *fit* slice and
    ``flat_coefficients`` its branch-major/knot-major complex coefficient
    vector.  Columns ``0..M-1`` hold derivatives w.r.t. the real part of
    each coefficient, columns ``M..2M-1`` w.r.t. the imaginary part.  The
    first ``warmup`` output samples (evaluator memory) are discarded.
    """

    _validate_evaluator(pa)
    design = np.asarray(design, dtype=np.complex128)
    if design.ndim != 2:
        raise ValueError("design must be a two-dimensional array")
    flat = _as_complex_vector(flat_coefficients, name="flat_coefficients")
    if flat.size != design.shape[1]:
        raise ValueError("coefficient count must match design columns")
    if warmup < 0 or design.shape[0] <= warmup:
        raise ValueError("warmup must leave at least one Jacobian row")
    base_drive = design @ flat
    base_output = np.asarray(pa.predict(base_drive), dtype=np.complex128)
    parameter_count = flat.size
    jacobian = np.empty(
        (2 * (design.shape[0] - warmup), 2 * parameter_count),
        dtype=float,
    )
    for parameter in range(parameter_count):
        direction = design[:, parameter]
        real_output = np.asarray(
            pa.predict(base_drive + epsilon * direction),
            dtype=np.complex128,
        )
        imag_output = np.asarray(
            pa.predict(base_drive + 1j * epsilon * direction),
            dtype=np.complex128,
        )
        real_derivative = (real_output - base_output) / epsilon
        imag_derivative = (imag_output - base_output) / epsilon
        real_derivative = real_derivative[warmup:]
        imag_derivative = imag_derivative[warmup:]
        jacobian[:, parameter] = np.concatenate(
            (real_derivative.real, real_derivative.imag)
        )
        jacobian[:, parameter_count + parameter] = np.concatenate(
            (imag_derivative.real, imag_derivative.imag)
        )
    return jacobian, base_drive, base_output


def bounded_direct_update_core(
    *,
    flat_coefficients: np.ndarray,
    design_fit: np.ndarray,
    design_advisor: np.ndarray,
    fit_x: np.ndarray,
    advisor_x: np.ndarray,
    pa: object,
    gain: complex,
    warmup: int,
    maximum_pa_input: float,
    config: DirectLearningConfig | None = None,
    secondary: tuple[object, complex] | None = None,
    joint_objective: bool = False,
) -> DirectUpdateResult:
    """Model-agnostic one-step damped direct update.

    ``design_fit``/``design_advisor`` are the complex dictionaries of the
    two disjoint train slices for the *current* linear-in-parameters model
    whose coefficients are ``flat_coefficients`` (branch-major layout of
    whatever structure the dictionary encodes).

    ``secondary`` optionally supplies a second frozen evaluator ``(pa_b,
    gain_b)``.  When present, candidates are ranked by the *worst-case*
    advisor NMSE across both evaluators, which suppresses improvements
    that exploit the idiosyncratic approximation error of one surrogate.

    With ``joint_objective`` and ``secondary`` set, the Gauss-Newton step
    itself is fitted to the *stacked* residuals and stacked forward-
    difference Jacobians of both evaluators on the fit block (the direct-
    learning analogue of the consensus residual target): shared
    coefficients that reduce both evaluators' errors simultaneously
    instead of the primary evaluator's alone.  Candidate ranking stays
    worst-case.
    """

    config = config or DirectLearningConfig()
    gain = complex(gain)
    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("gain must be finite and non-zero")
    fit_x = _as_complex_vector(fit_x, name="fit_x")
    advisor_x = _as_complex_vector(advisor_x, name="advisor_x")
    if not isinstance(warmup, (int, np.integer)) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    warmup = int(warmup)
    maximum_pa_input = float(maximum_pa_input)
    if not np.isfinite(maximum_pa_input) or maximum_pa_input <= 0.0:
        raise ValueError("maximum_pa_input must be finite and positive")
    started = time.perf_counter()

    flat = _as_complex_vector(flat_coefficients, name="flat_coefficients")
    design_fit = np.asarray(design_fit, dtype=np.complex128)
    if design_fit.ndim != 2 or design_fit.shape[0] != fit_x.size:
        raise ValueError("design_fit rows must match fit_x length")
    if design_fit.shape[1] != flat.size:
        raise ValueError("design_fit columns must match coefficient count")
    advisor_design = np.asarray(design_advisor, dtype=np.complex128)
    if advisor_design.ndim != 2 or advisor_design.shape[0] != advisor_x.size:
        raise ValueError("design_advisor rows must match advisor_x length")
    if advisor_design.shape[1] != flat.size:
        raise ValueError("design_advisor columns must match coefficient count")

    secondary_pa: object | None = None
    secondary_gain = 0.0 + 0.0j
    if secondary is not None:
        secondary_pa, secondary_gain = secondary
        secondary_gain = complex(secondary_gain)
        if not np.isfinite(secondary_gain) or abs(secondary_gain) == 0.0:
            raise ValueError("secondary gain must be finite and non-zero")
        if not hasattr(secondary_pa, "predict") or not callable(
            secondary_pa.predict
        ):
            raise TypeError("secondary evaluator must provide predict(drive)")

    jacobian, _, base_output = cascade_jacobian_forward_difference(
        design_fit,
        flat,
        pa,
        epsilon=config.epsilon,
        warmup=warmup,
    )
    fit_target = gain * fit_x
    complex_residual = fit_target[warmup:] - base_output[warmup:]
    target = np.concatenate(
        (complex_residual.real, complex_residual.imag)
    )
    if joint_objective and secondary_pa is not None:
        jacobian_b, _, base_output_b = cascade_jacobian_forward_difference(
            design_fit,
            flat,
            secondary_pa,
            epsilon=config.epsilon,
            warmup=warmup,
        )
        fit_target_b = secondary_gain * fit_x
        complex_residual_b = (
            fit_target_b[warmup:] - base_output_b[warmup:]
        )
        target = np.concatenate(
            (target, complex_residual_b.real, complex_residual_b.imag)
        )
        jacobian = np.vstack((jacobian, jacobian_b))
    normalization = np.sqrt(float(target.size))
    normalized_jacobian = jacobian / normalization
    normalized_target = target / normalization

    advisor_ideal = gain * advisor_x
    advisor_no_dpd_output = cascade_output(pa, advisor_x)
    advisor_no_dpd_nmse = nmse_db(
        advisor_no_dpd_output,
        advisor_ideal,
        warmup,
    )

    coefficient_norm = float(
        np.linalg.norm(
            np.concatenate(
                (
                    flat.real.ravel(),
                    flat.imag.ravel(),
                )
            )
        )
    )
    candidates: list[dict] = []
    for ridge in config.ridge_values:
        regularizer = np.sqrt(ridge) * np.eye(
            normalized_jacobian.shape[1], dtype=float
        )
        delta = np.linalg.lstsq(
            np.vstack((normalized_jacobian, regularizer)),
            np.concatenate(
                (
                    normalized_target,
                    np.zeros(regularizer.shape[0], dtype=float),
                )
            ),
            rcond=None,
        )[0]
        complex_delta = delta[: flat.size] + 1j * delta[flat.size :]
        delta_norm = float(np.linalg.norm(delta))
        for step in config.step_values:
            drive = advisor_design @ (flat + step * complex_delta)
            output = cascade_output(pa, drive)
            primary_score = nmse_db(output, advisor_ideal, warmup)
            secondary_score: float | None = None
            if secondary_pa is not None:
                secondary_output = np.asarray(
                    secondary_pa.predict(drive),
                    dtype=np.complex128,
                )
                secondary_score = nmse_db(
                    secondary_output,
                    secondary_gain * advisor_x,
                    warmup,
                )
                score = max(primary_score, secondary_score)
            else:
                score = primary_score
            maximum_drive = float(np.max(np.abs(drive)))
            support_valid = bool(
                maximum_drive
                <= maximum_pa_input * (1.0 + config.support_headroom)
            )
            candidates.append(
                {
                    "ridge": float(ridge),
                    "step": float(step),
                    "advisor_nmse_db": score,
                    "primary_nmse_db": primary_score,
                    "secondary_nmse_db": secondary_score,
                    "advisor_improvement_db": (
                        advisor_no_dpd_nmse - score
                    ),
                    "maximum_drive": maximum_drive,
                    "support_valid": support_valid,
                    "delta_relative_norm": float(
                        step
                        * delta_norm
                        / max(coefficient_norm, 1e-30)
                    ),
                    "delta": complex_delta,
                }
            )
    valid = [row for row in candidates if row["support_valid"]]
    if not valid:
        raise ValueError(
            "no direct-update candidate respects the drive support bound"
        )
    selected = min(valid, key=lambda row: row["advisor_nmse_db"])
    return DirectUpdateResult(
        delta=selected["delta"],
        selected_ridge=selected["ridge"],
        selected_step=selected["step"],
        advisor_nmse_db=selected["advisor_nmse_db"],
        advisor_no_dpd_nmse_db=advisor_no_dpd_nmse,
        advisor_improvement_db=selected["advisor_improvement_db"],
        maximum_drive=selected["maximum_drive"],
        delta_relative_norm=selected["delta_relative_norm"],
        support_valid=True,
        candidate_count=len(candidates),
        fit_sample_count=int(fit_x.size),
        coefficient_count=int(flat.size),
        elapsed_seconds=time.perf_counter() - started,
        selected_primary_nmse_db=selected["primary_nmse_db"],
        selected_secondary_nmse_db=selected["secondary_nmse_db"],
        candidates=[
            {key: value for key, value in row.items() if key != "delta"}
            for row in candidates
        ],
    )


def bounded_direct_update(
    *,
    model: SparseSplineMemoryDPD,
    pa: object,
    gain: complex,
    fit_x: np.ndarray,
    advisor_x: np.ndarray,
    warmup: int,
    maximum_pa_input: float,
    config: DirectLearningConfig | None = None,
    design_fit: np.ndarray | None = None,
) -> DirectUpdateResult:
    """Spline-memory convenience wrapper around the model-agnostic core.

    ``fit_x`` (Jacobian slice) and ``advisor_x`` (candidate-selection slice)
    must be two disjoint contiguous train segments.  Validation and test
    data must not be passed here under any circumstances.
    """

    if design_fit is None:
        design_fit = design_for_model(model, fit_x)
    return bounded_direct_update_core(
        flat_coefficients=model.coefficients.reshape(-1),
        design_fit=design_fit,
        design_advisor=design_for_model(model, advisor_x),
        fit_x=fit_x,
        advisor_x=advisor_x,
        pa=pa,
        gain=gain,
        warmup=warmup,
        maximum_pa_input=maximum_pa_input,
        config=config,
    )


def _slice_size(span: slice) -> int:
    return span.stop - span.start


def _validate_schedule_slices(
    train_x: np.ndarray,
    fit_slices: list[slice],
    advisor_slices: list[slice],
) -> None:
    if len(fit_slices) != len(advisor_slices):
        raise ValueError("fit and advisor schedules must have equal length")
    for fit_span, advisor_span in zip(fit_slices, advisor_slices):
        if fit_span.stop > train_x.size or advisor_span.stop > train_x.size:
            raise ValueError("schedule slices must stay inside train_x")
        if fit_span.start < 0 or advisor_span.start < 0:
            raise ValueError("schedule slices must be non-negative")
        if (
            max(fit_span.start, advisor_span.start)
            < min(fit_span.stop, advisor_span.stop)
        ):
            raise ValueError(
                "fit and advisor slices must be disjoint train segments"
            )


def iterative_direct_schedule_core(
    *,
    initial_flat_coefficients: np.ndarray,
    pa: object,
    gain: complex,
    train_x: np.ndarray,
    fit_slices: list[slice],
    advisor_slices: list[slice],
    warmup: int,
    maximum_pa_input: float,
    design_fn,
    config: DirectLearningConfig | None = None,
    minimum_improvement_db: float = 0.02,
    secondary: tuple[object, complex] | None = None,
    joint_objective: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Model-agnostic repeated bounded direct updates.

    ``design_fn(signal_slice)`` returns the dictionary of the fixed model
    structure for a train slice; it must depend only on the structure and
    the signal, never on the current coefficients (designs are cached).
    Returns the final flat coefficient vector and per-iteration records.
    Only the supplied train slices are read.  ``secondary`` optionally
    supplies a second frozen evaluator for worst-case candidate ranking;
    with ``joint_objective`` the Gauss-Newton step is fitted to the
    stacked residuals of both evaluators (consensus analogue).
    """

    config = config or DirectLearningConfig()
    train_x = _as_complex_vector(train_x, name="train_x")
    _validate_schedule_slices(train_x, fit_slices, advisor_slices)
    current_flat = _as_complex_vector(
        initial_flat_coefficients,
        name="initial_flat_coefficients",
    )
    design_cache: dict[tuple[int, int], np.ndarray] = {}
    for span in (*fit_slices, *advisor_slices):
        key = (span.start, span.stop)
        if key not in design_cache:
            design_cache[key] = np.asarray(
                design_fn(train_x[span]),
                dtype=np.complex128,
            )
    records: list[dict] = []
    previous_advisor_nmse: float | None = None
    for index, (fit_span, advisor_span) in enumerate(
        zip(fit_slices, advisor_slices), start=1
    ):
        result = bounded_direct_update_core(
            flat_coefficients=current_flat,
            design_fit=design_cache[(fit_span.start, fit_span.stop)],
            design_advisor=design_cache[(advisor_span.start, advisor_span.stop)],
            fit_x=train_x[fit_span],
            advisor_x=train_x[advisor_span],
            pa=pa,
            gain=gain,
            warmup=warmup,
            maximum_pa_input=maximum_pa_input,
            config=config,
            secondary=secondary,
            joint_objective=joint_objective,
        )
        current_flat = current_flat + result.selected_step * result.delta
        record: dict[str, object] = {
            "iteration": index,
            "fit_slice": [fit_span.start, fit_span.stop],
            "advisor_slice": [advisor_span.start, advisor_span.stop],
            "selected_ridge": result.selected_ridge,
            "selected_step": result.selected_step,
            "advisor_nmse_db": result.advisor_nmse_db,
            "selected_primary_nmse_db": result.selected_primary_nmse_db,
            "selected_secondary_nmse_db": result.selected_secondary_nmse_db,
            "advisor_no_dpd_nmse_db": result.advisor_no_dpd_nmse_db,
            "advisor_improvement_db": result.advisor_improvement_db,
            "advisor_gain_over_previous_db": (
                None
                if previous_advisor_nmse is None
                else previous_advisor_nmse - result.advisor_nmse_db
            ),
            "maximum_drive": result.maximum_drive,
            "delta_relative_norm": result.delta_relative_norm,
            "candidate_count": result.candidate_count,
            "elapsed_seconds": result.elapsed_seconds,
        }
        records.append(record)
        previous_advisor_nmse = result.advisor_nmse_db
        if result.advisor_improvement_db < minimum_improvement_db:
            break
    return current_flat, records


def iterative_direct_schedule(
    model: SparseSplineMemoryDPD,
    *,
    pa: object,
    gain: complex,
    train_x: np.ndarray,
    fit_slices: list[slice],
    advisor_slices: list[slice],
    warmup: int,
    maximum_pa_input: float,
    config: DirectLearningConfig | None = None,
    minimum_improvement_db: float = 0.02,
    validation_x: np.ndarray | None = None,
    secondary: tuple[object, complex] | None = None,
    joint_objective: bool = False,
) -> tuple[SparseSplineMemoryDPD, list[dict]]:
    """Spline-memory wrapper around the model-agnostic schedule core.

    Iterations stop when the advisor improvement falls below
    ``minimum_improvement_db`` (or the schedule is exhausted).  When
    ``validation_x`` is given it is *evaluated only* and recorded as a
    read-only diagnostic; it never influences coefficients or stopping.
    ``secondary`` optionally supplies a second frozen evaluator for
    worst-case candidate ranking (surrogate-exploitation guard); with
    ``joint_objective`` the GN step is fitted to the stacked residuals
    of both evaluators.
    """

    train_x = _as_complex_vector(train_x, name="train_x")
    knots = model.knots
    branches = model.branches

    def design_fn(signal: np.ndarray) -> np.ndarray:
        return spline_memory_design_matrix(signal, knots, branches)

    final_flat, records = iterative_direct_schedule_core(
        initial_flat_coefficients=model.coefficients.reshape(-1),
        pa=pa,
        gain=gain,
        train_x=train_x,
        fit_slices=fit_slices,
        advisor_slices=advisor_slices,
        warmup=warmup,
        maximum_pa_input=maximum_pa_input,
        design_fn=design_fn,
        config=config,
        minimum_improvement_db=minimum_improvement_db,
        secondary=secondary,
        joint_objective=joint_objective,
    )
    total_delta = final_flat - model.coefficients.reshape(-1)
    current = model_with_delta(model, total_delta, 1.0)
    if validation_x is not None:
        validation_x = _as_complex_vector(
            validation_x,
            name="validation_x",
        )
        drive = current.predict(validation_x)
        output = cascade_output(pa, drive)
        records.append(
            {
                "validation_diagnostic_only": True,
                "validation_nmse_db": nmse_db(
                    output,
                    gain * validation_x,
                    warmup,
                ),
                "validation_used_for_selection": False,
            }
        )
    return current, records


def ilc_waveform_refinement(
    desired: np.ndarray,
    *,
    pa: object,
    gain: complex,
    warmup: int,
    maximum_pa_input: float,
    beta: float = 1.0,
    maximum_iterations: int = 8,
    minimum_improvement_db: float = 0.05,
    initial_drive: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Gradient-free ILC on the drive waveform ``u <- u + beta*(g*x - P(u))``.

    Returns the best (lowest-NMSE) drive waveform and per-iteration records.
    The caller then fits the causal DPD on ``(desired, u_final)`` pairs; the
    deployed model never implements the ILC update itself.
    """

    desired = _as_complex_vector(desired, name="desired")
    gain = complex(gain)
    if not np.isfinite(gain) or abs(gain) == 0.0:
        raise ValueError("gain must be finite and non-zero")
    if not isinstance(warmup, (int, np.integer)) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must lie in (0, 1]")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be at least one")
    ideal = gain * desired
    drive = (
        _as_complex_vector(initial_drive, name="initial_drive")
        if initial_drive is not None
        else desired.copy()
    )
    output = cascade_output(pa, drive)
    best_nmse = nmse_db(output, ideal, warmup)
    best_drive = drive.copy()
    records: list[dict] = [
        {
            "iteration": 0,
            "cascade_nmse_db": best_nmse,
            "drive_peak": float(np.max(np.abs(drive))),
            "accepted": True,
        }
    ]
    for iteration in range(1, maximum_iterations + 1):
        correction = ideal - output
        candidate_drive = drive + beta * correction
        peak = float(np.max(np.abs(candidate_drive)))
        if peak > maximum_pa_input:
            candidate_drive = candidate_drive * (
                maximum_pa_input / peak
            )
            rescaled = True
        else:
            rescaled = False
        candidate_output = cascade_output(pa, candidate_drive)
        candidate_nmse = nmse_db(candidate_output, ideal, warmup)
        accepted = candidate_nmse < best_nmse
        records.append(
            {
                "iteration": iteration,
                "cascade_nmse_db": candidate_nmse,
                "drive_peak": float(np.max(np.abs(candidate_drive))),
                "rescaled_to_support": rescaled,
                "accepted": accepted,
            }
        )
        if accepted:
            best_nmse = candidate_nmse
            best_drive = candidate_drive.copy()
            drive = candidate_drive
            output = candidate_output
        else:
            break
        if (
            records[-2]["cascade_nmse_db"] - candidate_nmse
            < minimum_improvement_db
        ):
            break
    return best_drive, records
