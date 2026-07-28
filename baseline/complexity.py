"""Transparent operation and storage accounting for DPD candidates.

Counts are per complex sample.  Fused multiply-add is deliberately decomposed
into one real multiplication and one real addition.  The primary convention is

    one complex multiply = 4 real multiplications + 2 real additions.

The optional Gauss convention uses 3 multiplications + 5 additions.  Divisions,
square roots/trigonometric functions, comparisons, lookups and memory traffic
remain separate instead of being hidden in a generic FLOP number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Literal

ComplexMultiplyConvention = Literal["4m2a", "3m5a"]
GMPLeadingPolicy = Literal["causal_leading", "opendpd_exact"]


@dataclass(frozen=True)
class OperationCount:
    real_multiplications: int = 0
    real_additions: int = 0
    real_divisions: int = 0
    nonlinear_operations: int = 0
    comparisons: int = 0
    lookups: int = 0
    real_memory_reads: int = 0
    real_memory_writes: int = 0
    stored_real_coefficients: int = 0
    stored_real_constants: int = 0
    state_real_values: int = 0
    notes: tuple[str, ...] = ()

    def __add__(self, other: "OperationCount") -> "OperationCount":
        if not isinstance(other, OperationCount):
            return NotImplemented
        numeric = {
            key: getattr(self, key) + getattr(other, key)
            for key in asdict(self)
            if key != "notes"
        }
        return OperationCount(**numeric, notes=self.notes + other.notes)

    def scaled(self, factor: int) -> "OperationCount":
        if factor < 0:
            raise ValueError("factor must be non-negative")
        numeric = {
            key: factor * getattr(self, key)
            for key in asdict(self)
            if key != "notes"
        }
        return OperationCount(**numeric, notes=self.notes)

    def coefficient_bytes(self, bits_per_real: int) -> int:
        if bits_per_real <= 0:
            raise ValueError("bits_per_real must be positive")
        bits = self.stored_real_coefficients * bits_per_real
        return (bits + 7) // 8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def complex_multiply_cost(
    convention: ComplexMultiplyConvention = "4m2a",
) -> tuple[int, int]:
    if convention == "4m2a":
        return 4, 2
    if convention == "3m5a":
        return 3, 5
    raise ValueError(f"unknown complex multiply convention: {convention}")


def complex_spline_inference_cost(
    knot_count: int,
    *,
    convention: ComplexMultiplyConvention = "4m2a",
    indexing: Literal["binary", "uniform"] = "binary",
    reciprocal_widths: bool = True,
    amplitude_coordinate: bool = True,
) -> OperationCount:
    """Count one memoryless complex linear-spline sample.

    The count follows the implementation form
    ``c = c0 + t*(c1-c0); z = x*c``.  It includes magnitude/power formation and
    interval interpolation.  Precomputed reciprocal interval widths replace
    one division by one multiplication, a normal fixed-point implementation
    choice.  ``amplitude_coordinate=False`` describes a hypothetical spline
    that is trained/interpolated directly in ``q=I²+Q²``.  It is not a
    cost-free implementation of :class:`ComplexLinearSplineDPD`, whose knots
    are explicitly in amplitude.
    """

    if knot_count < 2:
        raise ValueError("knot_count must be at least two")
    cmul, cadd = complex_multiply_cost(convention)
    comparisons = (
        int(math.ceil(math.log2(knot_count))) if indexing == "binary" else 2
    )
    if indexing not in {"binary", "uniform"}:
        raise ValueError(f"unknown indexing mode: {indexing}")

    # |x|^2: 2M + 1A.  sqrt is explicit for a spline linear in amplitude.
    multiplications = 2
    additions = 1
    nonlinear = 1 if amplitude_coordinate else 0

    # t=(r-r0)*inv_width, or one explicit division.
    additions += 1
    divisions = 0
    stored_constants = knot_count
    if reciprocal_widths:
        multiplications += 1
        stored_constants += knot_count - 1
    else:
        divisions += 1

    # Complex c0 + t*(c1-c0): two real subtracts, two real multiplies,
    # two real adds.
    multiplications += 2
    additions += 4

    # Complex x*c.
    multiplications += cmul
    additions += cadd

    return OperationCount(
        real_multiplications=multiplications,
        real_additions=additions,
        real_divisions=divisions,
        nonlinear_operations=nonlinear,
        comparisons=comparisons,
        lookups=2,
        real_memory_reads=6,
        real_memory_writes=2,
        stored_real_coefficients=2 * knot_count,
        stored_real_constants=stored_constants,
        notes=(
            (
                "amplitude-coordinate ComplexLinearSplineDPD; sqrt counted separately"
                if amplitude_coordinate
                else (
                    "hypothetical power-coordinate spline; requires a "
                    "separately trained q-domain model"
                )
            ),
            f"complex multiply convention {convention}",
            f"{indexing} interval selection",
        ),
    )


def spline_memory_branch_cost(
    knot_count: int,
    branch_count: int,
    *,
    convention: ComplexMultiplyConvention = "4m2a",
    shared_envelope: bool = False,
) -> OperationCount:
    """Arithmetic lower bound for a sum of spline memory branches.

    Delay-line reads/writes and address/control overhead depend on the selected
    ``(m,d)`` pairs and are not inferable from ``branch_count`` alone.  They
    remain explicitly excluded instead of being disguised as zero-cost state.
    """

    if branch_count < 1:
        raise ValueError("branch_count must be positive")
    branch = complex_spline_inference_cost(
        knot_count,
        convention=convention,
    )
    result = branch.scaled(branch_count)
    # Sum branch complex outputs.
    result = result + OperationCount(real_additions=2 * (branch_count - 1))
    if shared_envelope and branch_count > 1:
        # Remove repeated magnitude formation (2M, 1A, 1 sqrt) for branches
        # known to use the same envelope delay.
        values = result.to_dict()
        values.update(
            real_multiplications=result.real_multiplications
            - 2 * (branch_count - 1),
            real_additions=result.real_additions - (branch_count - 1),
            nonlinear_operations=result.nonlinear_operations
            - (branch_count - 1),
            notes=result.notes + ("shared envelope magnitude",),
        )
        result = OperationCount(
            **values
        )
    values = result.to_dict()
    values["notes"] = result.notes + (
        "delay-buffer memory traffic/control excluded; arithmetic lower bound",
    )
    return OperationCount(**values)


def memory_polynomial_inference_cost(
    orders: Iterable[int],
    delays: Iterable[int],
    *,
    convention: ComplexMultiplyConvention = "4m2a",
) -> OperationCount:
    """Count a causal complex memory-polynomial sample.

    The dictionary is
    ``sum_d,p a[d,p] x[n-d] |x[n-d]|**(p-1)``.  For each delayed
    sample, ``q=|x|²`` and reusable scalar powers are formed once.  Arbitrary
    positive integer orders are supported: if an exponent ``p-1`` is odd, one
    shared ``sqrt(q)`` is counted as a nonlinear operation.  Each non-linear
    term then performs a complex-real product followed by a complex coefficient
    multiply.  Delay-line traffic and address generation are reported as an
    analytical lower bound, not a measured implementation result.
    """

    order_tuple = tuple(int(order) for order in orders)
    delay_tuple = tuple(int(delay) for delay in delays)
    if (
        not order_tuple
        or not delay_tuple
        or any(order < 1 for order in order_tuple)
        or any(delay < 0 for delay in delay_tuple)
        or len(set(order_tuple)) != len(order_tuple)
        or len(set(delay_tuple)) != len(delay_tuple)
    ):
        raise ValueError("orders must be unique positive; delays non-negative")
    cmul, cadd = complex_multiply_cost(convention)
    terms_per_delay = len(order_tuple)
    amplitude_exponents = tuple(order - 1 for order in order_tuple)
    maximum_q_power = max(exponent // 2 for exponent in amplitude_exponents)
    odd_exponents = tuple(
        exponent for exponent in amplitude_exponents if exponent % 2 == 1
    )
    needs_envelope = any(exponent > 0 for exponent in amplitude_exponents)

    # q=|x|² costs 2M+1A. q¹ is already available, so q²...q^s costs
    # max(s-1, 0) further scalar multiplications.
    multiplications_per_delay = (
        2 + max(maximum_q_power - 1, 0)
        if needs_envelope
        else 0
    )
    additions_per_delay = 1 if needs_envelope else 0
    # sqrt(q) supplies r for all odd exponents. r*q^s costs one scalar
    # multiplication except for r itself (exponent one).
    multiplications_per_delay += sum(
        exponent > 1 for exponent in odd_exponents
    )
    for exponent in amplitude_exponents:
        if exponent:
            multiplications_per_delay += 2
        multiplications_per_delay += cmul
        additions_per_delay += cadd
    term_count = len(delay_tuple) * terms_per_delay
    return OperationCount(
        real_multiplications=len(delay_tuple) * multiplications_per_delay,
        real_additions=(
            len(delay_tuple) * additions_per_delay
            + 2 * max(term_count - 1, 0)
        ),
        nonlinear_operations=(
            len(delay_tuple) if odd_exponents else 0
        ),
        comparisons=len(delay_tuple),
        lookups=0,
        real_memory_reads=(
            len(delay_tuple) * (2 + 2 * terms_per_delay)
        ),
        real_memory_writes=2,
        stored_real_coefficients=2 * term_count,
        stored_real_constants=len(order_tuple) + len(delay_tuple),
        # A causal delay line with maximum delay D stores D previous complex
        # input samples; the current sample is not additional persistent state.
        state_real_values=2 * max(delay_tuple),
        notes=(
            f"complex multiply convention {convention}",
            "shared |x|^2, sqrt, and envelope powers per delay",
            (
                "analytical arithmetic lower bound; delay-buffer capacity is "
                "counted, dynamic traffic and power/address control are not measured"
            ),
        ),
    )


def gmp_leading_coefficient_count(
    *,
    kc: int,
    lc: int,
    mc: int,
    leading_policy: GMPLeadingPolicy,
) -> int:
    """Return stored complex leading-branch coefficients.

    ``opendpd_exact`` stores every ``(k,q,lead)`` combination and can require
    future envelope samples. ``causal_leading`` stores only ``lead <= q``;
    structural future-term zeros are not counted as coefficients.
    """

    if kc < 0 or lc < 0 or mc < 0:
        raise ValueError("GMP leading dimensions must be non-negative")
    if kc == 0:
        if lc != 0 or mc != 0:
            raise ValueError("disabled leading branch requires lc=mc=0")
        return 0
    if lc < 1 or mc < 1:
        raise ValueError("enabled leading branch requires positive lc and mc")
    if leading_policy == "opendpd_exact":
        return kc * lc * mc
    if leading_policy == "causal_leading":
        return kc * sum(min(mc, q) for q in range(lc))
    raise ValueError(f"unknown GMP leading policy: {leading_policy}")


def gmp_inference_cost(
    *,
    ka: int,
    la: int,
    kb: int = 0,
    lb: int = 0,
    mb: int = 0,
    kc: int = 0,
    lc: int = 0,
    mc: int = 0,
    leading_policy: GMPLeadingPolicy = "causal_leading",
    convention: ComplexMultiplyConvention = "4m2a",
) -> OperationCount:
    """Count a factorized streaming GMP kernel per complex sample.

    The basis matches ``OpenDPD/benchmark/benchmark_volterra.py``.  Inference
    is factorized by complex-signal delay:

    ``y[n] = sum_q x[n-q] * h_q(envelope-power streams)``.

    Thus there is one complex multiplication per active base delay, not one
    per dictionary column.  The count is valid only for an implementation
    numerically equivalent to this factorization; a dense ``Phi @ c`` kernel
    is more expensive.
    """

    if ka < 1 or la < 1:
        raise ValueError("aligned GMP branch requires positive ka and la")
    if kb < 0 or lb < 0 or mb < 0:
        raise ValueError("GMP lagging dimensions must be non-negative")
    if kb == 0:
        if lb != 0 or mb != 0:
            raise ValueError("disabled lagging branch requires lb=mb=0")
    elif lb < 1 or mb < 1:
        raise ValueError("enabled lagging branch requires positive lb and mb")
    leading_count = gmp_leading_coefficient_count(
        kc=kc,
        lc=lc,
        mc=mc,
        leading_policy=leading_policy,
    )
    aligned_count = ka * la
    lagging_count = kb * lb * mb
    coefficient_count = aligned_count + lagging_count + leading_count
    base_delay_count = max(
        la,
        lb if kb else 0,
        lc if kc else 0,
    )
    nonlinear_coefficient_count = coefficient_count - la
    maximum_exponent = max(ka - 1, kb, kc)
    cmul, cadd = complex_multiply_cost(convention)

    # One current-sample magnitude/power stream is generated and delayed.
    # |x|² costs 2M+1A; sqrt is separate; r³...r^P cost P-2 products.
    generator_multiplications = (
        0
        if maximum_exponent == 0
        else 2 + max(maximum_exponent - 2, 0)
    )
    generator_additions = 1 if maximum_exponent > 0 else 0
    multiplications = (
        generator_multiplications
        + 2 * nonlinear_coefficient_count
        + cmul * base_delay_count
    )
    additions = (
        generator_additions
        + 2 * nonlinear_coefficient_count
        + cadd * base_delay_count
        + 2 * max(base_delay_count - 1, 0)
    )

    lookahead = mc if kc and leading_policy == "opendpd_exact" else 0
    maximum_raw_delay = base_delay_count - 1
    power_stream_delays: list[int] = []
    for exponent in range(1, maximum_exponent + 1):
        candidates: list[int] = []
        if exponent <= ka - 1:
            candidates.append(la - 1)
        if kb and exponent <= kb:
            candidates.append(lb - 1 + mb)
        if kc and exponent <= kc:
            candidates.append(max(lc - 2, 0))
        power_stream_delays.append(max(candidates, default=0))
    state_real_values = (
        2 * (lookahead + maximum_raw_delay)
        + sum(lookahead + delay for delay in power_stream_delays)
    )

    return OperationCount(
        real_multiplications=multiplications,
        real_additions=additions,
        nonlinear_operations=1 if maximum_exponent > 0 else 0,
        comparisons=0,
        lookups=0,
        real_memory_reads=(
            2 * coefficient_count
            + nonlinear_coefficient_count
            + 2 * base_delay_count
        ),
        real_memory_writes=(
            2 + maximum_exponent
            if maximum_exponent > 0
            else 2
        ),
        stored_real_coefficients=2 * coefficient_count,
        stored_real_constants=9,
        state_real_values=state_real_values,
        notes=(
            f"complex multiply convention {convention}",
            "factorized y=sum_q x[n-q]*h_q kernel",
            f"leading policy {leading_policy}; lookahead={lookahead} samples",
            (
                "persistent state stores raw I/Q and delayed amplitude-power "
                "streams; static addressing assumes no per-sample comparisons"
            ),
        ),
    )


def esn_fan_scalar_cost(
    reservoir_size: int,
    *,
    input_dimension: int = 2,
    polynomial_degree_two_features: int = 5,
    fan_terms: int = 8,
    dense_recurrence: bool = True,
    recurrent_nonzeros: int | None = None,
) -> OperationCount:
    """Count the current ``EnhancedESN_FAN`` scalar-output inference path.

    For the notebook configuration ``input_dimension=2``, the deterministic
    feature dimension is ``5 + 2*8*2 = 37``.  StandardScaler divisions are
    normalized to multiplications by stored reciprocals, which is favorable to
    a hardware implementation.  The source evaluates the two sin/cos argument
    expressions separately.
    """

    if reservoir_size < 1:
        raise ValueError("reservoir_size must be positive")
    if input_dimension < 1 or fan_terms < 1:
        raise ValueError("input_dimension and fan_terms must be positive")
    fourier_features = 2 * fan_terms * input_dimension
    feature_count = (
        reservoir_size + polynomial_degree_two_features + fourier_features
    )
    if dense_recurrence:
        recurrence_mult = reservoir_size * reservoir_size
        recurrence_add = reservoir_size * (reservoir_size - 1)
        recurrent_reads = reservoir_size * reservoir_size + reservoir_size
    else:
        if recurrent_nonzeros is None or recurrent_nonzeros < 0:
            raise ValueError("recurrent_nonzeros is required for sparse counting")
        recurrence_mult = recurrent_nonzeros
        recurrence_add = max(recurrent_nonzeros - reservoir_size, 0)
        recurrent_reads = 2 * recurrent_nonzeros

    # Win matvec, W matvec, vector sum, tanh and leaky update.
    mult = recurrence_mult + reservoir_size * (input_dimension + 1) + 2 * reservoir_size
    add = (
        recurrence_add
        + reservoir_size * input_dimension
        + reservoir_size
        + reservoir_size
    )

    # Raw scaler reciprocals, degree-2 products for d=2, FAN angles,
    # combined scaler reciprocals, and ridge readout.
    mult += input_dimension
    mult += polynomial_degree_two_features - input_dimension
    mult += fourier_features
    mult += feature_count
    mult += feature_count
    add += input_dimension + feature_count + feature_count

    stored = (
        (reservoir_size * reservoir_size if dense_recurrence else recurrent_nonzeros)
        + reservoir_size * (input_dimension + 1)
        + feature_count
        + 1
        + 2 * input_dimension
        + 2 * feature_count
    )
    # Coefficient/state operand reads under a simple non-cached scalar schedule.
    # Cache/broadcast reuse can reduce physical memory transactions, so this is
    # an algorithmic access convention rather than a measured bus count.
    win_reads = reservoir_size * (input_dimension + 1) + (input_dimension + 1)
    scaler_reads = 2 * input_dimension + 2 * feature_count
    feature_and_readout_reads = (
        input_dimension
        + reservoir_size
        + polynomial_degree_two_features
        + fourier_features
        + 2 * feature_count
    )
    return OperationCount(
        real_multiplications=mult,
        real_additions=add,
        nonlinear_operations=reservoir_size + fourier_features,
        comparisons=2 * input_dimension,
        real_memory_reads=(
            recurrent_reads
            + win_reads
            + scaler_reads
            + feature_and_readout_reads
        ),
        real_memory_writes=reservoir_size + 1,
        stored_real_coefficients=int(stored),
        state_real_values=reservoir_size,
        notes=(
            "StandardScaler divisions replaced by stored reciprocals",
            "FAN counts separate sin/cos angle expressions",
            "dense ndarray recurrence" if dense_recurrence else "ideal sparse recurrence",
            (
                "memory reads use an uncached scalar-operand convention; "
                "not measured DRAM traffic"
            ),
        ),
    )


def esn_fan_complex_pair_cost(
    reservoir_size: int,
    **kwargs: object,
) -> OperationCount:
    """Two independent scalar-output ESNs used for I and Q."""

    return esn_fan_scalar_cost(reservoir_size, **kwargs).scaled(2)
