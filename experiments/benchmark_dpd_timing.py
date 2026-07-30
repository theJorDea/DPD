"""Measure DPD-only streaming time against an explicit 1000-MUL reference.

The result is a same-host software timing diagnostic, not a hardware pass:
the Huawei target clock, implementation language, parallelism and reference
kernel are still unknown.  PA evaluation is intentionally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.spline_memory_dpd import (  # noqa: E402
    SparseSplineMemoryDPD,
    SplineMemoryState,
)
from baseline.metrics import as_complex  # noqa: E402

REFERENCE_REAL_MULS = 1000


def reference_1000_mul_kernel(signal: np.ndarray) -> np.ndarray:
    """Execute exactly 1000 scalar real products per complex sample.

    The two real components use 500 products each.  This deliberately simple
    Python reference is only a timing anchor on the current host; it is not a
    claimed FPGA/ASIC schedule.
    """

    samples = np.asarray(signal, dtype=np.complex128)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional vector")
    coefficients = np.linspace(
        0.25,
        1.25,
        REFERENCE_REAL_MULS,
        dtype=np.float64,
    )
    output = np.empty(samples.size, dtype=np.complex128)
    for index, sample in enumerate(samples):
        real_acc = 0.0
        imag_acc = 0.0
        for coefficient in coefficients[: REFERENCE_REAL_MULS // 2]:
            real_acc += float(sample.real) * float(coefficient)
        for coefficient in coefficients[REFERENCE_REAL_MULS // 2 :]:
            imag_acc += float(sample.imag) * float(coefficient)
        output[index] = real_acc + 1j * imag_acc
    return output


def _chunk_slices(length: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def _stream_model(
    model: SparseSplineMemoryDPD,
    signal: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    state: SplineMemoryState = model.initial_state()
    outputs: list[np.ndarray] = []
    for selection in _chunk_slices(signal.size, chunk_size):
        output, state = model.predict_chunk(signal[selection], state)
        outputs.append(np.asarray(output))
    return np.concatenate(outputs)


def _measure(
    function: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >=0 and repeats must be >0")
    for _ in range(warmup):
        function(signal)
    elapsed_ns: list[int] = []
    result: np.ndarray | None = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function(signal)
        elapsed_ns.append(time.perf_counter_ns() - started)
    assert result is not None
    values = np.asarray(elapsed_ns, dtype=np.float64)
    return {
        "repeats": repeats,
        "elapsed_ns": [int(value) for value in elapsed_ns],
        "median_ns_total": float(np.median(values)),
        "median_ns_per_sample": float(np.median(values) / signal.size),
        "min_ns_per_sample": float(np.min(values) / signal.size),
        "max_ns_per_sample": float(np.max(values) / signal.size),
        "output_shape": list(result.shape),
        "output_dtype": str(result.dtype),
    }


def benchmark_model(
    model: SparseSplineMemoryDPD,
    signal: np.ndarray,
    *,
    chunk_sizes: tuple[int, ...] = (1, 8, 64),
    warmup: int = 1,
    repeats: int = 3,
    reference_real_muls: int = REFERENCE_REAL_MULS,
) -> dict[str, Any]:
    """Benchmark a frozen DPD and the reference on identical input samples."""

    samples = np.asarray(signal)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional vector")
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.complex128)
    if any(
        not isinstance(size, (int, np.integer)) or int(size) <= 0
        for size in chunk_sizes
    ):
        raise ValueError("chunk_sizes must contain positive integers")
    if len(set(int(size) for size in chunk_sizes)) != len(chunk_sizes):
        raise ValueError("chunk_sizes must be unique")
    if int(reference_real_muls) != REFERENCE_REAL_MULS:
        raise ValueError(
            "this protocol binds the reference to exactly 1000 real MUL/sample"
        )

    full_output = model.predict(samples)
    rows: list[dict[str, Any]] = []
    for raw_size in chunk_sizes:
        chunk_size = int(raw_size)
        streamed = _stream_model(model, samples, chunk_size)
        if not np.array_equal(streamed, full_output):
            raise RuntimeError(
                f"streaming output differs from independent model output "
                f"for chunk size {chunk_size}"
            )
        timed = _measure(
            lambda values, size=chunk_size: _stream_model(
                model,
                values,
                size,
            ),
            samples,
            warmup=warmup,
            repeats=repeats,
        )
        timed.update(
            {
                "chunk_size": chunk_size,
                "state_reset": "zero once before stream; carried between chunks",
                "streaming_equivalent": True,
            }
        )
        rows.append(timed)

    reference = _measure(
        reference_1000_mul_kernel,
        samples,
        warmup=warmup,
        repeats=repeats,
    )
    reference.update(
        {
            "nominal_real_multiplications_per_sample": REFERENCE_REAL_MULS,
            "kernel": "scalar_python_500_real_plus_500_imag_products",
        }
    )
    operation_count = model.operation_count().to_dict()
    reference_ns = float(reference["median_ns_per_sample"])
    for row in rows:
        row["time_ratio_to_reference"] = (
            float(row["median_ns_per_sample"]) / reference_ns
            if reference_ns > 0.0
            else None
        )

    return {
        "schema_version": 1,
        "artifact_type": "dpd_only_timing_diagnostic",
        "claims_scope": {
            "pa_included": False,
            "hardware_pass_claim": False,
            "target_reference_known": False,
        },
        "host": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "signal": {
            "sample_count": int(samples.size),
            "input_dtype": str(samples.dtype),
        },
        "reference": reference,
        "dpd": {
            "model_metadata": model.metadata,
            "operation_vector": operation_count,
            "chunk_results": rows,
        },
        "interpretation": (
            "Only a target-specific streaming measurement against the "
            "customer reference kernel can establish the DPD time gate. "
            "This host-Python ratio is diagnostic."
        ),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-sizes", default="1,8,64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    with np.load(args.input, allow_pickle=False) as archive:
        if "desired_input" not in archive.files:
            raise ValueError("input archive must contain desired_input")
        signal = as_complex(
            np.asarray(archive["desired_input"]),
            name="desired_input",
        )
    chunk_sizes = tuple(
        int(value.strip())
        for value in args.chunk_sizes.split(",")
        if value.strip()
    )
    result = benchmark_model(
        SparseSplineMemoryDPD.load(args.model),
        signal,
        chunk_sizes=chunk_sizes,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result["provenance"] = {
        "model_path": str(args.model.resolve()),
        "model_sha256": _file_sha256(args.model.resolve()),
        "input_archive_path": str(args.input.resolve()),
        "input_archive_sha256": _file_sha256(args.input.resolve()),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "command": [sys.executable, *sys.argv],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
