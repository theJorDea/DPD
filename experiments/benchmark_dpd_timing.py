"""Measure DPD-only streaming time against an explicit 1000-MUL reference.

The result is a same-host software timing diagnostic, not a hardware pass:
the Huawei target clock, implementation language, parallelism and reference
kernel are still unknown.  PA evaluation is intentionally absent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
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
SOURCE_DEPENDENCIES = (
    Path("baseline/spline_memory_dpd.py"),
    Path("baseline/complex_spline_dpd.py"),
    Path("baseline/complexity.py"),
    Path("baseline/metrics.py"),
)


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


def _time_once(
    function: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
) -> tuple[np.ndarray, int]:
    started = time.perf_counter_ns()
    result = function(signal)
    return result, time.perf_counter_ns() - started


def _timing_summary(
    elapsed_ns: list[int],
    *,
    sample_count: int,
    result: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(elapsed_ns, dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return {
        "repeats": len(elapsed_ns),
        "elapsed_ns": [int(value) for value in elapsed_ns],
        "median_ns_total": float(np.median(values)),
        "median_ns_per_sample": float(np.median(values) / sample_count),
        "min_ns_per_sample": float(minimum / sample_count),
        "max_ns_per_sample": float(maximum / sample_count),
        "max_to_min_spread_ratio": (
            maximum / minimum if minimum > 0.0 else None
        ),
        "output_shape": list(result.shape),
        "output_dtype": str(result.dtype),
    }


def _measure_pair(
    dpd_function: Callable[[np.ndarray], np.ndarray],
    reference_function: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure DPD/reference pairs and alternate which function runs first."""

    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >=0 and repeats must be >0")
    for index in range(warmup):
        if index % 2 == 0:
            dpd_function(signal)
            reference_function(signal)
        else:
            reference_function(signal)
            dpd_function(signal)

    dpd_elapsed: list[int] = []
    reference_elapsed: list[int] = []
    execution_order: list[str] = []
    dpd_result: np.ndarray | None = None
    reference_result: np.ndarray | None = None
    for index in range(repeats):
        if index % 2 == 0:
            execution_order.append("dpd_then_reference")
            dpd_result, dpd_ns = _time_once(dpd_function, signal)
            reference_result, reference_ns = _time_once(
                reference_function,
                signal,
            )
        else:
            execution_order.append("reference_then_dpd")
            reference_result, reference_ns = _time_once(
                reference_function,
                signal,
            )
            dpd_result, dpd_ns = _time_once(dpd_function, signal)
        dpd_elapsed.append(dpd_ns)
        reference_elapsed.append(reference_ns)

    assert dpd_result is not None
    assert reference_result is not None
    ratios = [
        float(dpd_ns) / float(reference_ns)
        for dpd_ns, reference_ns in zip(
            dpd_elapsed,
            reference_elapsed,
            strict=True,
        )
    ]
    ratio_values = np.asarray(ratios, dtype=np.float64)
    return {
        "execution_order": execution_order,
        "dpd_numpy_reference": _timing_summary(
            dpd_elapsed,
            sample_count=signal.size,
            result=dpd_result,
        ),
        "scalar_1000_mul_reference": _timing_summary(
            reference_elapsed,
            sample_count=signal.size,
            result=reference_result,
        ),
        "host_python_time_ratio_to_scalar_reference": {
            "per_repeat": ratios,
            "median": float(np.median(ratio_values)),
            "minimum": float(np.min(ratio_values)),
            "maximum": float(np.max(ratio_values)),
        },
    }


def _load_average() -> list[float] | None:
    try:
        return [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _cpu_affinity() -> list[int] | None:
    try:
        return sorted(int(value) for value in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, UnicodeError, IndexError):
        return None
    return None


def _cpu_governors(affinity: list[int] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for cpu in affinity or []:
        path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
        )
        try:
            result[str(cpu)] = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return result


def benchmark_model(
    model: SparseSplineMemoryDPD,
    signal: np.ndarray,
    *,
    chunk_sizes: tuple[int, ...] = (1, 8, 64),
    warmup: int = 1,
    repeats: int = 3,
    reference_real_muls: int = REFERENCE_REAL_MULS,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Benchmark a frozen DPD and the reference on identical input samples.

    ``sample_limit`` is an explicit timing-only crop.  It is useful because
    the scalar Python reference intentionally performs 1000 products per
    sample and is not a practical way to time a complete waveform.  The
    prefix is never used for fitting or model selection.
    """

    samples = np.asarray(signal)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional vector")
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.complex128)
    source_sample_count = int(samples.size)
    if sample_limit is not None:
        if (
            isinstance(sample_limit, (bool, np.bool_))
            or not isinstance(sample_limit, (int, np.integer))
            or int(sample_limit) <= 0
        ):
            raise ValueError("sample_limit must be a positive integer or None")
        if int(sample_limit) > source_sample_count:
            raise ValueError("sample_limit cannot exceed signal length")
        samples = samples[: int(sample_limit)]
    if not chunk_sizes:
        raise ValueError("chunk_sizes must not be empty")
    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or int(size) <= 0
        for size in chunk_sizes
    ):
        raise ValueError("chunk_sizes must contain positive integers")
    if len(set(int(size) for size in chunk_sizes)) != len(chunk_sizes):
        raise ValueError("chunk_sizes must be unique")
    if any(int(size) > samples.size for size in chunk_sizes):
        raise ValueError(
            "chunk_sizes cannot exceed the timed sample count"
        )
    if int(reference_real_muls) != REFERENCE_REAL_MULS:
        raise ValueError(
            "this protocol binds the reference to exactly 1000 real MUL/sample"
        )

    affinity = _cpu_affinity()
    load_average_start = _load_average()
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
        timed = _measure_pair(
            lambda values, size=chunk_size: _stream_model(
                model,
                values,
                size,
            ),
            reference_1000_mul_kernel,
            samples,
            warmup=warmup,
            repeats=repeats,
        )
        timed.update(
            {
                "chunk_size": chunk_size,
                "chunk_invocations_per_repeat": int(
                    (samples.size + chunk_size - 1) // chunk_size
                ),
                "chunk_timing_semantics": (
                    "single-sample Python API invocation latency"
                    if chunk_size == 1
                    else "amortized Python/NumPy block throughput"
                ),
                "state_reset": "zero once before stream; carried between chunks",
                "streaming_equivalent": True,
            }
        )
        rows.append(timed)

    operation_count = model.operation_count().to_dict()

    return {
        "schema_version": 2,
        "artifact_type": "dpd_only_timing_diagnostic",
        "claims_scope": {
            "pa_included": False,
            "hardware_pass_claim": False,
            "target_reference_known": False,
            "customer_gate_evaluable": False,
            "customer_gate_pass": False,
        },
        "protocol": {
            "measurement_class": "host_python_diagnostic",
            "timed_input_key": "desired_input",
            "warmup_pairs_per_chunk": int(warmup),
            "measured_pairs_per_chunk": int(repeats),
            "pairing": (
                "DPD and scalar reference measured adjacently; first "
                "function alternates by repeat"
            ),
            "timed_implementation": (
                "Python control plus NumPy complex128 internal reference"
            ),
            "analytical_schedule_implemented_by_timed_path": False,
            "concurrent_workload_controlled": False,
            "cpu_affinity_single_core": (
                len(affinity) == 1 if affinity is not None else None
            ),
        },
        "host": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "cpu_affinity": affinity,
            "cpu_governors": _cpu_governors(affinity),
            "load_average_start_1_5_15_min": load_average_start,
            "load_average_end_1_5_15_min": _load_average(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "signal": {
            "sample_count": int(samples.size),
            "source_sample_count": source_sample_count,
            "timing_prefix": (
                "first samples from desired_input"
                if sample_limit is not None
                else "complete desired_input"
            ),
            "input_dtype": str(samples.dtype),
        },
        "reference_contract": {
            "nominal_real_multiplications_per_sample": REFERENCE_REAL_MULS,
            "kernel": "scalar_python_500_real_plus_500_imag_products",
            "also_executes": (
                "scalar additions, Python conversions/property access, "
                "complex construction and coefficient allocation"
            ),
            "multiplication_equivalent_budget": False,
        },
        "dpd": {
            "model_metadata": model.metadata,
            "model_storage": {
                "knot_strategy": model.knot_strategy,
                "knot_count": model.knot_count,
                "branch_count": model.branch_count,
                "knots_dtype": str(model.knots.dtype),
                "coefficients_dtype": str(model.coefficients.dtype),
                "stored_complex_coefficients": (
                    model.stored_complex_coefficients
                ),
            },
            "analytical_deployment_operation_vector": operation_count,
            "analytical_vector_is_not_timed_python_trace": True,
            "chunk_results": rows,
        },
        "interpretation": (
            "Only a target-specific streaming measurement against the "
            "customer reference kernel can establish the DPD time gate. "
            "The paired host-Python ratios are diagnostic and are not a "
            "multiplication-equivalent latency claim."
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


def _bound_file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        name: _file_sha256(path)
        for name, path in sorted(paths.items())
    }


def _git_provenance(git_dir: Path) -> dict[str, Any]:
    resolved = git_dir.resolve()
    if not resolved.is_dir():
        raise ValueError(f"git directory does not exist: {resolved}")
    prefix = [
        "git",
        f"--git-dir={resolved}",
        f"--work-tree={PROJECT_ROOT}",
    ]

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            [*prefix, *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status_text = run(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    status_lines = status_text.splitlines() if status_text else []
    return {
        "commit": commit,
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
    }


def _write_json_atomic(
    path: Path,
    value: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Publish strict JSON atomically and never overwrite unless requested."""

    path = path.absolute()
    if path.is_symlink():
        raise ValueError("output path must not be a symbolic link")
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(
            f"output already exists; pass --overwrite explicitly: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                _json_ready(value),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
            temporary_path.unlink()
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-sizes", default="1,8,64")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--max-samples",
        type=int,
        required=True,
        help=(
            "time only the first N desired-input samples; the full input "
            "archive remains bound in provenance"
        ),
    )
    parser.add_argument(
        "--git-dir",
        required=True,
        type=Path,
        help="real git directory used to bind commit and worktree state",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="refuse canonical timing when the source worktree is dirty",
    )
    parser.add_argument(
        "--workload-note",
        default="OS background activity not sealed",
        help="explicit statement about concurrent host workload",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing output artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.warmup < 2:
        raise ValueError("canonical CLI timing requires at least 2 warmup pairs")
    if args.repeats < 9 or args.repeats % 2 == 0:
        raise ValueError(
            "canonical CLI timing requires an odd repeat count of at least 9"
        )
    output_path = args.output.absolute()
    if output_path.is_symlink():
        raise ValueError("output path must not be a symbolic link")
    if os.path.lexists(output_path) and not args.overwrite:
        raise FileExistsError(
            "output already exists; pass --overwrite explicitly"
        )

    model_path = args.model.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    source_paths = {
        "experiments/benchmark_dpd_timing.py": Path(__file__).resolve(),
        **{
            str(relative): (PROJECT_ROOT / relative).resolve(strict=True)
            for relative in SOURCE_DEPENDENCIES
        },
    }
    bound_paths = {
        "model": model_path,
        "input_archive": input_path,
        **source_paths,
    }
    hashes_before = _bound_file_hashes(bound_paths)
    git_before = _git_provenance(args.git_dir)
    if args.require_clean_git and git_before["dirty"]:
        raise RuntimeError(
            "canonical timing requires a clean worktree; status was "
            f"{git_before['status_porcelain']}"
        )

    started_utc = datetime.now(timezone.utc).isoformat()
    with np.load(input_path, allow_pickle=False) as archive:
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
        sample_limit=args.max_samples,
    )
    finished_utc = datetime.now(timezone.utc).isoformat()
    hashes_after = _bound_file_hashes(bound_paths)
    if hashes_after != hashes_before:
        raise RuntimeError("a bound input/source file changed during timing")
    git_after = _git_provenance(args.git_dir)
    if git_after != git_before:
        raise RuntimeError("git commit/worktree state changed during timing")

    result["provenance"] = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "workload_note": args.workload_note,
        "model_path": str(model_path),
        "model_sha256": hashes_before["model"],
        "input_archive_path": str(input_path),
        "input_archive_sha256": hashes_before["input_archive"],
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": hashes_before[
            "experiments/benchmark_dpd_timing.py"
        ],
        "source_sha256": {
            name: hashes_before[name] for name in source_paths
        },
        "pre_post_hashes_equal": True,
        "git": git_before,
        "command": [sys.executable, *sys.argv],
    }
    _write_json_atomic(
        output_path,
        result,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
