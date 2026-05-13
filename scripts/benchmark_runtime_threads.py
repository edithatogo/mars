#!/usr/bin/env python3
"""Benchmark and smoke-check replay paths with runtime thread controls.

This script exercises ``pymars.runtime`` `design_matrix` and `predict` with
configurable thread hints and prints timing plus process-memory deltas.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from platform import system as platform_system
from typing import Any, Iterable, Iterator

try:
    import resource
except Exception:  # pragma: no cover - portability for systems without resource
    resource = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RUNTIME_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/model_spec_v1.json"
)


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values: list[int] = []
    for item in raw.split(","):
        value = int(item.strip())
        if value < 1:
            raise ValueError("row counts must be >= 1")
        values.append(value)
    return values


def parse_threads(raw: str) -> list[int]:
    """Parse comma-separated thread counts (`1`, `2`, ...)."""
    values: list[int] = []
    for item in raw.split(","):
        value = int(item.strip())
        if value < 1:
            raise ValueError("thread counts must be >= 1")
        values.append(value)
    return values


def make_rows(rows: int) -> list[list[float]]:
    """Generate a deterministic dense matrix for replay benchmarking."""
    return [[index * 0.01, index * 0.025, (index % 11) / 11.0] for index in range(rows)]


def read_fixture(runtime_module: Any) -> dict:
    """Load the reusable fixture path used by runtime tests."""
    return runtime_module.load_model_spec(RUNTIME_SPEC_PATH.as_posix())


@contextmanager
def _thread_context(runtime_module: Any, threads: int | None) -> Iterator[None]:
    if threads is None:
        with runtime_module.runtime_threads(None):
            yield
            return
    with runtime_module.runtime_threads(threads):
        yield


def _current_rss_kib() -> int | None:
    if resource is None:
        return None
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes. Normalize to KiB.
    if platform_system() == "Darwin":
        return int(ru_maxrss / 1024)
    return int(ru_maxrss)


def run_benchmarks(
    *,
    mode: str,
    thread_counts: Iterable[int],
    row_counts: Iterable[int],
    repeats: int,
    runtime_module: Any,
) -> list[dict]:
    """Run benchmark loops and return rows suitable for reporting."""
    spec = read_fixture(runtime_module)
    results: list[dict] = []
    for row_count in row_counts:
        rows = make_rows(row_count)
        for threads in thread_counts:
            with _thread_context(runtime_module, threads):
                durations: list[float] = []
                mem_before = _current_rss_kib()
                for _ in range(repeats):
                    _thread_start = time.perf_counter()
                    if mode == "predict":
                        runtime_module.predict(spec, rows)
                    else:
                        runtime_module.design_matrix(spec, rows)
                    durations.append(time.perf_counter() - _thread_start)
                mem_after = _current_rss_kib()

            duration_ms = [value * 1000 for value in durations]
            result = {
                "mode": mode,
                "rows": row_count,
                "threads": threads,
                "median_ms": statistics.median(duration_ms),
                "min_ms": min(duration_ms),
                "max_ms": max(duration_ms),
                "mean_ms": statistics.mean(duration_ms),
            }
            if mem_before is not None and mem_after is not None:
                result["rss_kib_delta"] = max(0, mem_after - mem_before)
            results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark runtime prediction/design-matrix replay paths."
    )
    parser.add_argument(
        "--mode",
        choices=("predict", "design_matrix"),
        default="predict",
        help="Which runtime API to exercise.",
    )
    parser.add_argument(
        "--rows",
        default="1024,8192",
        type=parse_int_list,
        help="Comma-separated row counts.",
    )
    parser.add_argument(
        "--threads",
        default="1,2",
        type=parse_threads,
        help="Comma-separated thread counts to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        default=5,
        type=int,
        help="Repeats per (rows, threads) configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the benchmark harness and print summary statistics."""
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    try:
        import pymars.runtime as runtime_module
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "pymars runtime import failed. Install project dependencies (including "
            "scikit-learn) and re-run this script."
            f"\nOriginal import error: {exc}"
        )

    results = run_benchmarks(
        mode=args.mode,
        thread_counts=args.threads,
        row_counts=args.rows,
        repeats=args.repeats,
        runtime_module=runtime_module,
    )

    for row in results:
        rss_note = ""
        if "rss_kib_delta" in row:
            rss_note = f", rss_kib_delta={row['rss_kib_delta']}"
        print(
            "{mode} rows={rows:>5} threads={threads:<2} "
            "median_ms={median_ms:8.3f} min_ms={min_ms:8.3f} "
            "max_ms={max_ms:8.3f}{rss}".format(
                rss=rss_note,
                mode=row["mode"],
                rows=row["rows"],
                threads=row["threads"],
                median_ms=row["median_ms"],
                min_ms=row["min_ms"],
                max_ms=row["max_ms"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
