"""Validation evidence helpers for accelerator backend contract coverage."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


@contextmanager
def _env_override(module: Any, key: str, value: str | None) -> Iterator[None]:
    previous = module.environ.get(key)
    if value is None:
        module.environ.pop(key, None)
    else:
        module.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            module.environ.pop(key, None)
        else:
            module.environ[key] = previous


def run_benchmarks(
    *,
    runtime_module: Any,
    accelerator_module: Any,
    iterations: int,
    requested_backends: Iterable[str | None],
) -> list[dict[str, Any]]:
    """Measure selection and fallback paths for the shared accelerator layer."""
    results: list[dict[str, Any]] = []
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/model_spec_v1.json"
    )
    spec = runtime_module.load_model_spec(fixture_path)
    rows = [[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]]

    for requested in requested_backends:
        with _env_override(
            accelerator_module.os,
            accelerator_module.ACCELERATOR_ENV_VAR,
            requested,
        ):
            durations: list[float] = []
            for _ in range(iterations):
                start = time.perf_counter()
                accelerator_module.accelerator_backend_summary()
                durations.append(time.perf_counter() - start)

            summary = accelerator_module.accelerator_backend_summary()
            cpu_durations: list[float] = []
            for _ in range(iterations):
                start = time.perf_counter()
                runtime_module.predict(spec, rows)
                cpu_durations.append(time.perf_counter() - start)

            results.append(
                {
                    "requested": requested or "cpu",
                    "selected": summary["selected"],
                    "fallback": summary["fallback"],
                    "registry_median_us": statistics.median(durations) * 1e6,
                    "cpu_predict_median_us": statistics.median(cpu_durations) * 1e6,
                }
            )
    return results
