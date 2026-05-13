"""Tests for accelerator validation evidence helpers."""

from __future__ import annotations

import pymars.accelerator as accelerator_module
import pymars.runtime as runtime_module
from pymars.accelerator_validation import run_benchmarks


def test_accelerator_validation_benchmark_reports_fallback(monkeypatch) -> None:
    """Benchmark evidence should record fallback when no backend is selected."""
    clear = accelerator_module.clear_accelerator_backends
    clear()
    monkeypatch.delenv(accelerator_module.ACCELERATOR_ENV_VAR, raising=False)

    rows = run_benchmarks(
        runtime_module=runtime_module,
        accelerator_module=accelerator_module,
        iterations=3,
        requested_backends=["", "cuda"],
    )

    assert [row["requested"] for row in rows] == ["cpu", "cuda"]
    assert all("registry_median_us" in row for row in rows)
    assert all("cpu_predict_median_us" in row for row in rows)
    assert rows[0]["selected"] == "cpu"
    assert rows[0]["fallback"] is True


def test_accelerator_validation_benchmark_uses_runtime_fixture() -> None:
    """Validation benchmark should load the shared replay fixture."""
    assert runtime_module.load_model_spec("tests/fixtures/model_spec_v1.json")
