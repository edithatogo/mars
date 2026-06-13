"""Tests for the shared accelerator backend contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymars import (
    AcceleratorCapabilities,
    accelerator_backend_summary,
    available_accelerator_backends,
    clear_accelerator_backends,
    detect_requested_accelerator_backend,
    register_accelerator_backend,
    select_accelerator_backend,
)
from pymars import accelerator as accelerator_module
from pymars.accelerator_backends import ArrayModuleAcceleratorBackend
from pymars.runtime import design_matrix, predict


@dataclass
class DummyBackend:
    """Simple backend double for registry and selection tests."""

    name: str
    available: bool = True

    def capabilities(self) -> AcceleratorCapabilities:
        """Return the dummy backend capability profile."""
        return AcceleratorCapabilities(
            backend_name=self.name,
            device_kind="test-device",
            supports_prediction=True,
            supports_design_matrix=True,
        )

    def is_available(self) -> bool:
        """Return whether the dummy backend is available."""
        return self.available


def teardown_module() -> None:
    """Reset shared registry state between test modules."""
    clear_accelerator_backends()


def test_accelerator_registry_selects_available_backend(monkeypatch) -> None:
    """Requested accelerator backends should be selected when available."""
    clear_accelerator_backends()
    register_accelerator_backend(DummyBackend("metal"))
    monkeypatch.setenv(accelerator_module.ACCELERATOR_ENV_VAR, "metal")

    backend = select_accelerator_backend()

    assert backend is not None
    assert backend.name == "metal"
    assert available_accelerator_backends() == ["metal"]
    assert detect_requested_accelerator_backend() == "metal"


def test_accelerator_registry_falls_back_to_cpu_when_unavailable(
    monkeypatch,
) -> None:
    """Unavailable backends should resolve to the CPU fallback path."""
    clear_accelerator_backends()
    register_accelerator_backend(DummyBackend("cuda", available=False))
    monkeypatch.setenv(accelerator_module.ACCELERATOR_ENV_VAR, "cuda")

    backend = select_accelerator_backend()
    summary = accelerator_backend_summary()

    assert backend is None
    assert summary["selected"] == "cpu"
    assert summary["fallback"] is True
    assert summary["available"] == []


def test_accelerator_registry_reports_capabilities(monkeypatch) -> None:
    """Registered backends should surface their capability metadata."""
    clear_accelerator_backends()
    register_accelerator_backend(DummyBackend("metal"))
    monkeypatch.setenv(accelerator_module.ACCELERATOR_ENV_VAR, "metal")

    summary = accelerator_backend_summary()

    assert summary["selected"] == "metal"
    assert summary["fallback"] is False
    assert summary["capabilities"]["backend_name"] == "metal"
    assert summary["capabilities"]["device_kind"] == "test-device"


def test_accelerated_prediction_matches_cpu_replay(monkeypatch) -> None:
    """Executable accelerator backends should match CPU portable replay."""
    clear_accelerator_backends()
    monkeypatch.setenv(accelerator_module.ACCELERATOR_ENV_VAR, "array-test")
    backend = ArrayModuleAcceleratorBackend(
        name="array-test",
        marker_module="numpy",
        device_kind="test-array",
    )
    register_accelerator_backend(backend)
    spec_path = "tests/fixtures/model_spec_v1.json"
    rows = np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]], dtype=float)

    accelerated = accelerator_module.predict_accelerated(spec_path, rows)
    cpu = predict(spec_path, rows)

    assert np.allclose(accelerated, cpu)
    assert accelerator_module.accelerator_backend_summary()["fallback"] is False


def test_accelerated_design_matrix_matches_cpu_replay(monkeypatch) -> None:
    """Accelerator design-matrix replay should match CPU fixture output."""
    clear_accelerator_backends()
    monkeypatch.setenv(accelerator_module.ACCELERATOR_ENV_VAR, "array-test")
    register_accelerator_backend(
        ArrayModuleAcceleratorBackend(
            name="array-test",
            marker_module="numpy",
            device_kind="test-array",
        )
    )
    spec_path = "tests/fixtures/model_spec_v1.json"
    rows = np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]], dtype=float)

    accelerated = accelerator_module.design_matrix_accelerated(spec_path, rows)
    cpu = design_matrix(spec_path, rows)

    assert np.allclose(accelerated, cpu)


def test_accelerated_replay_falls_back_to_cpu_when_backend_missing(monkeypatch) -> None:
    """Requested but unavailable accelerator paths should preserve CPU behavior."""
    clear_accelerator_backends()
    monkeypatch.setenv(accelerator_module.ACCELERATOR_ENV_VAR, "missing")
    spec_path = "tests/fixtures/model_spec_v1.json"
    rows = np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]], dtype=float)

    accelerated = accelerator_module.predict_accelerated(spec_path, rows)
    cpu = predict(spec_path, rows)

    assert np.allclose(accelerated, cpu)
    assert accelerator_module.accelerator_backend_summary()["fallback"] is True
