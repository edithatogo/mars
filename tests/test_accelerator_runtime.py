"""Tests for the shared accelerator backend contract."""

from __future__ import annotations

from dataclasses import dataclass

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
