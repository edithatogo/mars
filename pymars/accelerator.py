"""Shared accelerator backend selection and fallback utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

ACCELERATOR_ENV_VAR = "MARS_EARTH_ACCELERATOR_BACKEND"
CPU_BACKEND_NAME = "cpu"


@dataclass(frozen=True, slots=True)
class AcceleratorCapabilities:
    """Describe the accelerator features a backend exposes."""

    backend_name: str
    device_kind: str
    supports_prediction: bool = True
    supports_design_matrix: bool = True
    supports_training: bool = False
    supports_distributed: bool = False
    supports_batch_replay: bool = True


@runtime_checkable
class AcceleratorBackend(Protocol):
    """Protocol for optional accelerator replay backends."""

    name: str

    def capabilities(self) -> AcceleratorCapabilities:
        """Return backend capability metadata."""

    def is_available(self) -> bool:
        """Return whether the backend is usable in the current process."""


_ACCELERATOR_BACKENDS: dict[str, AcceleratorBackend] = {}


def register_accelerator_backend(backend: AcceleratorBackend) -> None:
    """Register an accelerator backend for later selection."""
    _ACCELERATOR_BACKENDS[backend.name] = backend


def clear_accelerator_backends() -> None:
    """Clear all registered accelerator backends.

    This is primarily intended for tests.
    """
    _ACCELERATOR_BACKENDS.clear()


def available_accelerator_backends() -> list[str]:
    """Return the names of registered, available accelerator backends."""
    return [
        name
        for name, backend in sorted(_ACCELERATOR_BACKENDS.items())
        if backend.is_available()
    ]


def get_registered_accelerator_backend(
    name: str,
) -> AcceleratorBackend | None:
    """Return a registered accelerator backend by name."""
    backend = _ACCELERATOR_BACKENDS.get(name)
    if backend is None or not backend.is_available():
        return None
    return backend


def detect_requested_accelerator_backend() -> str | None:
    """Return the backend requested via environment configuration."""
    value = os.environ.get(ACCELERATOR_ENV_VAR)
    if value is None:
        return None
    backend_name = value.strip().lower()
    return backend_name or None


def select_accelerator_backend(
    preferred: str | None = None,
) -> AcceleratorBackend | None:
    """Select an accelerator backend or fall back to CPU."""
    requested = (preferred or detect_requested_accelerator_backend() or "").strip()
    if requested:
        backend = get_registered_accelerator_backend(requested.lower())
        if backend is not None:
            return backend
    if requested and requested.lower() != CPU_BACKEND_NAME:
        return None
    return None


def accelerator_backend_summary() -> dict[str, Any]:
    """Return a normalized view of accelerator availability for docs/tests."""
    requested = detect_requested_accelerator_backend()
    selected = select_accelerator_backend(requested)
    if selected is None:
        return {
            "requested": requested or CPU_BACKEND_NAME,
            "selected": CPU_BACKEND_NAME,
            "available": available_accelerator_backends(),
            "fallback": True,
        }
    capabilities = selected.capabilities()
    return {
        "requested": requested or selected.name,
        "selected": selected.name,
        "available": available_accelerator_backends(),
        "fallback": False,
        "capabilities": {
            "backend_name": capabilities.backend_name,
            "device_kind": capabilities.device_kind,
            "supports_prediction": capabilities.supports_prediction,
            "supports_design_matrix": capabilities.supports_design_matrix,
            "supports_training": capabilities.supports_training,
            "supports_distributed": capabilities.supports_distributed,
            "supports_batch_replay": capabilities.supports_batch_replay,
        },
    }


def predict_accelerated(
    spec_or_path: dict[str, Any] | str,
    X: Any,
    *,
    preferred: str | None = None,
) -> np.ndarray:
    """Predict through an optional H3 backend with CPU fallback.

    Accelerator dependencies remain optional. If no requested backend is
    available, or if the selected backend does not expose executable replay, the
    portable CPU runtime is used.
    """
    from . import runtime

    backend = select_accelerator_backend(preferred)
    predict_fn = getattr(backend, "predict", None) if backend is not None else None
    if callable(predict_fn):
        return np.asarray(predict_fn(spec_or_path, X), dtype=float)
    return runtime.predict(spec_or_path, X)


def design_matrix_accelerated(
    spec_or_path: dict[str, Any] | str,
    X: Any,
    *,
    preferred: str | None = None,
) -> np.ndarray:
    """Build a design matrix through an optional H3 backend with CPU fallback."""
    from . import runtime

    backend = select_accelerator_backend(preferred)
    design_matrix_fn = (
        getattr(backend, "design_matrix", None) if backend is not None else None
    )
    if callable(design_matrix_fn):
        return np.asarray(design_matrix_fn(spec_or_path, X), dtype=float)
    return runtime.design_matrix(spec_or_path, X)
