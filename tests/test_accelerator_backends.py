"""Tests for optional accelerator-family backend adapters."""

from __future__ import annotations

import pymars.accelerator_backends as accelerator_backends
from pymars.accelerator_backends import (
    make_cuda_backend,
    make_metal_backend,
    make_rocm_backend,
)


def test_optional_backend_factories_expose_expected_names(monkeypatch) -> None:
    """Optional backends should register stable names and device kinds."""
    monkeypatch.setattr(
        accelerator_backends.importlib.util,
        "find_spec",
        lambda module_name: object() if module_name == "cupy" else None,
    )

    cuda = make_cuda_backend()
    rocm = make_rocm_backend()
    metal = make_metal_backend()

    assert cuda.name == "cuda"
    assert cuda.capabilities().device_kind == "cuda"
    assert cuda.is_available() is True
    assert rocm.name == "rocm"
    assert rocm.capabilities().device_kind == "rocm"
    assert rocm.is_available() is True
    assert metal.name == "metal"
    assert metal.capabilities().device_kind == "metal"
    assert metal.is_available() is False


def test_optional_backends_fall_back_when_module_missing(monkeypatch) -> None:
    """Optional backends should report unavailable when their marker module is absent."""
    monkeypatch.setattr(accelerator_backends.importlib.util, "find_spec", lambda _module_name: None)

    assert make_cuda_backend().is_available() is False
    assert make_rocm_backend().is_available() is False
    assert make_metal_backend().is_available() is False
