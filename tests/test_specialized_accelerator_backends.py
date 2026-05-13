"""Tests for optional specialized accelerator backend adapters."""

from __future__ import annotations

import pymars.specialized_accelerator_backends as specialized_backends
from pymars.specialized_accelerator_backends import (
    SPECIALIZED_DEFERRED_TARGETS,
    make_asic_backend,
    make_fpga_backend,
    make_tpu_backend,
)


def test_specialized_backend_factories_report_expected_targets(monkeypatch) -> None:
    """Specialized backends should expose stable names and device kinds."""
    monkeypatch.setattr(
        specialized_backends.importlib.util,
        "find_spec",
        lambda module_name: object() if module_name in {"jax", "amaranth"} else None,
    )

    tpu = make_tpu_backend()
    fpga = make_fpga_backend()
    asic = make_asic_backend()

    assert SPECIALIZED_DEFERRED_TARGETS == ("tpu", "fpga", "asic")
    assert tpu.name == "tpu"
    assert tpu.capabilities().device_kind == "tpu"
    assert tpu.is_available() is True
    assert fpga.name == "fpga"
    assert fpga.capabilities().device_kind == "fpga"
    assert fpga.is_available() is True
    assert asic.name == "asic"
    assert asic.capabilities().device_kind == "asic"
    assert asic.is_available() is False


def test_specialized_backends_fall_back_when_modules_missing(monkeypatch) -> None:
    """Specialized backends should report unavailable when marker modules are absent."""
    monkeypatch.setattr(specialized_backends.importlib.util, "find_spec", lambda _module_name: None)

    assert make_tpu_backend().is_available() is False
    assert make_fpga_backend().is_available() is False
    assert make_asic_backend().is_available() is False
