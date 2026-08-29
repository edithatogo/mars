"""Optional specialized accelerator backend adapters."""

from __future__ import annotations

from dataclasses import dataclass

from .accelerator import BaseModuleBackend

SPECIALIZED_DEFERRED_TARGETS = ("tpu", "fpga", "asic")


@dataclass(frozen=True, slots=True)
class SpecializedModuleBackend(BaseModuleBackend):
    """Backend adapter that is available only when a marker module exists."""


def make_tpu_backend() -> SpecializedModuleBackend:
    """Create a TPU-family backend adapter."""
    return SpecializedModuleBackend(
        name="tpu",
        marker_module="jax",
        device_kind="tpu",
    )


def make_fpga_backend() -> SpecializedModuleBackend:
    """Create an FPGA-family backend adapter."""
    return SpecializedModuleBackend(
        name="fpga",
        marker_module="amaranth",
        device_kind="fpga",
    )


def make_asic_backend() -> SpecializedModuleBackend:
    """Create an ASIC-family backend adapter."""
    return SpecializedModuleBackend(
        name="asic",
        marker_module="torch",
        device_kind="asic",
    )
