"""Optional specialized accelerator backend adapters."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .accelerator import AcceleratorCapabilities

SPECIALIZED_DEFERRED_TARGETS = ("tpu", "fpga", "asic")


@dataclass(frozen=True, slots=True)
class SpecializedModuleBackend:
    """Backend adapter that is available only when a marker module exists."""

    name: str
    marker_module: str
    device_kind: str

    def capabilities(self) -> AcceleratorCapabilities:
        """Return the backend capability profile."""
        return AcceleratorCapabilities(
            backend_name=self.name,
            device_kind=self.device_kind,
            supports_prediction=True,
            supports_design_matrix=True,
        )

    def is_available(self) -> bool:
        """Return whether the backing module can be imported."""
        return importlib.util.find_spec(self.marker_module) is not None


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
