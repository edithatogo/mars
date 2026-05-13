"""Optional accelerator-family backend adapters."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .accelerator import AcceleratorCapabilities


@dataclass(frozen=True, slots=True)
class OptionalModuleBackend:
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


def make_cuda_backend() -> OptionalModuleBackend:
    """Create a CUDA-family backend adapter."""
    return OptionalModuleBackend(
        name="cuda",
        marker_module="cupy",
        device_kind="cuda",
    )


def make_rocm_backend() -> OptionalModuleBackend:
    """Create a ROCm-family backend adapter."""
    return OptionalModuleBackend(
        name="rocm",
        marker_module="cupy",
        device_kind="rocm",
    )


def make_metal_backend() -> OptionalModuleBackend:
    """Create a Metal-family backend adapter."""
    return OptionalModuleBackend(
        name="metal",
        marker_module="metal",
        device_kind="metal",
    )
