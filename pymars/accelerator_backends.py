"""Optional accelerator-family backend adapters."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

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


def make_cuda_backend() -> ArrayModuleAcceleratorBackend:
    """Create a CUDA-family backend adapter."""
    return ArrayModuleAcceleratorBackend(
        name="cuda",
        marker_module="cupy",
        device_kind="cuda",
    )


def make_rocm_backend() -> ArrayModuleAcceleratorBackend:
    """Create a ROCm-family backend adapter."""
    return ArrayModuleAcceleratorBackend(
        name="rocm",
        marker_module="cupy",
        device_kind="rocm",
    )


def make_metal_backend() -> ArrayModuleAcceleratorBackend:
    """Create a Metal-family backend adapter."""
    return ArrayModuleAcceleratorBackend(
        name="metal",
        marker_module="metal",
        device_kind="metal",
    )


@dataclass(frozen=True, slots=True)
class ArrayModuleAcceleratorBackend(OptionalModuleBackend):
    """Optional array-namespace backend for H3 replay workloads.

    The backend imports its array module lazily when replay is requested. That
    keeps package import side-effect free and allows CUDA/ROCm/Metal-family
    packages to remain optional.
    """

    def _array_module(self) -> Any:
        """Import and return the backend array module lazily."""
        return importlib.import_module(self.marker_module)

    def design_matrix(self, spec_or_path: dict[str, Any] | str, X: Any) -> np.ndarray:
        """Build a portable ModelSpec design matrix through the array module."""
        from . import runtime

        spec = runtime.load_model_spec(spec_or_path)
        xp = self._array_module()
        rows = xp.asarray(X, dtype=float)
        if rows.ndim != 2:
            raise ValueError("X must be a 2D array-like input for accelerator replay.")

        terms = spec.get("basis_terms", [])
        if not isinstance(terms, list):
            raise TypeError("Model spec field 'basis_terms' must be an array.")
        columns = [_evaluate_basis_term(term, rows, xp) for term in terms]
        if not columns:
            return np.empty((int(rows.shape[0]), 0), dtype=float)
        matrix = xp.stack(columns, axis=1)
        return cast("np.ndarray", np.asarray(_to_host_array(matrix), dtype=float))

    def predict(self, spec_or_path: dict[str, Any] | str, X: Any) -> np.ndarray:
        """Predict from a portable ModelSpec through the array module."""
        from . import runtime

        spec = runtime.load_model_spec(spec_or_path)
        xp = self._array_module()
        matrix = xp.asarray(self.design_matrix(spec, X), dtype=float)
        coefficients = xp.asarray(spec.get("coefficients", []), dtype=float)
        if matrix.shape[1] != coefficients.shape[0]:
            raise ValueError("Model spec must contain one coefficient per basis term.")
        predictions = matrix @ coefficients
        return cast("np.ndarray", np.asarray(_to_host_array(predictions), dtype=float))


def _to_host_array(value: Any) -> Any:
    """Return a NumPy-compatible host value from an optional device array."""
    if hasattr(value, "get"):
        return value.get()
    return value


def _evaluate_basis_term(term: dict[str, Any], rows: Any, xp: Any) -> Any:
    """Evaluate a supported portable basis term with an array namespace."""
    kind = term.get("kind")
    if kind == "constant":
        return xp.ones(rows.shape[0], dtype=float)
    if kind == "linear":
        parent = _evaluate_parent(term, rows, xp)
        return parent * rows[:, int(term["variable_idx"])]
    if kind == "hinge":
        parent = _evaluate_parent(term, rows, xp)
        values = rows[:, int(term["variable_idx"])]
        knot = float(term["knot_val"])
        if bool(term.get("is_right_hinge")):
            hinge_values = xp.maximum(0.0, values - knot)
        else:
            hinge_values = xp.maximum(0.0, knot - values)
        return parent * hinge_values
    if kind == "interaction":
        left = _evaluate_basis_term(term["parent1"], rows, xp)
        right = _evaluate_basis_term(term["parent2"], rows, xp)
        return left * right
    if kind == "missingness":
        return xp.isnan(rows[:, int(term["variable_idx"])]).astype(float)
    if kind == "categorical":
        return (rows[:, int(term["variable_idx"])] == term.get("category")).astype(
            float
        )
    raise ValueError(f"Unsupported accelerator basis term kind: {kind!r}.")


def _evaluate_parent(term: dict[str, Any], rows: Any, xp: Any) -> Any:
    """Evaluate a parent basis term or return the implicit constant parent."""
    parent = term.get("parent1")
    if parent is None:
        return xp.ones(rows.shape[0], dtype=float)
    return _evaluate_basis_term(parent, rows, xp)
