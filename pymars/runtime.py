from __future__ import annotations

"""Spec-driven runtime helpers for portable pymars models."""

from pathlib import Path
from typing import Any, cast

import numpy as np

from ._model_spec import (
    model_to_spec,
    spec_to_json,
    spec_from_json,
    validate_model_spec,
)
from .earth import Earth

try:
    import pymars_runtime as _rust_backend
    if not getattr(_rust_backend, "_IS_COMPILED", False):
        _rust_backend = None
except Exception:  # pragma: no cover - optional compiled extension
    _rust_backend = None

_RUST_RUNTIME_SUPPORTED_KINDS = {
    "constant",
    "categorical",
    "hinge",
    "interaction",
    "linear",
    "missingness",
}


def load_model_spec(spec_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load a model spec from a dict, JSON string, or JSON file path."""
    if isinstance(spec_or_path, dict):
        return spec_or_path

    if isinstance(spec_or_path, Path):
        return spec_from_json(spec_or_path.read_text())

    text = spec_or_path.strip()
    if text.startswith("{"):
        return spec_from_json(text)

    return spec_from_json(Path(spec_or_path).read_text())


def load_model(spec_or_path: dict[str, Any] | str | Path) -> Earth:
    """Load a portable model as an ``Earth`` instance."""
    return Earth.from_model(load_model_spec(spec_or_path))


def validate(spec_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Validate and return a portable model spec."""
    spec = load_model_spec(spec_or_path)
    if _rust_backend is not None and _spec_is_rust_runtime_compatible(spec):
        try:
            _rust_backend.validate_model_spec_json(spec_to_json(spec))
        except Exception:
            pass
    validate_model_spec(spec)
    return spec


def save_model(model_or_spec: Earth | dict[str, Any], path: str | Path) -> Path:
    """Save a fitted model or normalized spec to a JSON file."""
    target = Path(path)
    spec = (
        model_to_spec(model_or_spec)
        if isinstance(model_or_spec, Earth)
        else model_or_spec
    )
    target.write_text(spec_to_json(spec))
    return target


def predict(spec_or_path: dict[str, Any] | str | Path, X: Any) -> np.ndarray:
    """Predict using a portable model spec."""
    spec = load_model_spec(spec_or_path)
    if _rust_backend is not None and _spec_is_rust_runtime_compatible(spec):
        try:
            rows = _coerce_rows_for_rust(X)
        except (TypeError, ValueError):
            rows = None
        if rows is not None:
            try:
                return cast(
                    np.ndarray,
                    np.asarray(
                        _rust_backend.predict_json(spec_to_json(spec), rows),
                        dtype=float,
                    ),
                )
            except Exception:
                pass
    model = load_model(spec)
    return cast(np.ndarray, model.predict(X))


def design_matrix(spec_or_path: dict[str, Any] | str | Path, X: Any) -> np.ndarray:
    """Build the basis matrix for a portable model spec."""
    spec = load_model_spec(spec_or_path)
    if _rust_backend is not None and _spec_is_rust_runtime_compatible(spec):
        try:
            rows = _coerce_rows_for_rust(X)
        except (TypeError, ValueError):
            rows = None
        if rows is not None:
            try:
                return cast(
                    np.ndarray,
                    np.asarray(
                        _rust_backend.design_matrix_json(spec_to_json(spec), rows),
                        dtype=float,
                    ),
                )
            except Exception:
                pass
    model = load_model(spec)
    X_processed, missing_mask = model._prepare_prediction_data(X)
    basis = model.basis_
    if basis is None:
        raise ValueError("Portable model is missing basis functions.")
    return model._build_basis_matrix(X_processed, basis, missing_mask)


def inspect(spec_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Return a normalized view of a portable model spec."""
    spec = load_model_spec(spec_or_path)
    return {
        "spec_version": spec.get("spec_version"),
        "model_type": spec.get("model_type"),
        "n_features": spec.get("feature_schema", {}).get("n_features"),
        "n_basis_terms": len(spec.get("basis_terms", [])),
        "metrics": spec.get("metrics", {}),
    }


def _spec_is_rust_runtime_compatible(spec: dict[str, Any]) -> bool:
    """Return whether the Rust runtime can evaluate the spec without fallback."""
    if spec.get("categorical_imputer") is not None:
        return False

    basis_terms = spec.get("basis_terms", [])
    if not isinstance(basis_terms, list):
        return False

    for term in basis_terms:
        if not isinstance(term, dict):
            return False
        kind = term.get("kind")
        if kind not in _RUST_RUNTIME_SUPPORTED_KINDS:
            return False
        if kind == "categorical":
            category = term.get("category")
            if not isinstance(category, (int, float, bool)):
                return False

    return True


def _coerce_rows_for_rust(X: Any) -> list[list[float]]:
    """Coerce runtime input to a Rust-friendly row-major float matrix."""
    rows = np.asarray(X, dtype=float)
    if rows.ndim != 2:
        raise ValueError("X must be a 2D array-like input for runtime evaluation.")
    return cast(list[list[float]], rows.tolist())
