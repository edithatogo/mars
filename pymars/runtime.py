from __future__ import annotations

"""Spec-driven runtime helpers for portable pymars models."""

from pathlib import Path
from typing import Any, cast

import numpy as np

from ._model_spec import (
    model_to_spec,
    spec_from_json,
    spec_to_json,
    validate_model_spec,
)
from .earth import Earth


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
    model = load_model(spec_or_path)
    return cast(np.ndarray, model.predict(X))


def design_matrix(spec_or_path: dict[str, Any] | str | Path, X: Any) -> np.ndarray:
    """Build the basis matrix for a portable model spec."""
    model = load_model(spec_or_path)
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
