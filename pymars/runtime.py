from __future__ import annotations

"""Spec-driven runtime helpers for portable pymars models."""

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._basis import HingeBasisFunction
from ._model_spec import (
    model_to_spec,
    spec_from_json,
    spec_to_json,
    validate_model_spec,
)
from ._util import calculate_gcv, gcv_penalty_cost_effective_parameters
from .earth import Earth

logger = logging.getLogger(__name__)

_rust_backend: Any = None
try:
    import pymars_runtime as _native_module

    if _native_module._IS_COMPILED:
        _rust_backend = _native_module
except Exception as exc:  # pragma: no cover - optional compiled extension
    logger.debug("Rust runtime backend unavailable: %s", exc)

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

    if _rust_backend is not None:
        try:
            return spec_from_json(
                _rust_backend.load_model_spec_canonical_json(str(spec_or_path))
            )
        except Exception:
            pass

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
    try:
        if _validate_with_rust(spec):
            return spec
    except ValueError:
        pass
    validate_model_spec(spec)
    return spec


def save_model(model_or_spec: Earth | dict[str, Any], path: str | Path) -> Path:
    """Save a fitted model or normalized spec to a JSON file."""
    target = Path(path)
    target.write_text(export_model_json(model_or_spec))
    return target


def export_model_json(model_or_spec: Earth | dict[str, Any]) -> str:
    """Export a fitted model or normalized spec as canonical JSON."""
    spec = (
        model_to_spec(model_or_spec)
        if isinstance(model_or_spec, Earth)
        else model_or_spec
    )
    if _rust_backend is not None and _spec_is_rust_runtime_compatible(spec):
        return cast("str", _rust_backend.export_model_json(spec_to_json(spec)))
    return spec_to_json(spec)


def fit_model(
    model: Earth, X: Any, y: Any, sample_weight: Any | None = None
) -> Earth | None:
    """Fit an Earth model through the Rust training bridge when available."""
    if _rust_backend is None:
        return None
    if model.categorical_features or model.allow_missing:
        return None
    rows = _coerce_rows_for_rust(X)
    y_values = cast("list[float]", np.asarray(y, dtype=float).reshape(-1).tolist())
    weights = None
    if sample_weight is not None:
        weights = cast(
            "list[float]",
            np.asarray(sample_weight, dtype=float).reshape(-1).tolist(),
        )
    feature_names = getattr(model, "feature_names_in_", None)
    feature_names_list = None
    if feature_names is not None:
        feature_names_list = list(feature_names)
    payload: dict[str, Any] = {
        "x": rows,
        "y": y_values,
        "sample_weight": weights,
        "params": {
            "max_terms": model.max_terms or max(2, 2 * len(rows[0]) + 1),
            "max_degree": model.max_degree,
            "penalty": model.penalty,
            "minspan": model.minspan,
            "endspan": model.endspan,
            "threshold": 0.001,
            "allow_linear": model.allow_linear,
            "allow_missing": model.allow_missing,
            "categorical_features": list(model.categorical_features or []),
            "feature_names": feature_names_list,
        },
    }
    trained_spec_json = _rust_backend.fit_model_json(spec_to_json(payload))

    from ._model_spec import spec_to_model

    trained_model = spec_to_model(spec_from_json(trained_spec_json), Earth)
    trained_model.feature_importance_type = model.feature_importance_type
    if trained_model.rss_ is None or trained_model.mse_ is None or trained_model.gcv_ is None:
        y_array = np.asarray(y_values, dtype=float)
        X_array = np.asarray(rows, dtype=float)
        predictions = np.asarray(trained_model.predict(X_array), dtype=float)
        residuals = y_array - predictions
        if weights is None:
            rss = float(np.sum(residuals**2))
            mse = rss / len(y_array) if len(y_array) else np.inf
        else:
            sample_weight_array = np.asarray(weights, dtype=float)
            rss = float(np.sum(sample_weight_array * residuals**2))
            weight_sum = float(np.sum(sample_weight_array))
            mse = rss / weight_sum if weight_sum > 0.0 else np.inf

        num_terms = len(trained_model.basis_ or [])
        num_hinge_terms = sum(
            isinstance(bf, HingeBasisFunction) for bf in (trained_model.basis_ or [])
        )
        n_samples_for_gcv = len(y_array)
        eff_params = gcv_penalty_cost_effective_parameters(
            num_terms=num_terms,
            num_hinge_terms=num_hinge_terms,
            penalty=trained_model.penalty,
            num_samples=n_samples_for_gcv,
        )
        gcv = calculate_gcv(rss, n_samples_for_gcv, eff_params)
        trained_model.rss_ = rss
        trained_model.mse_ = mse
        trained_model.gcv_ = gcv
        if trained_model.record_ is not None:
            trained_model.record_.final_rss_ = rss
            trained_model.record_.final_mse_ = mse
            trained_model.record_.final_gcv_ = gcv
            trained_model.record_.n_samples = len(y_array)
    model.__dict__.update(trained_model.__dict__)
    return model


def predict(spec_or_path: dict[str, Any] | str | Path, X: Any) -> np.ndarray:
    """Predict using a portable model spec."""
    spec = load_model_spec(spec_or_path)
    rust_prediction = _predict_with_rust(spec, X)
    if rust_prediction is not None:
        return rust_prediction
    model = load_model(spec)
    return cast("np.ndarray", model.predict(X))


def design_matrix(spec_or_path: dict[str, Any] | str | Path, X: Any) -> np.ndarray:
    """Build the basis matrix for a portable model spec."""
    spec = load_model_spec(spec_or_path)
    rust_matrix = _design_matrix_with_rust(spec, X)
    if rust_matrix is not None:
        return rust_matrix
    model = load_model(spec)
    X_processed, missing_mask = model._prepare_prediction_data(X)
    basis = model.basis_
    if basis is None:
        raise ValueError("Portable model is missing basis functions.")
    return model._build_basis_matrix(X_processed, basis, missing_mask)


def inspect(spec_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Return a normalized view of a portable model spec."""
    spec = load_model_spec(spec_or_path)
    rust_summary = _inspect_with_rust(spec)
    if rust_summary is not None:
        return rust_summary
    return {
        "spec_version": spec.get("spec_version"),
        "model_type": spec.get("model_type"),
        "n_features": spec.get("feature_schema", {}).get("n_features"),
        "n_basis_terms": len(spec.get("basis_terms", [])),
        "metrics": spec.get("metrics", {}),
    }


def _validate_with_rust(spec: dict[str, Any]) -> bool:
    """Validate a portable spec with the Rust backend if possible."""
    if _rust_backend is None:
        return False
    if not _spec_is_rust_runtime_compatible(spec):
        return False
    _rust_backend.validate_model_spec_json(spec_to_json(spec))
    return True


def _inspect_with_rust(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Inspect a portable spec with the Rust backend if possible."""
    if _rust_backend is None:
        return None
    if not _spec_is_rust_runtime_compatible(spec):
        return None
    return cast(
        "dict[str, Any]",
        json.loads(_rust_backend.inspect_model_spec_json(spec_to_json(spec))),
    )


def _predict_with_rust(spec: dict[str, Any], X: Any) -> np.ndarray | None:
    """Predict through Rust when the runtime and inputs are compatible."""
    if not _spec_is_rust_runtime_compatible(spec):
        return None
    if _rust_backend is None:
        return None
    try:
        rows = _coerce_rows_for_rust(X)
    except (TypeError, ValueError):
        return None
    return cast(
        "np.ndarray",
        np.asarray(
            _rust_backend.predict_json(spec_to_json(spec), rows),
            dtype=float,
        ),
    )


def _design_matrix_with_rust(spec: dict[str, Any], X: Any) -> np.ndarray | None:
    """Build a basis matrix through Rust when the runtime and inputs are compatible."""
    if not _spec_is_rust_runtime_compatible(spec):
        return None
    if _rust_backend is None:
        return None
    try:
        rows = _coerce_rows_for_rust(X)
    except (TypeError, ValueError):
        return None
    return cast(
        "np.ndarray",
        np.asarray(
            _rust_backend.design_matrix_json(spec_to_json(spec), rows),
            dtype=float,
        ),
    )


def _coerce_rows_for_rust(X: Any) -> list[list[float]]:
    """Coerce runtime input to a Rust-friendly row-major float matrix."""
    rows = np.asarray(X, dtype=float)
    if rows.ndim != 2:
        raise ValueError("X must be a 2D array-like input for runtime evaluation.")
    return cast("list[list[float]]", rows.tolist())


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
