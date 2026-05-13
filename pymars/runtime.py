"""Spec-driven runtime helpers for portable pymars models."""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, cast

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
if sys.platform != "darwin":
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
_RUNTIME_THREAD_ENV_VAR = "MARS_EARTH_RUNTIME_THREADS"


def _normalize_workers(workers: int | None) -> int:
    if workers is None:
        return 1
    if workers < 1:
        raise ValueError("workers must be >= 1")
    return workers


def _normalize_chunk_size(
    chunk_size: int | None, rows: list[list[float]], workers: int
) -> int:
    if chunk_size is None:
        return max(1, len(rows) // workers) if len(rows) >= workers else 1
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return chunk_size


def _chunked_row_indices(n_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    if n_rows <= 0:
        return []
    chunks: list[tuple[int, int]] = []
    for start in range(0, n_rows, chunk_size):
        end = min(n_rows, start + chunk_size)
        chunks.append((start, end))
    return chunks


def _distributed_map(
    rows: list[list[float]],
    workers: int,
    chunk_size: int,
    worker_fn: Callable[[list[list[float]]], list[Any]],
    preserve_order: bool,
) -> list[Any]:
    indices = _chunked_row_indices(len(rows), chunk_size)
    if not indices:
        return []

    if workers <= 1 or len(indices) == 1:
        ordered_results: list[Any] = []
        for start, end in indices:
            ordered_results.extend(worker_fn(rows[start:end]))
        return ordered_results[: len(rows)]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(worker_fn, rows[start:end]): (start, end)
            for start, end in indices
        }
        if preserve_order:
            ordered_pairs = sorted(
                (
                    (start, end, future.result())
                    for future, (start, end) in futures.items()
                ),
                key=lambda item: item[0],
            )
            ordered_results: list[Any] = []
            for _, _, chunk_values in ordered_pairs:
                ordered_results.extend(chunk_values)
            return ordered_results

        flat: list[Any] = []
        for _, _, chunk_values in [
            (start, end, future.result()) for future, (start, end) in futures.items()
        ]:
            flat.extend(chunk_values)
        return flat


def _process_cluster_map(
    rows: list[list[float]],
    workers: int,
    chunk_size: int,
    worker_fn: Callable[[list[list[float]]], list[Any]],
    preserve_order: bool,
) -> list[Any]:
    indices = _chunked_row_indices(len(rows), chunk_size)
    if not indices:
        return []

    if workers <= 1 or len(indices) == 1:
        ordered_results: list[Any] = []
        for start, end in indices:
            ordered_results.extend(worker_fn(rows[start:end]))
        return ordered_results[: len(rows)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(worker_fn, rows[start:end]): (start, end)
            for start, end in indices
        }
        if preserve_order:
            ordered_pairs = sorted(
                (
                    (start, end, future.result())
                    for future, (start, end) in futures.items()
                ),
                key=lambda item: item[0],
            )
            ordered_results: list[Any] = []
            for _, _, chunk_values in ordered_pairs:
                ordered_results.extend(chunk_values)
            return ordered_results

        flat: list[Any] = []
        for _, _, chunk_values in [
            (start, end, future.result()) for future, (start, end) in futures.items()
        ]:
            flat.extend(chunk_values)
        return flat


def _predict_cpu_cluster_chunk(
    spec_json: str, batch: list[list[float]]
) -> list[float]:
    """Predict a batch in a process worker using the portable Python model."""
    model = load_model(spec_from_json(spec_json))
    return cast("list[float]", model.predict(np.asarray(batch)).tolist())


def _design_matrix_cpu_cluster_chunk(
    spec_json: str, batch: list[list[float]]
) -> list[list[float]]:
    """Build a design matrix batch in a process worker using the portable model."""
    model = load_model(spec_from_json(spec_json))
    X_processed, missing_mask = model._prepare_prediction_data(
        np.asarray(batch, dtype=float)
    )
    basis = model.basis_
    if basis is None:
        raise ValueError("Portable model is missing basis functions.")
    return cast(
        "list[list[float]]",
        model._build_basis_matrix(X_processed, basis, missing_mask).tolist(),
    )


def set_runtime_threads(thread_count: int | None) -> None:
    """Set runtime thread hint for Rust replay.

    Args:
        thread_count: If ``None``, remove the override and use runtime defaults.
            If ``1``, force deterministic single-thread execution.
    """
    if thread_count is None:
        os.environ.pop(_RUNTIME_THREAD_ENV_VAR, None)
        return
    if thread_count < 1:
        raise ValueError("thread_count must be >= 1")
    os.environ[_RUNTIME_THREAD_ENV_VAR] = str(thread_count)


@contextmanager
def runtime_threads(thread_count: int | None):
    """Temporarily configure runtime thread hint and restore afterwards.

    Args:
        thread_count: Optional thread count. ``1`` enforces deterministic
            single-thread replay. ``None`` removes the override.
    """
    previous = os.environ.get(_RUNTIME_THREAD_ENV_VAR)
    set_runtime_threads(thread_count)
    try:
        yield thread_count
    finally:
        if previous is None:
            os.environ.pop(_RUNTIME_THREAD_ENV_VAR, None)
        else:
            os.environ[_RUNTIME_THREAD_ENV_VAR] = previous


def _should_use_rust_backend() -> bool:
    """Return whether runtime helpers should route through the active backend."""
    return _rust_backend is not None and getattr(_rust_backend, "_IS_COMPILED", False)


def _rust_backend_supports(method_name: str) -> bool:
    """Return whether the active backend provides a callable method."""
    method = getattr(_rust_backend, method_name, None)
    return callable(method)


def load_model_spec(spec_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load a model spec from a dict, JSON string, or JSON file path."""
    if isinstance(spec_or_path, dict):
        return spec_or_path

    if _should_use_rust_backend() and _rust_backend_supports(
        "load_model_spec_canonical_json"
    ):
        try:
            return spec_from_json(
                _rust_backend.load_model_spec_canonical_json(str(spec_or_path))
            )
        except Exception as exc:  # pragma: no cover - backend fallback path
            logger.debug("Rust spec loader failed, falling back to Python: %s", exc)

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
    if (
        _should_use_rust_backend()
        and _rust_backend_supports("export_model_json")
        and _spec_is_rust_runtime_compatible(spec)
    ):
        return cast("str", _rust_backend.export_model_json(spec_to_json(spec)))
    return spec_to_json(spec)


def fit_model(
    model: Earth, X: Any, y: Any, sample_weight: Any | None = None
) -> Earth | None:
    """Fit an Earth model through the Rust training bridge when available."""
    if not (_should_use_rust_backend() and _rust_backend_supports("fit_model_json")):
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
    if (
        trained_model.rss_ is None
        or trained_model.mse_ is None
        or trained_model.gcv_ is None
    ):
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


def predict_distributed(
    spec_or_path: dict[str, Any] | str | Path,
    X: Any,
    *,
    workers: int | None = 1,
    chunk_size: int | None = None,
    preserve_order: bool = True,
) -> np.ndarray:
    """Replay a spec with local row chunking for preview distributed execution.

    This helper is a replay-only, opt-in H4 preview surface. It does not require
    cluster provisioning and must not initialize external services.
    """
    spec = load_model_spec(spec_or_path)
    rows = _coerce_rows_for_rust(X)
    resolved_workers = _normalize_workers(workers)
    resolved_chunk_size = _normalize_chunk_size(chunk_size, rows, resolved_workers)
    use_rust = _predict_with_rust(spec, rows[:0]) is not None
    if use_rust and not _spec_is_rust_runtime_compatible(spec):
        use_rust = False

    worker_model = None if use_rust else load_model(spec)

    def predict_chunk(batch: list[list[float]]) -> list[float]:
        if use_rust:
            return cast("list[float]", _predict_with_rust(spec, batch).tolist())
        return cast("list[float]", worker_model.predict(np.asarray(batch)).tolist())

    return np.asarray(
        _distributed_map(
            rows=rows,
            workers=resolved_workers,
            chunk_size=resolved_chunk_size,
            worker_fn=predict_chunk,
            preserve_order=preserve_order,
        ),
        dtype=float,
    )


def design_matrix_distributed(
    spec_or_path: dict[str, Any] | str | Path,
    X: Any,
    *,
    workers: int | None = 1,
    chunk_size: int | None = None,
    preserve_order: bool = True,
) -> np.ndarray:
    """Build model matrices with local chunked replay execution.

    This helper is a replay-only, opt-in H4 preview surface. It does not require
    cluster provisioning and must not initialize external services.
    """
    spec = load_model_spec(spec_or_path)
    rows = _coerce_rows_for_rust(X)
    resolved_workers = _normalize_workers(workers)
    resolved_chunk_size = _normalize_chunk_size(chunk_size, rows, resolved_workers)
    use_rust = _design_matrix_with_rust(spec, rows[:0]) is not None
    if use_rust and not _spec_is_rust_runtime_compatible(spec):
        use_rust = False

    worker_model = None if use_rust else load_model(spec)

    def design_chunk(batch: list[list[float]]) -> list[list[float]]:
        if use_rust:
            matrix = _design_matrix_with_rust(spec, batch)
            return cast("list[list[float]]", matrix.tolist())
        X_processed, missing_mask = worker_model._prepare_prediction_data(
            np.asarray(batch, dtype=float)
        )
        basis = worker_model.basis_
        if basis is None:
            raise ValueError("Portable model is missing basis functions.")
        return cast(
            "list[list[float]]",
            worker_model._build_basis_matrix(X_processed, basis, missing_mask).tolist(),
        )

    return np.asarray(
        _distributed_map(
            rows=rows,
            workers=resolved_workers,
            chunk_size=resolved_chunk_size,
            worker_fn=design_chunk,
            preserve_order=preserve_order,
        ),
        dtype=float,
    )


def predict_cpu_cluster(
    spec_or_path: dict[str, Any] | str | Path,
    X: Any,
    *,
    workers: int | None = 1,
    chunk_size: int | None = None,
    preserve_order: bool = True,
) -> np.ndarray:
    """Replay a spec through process-based CPU cluster parallelism.

    This is an opt-in H4 CPU-cluster preview surface. It uses process workers
    for partitioned CPU replay and falls back to serial execution when workers
    are not greater than one.
    """
    spec = load_model_spec(spec_or_path)
    rows = _coerce_rows_for_rust(X)
    resolved_workers = _normalize_workers(workers)
    resolved_chunk_size = _normalize_chunk_size(chunk_size, rows, resolved_workers)
    spec_json = spec_to_json(spec)

    indices = _chunked_row_indices(len(rows), resolved_chunk_size)
    if not indices:
        return np.asarray([], dtype=float)
    if resolved_workers <= 1 or len(indices) == 1:
        ordered_results: list[float] = []
        for start, end in indices:
            ordered_results.extend(_predict_cpu_cluster_chunk(spec_json, rows[start:end]))
        return np.asarray(ordered_results[: len(rows)], dtype=float)

    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        futures = {
            executor.submit(_predict_cpu_cluster_chunk, spec_json, rows[start:end]): (
                start,
                end,
            )
            for start, end in indices
        }
        if preserve_order:
            ordered_pairs = sorted(
                (
                    (start, end, future.result())
                    for future, (start, end) in futures.items()
                ),
                key=lambda item: item[0],
            )
        else:
            ordered_pairs = [
                (start, end, future.result()) for future, (start, end) in futures.items()
            ]
    ordered_results: list[float] = []
    for _, _, chunk_values in ordered_pairs:
        ordered_results.extend(chunk_values)
    return np.asarray(ordered_results, dtype=float)


def design_matrix_cpu_cluster(
    spec_or_path: dict[str, Any] | str | Path,
    X: Any,
    *,
    workers: int | None = 1,
    chunk_size: int | None = None,
    preserve_order: bool = True,
) -> np.ndarray:
    """Build model matrices through process-based CPU cluster parallelism."""
    spec = load_model_spec(spec_or_path)
    rows = _coerce_rows_for_rust(X)
    resolved_workers = _normalize_workers(workers)
    resolved_chunk_size = _normalize_chunk_size(chunk_size, rows, resolved_workers)
    spec_json = spec_to_json(spec)

    indices = _chunked_row_indices(len(rows), resolved_chunk_size)
    if not indices:
        return np.asarray([], dtype=float)
    if resolved_workers <= 1 or len(indices) == 1:
        ordered_results: list[list[float]] = []
        for start, end in indices:
            ordered_results.extend(
                _design_matrix_cpu_cluster_chunk(spec_json, rows[start:end])
            )
        return np.asarray(ordered_results, dtype=float)

    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        futures = {
            executor.submit(
                _design_matrix_cpu_cluster_chunk, spec_json, rows[start:end]
            ): (start, end)
            for start, end in indices
        }
        if preserve_order:
            ordered_pairs = sorted(
                (
                    (start, end, future.result())
                    for future, (start, end) in futures.items()
                ),
                key=lambda item: item[0],
            )
        else:
            ordered_pairs = [
                (start, end, future.result()) for future, (start, end) in futures.items()
            ]
    ordered_results: list[list[float]] = []
    for _, _, chunk_values in ordered_pairs:
        ordered_results.extend(chunk_values)
    return np.asarray(ordered_results, dtype=float)


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
    if not _should_use_rust_backend():
        return False
    if not _rust_backend_supports("validate_model_spec_json"):
        return False
    if not _spec_is_rust_runtime_compatible(spec):
        return False
    _rust_backend.validate_model_spec_json(spec_to_json(spec))
    return True


def _inspect_with_rust(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Inspect a portable spec with the Rust backend if possible."""
    if not _should_use_rust_backend():
        return None
    if not _rust_backend_supports("inspect_model_spec_json"):
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
    if not (_should_use_rust_backend() and _rust_backend_supports("predict_json")):
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
    if not (
        _should_use_rust_backend() and _rust_backend_supports("design_matrix_json")
    ):
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
