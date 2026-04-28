"""Python entrypoint for the Rust-backed pymars runtime extension.

When the compiled extension is available, the package re-exports it as the
runtime backend. Otherwise, it provides a small fallback shim so editable
installs and source-only environments can still import the package.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised implicitly when the extension is built
    from . import pymars_runtime as _native
except Exception:  # pragma: no cover - optional compiled extension
    _native = None

_IS_COMPILED = _native is not None


def validate_model_spec_json(spec_json: str) -> None:
    if _native is None:
        raise NotImplementedError(
            "The compiled pymars_runtime extension is not built."
        )
    return _native.validate_model_spec_json(spec_json)


def design_matrix_json(spec_json: str, rows: list[list[float]]) -> list[list[float]]:
    if _native is None:
        raise NotImplementedError(
            "The compiled pymars_runtime extension is not built."
        )
    return _native.design_matrix_json(spec_json, rows)


def predict_json(spec_json: str, rows: list[list[float]]) -> list[float]:
    if _native is None:
        raise NotImplementedError(
            "The compiled pymars_runtime extension is not built."
        )
    return _native.predict_json(spec_json, rows)
