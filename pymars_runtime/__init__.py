"""Python entrypoint for the Rust-backed pymars runtime extension.

When the compiled extension is available, the package re-exports it as the
runtime backend. Otherwise, it provides a small fallback shim so editable
installs and source-only environments can still import the package.
"""

from __future__ import annotations

import json

try:  # pragma: no cover - exercised implicitly when the extension is built
    from . import pymars_runtime as _native
except Exception:  # pragma: no cover - optional compiled extension
    _native = None

_IS_COMPILED = _native is not None


def _require_native() -> object:
    if _native is None:
        raise NotImplementedError(
            "The compiled pymars_runtime extension is not built."
        )
    return _native


def load_model_spec_canonical_json(spec_json: str) -> str:
    native = _require_native()
    return native.load_model_spec_canonical_json(spec_json)


def load_model_spec_path_json(path: str) -> str:
    native = _require_native()
    return native.load_model_spec_path_json(path)


def validate_model_spec_json(spec_json: str) -> None:
    native = _require_native()
    return native.validate_model_spec_json(spec_json)


def design_matrix_json(spec_json: str, rows: list[list[float]]) -> list[list[float]]:
    native = _require_native()
    return native.design_matrix_json(spec_json, rows)


def predict_json(spec_json: str, rows: list[list[float]]) -> list[float]:
    native = _require_native()
    return native.predict_json(spec_json, rows)


def inspect_model_spec_json(spec_json: str) -> str:
    native = _require_native()
    return native.inspect_model_spec_json(spec_json)


def export_model_json(spec_json: str) -> str:
    native = _require_native()
    if hasattr(native, "export_model_json"):
        return native.export_model_json(spec_json)
    return json.dumps(json.loads(spec_json), indent=2, sort_keys=True)


def fit_model_json(request_json: str) -> str:
    native = _require_native()
    return native.fit_model_json(request_json)
