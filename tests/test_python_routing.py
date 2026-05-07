from __future__ import annotations

"""Tests for the current Python training routing contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from pymars import Earth, runtime

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MODEL_SPEC_V1_PATH = FIXTURES_DIR / "model_spec_v1.json"


def test_public_rust_training_flag_is_not_exposed_yet() -> None:
    """Rust training routing is still a private migration detail."""
    model = Earth()

    assert not hasattr(model, "_use_rust_training")
    assert not hasattr(model, "use_rust_training")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_earth_preserves_constructor_params() -> None:
    """Earth constructor parameters should remain stable."""
    model = Earth(max_terms=15, max_degree=2, penalty=2.5)

    assert model.max_terms == 15
    assert model.max_degree == 2
    assert model.penalty == 2.5


def test_python_fallback_available() -> None:
    """The Python training path should still fit and predict successfully."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])

    model = Earth(max_terms=5)
    model.fit(X, y)

    assert model.fitted_
    assert hasattr(model, "basis_")
    assert model.predict(X).shape == (3,)


def test_rust_training_bridge_can_fit_without_environment_gate(monkeypatch) -> None:
    """Rust training routing should work without a private environment gate."""
    fixture_path = Path("tests/fixtures/training_full_fit_baseline_v1.json")
    spec_json = fixture_path.read_text()
    calls: list[str] = []

    class DummyRustBackend:
        def fit_model_json(self, request_json: str) -> str:
            calls.append(request_json)
            return spec_json

        def export_model_json(self, spec_json: str) -> str:
            return spec_json

        def predict_json(self, spec_json: str, rows: list[list[float]]) -> list[float]:
            del spec_json, rows
            return [1.0, 3.0, 5.0]

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    model = Earth(max_terms=5, max_degree=1, penalty=3.0)
    fitted = model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))

    assert fitted is model
    assert model.fitted_
    assert len(calls) == 1
    request = json.loads(calls[0])
    assert request["params"]["max_degree"] == 1
    assert request["params"]["penalty"] == 3.0
    exported = json.loads(model.export_model())
    assert exported["basis_terms"]
    assert exported["coefficients"]
    assert model.predict(np.array([[0.0], [1.0], [2.0]])).shape == (3,)


def test_rust_training_bridge_sends_routing_flags(monkeypatch) -> None:
    """Rust training routing should forward fit flags in the request payload."""
    fixture_path = Path("tests/fixtures/training_full_fit_baseline_v1.json")
    spec_json = fixture_path.read_text()
    calls: list[str] = []

    class DummyRustBackend:
        def fit_model_json(self, request_json: str) -> str:
            calls.append(request_json)
            return spec_json

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    model = Earth(
        max_terms=5,
        max_degree=1,
        penalty=3.0,
        allow_missing=True,
        categorical_features=[0],
        allow_linear=False,
    )
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))

    assert len(calls) == 1
    request = json.loads(calls[0])
    assert request["params"]["allow_missing"] is True
    assert request["params"]["allow_linear"] is False
    assert request["params"]["categorical_features"] == [0]


def test_rust_training_bridge_preserves_diagnostics(monkeypatch) -> None:
    """Rust training routing should keep the Python diagnostics surface usable."""
    fixture_path = Path("tests/fixtures/training_full_fit_baseline_v1.json")
    spec_json = fixture_path.read_text()

    class DummyRustBackend:
        def fit_model_json(self, request_json: str) -> str:
            del request_json
            return spec_json

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    model = Earth(
        max_terms=5, max_degree=1, penalty=3.0, feature_importance_type="nb_subsets"
    )
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))

    assert model.record_ is not None
    assert getattr(model.record_, "pruning_trace_basis_functions_", None)
    assert model.feature_importances_ is not None
    assert np.any(model.feature_importances_ > 0.0)

    summary = model.summary_feature_importances()
    assert "Feature Importances (nb_subsets)" in summary
    assert "x0" in summary


def test_runtime_validate_uses_rust_backend_for_portable_specs(monkeypatch) -> None:
    """Rust-backed validation should run before the Python validator."""
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    calls: list[str] = []

    class DummyRustBackend:
        def validate_model_spec_json(self, spec_json: str) -> None:
            calls.append(spec_json)

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    validated = runtime.validate(spec)

    assert validated == spec
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("runtime_call", "expected_error"),
    [
        (
            lambda spec: runtime.validate(spec),
            RuntimeError,
        ),
        (
            lambda spec: runtime.inspect(spec),
            RuntimeError,
        ),
        (
            lambda spec: runtime.export_model_json(spec),
            RuntimeError,
        ),
    ],
)
def test_runtime_supported_rust_paths_propagate_backend_errors(
    monkeypatch, runtime_call, expected_error
) -> None:
    """Supported Rust paths should not silently fall back on backend errors."""
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    spec["categorical_imputer"] = {"sentinel": True}

    class DummyRustBackend:
        def validate_model_spec_json(self, spec_json: str) -> None:
            del spec_json
            raise RuntimeError("boom")

        def inspect_model_spec_json(self, spec_json: str) -> str:
            del spec_json
            raise RuntimeError("boom")

        def export_model_json(self, spec_json: str) -> str:
            del spec_json
            raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())
    monkeypatch.setattr(runtime, "_spec_is_rust_runtime_compatible", lambda _spec: True)

    with pytest.raises(expected_error):
        runtime_call(spec)


def test_runtime_inspect_uses_rust_backend_for_supported_specs(monkeypatch) -> None:
    """Rust-backed inspect should match the Python summary for portable specs."""
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    spec["categorical_imputer"] = {"sentinel": True}
    expected = {
        "spec_version": spec["spec_version"],
        "model_type": spec["model_type"],
        "n_features": spec["feature_schema"]["n_features"],
        "n_basis_terms": len(spec["basis_terms"]),
        "metrics": spec["metrics"],
    }
    calls: list[str] = []

    class DummyRustBackend:
        def inspect_model_spec_json(self, spec_json: str) -> str:
            calls.append(spec_json)
            return json.dumps(expected)

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    assert runtime.inspect(MODEL_SPEC_V1_PATH) == expected
    assert len(calls) == 1


def test_runtime_inspect_falls_back_to_python_for_incompatible_specs(
    monkeypatch,
) -> None:
    """If a spec is incompatible, the Python inspect summary should still work."""
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    spec["categorical_imputer"] = {"kind": "sentinel"}
    expected = {
        "spec_version": spec["spec_version"],
        "model_type": spec["model_type"],
        "n_features": spec["feature_schema"]["n_features"],
        "n_basis_terms": len(spec["basis_terms"]),
        "metrics": spec["metrics"],
    }
    calls: list[str] = []

    class DummyRustBackend:
        def inspect_model_spec_json(self, spec_json: str) -> str:
            calls.append(spec_json)
            raise AssertionError("Rust should not be called for incompatible specs")

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    assert runtime.inspect(spec) == expected
    assert calls == []


def test_runtime_export_model_json_routes_through_rust(monkeypatch) -> None:
    """Supported specs should use the Rust export-normalization path."""
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    expected_json = json.dumps(spec, indent=2, sort_keys=True)
    calls: list[str] = []

    class DummyRustBackend:
        def export_model_json(self, spec_json: str) -> str:
            calls.append(spec_json)
            return expected_json

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())
    monkeypatch.setattr(runtime, "_spec_is_rust_runtime_compatible", lambda _spec: True)

    exported = runtime.export_model_json(spec)

    assert exported == expected_json
    assert len(calls) == 1


def test_runtime_predict_supported_rust_paths_propagate_backend_errors(
    monkeypatch,
) -> None:
    """Supported Rust prediction should not fall back when the backend errors."""
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    spec["categorical_imputer"] = {"sentinel": True}

    class DummyRustBackend:
        def predict_json(self, spec_json: str, rows: list[list[float]]) -> list[float]:
            del spec_json, rows
            raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())
    monkeypatch.setattr(runtime, "_spec_is_rust_runtime_compatible", lambda _spec: True)

    with pytest.raises(RuntimeError):
        runtime.predict(spec, np.array([[0.0], [1.0], [2.0]]))


def test_runtime_design_matrix_supported_rust_paths_propagate_backend_errors(
    monkeypatch,
) -> None:
    """Supported Rust design-matrix generation should not silently fall back."""
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    spec["categorical_imputer"] = {"sentinel": True}

    class DummyRustBackend:
        def design_matrix_json(
            self, spec_json: str, rows: list[list[float]]
        ) -> list[list[float]]:
            del spec_json, rows
            raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())
    monkeypatch.setattr(runtime, "_spec_is_rust_runtime_compatible", lambda _spec: True)

    with pytest.raises(RuntimeError):
        runtime.design_matrix(spec, np.array([[0.0], [1.0], [2.0]]))


def test_runtime_save_model_routes_through_rust_export(monkeypatch, tmp_path) -> None:
    """Saving a compatible model should use the Rust export-normalization path."""
    model = runtime.load_model(runtime.load_model_spec(MODEL_SPEC_V1_PATH))
    expected_json = json.dumps(
        model.export_model(format="dict"), indent=2, sort_keys=True
    )
    target = tmp_path / "model.json"
    calls: list[str] = []

    class DummyRustBackend:
        def export_model_json(self, spec_json: str) -> str:
            calls.append(spec_json)
            return expected_json

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())
    monkeypatch.setattr(runtime, "_spec_is_rust_runtime_compatible", lambda _spec: True)

    saved = runtime.save_model(model, target)

    assert saved == target
    assert target.read_text() == expected_json
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("spec_or_path", "expected_call"),
    [
        (MODEL_SPEC_V1_PATH.read_text(), MODEL_SPEC_V1_PATH.read_text()),
        (MODEL_SPEC_V1_PATH, str(MODEL_SPEC_V1_PATH)),
    ],
    ids=["json-string", "path"],
)
def test_runtime_load_model_spec_routes_through_rust(
    monkeypatch, spec_or_path: str | Path, expected_call: str
):
    """JSON strings and paths should round-trip through the Rust loader."""
    calls: list[str] = []

    class DummyRustBackend:
        def load_model_spec_canonical_json(self, spec_input: str) -> str:
            calls.append(spec_input)
            try:
                return json.dumps(json.loads(spec_input))
            except json.JSONDecodeError:
                return json.dumps(json.loads(Path(spec_input).read_text()))

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    spec = runtime.load_model_spec(spec_or_path)

    assert spec["spec_version"] == "1.0"
    assert calls == [expected_call]


def test_runtime_load_model_spec_falls_back_when_rust_loader_fails(monkeypatch):
    """If Rust fails, the Python loader should still handle paths."""
    calls: list[str] = []

    class FailingRustBackend:
        def load_model_spec_canonical_json(self, spec_input: str) -> str:
            calls.append(spec_input)
            raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "_rust_backend", FailingRustBackend())

    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)

    assert spec["spec_version"] == "1.0"
    assert calls == [str(MODEL_SPEC_V1_PATH)]


def test_runtime_load_model_spec_dict_input_stays_on_python_path(monkeypatch):
    """Dictionary payloads should not be diverted to Rust."""
    payload = runtime.load_model_spec(MODEL_SPEC_V1_PATH)

    class FailingRustBackend:
        def load_model_spec_canonical_json(self, spec_input: str) -> str:
            raise AssertionError(f"unexpected Rust call for {spec_input!r}")

    monkeypatch.setattr(runtime, "_rust_backend", FailingRustBackend())

    restored = runtime.load_model_spec(payload)

    assert restored == payload


def test_earth_predict_uses_rust_backend_when_available(monkeypatch) -> None:
    """Fitted Earth models should route portable prediction through Rust first."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])
    model = Earth(max_terms=5)
    model.fit(X, y)

    calls: list[str] = []

    class DummyRustBackend:
        def predict_json(self, spec_json: str, rows: list[list[float]]) -> list[float]:
            calls.append(spec_json)
            assert rows == [[0.0], [1.0], [2.0]]
            return [1.0, 3.0, 5.0]

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    predicted = model.predict(X)

    assert np.allclose(predicted, y)
    assert len(calls) == 1
