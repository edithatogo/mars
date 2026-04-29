"""Tests for the current Python training routing contract."""

import json
from pathlib import Path

import numpy as np

from pymars import Earth
from pymars import runtime


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


def test_rust_training_bridge_can_fit_when_enabled(monkeypatch) -> None:
    """Rust training routing should work behind the private environment gate."""
    fixture_path = Path("tests/fixtures/training_full_fit_baseline_v1.json")
    spec_json = fixture_path.read_text()
    calls: list[str] = []

    class DummyRustBackend:
        def fit_model_json(self, request_json: str) -> str:
            calls.append(request_json)
            return spec_json

    monkeypatch.setenv("PYMARS_USE_RUST_TRAINING", "1")
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

    monkeypatch.setenv("PYMARS_USE_RUST_TRAINING", "1")
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

    monkeypatch.setenv("PYMARS_USE_RUST_TRAINING", "1")
    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())

    model = Earth(max_terms=5, max_degree=1, penalty=3.0, feature_importance_type="nb_subsets")
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))

    assert model.record_ is not None
    assert getattr(model.record_, "pruning_trace_basis_functions_", None)
    assert model.feature_importances_ is not None
    assert np.any(model.feature_importances_ > 0.0)

    summary = model.summary_feature_importances()
    assert "Feature Importances (nb_subsets)" in summary
    assert "x0" in summary
