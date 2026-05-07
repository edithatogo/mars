"""Tests for estimator compatibility and the current Rust training boundary."""

from pathlib import Path

import numpy as np

from pymars import Earth
from pymars._model_spec import spec_to_model


def test_sklearn_compat_fallback():
    """Test that the estimator interface remains intact on the Python path."""
    model = Earth()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "get_params")
    assert hasattr(model, "set_params")


def test_rust_training_routing_is_not_public_yet():
    """Rust training routing should remain an internal migration detail."""
    model = Earth()
    assert not hasattr(model, "_use_rust_training")
    assert not hasattr(model, "use_rust_training")
    assert Path("tests/test_sklearn_compat.py").exists()


def test_rust_backed_exports_compatible_model_spec():
    """Test that Rust-backed estimators export compatible ModelSpec."""
    # Train a simple model
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])

    model = Earth(max_terms=5)
    model.fit(X, y)

    # Export model spec
    spec = model.get_model_spec()

    # Verify spec has required fields
    assert "basis_terms" in spec, "Spec should have basis_terms"
    assert "coefficients" in spec, "Spec should have coefficients"
    assert "params" in spec, "Spec should have params"
    assert spec["feature_schema"]["n_features"] == model.n_features_in_
    assert len(spec["feature_schema"]["feature_names"]) == model.n_features_in_

    # Verify spec can be used to recreate model
    model2 = Earth.from_model(spec)
    assert model2.fitted_, "Recreated model should be fitted"

    # Verify predictions match
    y_pred1 = model.predict(X)
    y_pred2 = model2.predict(X)
    np.testing.assert_almost_equal(y_pred1, y_pred2, decimal=10)


def test_spec_rehydration_prefers_feature_names_when_counts_disagree() -> None:
    """Rehydration should preserve the actual input schema width."""
    payload = {
        "spec_version": "1.0",
        "model_type": "Earth",
        "params": {},
        "feature_schema": {
            "n_features": 3,
            "feature_names": ["x0", "x1", "x2", "x3", "x4"],
        },
        "basis_terms": [
            {"kind": "constant", "gcv_score": 0.0, "rss_score": 0.0},
        ],
        "coefficients": [1.0],
        "metrics": {"rss": 0.0, "mse": 0.0, "gcv": 0.0},
    }

    model = spec_to_model(payload, Earth)

    assert model.n_features_in_ == 5
    assert list(model.feature_names_in_) == ["x0", "x1", "x2", "x3", "x4"]


def test_rust_routing_does_not_break_python_fit() -> None:
    """Rust-backed training routing should not break the Python fallback path."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 3.0, 5.0])

    model = Earth(max_terms=5)
    model.fit(X, y)

    assert model.fitted_
    assert model.predict(X).shape == (3,)
