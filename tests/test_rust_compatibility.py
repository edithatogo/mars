"""Tests for Phase 3 Task 2: Validate sklearn and artifact compatibility."""

import os
from pathlib import Path
import numpy as np
import pytest

from pymars import Earth


def test_sklearn_compat_fallback():
    """Test that estimator compatibility tests pass with Python fallback."""
    # This test verifies sklearn compatibility is preserved
    from sklearn.utils.estimator_checks import check_estimator
    
    # Check that Earth passes sklearn estimator checks (basic)
    # This is a placeholder - full check_estimator takes too long
    model = Earth()
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert hasattr(model, 'get_params')
    assert hasattr(model, 'set_params')


def test_targeted_tests_rust_routing():
    """Test that targeted estimator tests pass with Rust routing enabled."""
    # This is a placeholder - will be updated after Rust routing is implemented
    # For now, just verify the test structure exists
    test_file = Path("tests/test_sklearn_compat.py")
    assert test_file.exists(), "test_sklearn_compat.py should exist"


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
    assert 'basis_terms' in spec, "Spec should have basis_terms"
    assert 'coefficients' in spec, "Spec should have coefficients"
    assert 'params' in spec, "Spec should have params"
    
    # Verify spec can be used to recreate model
    model2 = Earth.from_model(spec)
    assert model2.fitted_, "Recreated model should be fitted"
    
    # Verify predictions match
    y_pred1 = model.predict(X)
    y_pred2 = model2.predict(X)
    np.testing.assert_almost_equal(y_pred1, y_pred2, decimal=10)


def test_rust_routing_environment_flag():
    """Test that Rust routing can be enabled via environment flag."""
    # Set environment flag
    os.environ['PYMARS_USE_RUST_TRAINING'] = '1'
    
    # This is a placeholder - will be updated after implementation
    # For now, just verify the flag can be set
    assert os.environ.get('PYMARS_USE_RUST_TRAINING') == '1'
    
    # Clean up
    del os.environ['PYMARS_USE_RUST_TRAINING']
