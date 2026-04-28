"""Tests for Phase 0 Task 2: Create failing Rust training-orchestration fixtures."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pymars import Earth


def test_python_baseline_fixture_exists():
    """Test that Python-generated baseline fixture for full fit exists."""
    fixture_path = Path("tests/fixtures/training_full_fit_baseline_v1.json")
    assert fixture_path.exists(), (
        "Python baseline fixture not found at "
        "tests/fixtures/training_full_fit_baseline_v1.json"
    )


def test_python_baseline_fixture_structure():
    """Test that the baseline fixture has the correct structure for full fit."""
    fixture_path = Path("tests/fixtures/training_full_fit_baseline_v1.json")
    with open(fixture_path) as f:
        fixture = json.load(f)

    required_keys = [
        "spec_version",
        "params",
        "feature_schema",
        "basis_terms",
        "coefficients",
    ]
    for key in required_keys:
        assert key in fixture, f"Missing key '{key}' in baseline fixture"

    assert "metrics" in fixture or "rss" in fixture, (
        "Missing metrics in baseline fixture"
    )


def test_python_baseline_fixture_reproducible():
    """Test that the baseline fixture can be loaded and used for prediction."""
    fixture_path = Path("tests/fixtures/training_full_fit_baseline_v1.json")
    with open(fixture_path) as f:
        fixture = json.load(f)

    # Load the model spec and verify it works
    model = Earth.from_model(fixture)
    assert model.fitted_, "Model from baseline fixture should be fitted"

    # Test prediction on a simple case
    X_test = np.array([[0.0], [1.0], [2.0]])
    y_pred = model.predict(X_test)
    assert len(y_pred) == 3, "Prediction should return 3 values"


def test_sample_weight_fixture_exists():
    """Test that sample-weight fixture exists."""
    fixture_path = Path("tests/fixtures/training_sample_weight_baseline_v1.json")
    assert fixture_path.exists(), (
        "Sample-weight baseline fixture not found at "
        "tests/fixtures/training_sample_weight_baseline_v1.json"
    )


def test_sample_weight_fixture_has_weights():
    """Test that the sample-weight fixture includes sample weights."""
    fixture_path = Path("tests/fixtures/training_sample_weight_baseline_v1.json")
    with open(fixture_path) as f:
        fixture = json.load(f)

    assert "sample_weight" in fixture, "Missing sample_weight in sample-weight fixture"
    assert "y" in fixture, "Missing y values in sample-weight fixture"
    assert "X" in fixture, "Missing X values in sample-weight fixture"


def test_interaction_fixture_exists():
    """Test that interaction term fixture exists."""
    fixture_path = Path("tests/fixtures/training_interaction_baseline_v1.json")
    assert fixture_path.exists(), (
        "Interaction baseline fixture not found at "
        "tests/fixtures/training_interaction_baseline_v1.json"
    )


def test_interaction_fixture_has_interactions():
    """Test that the interaction fixture includes interaction terms."""
    fixture_path = Path("tests/fixtures/training_interaction_baseline_v1.json")
    with open(fixture_path) as f:
        fixture = json.load(f)

    # Check that basis_terms includes at least one term with parents (interaction)
    basis_terms = fixture.get("basis_terms", [])
    has_interaction = any(term.get("parent1") is not None for term in basis_terms)
    assert has_interaction, (
        "Interaction fixture should contain at least one term with parent1 (interaction)"
    )


def test_rust_training_fixture_tests_exist():
    """Test that Rust tests for the full fit/export path exist."""
    rust_test_path = Path("rust-runtime/tests/training_fixture_tests.rs")
    assert rust_test_path.exists(), (
        "Rust fixture tests not found at rust-runtime/tests/training_fixture_tests.rs"
    )


def test_rust_can_load_python_baseline():
    """Test that Rust can load and validate the Python baseline fixture."""
    rust_test_path = Path("rust-runtime/tests/training_fixture_tests.rs")
    with open(rust_test_path) as f:
        content = f.read()

    assert "training_full_fit_baseline" in content, (
        "Rust tests should reference the Python baseline fixture"
    )
    assert "load_model_spec" in content or "from_json" in content, (
        "Rust tests should load the fixture"
    )
    assert "predict" in content, (
        "Rust tests should verify prediction from loaded fixture"
    )
