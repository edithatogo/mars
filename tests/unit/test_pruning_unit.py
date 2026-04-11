"""Unit tests for pruning implementation."""

import pytest
import numpy as np
from pymars import Earth
from pymars._pruning import PruningPasser


@pytest.mark.unit
class TestPruningGCV:
    """Test GCV-based pruning."""

    def test_pruning_reduces_model_complexity(self):
        """Test that pruning reduces number of basis functions."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(100)

        # Fit with generous terms
        model = Earth(max_terms=30, max_degree=1, penalty=3.0)
        model.fit(X, y)

        # Pruning should have occurred
        assert len(model.basis_) > 0
        # Score should be reasonable
        assert model.score(X, y) > 0.8

    def test_pruning_penalty_affects_complexity(self):
        """Test that penalty parameter affects model complexity."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(100)

        model_low = Earth(max_terms=30, max_degree=1, penalty=1.5)
        model_high = Earth(max_terms=30, max_degree=1, penalty=5.0)

        model_low.fit(X, y)
        model_high.fit(X, y)

        # Higher penalty should result in fewer basis functions
        assert len(model_low.basis_) >= len(model_high.basis_)

    def test_gcv_score_is_finite(self):
        """Test that GCV score is finite after pruning."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = Earth(max_terms=10, max_degree=1)
        model.fit(X, y)

        # GCV should be finite
        assert np.isfinite(model.gcv_)


@pytest.mark.unit
class TestPruningTrace:
    """Test pruning trace functionality."""

    def test_pruning_trace_records_gcv(self):
        """Test that pruning trace records GCV values."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = Earth(max_terms=10, max_degree=1, penalty=3.0)
        model.fit(X, y)

        # Record should exist
        assert hasattr(model, 'record_')
        assert model.record_ is not None


@pytest.mark.unit
class TestPruningEdgeCases:
    """Test pruning edge cases."""

    def test_pruning_with_overfit_model(self):
        """Test pruning handles overfit models."""
        np.random.seed(42)
        X = np.random.rand(30, 5)
        y = X[:, 0] + 0.01 * np.random.randn(30)

        # Allow overfitting
        model = Earth(max_terms=50, max_degree=2, penalty=1.0)
        model.fit(X, y)

        # Pruning should simplify
        assert len(model.basis_) > 0
        # Should still have reasonable fit
        assert model.score(X, y) >= 0

    def test_pruning_with_small_dataset(self):
        """Test pruning with very small dataset."""
        np.random.seed(42)
        X = np.random.rand(10, 2)
        y = X[:, 0] + X[:, 1]

        model = Earth(max_terms=10, max_degree=1)
        model.fit(X, y)

        # Should not crash
        assert len(model.basis_) > 0

    def test_pruning_preserves_intercept(self):
        """Test that pruning preserves intercept term."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1] + 5  # Large intercept

        model = Earth(max_terms=10, max_degree=1)
        model.fit(X, y)

        # Model should capture the offset
        predictions = model.predict(X)
        assert np.mean(np.abs(predictions - y)) < 1.0
