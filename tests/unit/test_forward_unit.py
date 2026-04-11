"""Unit tests for forward pass implementation."""

import pytest
import numpy as np
from pymars import Earth
from pymars._forward import ForwardPasser


@pytest.mark.unit
class TestForwardPassBasic:
    """Test basic forward pass functionality."""

    def test_forward_pass_creates_basis_functions(self):
        """Test that forward pass creates basis functions."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = Earth(max_terms=10, max_degree=1)
        model.fit(X, y)

        assert len(model.basis_) > 1  # At least intercept + some basis functions

    def test_forward_pass_respects_max_terms(self):
        """Test that forward pass stops at max_terms."""
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = np.sin(X[:, 0]) + X[:, 1]

        model = Earth(max_terms=5, max_degree=1)
        model.fit(X, y)

        assert len(model.basis_) <= 5

    def test_forward_pass_linear_terms(self):
        """Test forward pass with linear terms only."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] * 2 + X[:, 1] * 3

        model = Earth(max_degree=1, allow_linear=True)
        model.fit(X, y)

        # Should find linear relationship
        assert model.score(X, y) > 0.95


@pytest.mark.unit
class TestForwardPassHinge:
    """Test forward pass with hinge terms."""

    def test_forward_pass_creates_hinge_terms(self):
        """Test that forward pass creates hinge basis functions."""
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = np.maximum(X[:, 0] - 0.5, 0)

        model = Earth(max_degree=1, max_terms=10)
        model.fit(X, y)

        assert len(model.basis_) > 1

    def test_forward_pass_with_nonlinear_relationship(self):
        """Test forward pass captures nonlinear relationships."""
        np.random.seed(42)
        X = np.random.rand(100, 1)
        y = np.abs(X[:, 0] - 0.5)

        model = Earth(max_degree=1, max_terms=20, penalty=2.0)
        model.fit(X, y)

        # Should capture the V-shaped relationship
        assert model.score(X, y) > 0.8


@pytest.mark.unit
class TestForwardPassInteractions:
    """Test forward pass with interaction terms."""

    def test_forward_pass_interactions_degree2(self):
        """Test forward pass with degree 2 interactions."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = X[:, 0] * X[:, 1]

        model = Earth(max_degree=2, max_terms=30, penalty=2.0)
        model.fit(X, y)

        # Should capture interaction
        assert model.score(X, y) > 0.9

    def test_forward_pass_interactions_degree3(self):
        """Test forward pass with degree 3 interactions."""
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = X[:, 0] * X[:, 1] * X[:, 2]

        model = Earth(max_degree=3, max_terms=50, penalty=2.0)
        model.fit(X, y)

        # Should capture 3-way interaction
        assert model.score(X, y) > 0.85


@pytest.mark.unit
class TestForwardPassMinspanEndspan:
    """Test minspan and endspan controls."""

    def test_minspan_alpha_affects_knot_placement(self):
        """Test that minspan_alpha affects knot placement."""
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = X[:, 0] + X[:, 1]

        model_tight = Earth(max_degree=1, minspan_alpha=0.001)
        model_loose = Earth(max_degree=1, minspan_alpha=0.5)

        model_tight.fit(X, y)
        model_loose.fit(X, y)

        # Tighter minspan should allow more basis functions
        assert len(model_tight.basis_) >= len(model_loose.basis_)

    def test_endspan_alpha_affects_knot_placement(self):
        """Test that endspan_alpha affects knot placement."""
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = X[:, 0] + X[:, 1]

        model_tight = Earth(max_degree=1, endspan_alpha=0.001)
        model_loose = Earth(max_degree=1, endspan_alpha=0.5)

        model_tight.fit(X, y)
        model_loose.fit(X, y)

        # Tighter endspan should allow more basis functions
        assert len(model_tight.basis_) >= len(model_loose.basis_)


@pytest.mark.unit
class TestForwardPassEdgeCases:
    """Test forward pass edge cases."""

    def test_empty_data_raises_error(self):
        """Test that empty data raises error."""
        model = Earth()
        with pytest.raises(ValueError):
            model.fit(np.array([]).reshape(0, 1), np.array([]))

    def test_single_feature_works(self):
        """Test forward pass with single feature."""
        np.random.seed(42)
        X = np.random.rand(50, 1)
        y = X[:, 0] * 2

        model = Earth(max_degree=1)
        model.fit(X, y)

        assert model.score(X, y) > 0.95

    def test_many_features_works(self):
        """Test forward pass with many features."""
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = X[:, 0] + X[:, 1]

        model = Earth(max_degree=1, max_terms=20)
        model.fit(X, y)

        assert model.score(X, y) > 0.9

    def test_constant_feature_ignored(self):
        """Test that constant features are ignored."""
        np.random.seed(42)
        X = np.column_stack([np.random.rand(50, 2), np.ones(50)])
        y = X[:, 0] + X[:, 1]

        model = Earth(max_degree=1)
        model.fit(X, y)

        # Should still work
        assert model.score(X, y) > 0.9
