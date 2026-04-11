"""Unit tests for basis functions - integration level."""

import pytest
import numpy as np
from pymars import Earth


@pytest.mark.unit
class TestBasisFunctionsIntegration:
    """Test basis functions through Earth model integration."""

    def test_model_creates_constant_basis(self):
        """Test that Earth model creates constant (intercept) basis."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1] + 10  # Large intercept

        model = Earth(max_degree=1, max_terms=10)
        model.fit(X, y)

        # Model should have intercept
        assert len(model.basis_) > 0
        # Predictions should be close
        assert np.mean(np.abs(model.predict(X) - y)) < 1.0

    def test_model_creates_linear_basis(self):
        """Test that Earth model creates linear basis functions."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] * 2 + X[:, 1] * 3

        model = Earth(max_degree=1, max_terms=10, allow_linear=True)
        model.fit(X, y)

        # Should capture linear relationship
        assert model.score(X, y) > 0.95

    def test_model_creates_hinge_basis(self):
        """Test that Earth model creates hinge basis functions."""
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = np.maximum(X[:, 0] - 0.5, 0) + X[:, 1]

        model = Earth(max_degree=1, max_terms=20)
        model.fit(X, y)

        # Should capture hinge relationship
        assert model.score(X, y) > 0.9

    def test_basis_function_degree_0(self):
        """Test constant basis function has degree 0."""
        from pymars._basis import ConstantBasisFunction
        bf = ConstantBasisFunction()
        assert bf.degree() == 0

    def test_basis_function_degree_1(self):
        """Test linear/hinge basis functions have degree 1."""
        from pymars._basis import LinearBasisFunction, HingeBasisFunction
        linear = LinearBasisFunction(variable_idx=0)
        hinge = HingeBasisFunction(variable_idx=0, knot_val=0.5)
        assert linear.degree() == 1
        assert hinge.degree() == 1

    def test_multiple_variables(self):
        """Test basis functions with multiple variables."""
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = X[:, 0] + X[:, 2] * 2

        model = Earth(max_degree=1, max_terms=15)
        model.fit(X, y)

        # Should identify important variables
        assert model.score(X, y) > 0.95

    def test_hinge_and_linear_mixed(self):
        """Test model with both hinge and linear terms."""
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = np.maximum(X[:, 0] - 0.5, 0) + X[:, 1]

        model = Earth(max_degree=1, max_terms=20)
        model.fit(X, y)

        assert model.score(X, y) > 0.9

    def test_basis_function_evaluation_consistency(self):
        """Test that basis functions evaluate consistently."""
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = X[:, 0] ** 2

        model1 = Earth(max_degree=1, max_terms=10, penalty=2.0)
        model2 = Earth(max_degree=1, max_terms=10, penalty=2.0)

        model1.fit(X, y)
        model2.fit(X, y)

        # Same seed and data should give similar results
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        # Results should be very similar
        assert np.corrcoef(pred1, pred2)[0, 1] > 0.95
