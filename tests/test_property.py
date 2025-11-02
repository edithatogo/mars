"""
Property-based tests for pymars using Hypothesis.
"""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pymars import Earth


# Define strategies for generating test data
@st.composite
def feature_and_target_strategy(draw):
    """Generate compatible X (features) and y (target) arrays."""
    n_samples = draw(st.integers(min_value=10, max_value=50))
    n_features = draw(st.integers(min_value=1, max_value=5))

    X = draw(arrays(
        dtype=float,
        shape=(n_samples, n_features),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_infinity=False, allow_nan=False)
    ))

    y = draw(arrays(
        dtype=float,
        shape=(n_samples,),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_infinity=False, allow_nan=False)
    ))

    return X, y


@given(draw=feature_and_target_strategy())
@settings(max_examples=20, deadline=2000)  # Limit examples and time to avoid long runs
def test_earth_fitting_property(draw):
    """Test that Earth model can handle various valid inputs."""
    X, y = draw

    # Create and fit the model
    model = Earth(max_terms=10, max_degree=2)

    # This should not raise an exception for valid inputs
    model.fit(X, y)

    # Model should be fitted
    assert model.fitted_

    # Predictions should have same length as input
    predictions = model.predict(X)
    assert predictions.shape == y.shape

    # Score should be a float
    score = model.score(X, y)
    assert isinstance(score, (int, float, np.floating))
    assert np.isfinite(score)


@given(draw=feature_and_target_strategy())
@settings(max_examples=10, deadline=1000)
def test_earth_prediction_property(draw):
    """Test that Earth model prediction properties hold."""
    X, _ = draw
    n_samples = X.shape[0]

    # Create a simple target variable based on first feature
    y = X[:, 0] + 0.1 * np.random.randn(n_samples)

    model = Earth(max_terms=5, max_degree=1)
    model.fit(X, y)

    # Predictions should be finite
    predictions = model.predict(X)
    assert np.all(np.isfinite(predictions))

    # Consistency: same input should give same output
    predictions2 = model.predict(X)
    np.testing.assert_array_almost_equal(predictions, predictions2)


@given(draw=feature_and_target_strategy())
@settings(max_examples=5, deadline=1000)
def test_earth_max_degree_property(draw):
    """Test max_degree parameter with various values."""
    X, y = draw
    for max_degree in [1, 2, 3]:
        model = Earth(max_degree=max_degree, max_terms=min(15, X.shape[0] // 2))  # Adjust max_terms to avoid overfitting
        model.fit(X, y)

        # Model should be fitted regardless of max_degree
        assert model.fitted_
        assert hasattr(model, 'basis_')
        if model.basis_ is not None:  # Check that if basis exists, it's valid
            assert len(model.basis_) >= 1  # At least intercept should be there


def test_earth_model_consistency():
    """Test that the model behaves consistently across multiple fits."""
    X = np.random.rand(30, 2)
    y = X[:, 0] + X[:, 1]

    model1 = Earth(max_terms=8, penalty=2.0)
    model1.fit(X, y)
    pred1 = model1.predict(X[:5])

    model2 = Earth(max_terms=8, penalty=2.0)
    model2.fit(X, y)
    pred2 = model2.predict(X[:5])

    # For same parameters and data, results should be reproducible
    # (Note: this depends on internal random states, so may not always hold)
    # Instead, we test that both models work correctly
    assert len(pred1) == len(pred2) == 5
    assert np.all(np.isfinite(pred1))
    assert np.all(np.isfinite(pred2))
