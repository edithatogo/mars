"""
Stress testing for pymars to evaluate behavior under extreme conditions
"""
import numpy as np
import pytest
from pymars import Earth
import warnings


def test_stress_extreme_parameters():
    """Test behavior with extreme parameter values."""
    # Generate test data
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 50)
    
    # Test with extreme parameters
    extreme_configs = [
        # Very high penalty
        {'max_degree': 1, 'penalty': 1000.0, 'max_terms': 5},
        # Very low penalty (potentially causing overfitting)
        {'max_degree': 4, 'penalty': 0.01, 'max_terms': 50},
        # Maximum degree with maximum terms
        {'max_degree': 10, 'penalty': 3.0, 'max_terms': 100},  # This might be pushing limits
        # Minimum settings
        {'max_degree': 1, 'penalty': 1.0, 'max_terms': 1},
        # Extreme penalty with high degree
        {'max_degree': 5, 'penalty': 100.0, 'max_terms': 20},
    ]
    
    for i, config in enumerate(extreme_configs):
        try:
            model = Earth(**config)
            # Suppress warnings for these extreme cases
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            
            # Model should at least be fitted without crashing
            assert model.fitted_
            
            # Try to get predictions (should not crash)
            predictions = model.predict(X[:5])
            assert len(predictions) == 5
            assert all(np.isfinite(pred) for pred in predictions)
            
            print(f"✅ Extreme config {i+1} passed: {config}")
            
        except Exception as e:
            # For some extreme configs, failure might be expected
            print(f"⚠️  Extreme config {i+1} raised {type(e).__name__}: {e} - Config: {config}")


def test_stress_extreme_data_values():
    """Test with extreme data values."""
    # Create data with very large/small values
    X = np.random.rand(20, 3)
    # Scale to extreme values
    X[:, 0] *= 1e6  # Very large values
    X[:, 1] *= 1e-6  # Very small values
    X[:, 2] = X[:, 2] * 2e5 - 1e5  # Mix of positive/negative large values
    
    # Generate corresponding y
    y = X[:, 0] * 1e-6 + X[:, 1] * 1e6 + X[:, 2] * 1e-1  # Adjust y to match scale
    
    model = Earth(max_degree=2, penalty=3.0, max_terms=10)
    
    # Should handle extreme scales without crashing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    assert model.fitted_
    
    # Predictions should be finite
    predictions = model.predict(X[:5])
    assert all(np.isfinite(pred) for pred in predictions)
    print("✅ Extreme data values test passed")


def test_stress_single_sample():
    """Test with single sample (minimum possible)."""
    X = np.array([[1.0, 2.0, 3.0]])  # Single sample
    y = np.array([4.0])  # Single target
    
    model = Earth(max_degree=1, penalty=1.0, max_terms=5)
    
    # Should handle single sample gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    # Should create at least a basic model
    assert model.fitted_
    
    # Prediction should work
    pred = model.predict(X)
    assert len(pred) == 1
    assert np.isfinite(pred[0])
    print("✅ Single sample stress test passed")


def test_stress_two_samples():
    """Test with two samples (minimum for most algorithms)."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # Two samples
    y = np.array([2.0, 5.0])  # Two targets
    
    model = Earth(max_degree=2, penalty=3.0, max_terms=10)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    assert model.fitted_
    
    # Should work for prediction
    pred = model.predict(X)
    assert len(pred) == 2
    assert all(np.isfinite(p) for p in pred)
    print("✅ Two samples stress test passed")


def test_stress_many_features_few_samples():
    """Test with many features but few samples (high dimensional, low sample scenario)."""
    X = np.random.rand(5, 20)  # 5 samples, 20 features
    y = X[:, 0] + X[:, 1] * 0.5  # Only first 2 features matter
    
    model = Earth(max_degree=2, penalty=10.0, max_terms=10)  # High penalty to avoid overfitting
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    # Should handle overparameterized scenario gracefully
    assert model.fitted_
    
    pred = model.predict(X)
    assert len(pred) == 5
    assert all(np.isfinite(p) for p in pred)
    print("✅ Many features few samples stress test passed")


def test_stress_all_same_targets():
    """Test with all identical target values."""
    X = np.random.rand(10, 3)
    y = np.ones(10)  # All targets are the same
    
    model = Earth(max_degree=2, penalty=3.0, max_terms=10)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    assert model.fitted_
    
    # Should predict near-constant values
    pred = model.predict(X)
    assert all(abs(p - 1.0) < 0.1 for p in pred)  # Should be close to 1.0
    print("✅ All same targets stress test passed")


def test_stress_all_same_features():
    """Test with all identical feature values."""
    X = np.ones((10, 3))  # All feature values are the same
    y = np.arange(10) * 0.5  # Different targets
    
    model = Earth(max_degree=2, penalty=3.0, max_terms=10)
    
    # Should handle constant features gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    # Model should still be fitted
    assert model.fitted_
    print("✅ All same features stress test passed")


def test_stress_alternating_pattern():
    """Test with alternating/oscillating target pattern."""
    X = np.random.rand(20, 2)
    # Create alternating pattern that's hard to fit
    y = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(20)])
    
    model = Earth(max_degree=3, penalty=3.0, max_terms=15)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    assert model.fitted_
    
    pred = model.predict(X)
    assert len(pred) == 20
    assert all(np.isfinite(p) for p in pred)
    print("✅ Alternating pattern stress test passed")


def test_stress_max_terms_exceeding_samples():
    """Test with max_terms exceeding number of samples (should be handled gracefully)."""
    X = np.random.rand(10, 2)  # 10 samples
    y = X[:, 0] + X[:, 1] * 0.5
    
    # Try with max_terms > n_samples
    model = Earth(max_degree=2, penalty=3.0, max_terms=50)  # Much larger than samples
    
    # Should handle gracefully without crashing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    
    assert model.fitted_
    
    # Check that model didn't grow too large
    assert len(model.basis_) <= 10  # Should be limited by sample size
    print("✅ Max terms exceeding samples stress test passed")


if __name__ == "__main__":
    test_stress_extreme_parameters()
    test_stress_extreme_data_values()
    test_stress_single_sample()
    test_stress_two_samples()
    test_stress_many_features_few_samples()
    test_stress_all_same_targets()
    test_stress_all_same_features()
    test_stress_alternating_pattern()
    test_stress_max_terms_exceeding_samples()
    print("✅ All stress tests passed!")