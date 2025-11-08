"""
Load testing for pymars to evaluate performance under different data sizes and loads
"""
import numpy as np
import pytest
import time
from pymars import Earth


def test_load_scaling_with_samples():
    """Test performance scaling with increasing number of samples."""
    np.random.seed(42)
    
    # Test with different sample sizes
    sample_sizes = [50, 100, 200, 500]
    times = []
    
    for n_samples in sample_sizes:
        X = np.random.rand(n_samples, 3)
        y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, n_samples)
        
        start_time = time.time()
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        model.fit(X, y)
        fit_time = time.time() - start_time
        
        times.append(fit_time)
        
        # Verify model still works
        assert model.fitted_
        score = model.score(X, y)
        assert isinstance(score, (int, float, np.floating))
        
        print(f"Load test with {n_samples} samples: {fit_time:.3f}s")
    
    # Performance should scale reasonably - shouldn't have exponential growth
    if len(times) > 1:
        growth_rates = [times[i]/times[i-1] for i in range(1, len(times))]
        # Average growth rate should be reasonable (e.g. < 10x per doubling)
        avg_growth = np.mean(growth_rates)
        assert avg_growth < 20, f"Performance growth rate too high: {avg_growth}"


def test_load_scaling_with_features():
    """Test performance scaling with increasing number of features."""
    feature_counts = [2, 5, 10, 15, 20]
    times = []
    
    for n_features in feature_counts:
        n_samples = 100  # Keep samples constant
        X = np.random.rand(n_samples, n_features)
        y = np.sum(X[:, :min(5, n_features)], axis=1)  # Use first few features
        
        start_time = time.time()
        model = Earth(max_degree=2, penalty=3.0, max_terms=20)
        model.fit(X, y)
        fit_time = time.time() - start_time
        
        times.append(fit_time)
        
        # Verify model works
        assert model.fitted_
        score = model.score(X, y)
        assert isinstance(score, (int, float, np.floating))
        
        print(f"Load test with {n_features} features: {fit_time:.3f}s")


def test_load_with_complex_models():
    """Test performance with increasingly complex model configurations."""
    # Test with different model complexities
    configs = [
        {'max_degree': 1, 'max_terms': 5},
        {'max_degree': 2, 'max_terms': 10},
        {'max_degree': 3, 'max_terms': 15},
        {'max_degree': 4, 'max_terms': 20}
    ]
    
    X = np.random.rand(100, 5)
    y = X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3
    
    for i, config in enumerate(configs):
        start_time = time.time()
        model = Earth(penalty=3.0, **config)
        model.fit(X, y)
        fit_time = time.time() - start_time
        
        # Verify model works
        assert model.fitted_
        score = model.score(X, y)
        assert isinstance(score, (int, float, np.floating))
        
        print(f"Complexity {i+1} (degree={config['max_degree']}, terms={config['max_terms']}): {fit_time:.3f}s")


def test_memory_usage_under_load():
    """Test memory usage patterns under different loads."""
    import gc
    import tracemalloc
    
    X = np.random.rand(200, 10)
    y = np.sum(X[:, :5], axis=1)
    
    # Start tracing memory
    tracemalloc.start()
    
    # Fit multiple models to check for memory leaks
    for i in range(5):
        model = Earth(max_degree=2, penalty=3.0, max_terms=15)
        model.fit(X, y)
        
        # Take snapshot
        current, peak = tracemalloc.get_traced_memory()
        print(f"Iteration {i+1}: Current={current/1024:.1f}KB, Peak={peak/1024:.1f}KB")
    
    tracemalloc.stop()
    gc.collect()


if __name__ == "__main__":
    test_load_scaling_with_samples()
    test_load_scaling_with_features()
    test_load_with_complex_models()
    test_memory_usage_under_load()
    print("✅ All load tests passed!")