#!/usr/bin/env python
"""
Performance benchmarking script for pymars using pytest-benchmark.

This script benchmarks pymars performance to identify bottlenecks and track improvements.
"""
import numpy as np
import pytest


def generate_benchmark_data(n_samples=100, n_features=5, noise_level=0.1, seed=42):
    """Generate synthetic data for benchmarking."""
    np.random.seed(seed)
    X = np.random.rand(n_samples, n_features)
    # Create a complex target with interactions
    y = (np.sin(X[:, 0] * np.pi) + 
         X[:, 1] * X[:, 2] + 
         0.5 * X[:, 3] + 
         np.random.normal(0, noise_level, n_samples))
    return X, y


def test_earth_fit_performance_small(benchmark):
    """Benchmark Earth model fitting on small datasets."""
    X, y = generate_benchmark_data(50, 3, 0.1)
    
    def fit_earth():
        from pymars import Earth
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        model.fit(X, y)
        return model
    
    result = benchmark(fit_earth)
    assert result.fitted_


def test_earth_fit_performance_medium(benchmark):
    """Benchmark Earth model fitting on medium datasets."""
    X, y = generate_benchmark_data(200, 5, 0.1)
    
    def fit_earth():
        from pymars import Earth
        model = Earth(max_degree=2, penalty=3.0, max_terms=15)
        model.fit(X, y)
        return model
    
    result = benchmark(fit_earth)
    assert result.fitted_


def test_earth_predict_performance(benchmark):
    """Benchmark Earth model prediction."""
    X, y = generate_benchmark_data(100, 4, 0.1)
    
    # Fit model first
    from pymars import Earth
    model = Earth(max_degree=2, penalty=3.0, max_terms=12)
    model.fit(X, y)
    
    # Benchmark prediction
    def predict_earth():
        return model.predict(X[:50])
    
    predictions = benchmark(predict_earth)
    assert predictions.shape == (50,)


def test_earth_score_performance(benchmark):
    """Benchmark Earth model scoring."""
    X, y = generate_benchmark_data(100, 3, 0.1)
    
    # Fit model first
    from pymars import Earth
    model = Earth(max_degree=1, penalty=3.0, max_terms=8)
    model.fit(X, y)
    
    # Benchmark scoring
    def score_earth():
        return model.score(X, y)
    
    score = benchmark(score_earth)
    assert isinstance(score, (int, float, np.floating))


def test_scaling_with_features(benchmark, n_features=5):
    """Benchmark how performance scales with number of features."""
    X, y = generate_benchmark_data(100, n_features, 0.1)
    
    def fit_earth():
        from pymars import Earth
        model = Earth(max_degree=2, penalty=3.0, max_terms=min(20, n_features * 3))
        model.fit(X, y)
        return model
    
    result = benchmark(fit_earth)
    assert result.fitted_
    assert len(result.basis_) >= 1


def test_forward_passer_performance(benchmark):
    """Benchmark ForwardPasser performance."""
    X, y = generate_benchmark_data(100, 4, 0.1)
    missing_mask = np.zeros_like(X, dtype=bool)
    
    from pymars import Earth
    from pymars._forward import ForwardPasser
    
    earth_model = Earth(max_degree=2, penalty=3.0, max_terms=15)
    
    def run_forward():
        forward_passer = ForwardPasser(earth_model)
        return forward_passer.run(
            X_fit_processed=X,
            y_fit=y,
            missing_mask=missing_mask,
            X_fit_original=X
        )
    
    result = benchmark(run_forward)
    basis_functions, coefficients = result
    assert len(basis_functions) >= 1


def test_pruning_passer_performance(benchmark):
    """Benchmark PruningPasser performance."""
    X, y = generate_benchmark_data(100, 4, 0.1)
    missing_mask = np.zeros_like(X, dtype=bool)
    
    # First run forward pass to get initial basis functions
    from pymars import Earth
    from pymars._forward import ForwardPasser
    from pymars._pruning import PruningPasser
    
    earth_model = Earth(max_degree=2, penalty=3.0, max_terms=15)
    forward_passer = ForwardPasser(earth_model)
    initial_bfs, initial_coefs = forward_passer.run(
        X_fit_processed=X,
        y_fit=y,
        missing_mask=missing_mask,
        X_fit_original=X
    )
    
    def run_pruning():
        pruning_passer = PruningPasser(earth_model)
        return pruning_passer.run(
            X_fit_processed=X,
            y_fit=y,
            missing_mask=missing_mask,
            initial_basis_functions=initial_bfs,
            initial_coefficients=initial_coefs,
            X_fit_original=X
        )
    
    result = benchmark(run_pruning)
    pruned_bfs, pruned_coefs, best_gcv = result
    assert len(pruned_bfs) >= 1


if __name__ == "__main__":
    # Run benchmarks directly if script is executed
    pytest.main([__file__, "-v", "--benchmark-only"])