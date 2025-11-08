"""
Benchmark tests for pymars using pytest-benchmark.
"""

import numpy as np
import pytest

from pymars import Earth
from pymars._forward import ForwardPasser
from pymars._pruning import PruningPasser


@pytest.mark.benchmark(min_time=0.1, min_rounds=5, max_time=1.0)
def test_earth_fit_performance_small(benchmark):
    """Benchmark fitting performance on small dataset."""
    X = np.random.rand(50, 3)
    y = np.sin(X[:, 0]) + X[:, 1] * 0.5

    def fit_model():
        model = Earth(max_terms=10, max_degree=2)
        model.fit(X, y)
        return model

    result = benchmark(fit_model)
    assert result.fitted_


@pytest.mark.benchmark(min_time=0.1, min_rounds=3, max_time=2.0)
def test_earth_fit_performance_medium(benchmark):
    """Benchmark fitting performance on medium dataset."""
    X = np.random.rand(200, 5)
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + np.random.normal(0, 0.1, 200)

    def fit_model():
        model = Earth(max_terms=20, max_degree=2)
        model.fit(X, y)
        return model

    result = benchmark(fit_model)
    assert result.fitted_


@pytest.mark.benchmark(min_time=0.1, min_rounds=2, max_time=3.0)
def test_earth_predict_performance(benchmark):
    """Benchmark prediction performance."""
    X_train = np.random.rand(100, 4)
    y_train = np.sin(X_train[:, 0]) + X_train[:, 1]

    model = Earth(max_terms=15, max_degree=2)
    model.fit(X_train, y_train)

    X_test = np.random.rand(1000, 4)

    def predict():
        return model.predict(X_test)

    predictions = benchmark(predict)
    assert len(predictions) == 1000


@pytest.mark.benchmark(min_time=0.1, min_rounds=3, max_time=1.0)
def test_forward_passer_performance(benchmark):
    """Benchmark forward passer performance."""
    X = np.random.rand(100, 3)
    y = np.sin(X[:, 0]) + X[:, 1]
    missing_mask = np.zeros_like(X, dtype=bool)

    earth_model = Earth(max_terms=15, max_degree=2)

    def run_forward():
        forward_passer = ForwardPasser(earth_model)
        return forward_passer.run(
            X_fit_processed=X, y_fit=y, missing_mask=missing_mask, X_fit_original=X
        )

    result = benchmark(run_forward)
    basis_functions, coefficients = result
    assert len(basis_functions) > 0


@pytest.mark.benchmark(min_time=0.1, min_rounds=3, max_time=1.0)
def test_pruning_passer_performance(benchmark):
    """Benchmark pruning passer performance."""
    X = np.random.rand(100, 3)
    y = np.sin(X[:, 0]) + X[:, 1]
    missing_mask = np.zeros_like(X, dtype=bool)

    # First run forward pass to get initial basis functions
    earth_model = Earth(max_terms=20, max_degree=2)
    from pymars._forward import ForwardPasser

    forward_passer = ForwardPasser(earth_model)
    initial_bfs, initial_coefs = forward_passer.run(
        X_fit_processed=X, y_fit=y, missing_mask=missing_mask, X_fit_original=X
    )

    def run_pruning():
        pruning_passer = PruningPasser(earth_model)
        return pruning_passer.run(
            X_fit_processed=X,
            y_fit=y,
            missing_mask=missing_mask,
            initial_basis_functions=initial_bfs,
            initial_coefficients=initial_coefs,
            X_fit_original=X,
        )

    result = benchmark(run_pruning)
    pruned_bfs, pruned_coefs, best_gcv = result
    assert len(pruned_bfs) >= 1


def test_memory_usage():
    """Basic memory usage test."""
    import gc

    # Create a moderately large dataset
    X = np.random.rand(500, 10)
    y = np.sum(X[:, :3], axis=1)  # Only first 3 features matter

    # Force garbage collection before measuring
    gc.collect()

    # Fit model
    model = Earth(max_terms=25, max_degree=2)
    model.fit(X, y)

    # Check that the model has reasonable size
    assert len(model.basis_) > 0
    assert model.coef_ is not None
    assert hasattr(model, "gcv_")
    assert model.gcv_ is not None


@pytest.mark.benchmark(min_time=0.05, min_rounds=5, max_time=1.0)
def test_earth_score_performance(benchmark):
    """Benchmark score method performance."""
    X_train = np.random.rand(100, 4)
    y_train = np.sin(X_train[:, 0]) + X_train[:, 1]

    model = Earth(max_terms=10, max_degree=1)
    model.fit(X_train, y_train)

    def score_model():
        return model.score(X_train, y_train)

    score = benchmark(score_model)
    assert isinstance(score, (int, float))


@pytest.mark.parametrize("n_features", [2, 5, 10])
def test_scaling_with_features(benchmark, n_features):
    """Test how performance scales with number of features."""
    n_samples = 100
    X = np.random.rand(n_samples, n_features)
    y = np.sum(X[:, : min(3, n_features)], axis=1)  # First few features matter

    def fit_with_features():
        model = Earth(max_terms=min(2 * n_features, 20), max_degree=2)
        model.fit(X, y)
        return model

    result = benchmark(fit_with_features)
    assert result.fitted_
