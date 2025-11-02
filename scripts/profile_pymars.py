#!/usr/bin/env python
"""
Script to profile the pymars implementation and identify performance bottlenecks.
"""
import cProfile
import io
import pstats

import numpy as np

from pymars import Earth


def profile_earth_model():
    """Profile the Earth model with sample data."""
    # Generate sample data
    X = np.random.rand(100, 5)
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]  # Add some interactions

    # Create and fit model
    model = Earth(max_degree=2, penalty=3.0, max_terms=20)

    # Profile the fit method
    pr = cProfile.Profile()
    pr.enable()

    model.fit(X, y)

    pr.disable()

    # Create a stats object and print top functions
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')

    print("Profiling Earth model fitting...")
    print("=" * 50)
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())

    # Also profile prediction
    pr2 = cProfile.Profile()
    pr2.enable()

    predictions = model.predict(X)

    pr2.disable()

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=s2)
    ps2.sort_stats('cumulative')

    print("\\nProfiling Earth model prediction...")
    print("=" * 50)
    ps2.print_stats(10)  # Print top 10 functions
    print(s2.getvalue())

def profile_forward_pass():
    """Profile just the forward pass."""
    from pymars._forward import ForwardPasser
    from pymars.earth import Earth

    X = np.random.rand(100, 5)
    y = np.sin(X[:, 0]) + X[:, 1]

    earth_model = Earth(max_degree=2, penalty=3.0)

    # Profile forward passer
    pr = cProfile.Profile()
    pr.enable()

    forward_passer = ForwardPasser(earth_model)
    basis_functions, coefficients = forward_passer.run(
        X_fit_processed=X,
        y_fit=y,
        missing_mask=np.zeros_like(X, dtype=bool),
        X_fit_original=X
    )

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')

    print("Profiling Forward Pass...")
    print("=" * 50)
    ps.print_stats(15)
    print(s.getvalue())

def profile_pruning_pass():
    """Profile just the pruning pass."""
    from pymars._forward import ForwardPasser
    from pymars._pruning import PruningPasser
    from pymars.earth import Earth

    X = np.random.rand(100, 5)
    y = np.sin(X[:, 0]) + X[:, 1]

    earth_model = Earth(max_degree=2, penalty=3.0, max_terms=15)

    # First run forward pass to get basis functions
    forward_passer = ForwardPasser(earth_model)
    fwd_basis_functions, fwd_coefficients = forward_passer.run(
        X_fit_processed=X,
        y_fit=y,
        missing_mask=np.zeros_like(X, dtype=bool),
        X_fit_original=X
    )

    # Now profile pruning pass
    pr = cProfile.Profile()
    pr.enable()

    pruning_passer = PruningPasser(earth_model)
    pruned_bfs, pruned_coeffs, best_gcv = pruning_passer.run(
        X_fit_processed=X,
        y_fit=y,
        missing_mask=np.zeros_like(X, dtype=bool),
        initial_basis_functions=fwd_basis_functions,
        initial_coefficients=fwd_coefficients,
        X_fit_original=X
    )

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')

    print("Profiling Pruning Pass...")
    print("=" * 50)
    ps.print_stats(15)
    print(s.getvalue())

if __name__ == "__main__":
    print("Starting profiling of pymars components...")
    print("=" * 60)

    profile_earth_model()
    print()
    profile_forward_pass()
    print()
    profile_pruning_pass()

    print("\\nProfiling complete.")
