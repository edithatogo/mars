#!/usr/bin/env python
"""
Line profiling script for pymars using line_profiler.

This script profiles individual lines of key pymars functions to identify optimization opportunities.
"""
import numpy as np
from line_profiler import LineProfiler
from pymars import Earth
from pymars._forward import ForwardPasser
from pymars._pruning import PruningPasser
from pymars.earth import Earth as CoreEarth


def generate_test_data(n_samples=100, n_features=5, noise_level=0.1, seed=42):
    """Generate synthetic test data for profiling."""
    np.random.seed(seed)
    X = np.random.rand(n_samples, n_features)
    # Create a complex target with interactions
    y = (np.sin(X[:, 0] * np.pi) + 
         X[:, 1] * X[:, 2] + 
         0.5 * X[:, 3] + 
         np.random.normal(0, noise_level, n_samples))
    return X, y


def line_profile_earth_fit():
    """Line profile the Earth.fit method."""
    print("🔧 Line profiling Earth.fit method...")
    
    X, y = generate_test_data(50, 3, 0.1)
    
    # Create Earth model
    model = CoreEarth(max_degree=2, penalty=3.0, max_terms=10)
    
    # Create line profiler
    profiler = LineProfiler()
    profiler.add_function(model.fit)
    
    # Wrap and run
    wrapped_fit = profiler(model.fit)
    wrapped_fit(X, y)
    
    # Print results
    profiler.print_stats()


def line_profile_forward_passer_run():
    """Line profile the ForwardPasser.run method."""
    print("🔧 Line profiling ForwardPasser.run method...")
    
    X, y = generate_test_data(50, 3, 0.1)
    
    # Create Earth model and ForwardPasser
    earth_model = CoreEarth(max_degree=2, penalty=3.0, max_terms=10)
    forward_passer = ForwardPasser(earth_model)
    
    # Create line profiler
    profiler = LineProfiler()
    profiler.add_function(forward_passer.run)
    
    # Wrap and run
    wrapped_run = profiler(forward_passer.run)
    wrapped_run(
        X_fit_processed=X,
        y_fit=y,
        missing_mask=np.zeros_like(X, dtype=bool),
        X_fit_original=X
    )
    
    # Print results
    profiler.print_stats()


def line_profile_pruning_passer_run():
    """Line profile the PruningPasser.run method."""
    print("🔧 Line profiling PruningPasser.run method...")
    
    X, y = generate_test_data(50, 3, 0.1)
    
    # First run forward pass
    earth_model = CoreEarth(max_degree=2, penalty=3.0, max_terms=10)
    forward_passer = ForwardPasser(earth_model)
    initial_bfs, initial_coeffs = forward_passer.run(
        X_fit_processed=X,
        y_fit=y,
        missing_mask=np.zeros_like(X, dtype=bool),
        X_fit_original=X
    )
    
    # Create PruningPasser
    pruning_passer = PruningPasser(earth_model)
    
    # Create line profiler
    profiler = LineProfiler()
    profiler.add_function(pruning_passer.run)
    
    # Wrap and run
    wrapped_run = profiler(pruning_passer.run)
    wrapped_run(
        X_fit_processed=X,
        y_fit=y,
        missing_mask=np.zeros_like(X, dtype=bool),
        initial_basis_functions=initial_bfs,
        initial_coefficients=initial_coeffs,
        X_fit_original=X
    )
    
    # Print results
    profiler.print_stats()


def line_profile_basis_transform():
    """Line profile the basis function transform methods."""
    print("🔧 Line profiling basis function transform methods...")
    
    from pymars._basis import (
        ConstantBasisFunction,
        LinearBasisFunction,
        HingeBasisFunction
    )
    
    X, y = generate_test_data(100, 3, 0.1)
    missing_mask = np.zeros_like(X, dtype=bool)
    
    # Create basis functions
    bf_const = ConstantBasisFunction()
    bf_linear = LinearBasisFunction(variable_idx=0, variable_name="x0")
    bf_hinge = HingeBasisFunction(variable_idx=1, knot_val=0.5, is_right_hinge=True, variable_name="x1")
    
    # Create line profiler
    profiler = LineProfiler()
    profiler.add_function(bf_const.transform)
    profiler.add_function(bf_linear.transform)
    profiler.add_function(bf_hinge.transform)
    
    # Profile each transform
    wrapped_const_transform = profiler(bf_const.transform)
    wrapped_linear_transform = profiler(bf_linear.transform)
    wrapped_hinge_transform = profiler(bf_hinge.transform)
    
    # Run transforms
    result_const = wrapped_const_transform(X, missing_mask)
    result_linear = wrapped_linear_transform(X, missing_mask)
    result_hinge = wrapped_hinge_transform(X, missing_mask)
    
    # Print results
    profiler.print_stats()


def line_profile_build_basis_matrix():
    """Line profile the _build_basis_matrix method."""
    print("🔧 Line profiling _build_basis_matrix method...")
    
    from pymars._forward import ForwardPasser
    
    X, y = generate_test_data(50, 3, 0.1)
    
    # Create Earth model and ForwardPasser
    earth_model = CoreEarth(max_degree=2, penalty=3.0, max_terms=10)
    forward_passer = ForwardPasser(earth_model)
    
    # Create some basis functions
    from pymars._basis import (
        ConstantBasisFunction,
        LinearBasisFunction,
        HingeBasisFunction
    )
    
    bf_const = ConstantBasisFunction()
    bf_linear = LinearBasisFunction(variable_idx=0, variable_name="x0")
    bf_hinge1 = HingeBasisFunction(variable_idx=1, knot_val=0.3, is_right_hinge=True, variable_name="x1")
    bf_hinge2 = HingeBasisFunction(variable_idx=2, knot_val=0.7, is_right_hinge=False, variable_name="x2")
    
    basis_functions = [bf_const, bf_linear, bf_hinge1, bf_hinge2]
    
    # Create line profiler
    profiler = LineProfiler()
    profiler.add_function(forward_passer._build_basis_matrix)
    
    # Wrap and run
    wrapped_build = profiler(forward_passer._build_basis_matrix)
    result = wrapped_build(X, basis_functions, missing_mask=np.zeros_like(X, dtype=bool))
    
    # Print results
    profiler.print_stats()


def main():
    """Run all line profiling tests."""
    print("=" * 80)
    print("📏 LINE PROFILING FOR PYMARS")
    print("=" * 80)
    
    try:
        line_profile_earth_fit()
        print("\n" + "-" * 80)
        
        line_profile_forward_passer_run()
        print("\n" + "-" * 80)
        
        line_profile_pruning_passer_run()
        print("\n" + "-" * 80)
        
        line_profile_basis_transform()
        print("\n" + "-" * 80)
        
        line_profile_build_basis_matrix()
        print("\n" + "-" * 80)
        
    except Exception as e:
        print(f"❌ Error during line profiling: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("📏 Line profiling completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()