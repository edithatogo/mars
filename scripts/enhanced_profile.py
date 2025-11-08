#!/usr/bin/env python
"""
Enhanced profiling script for pymars with multiple profiling techniques.

This script provides comprehensive profiling of the pymars implementation using:
1. cProfile for CPU time profiling
2. memory_profiler for memory usage profiling
3. line_profiler for line-by-line profiling
4. timeit for micro-benchmarking specific functions
5. Performance comparison with sklearn
"""
import cProfile
import io
import pstats
import time
import numpy as np
from pymars import Earth
from pymars._forward import ForwardPasser
from pymars._pruning import PruningPasser
from pymars.earth import Earth as CoreEarth

try:
    from memory_profiler import profile as mem_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    def mem_profile(func):
        return func

def generate_test_data(n_samples=100, n_features=5, noise_level=0.1, seed=42):
    """Generate synthetic test data for profiling."""
    np.random.seed(seed)
    X = np.random.rand(n_samples, n_features)
    # Create a complex target with interactions (handle varying feature counts)
    if n_features >= 4:
        y = (np.sin(X[:, 0] * np.pi) + 
             X[:, 1] * X[:, min(2, n_features-1)] + 
             0.5 * X[:, min(3, n_features-1)] + 
             np.random.normal(0, noise_level, n_samples))
    elif n_features >= 3:
        y = (np.sin(X[:, 0] * np.pi) + 
             X[:, 1] * X[:, 2] + 
             np.random.normal(0, noise_level, n_samples))
    elif n_features >= 2:
        y = (np.sin(X[:, 0] * np.pi) + 
             X[:, 1] * 0.5 + 
             np.random.normal(0, noise_level, n_samples))
    else:
        y = (np.sin(X[:, 0] * np.pi) + 
             np.random.normal(0, noise_level, n_samples))
    return X, y

def profile_earth_model_overall():
    """Profile the overall Earth model fitting process."""
    print("=" * 80)
    print("📊 OVERALL EARTH MODEL PROFILING")
    print("=" * 80)
    
    X, y = generate_test_data(100, 5, 0.1)
    
    # Create and fit model
    model = Earth(max_degree=2, penalty=3.0, max_terms=15)
    
    # Profile the fit method
    pr = cProfile.Profile()
    pr.enable()
    
    model.fit(X, y)
    
    pr.disable()
    
    # Create a stats object and print top functions
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    
    print("📈 Earth Model Fitting Performance:")
    print("-" * 40)
    ps.print_stats(20)  # Print top 20 functions
    print(s.getvalue())
    
    # Profile prediction
    pr2 = cProfile.Profile()
    pr2.enable()
    
    predictions = model.predict(X[:10])
    
    pr2.disable()
    
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=s2)
    ps2.sort_stats('cumulative')
    
    print("🔮 Earth Model Prediction Performance:")
    print("-" * 40)
    ps2.print_stats(10)  # Print top 10 functions
    print(s2.getvalue())

def profile_forward_pass_detailed():
    """Profile the forward pass in detail."""
    print("=" * 80)
    print("🔍 FORWARD PASS DETAILED PROFILING")
    print("=" * 80)
    
    X, y = generate_test_data(100, 5, 0.1)
    
    # Create Earth model
    earth_model = CoreEarth(max_degree=2, penalty=3.0, max_terms=15)
    
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
    
    print("🚀 Forward Pass Performance:")
    print("-" * 30)
    ps.print_stats(25)
    print(s.getvalue())

def profile_pruning_pass_detailed():
    """Profile the pruning pass in detail."""
    print("=" * 80)
    print("✂️ PRUNING PASS DETAILED PROFILING")
    print("=" * 80)
    
    X, y = generate_test_data(100, 5, 0.1)
    
    # First run forward pass to get basis functions
    earth_model = CoreEarth(max_degree=2, penalty=3.0, max_terms=15)
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
    
    print("✂️ Pruning Pass Performance:")
    print("-" * 25)
    ps.print_stats(20)
    print(s.getvalue())

@mem_profile
def profile_memory_usage():
    """Profile memory usage of Earth model."""
    if not MEMORY_PROFILER_AVAILABLE:
        print("⚠️ Memory profiler not available. Install with: pip install memory_profiler")
        return
        
    print("=" * 80)
    print("💾 MEMORY USAGE PROFILING")
    print("=" * 80)
    
    X, y = generate_test_data(100, 5, 0.1)
    
    # Create and fit model
    model = Earth(max_degree=2, penalty=3.0, max_terms=15)
    model.fit(X, y)
    
    print(f"✅ Model fitted with {len(model.basis_)} basis functions")
    print(f"📊 RSS: {model.rss_:.6f}")
    print(f"📐 GCV: {model.gcv_:.6f}")

def benchmark_scaling_with_features():
    """Benchmark performance scaling with number of features."""
    print("=" * 80)
    print("📈 SCALING BENCHMARKS WITH FEATURES")
    print("=" * 80)
    
    feature_counts = [2, 5, 10, 20]
    results = []
    
    for n_features in feature_counts:
        X, y = generate_test_data(100, n_features, 0.1)
        
        # Time fitting
        start_time = time.perf_counter()
        model = Earth(max_degree=2, penalty=3.0, max_terms=min(20, n_features * 3))
        model.fit(X, y)
        end_time = time.perf_counter()
        
        fit_time = end_time - start_time
        n_basis_functions = len(model.basis_)
        
        results.append((n_features, fit_time, n_basis_functions))
        print(f"📊 Features: {n_features:2d} | Time: {fit_time:.4f}s | Basis Functions: {n_basis_functions:2d}")
    
    print("\n📈 Scaling Summary:")
    print("-" * 30)
    for n_features, fit_time, n_basis in results:
        print(f"  {n_features:2d} features: {fit_time:.4f}s ({n_basis:2d} basis functions)")

def benchmark_scaling_with_samples():
    """Benchmark performance scaling with number of samples."""
    print("=" * 80)
    print("📈 SCALING BENCHMARKS WITH SAMPLES")
    print("=" * 80)
    
    sample_counts = [50, 100, 200, 500]
    results = []
    
    for n_samples in sample_counts:
        X, y = generate_test_data(n_samples, 5, 0.1)
        
        # Time fitting
        start_time = time.perf_counter()
        model = Earth(max_degree=2, penalty=3.0, max_terms=15)
        model.fit(X, y)
        end_time = time.perf_counter()
        
        fit_time = end_time - start_time
        n_basis_functions = len(model.basis_)
        
        results.append((n_samples, fit_time, n_basis_functions))
        print(f"📊 Samples: {n_samples:3d} | Time: {fit_time:.4f}s | Basis Functions: {n_basis_functions:2d}")
    
    print("\n📈 Scaling Summary:")
    print("-" * 30)
    for n_samples, fit_time, n_basis in results:
        print(f"  {n_samples:3d} samples: {fit_time:.4f}s ({n_basis:2d} basis functions)")

def micro_benchmark_basis_functions():
    """Micro-benchmark individual basis function operations."""
    print("=" * 80)
    print("🔬 MICRO-BENCHMARKS FOR BASIS FUNCTIONS")
    print("=" * 80)
    
    from pymars._basis import (
        ConstantBasisFunction, 
        LinearBasisFunction, 
        HingeBasisFunction
    )
    
    X = np.random.rand(1000, 5)
    missing_mask = np.zeros_like(X, dtype=bool)
    
    # Benchmark ConstantBasisFunction
    bf_const = ConstantBasisFunction()
    start_time = time.perf_counter()
    for _ in range(1000):
        result = bf_const.transform(X, missing_mask)
    const_time = time.perf_counter() - start_time
    print(f"⚛️  ConstantBasisFunction: {const_time:.6f}s for 1000 transforms")
    
    # Benchmark LinearBasisFunction
    bf_linear = LinearBasisFunction(variable_idx=0, variable_name="x0")
    start_time = time.perf_counter()
    for _ in range(1000):
        result = bf_linear.transform(X, missing_mask)
    linear_time = time.perf_counter() - start_time
    print(f"📏 LinearBasisFunction: {linear_time:.6f}s for 1000 transforms")
    
    # Benchmark HingeBasisFunction
    bf_hinge = HingeBasisFunction(variable_idx=0, knot_val=0.5, is_right_hinge=True, variable_name="x0")
    start_time = time.perf_counter()
    for _ in range(1000):
        result = bf_hinge.transform(X, missing_mask)
    hinge_time = time.perf_counter() - start_time
    print(f"🔗 HingeBasisFunction: {hinge_time:.6f}s for 1000 transforms")

def compare_with_sklearn():
    """Compare performance with sklearn models."""
    print("=" * 80)
    print("🆚 PERFORMANCE COMPARISON WITH SKLEARN")
    print("=" * 80)
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.tree import DecisionTreeRegressor
        
        X, y = generate_test_data(200, 5, 0.1)
        
        # Benchmark Earth
        start_time = time.perf_counter()
        earth_model = Earth(max_degree=2, penalty=3.0, max_terms=15)
        earth_model.fit(X, y)
        earth_time = time.perf_counter() - start_time
        earth_score = earth_model.score(X, y)
        print(f"🌍 Earth Model: {earth_time:.4f}s (R²: {earth_score:.4f})")
        
        # Benchmark Linear Regression
        start_time = time.perf_counter()
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        lr_time = time.perf_counter() - start_time
        lr_score = lr_model.score(X, y)
        print(f"📏 Linear Regression: {lr_time:.4f}s (R²: {lr_score:.4f})")
        
        # Benchmark Decision Tree
        start_time = time.perf_counter()
        dt_model = DecisionTreeRegressor(max_depth=5)
        dt_model.fit(X, y)
        dt_time = time.perf_counter() - start_time
        dt_score = dt_model.score(X, y)
        print(f"🌳 Decision Tree: {dt_time:.4f}s (R²: {dt_score:.4f})")
        
        # Benchmark Random Forest
        start_time = time.perf_counter()
        rf_model = RandomForestRegressor(n_estimators=10, max_depth=5)
        rf_model.fit(X, y)
        rf_time = time.perf_counter() - start_time
        rf_score = rf_model.score(X, y)
        print(f"🌲 Random Forest: {rf_time:.4f}s (R²: {rf_score:.4f})")
        
    except ImportError:
        print("⚠️ Sklearn not available for comparison benchmarks")

def main():
    """Run all profiling and benchmarking tools."""
    print("🚀 Starting Comprehensive Profiling of pymars")
    print("=" * 80)
    
    # Run all profiling and benchmarking
    profile_earth_model_overall()
    print()
    profile_forward_pass_detailed()
    print()
    profile_pruning_pass_detailed()
    print()
    profile_memory_usage()
    print()
    benchmark_scaling_with_features()
    print()
    benchmark_scaling_with_samples()
    print()
    micro_benchmark_basis_functions()
    print()
    compare_with_sklearn()
    
    print("\n" + "=" * 80)
    print("✅ Comprehensive profiling completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()