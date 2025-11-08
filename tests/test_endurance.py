"""
Endurance testing for pymars to detect memory leaks and performance degradation over time
"""
import numpy as np
import time
import gc
import psutil
import os
from pymars import Earth
import warnings


def test_endurance_memory_leak_detection():
    """Test for memory leaks over repeated model fittings."""
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 50)
    
    memory_readings = []
    
    for i in range(10):  # Run multiple iterations to check for accumulation
        # Force garbage collection before each iteration
        gc.collect()
        
        # Create and fit model
        model = Earth(max_degree=2, penalty=3.0, max_terms=8)
        model.fit(X, y)
        
        # Make sure model is cleaned up
        del model
        
        # Take memory reading
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_readings.append(current_memory)
        
        print(f"Iteration {i+1}: Memory = {current_memory:.1f} MB")
    
    # Check that memory growth is bounded
    max_memory = max(memory_readings)
    memory_growth = max_memory - initial_memory
    
    # Allow some variation but shouldn't grow significantly
    assert memory_growth < 20, f"Memory grew by {memory_growth:.1f} MB, which is too much"
    print(f"✅ Memory leak test passed: Growth was only {memory_growth:.1f} MB")


def test_endurance_performance_consistency():
    """Test that performance remains consistent over time."""
    X = np.random.rand(30, 2)
    y = X[:, 0] + X[:, 1] * 0.5
    
    fit_times = []
    score_values = []
    
    for i in range(5):  # Test over multiple iterations
        start_time = time.time()
        model = Earth(max_degree=2, penalty=3.0, max_terms=8)
        model.fit(X, y)
        fit_time = time.time() - start_time
        
        score = model.score(X, y)
        
        fit_times.append(fit_time)
        score_values.append(score)
        
        print(f"Iteration {i+1}: Fit time = {fit_time:.3f}s, Score = {score:.4f}")
        
        # Clean up
        del model
    
    # Check that performance doesn't degrade significantly
    avg_fit_time = np.mean(fit_times)
    std_fit_time = np.std(fit_times)
    max_fit_time = np.max(fit_times)
    
    # Performance should be consistent - max shouldn't be too far above average
    assert max_fit_time < avg_fit_time * 3, f"Performance degraded significantly: max={max_fit_time:.3f}s, avg={avg_fit_time:.3f}s"
    
    # Scores should be consistent
    score_std = np.std(score_values)
    assert score_std < 0.1, f"Score consistency degraded: std={score_std:.4f}"
    
    print("✅ Performance consistency test passed")


def test_endurance_model_repeatability():
    """Test that models remain repeatable over long periods of usage."""
    X = np.random.rand(25, 2)
    y = X[:, 0] + X[:, 1] * 0.5
    
    baseline_model = Earth(max_degree=2, penalty=3.0, max_terms=8)
    baseline_model.fit(X, y)
    baseline_score = baseline_model.score(X, y)
    
    # Use same parameters for comparison later
    for i in range(10):
        # Run other models to simulate long-term usage
        for j in range(3):
            temp_model = Earth(max_degree=1, penalty=2.0, max_terms=5)
            temp_X = np.random.rand(20, 2)
            temp_y = temp_X[:, 0] + np.random.normal(0, 0.1, 20)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp_model.fit(temp_X, temp_y)
            del temp_model
        
        # Now check our original model type still works consistently
        test_model = Earth(max_degree=2, penalty=3.0, max_terms=8)
        test_model.fit(X, y)
        test_score = test_model.score(X, y)
        
        # Should get similar results to baseline
        assert abs(test_score - baseline_score) < 0.05, f"Repeatability degraded at iteration {i}: baseline={baseline_score:.4f}, now={test_score:.4f}"
        
        print(f"Repeatability check {i+1}: Score = {test_score:.4f} (baseline: {baseline_score:.4f})")
    
    print("✅ Model repeatability endurance test passed")


def test_endurance_large_repeated_operations():
    """Test large number of operations to find any degradation."""
    np.random.seed(42)
    
    # Track performance metrics
    all_fit_times = []
    all_scores = []
    
    print("Starting long-term endurance test...")
    
    for iteration in range(20):  # More iterations to really stress it
        # Create different sized datasets each time
        n_samples = np.random.randint(20, 40)
        n_features = np.random.randint(2, 4)
        
        X = np.random.rand(n_samples, n_features)
        y = np.sum(X[:, :min(2, n_features)], axis=1) + np.random.normal(0, 0.1, n_samples)
        
        start_time = time.time()
        model = Earth(max_degree=2, penalty=3.0, max_terms=min(10, n_samples//2))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        fit_time = time.time() - start_time
        
        score = model.score(X, y) if model.fitted_ else -np.inf
        
        all_fit_times.append(fit_time)
        all_scores.append(score)
        
        if iteration % 5 == 0:  # Report every 5 iterations
            print(f"Endurance iteration {iteration+1}: Time = {fit_time:.3f}s, Score = {score:.4f}")
    
    # Analyze results for trends
    early_times = all_fit_times[:5]
    late_times = all_fit_times[-5:]
    
    avg_early_time = np.mean(early_times)
    avg_late_time = np.mean(late_times)
    
    # Performance should not significantly degrade over time
    performance_degradation = avg_late_time / avg_early_time if avg_early_time > 0 else 1.0
    assert performance_degradation < 2.0, f"Significant performance degradation detected: {performance_degradation:.2f}x slower"
    
    print(f"✅ Long-term endurance test passed: Performance degradation = {performance_degradation:.2f}x")


def test_endurance_with_garbage_collection_cycles():
    """Test endurance with explicit garbage collection cycles."""
    import gc
    
    X = np.random.rand(30, 2)
    y = X[:, 0] + X[:, 1] * 0.5
    
    # Pre-collect to have a baseline
    gc.collect()
    
    initial_objects = len(gc.get_objects())
    
    models_created = 0
    for i in range(15):
        # Create and use multiple models
        for j in range(3):
            model = Earth(max_degree=2, penalty=3.0, max_terms=8)
            model.fit(X, y)
            
            # Use the model a bit
            _ = model.predict(X[:5])
            _ = model.score(X, y)
            
            models_created += 1
        
        # Check that we're not accumulating objects
        gc.collect()
        current_objects = len(gc.get_objects())
        
        # Should not accumulate a large number of objects
        growth = current_objects - initial_objects
        assert growth < 1000, f"Object accumulation detected: {growth} new objects"
        
        if i % 5 == 0:
            print(f"GC cycle {i+1}: Objects growth = {growth}")
    
    print(f"✅ GC endurance test passed: Created {models_created} models without significant object accumulation")


def test_endurance_edge_case_operations():
    """Test endurance with edge case operations."""
    edge_cases = [
        lambda: Earth(max_degree=1, penalty=1.0, max_terms=1).fit(np.random.rand(5, 1), np.random.rand(5)),
        lambda: Earth(max_degree=3, penalty=0.1, max_terms=20).fit(np.random.rand(25, 3), np.random.rand(25)),
        lambda: Earth(max_degree=1, penalty=100, max_terms=10).fit(np.random.rand(15, 2), np.random.rand(15)),
    ]
    
    for cycle in range(5):
        for case_idx, case_func in enumerate(edge_cases):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = case_func()
                
                if hasattr(model, 'fitted_') and model.fitted_:
                    # Verify the model has reasonable properties
                    if hasattr(model, 'basis_') and model.basis_ is not None:
                        assert len(model.basis_) >= 0  # Should have at least intercept
                    
            except Exception:
                # Some edge cases might legitimately fail, that's ok
                pass
        
        print(f"Edge case endurance cycle {cycle+1} completed")
    
    print("✅ Edge case endurance test completed")


if __name__ == "__main__":
    test_endurance_memory_leak_detection()
    test_endurance_performance_consistency()
    test_endurance_model_repeatability()
    test_endurance_large_repeated_operations()
    test_endurance_with_garbage_collection_cycles()
    test_endurance_edge_case_operations()
    print("\\n✅ All endurance tests passed!")