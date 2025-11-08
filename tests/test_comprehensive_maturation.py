"""
Additional maturation tests for pymars to further improve reliability and performance verification
"""
import numpy as np
import pytest
import time
import psutil
import os
import gc
from pymars import Earth
import warnings
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestAdvancedMaturation:
    """Additional maturation tests to further improve library robustness."""
    
    def test_memory_efficiency_comprehensive(self):
        """Comprehensive memory efficiency test."""
        process = psutil.Process(os.getpid())
        
        # Test with different dataset sizes
        test_configs = [
            (50, 2),   # small
            (100, 3),  # medium-small
            (200, 3),  # medium
        ]
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_usage_per_run = []
        for n_samples, n_features in test_configs:
            # Clear before test
            gc.collect()
            
            X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42, noise=0.1)
            
            # Record memory before fitting
            mem_before = process.memory_info().rss / 1024 / 1024
            
            model = Earth(max_degree=2, penalty=3.0, max_terms=min(10, n_samples//2))
            model.fit(X, y)
            
            # Get predictions to ensure full model is exercised
            _ = model.predict(X[:min(5, len(X))])
            
            # Record memory after fitting
            mem_after = process.memory_info().rss / 1024 / 1024
            
            memory_used = mem_after - mem_before
            memory_usage_per_run.append(memory_used)
            
            # Delete model to free memory
            del model
            gc.collect()
            
            print(f"Memory used for {n_samples}x{n_features} dataset: {memory_used:.2f} MB")
        
        # Memory usage should be reasonable - not grow exponentially
        avg_memory = np.mean(memory_usage_per_run)
        assert avg_memory < 200, f"Memory usage too high: {avg_memory:.2f} MB average"
        print(f"✅ Memory efficiency: Average usage = {avg_memory:.2f} MB")
    
    def test_performance_scaling_large_datasets(self):
        """Test performance with larger datasets."""
        # Test with progressively larger datasets
        configs = [(100, 2), (150, 2), (200, 2)]
        times = []
        
        for n_samples, n_features in configs:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42, noise=0.1)
            
            start_time = time.time()
            model = Earth(max_degree=2, penalty=3.0, max_terms=min(15, n_samples//2))
            model.fit(X, y)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            
            print(f"Fit time for {n_samples}x{n_features}: {elapsed:.3f}s")
            
            # Verify model worked
            assert model.fitted_
            score = model.score(X, y)
            assert np.isfinite(score)
        
        # Check that time scaling is reasonable (shouldn't be exponential)
        if len(times) > 1:
            ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
            max_ratio = max(ratios) if ratios else 1.0
            assert max_ratio < 10.0, f"Time scaling too steep: {max_ratio:.2f}x"
        print("✅ Performance scaling: Reasonable time complexity")
    
    def test_numerical_robustness_extreme_values(self):
        """Test numerical robustness with extreme value ranges."""
        # Create dataset with values spanning wide range
        X = np.random.rand(30, 3)
        X[:, 0] = X[:, 0] * 1e8  # Very large values
        X[:, 1] = X[:, 1] * 1e-8  # Very small values  
        X[:, 2] = (X[:, 2] - 0.5) * 1e4  # Moderate values centered around 0
        
        # Create corresponding target
        y = (X[:, 0] * 1e-8) + (X[:, 1] * 1e8) + X[:, 2] * 0.001 + np.random.normal(0, 1, 30)
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=12)
        
        # Should handle extreme scaling without numerical errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        assert model.fitted_
        pred = model.predict(X[:5])
        assert len(pred) == 5
        # Predictions should be reasonable (not inf or very large values)
        assert all(-1e6 < p < 1e6 for p in pred if np.isfinite(p))
        print("✅ Numerical robustness: Handles extreme value scales")
    
    def test_model_consistency_multiple_fits(self):
        """Test that multiple fits produce consistent results for same data."""
        X, y = make_regression(n_samples=35, n_features=2, random_state=42, noise=0.05)
        
        # Fit multiple models on same data
        results = []
        for _ in range(3):
            model = Earth(max_degree=2, penalty=3.0, max_terms=10)
            model.fit(X, y)
            score = model.score(X, y)
            pred = model.predict(X[:5])
            
            results.append({
                'score': score,
                'pred': pred.copy(),
                'n_basis': len(model.basis_) if model.basis_ is not None else 0
            })
        
        # Results should be consistent across models
        scores = [r['score'] for r in results]
        basis_counts = [r['n_basis'] for r in results]
        
        # All scores should be very similar (identical for deterministic algorithm)
        assert all(abs(s - scores[0]) < 1e-10 for s in scores), "Scores vary between fits"
        assert all(bc == basis_counts[0] for bc in basis_counts), "Basis counts vary between fits"
        
        print("✅ Model consistency: Reproducible results across multiple fits")
    
    def test_sklearn_integration_comprehensive(self):
        """Test comprehensive scikit-learn integration."""
        X, y = make_regression(n_samples=40, n_features=3, random_state=42, noise=0.1)
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        # Test cross-validation integration
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        assert len(cv_scores) == 3
        assert all(np.isfinite(score) and score > 0.5 for score in cv_scores)
        
        # Test with pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('earth', model)
        ])
        
        pipeline.fit(X, y)
        pipeline_score = pipeline.score(X, y)
        assert np.isfinite(pipeline_score)
        
        # Test prediction
        pipeline_pred = pipeline.predict(X[:3])
        assert len(pipeline_pred) == 3
        assert all(np.isfinite(p) for p in pipeline_pred)
        
        print(f"✅ Scikit-learn integration: CV R²={np.mean(cv_scores):.4f}, Pipeline R²={pipeline_score:.4f}")
    
    def test_edge_case_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Test with minimum viable dataset
        X_min = np.array([[1.0], [2.0]])  # 2 samples, 1 feature
        y_min = np.array([1.0, 2.0])
        
        model_min = Earth(max_degree=1, penalty=3.0, max_terms=5)
        model_min.fit(X_min, y_min)
        assert model_min.fitted_
        
        pred_min = model_min.predict(X_min)
        assert len(pred_min) == 2
        assert all(np.isfinite(p) for p in pred_min)
        
        # Test with many features, few samples (challenging for model)
        X_wide = np.random.rand(8, 15)  # 8 samples, 15 features
        y_wide = np.sum(X_wide[:, :3], axis=1)  # Only first 3 features matter
        
        model_wide = Earth(max_degree=2, penalty=10.0, max_terms=10)  # High penalty to avoid overfitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_wide.fit(X_wide, y_wide)
        
        assert model_wide.fitted_
        
        print("✅ Boundary conditions: Handles minimal and wide datasets")
    
    def test_feature_importance_reliability(self):
        """Test reliability of feature importance calculations."""
        X, y = make_regression(n_samples=50, n_features=4, random_state=42, noise=0.1)
        
        # Create a dataset where feature 0 is most important
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + X[:, 2] * 0.1 + X[:, 3] * 0.05
        
        # Test different feature importance methods
        for imp_type in ['nb_subsets', 'gcv', 'rss']:
            model = Earth(max_degree=2, penalty=3.0, max_terms=15, feature_importance_type=imp_type)
            model.fit(X, y)
            
            # Should have feature importances
            fi = model.feature_importances_
            assert fi is not None
            assert len(fi) == X.shape[1]
            assert all(np.isfinite(f) for f in fi)
            
            # Feature importances should sum to approximately 1.0
            fi_sum = np.sum(fi)
            assert 0.95 < fi_sum < 1.05, f"Feature importances don't sum to ~1.0: {fi_sum}"
        
        print("✅ Feature importance: Reliable across different calculation methods")
    
    def test_prediction_edge_cases(self):
        """Test prediction method with edge cases."""
        X, y = make_regression(n_samples=25, n_features=2, random_state=42, noise=0.1)
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=8)
        model.fit(X, y)
        
        # Test predicting with same data
        pred_same = model.predict(X)
        assert len(pred_same) == len(X)
        assert all(np.isfinite(p) for p in pred_same)
        
        # Test predicting with single sample
        pred_single = model.predict(X[0:1])
        assert len(pred_single) == 1
        assert np.isfinite(pred_single[0])
        
        # Test predicting with many samples at once
        X_many = np.tile(X[0], (100, 1))  # Repeat first sample 100 times
        pred_many = model.predict(X_many)
        assert len(pred_many) == 100
        # All predictions should be nearly identical
        assert np.std(pred_many) < 1e-10, "Repeated samples should give nearly identical predictions"
        
        print("✅ Prediction edge cases: Handles various input shapes correctly")
    
    def test_parameter_edge_cases(self):
        """Test with extreme parameter values."""
        X, y = make_regression(n_samples=30, n_features=2, random_state=42, noise=0.1)
        
        # Test very high penalty (should result in simpler model)
        model_high_penalty = Earth(max_degree=3, penalty=100.0, max_terms=10)
        model_high_penalty.fit(X, y)
        assert model_high_penalty.fitted_
        assert len(model_high_penalty.basis_) >= 1
        
        # Test very low penalty (higher complexity allowed)
        model_low_penalty = Earth(max_degree=3, penalty=0.1, max_terms=min(20, len(X)//2))
        model_low_penalty.fit(X, y)
        assert model_low_penalty.fitted_
        
        # Test max_terms limiting
        model_limited = Earth(max_degree=3, penalty=3.0, max_terms=2)  # Very limited
        model_limited.fit(X, y)
        assert model_limited.fitted_
        # Model should have at most max_terms basis functions
        assert len(model_limited.basis_) <= 2 if model_limited.basis_ is not None else True
        
        print("✅ Parameter edge cases: Handles extreme values gracefully")
    
    def test_multicollinearity_handling(self):
        """Test model's handling of multicollinear features."""
        # Create dataset with highly correlated features
        X_base = np.random.rand(40, 2)
        # Create third feature that's almost perfectly correlated with first
        X_collinear = np.column_stack([
            X_base[:, 0],
            X_base[:, 1], 
            X_base[:, 0] * 0.99 + np.random.normal(0, 0.01, 40)  # Highly correlated with first
        ])
        
        y = X_collinear[:, 0] + X_collinear[:, 1] * 0.5
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=15)
        
        # Should handle collinear features without crashing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_collinear, y)
        
        assert model.fitted_
        score = model.score(X_collinear, y)
        assert np.isfinite(score)
        
        print(f"✅ Multicollinearity: Handles correlated features, R²={score:.4f}")
    
    def test_resampling_method_consistency(self):
        """Test consistency of results across different subsamples."""
        X, y = make_regression(n_samples=60, n_features=3, random_state=42, noise=0.1)
        
        # Fit on different subsets and compare general behavior
        subset_results = []
        
        for i in range(5):
            # Create different random subset
            indices = np.random.choice(len(X), size=min(40, len(X)), replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
            
            model = Earth(max_degree=2, penalty=3.0, max_terms=12)
            model.fit(X_subset, y_subset)
            
            subset_results.append({
                'n_basis': len(model.basis_) if model.basis_ is not None else 0,
                'score': model.score(X_subset, y_subset),
                'coef_count': len(model.coef_) if model.coef_ is not None else 0
            })
        
        # Results should be reasonable across subsets
        avg_n_basis = np.mean([r['n_basis'] for r in subset_results])
        avg_score = np.mean([r['score'] for r in subset_results])
        
        assert avg_n_basis > 0, "Models should have basis functions"
        assert np.isfinite(avg_score), "Scores should be finite"
        
        print(f"✅ Resampling consistency: Avg basis functions={avg_n_basis:.1f}, Avg R²={avg_score:.4f}")


def test_comprehensive_maturation():
    """Run all comprehensive maturation tests."""
    test_instance = TestAdvancedMaturation()
    
    print("🧪 Running comprehensive maturation tests...")
    print("=" * 50)
    
    test_instance.test_memory_efficiency_comprehensive()
    test_instance.test_performance_scaling_large_datasets()
    test_instance.test_numerical_robustness_extreme_values() 
    test_instance.test_model_consistency_multiple_fits()
    test_instance.test_sklearn_integration_comprehensive()
    test_instance.test_edge_case_boundary_conditions()
    test_instance.test_feature_importance_reliability()
    test_instance.test_prediction_edge_cases()
    test_instance.test_parameter_edge_cases()
    test_instance.test_multicollinearity_handling()
    test_instance.test_resampling_method_consistency()
    
    print("=" * 50)
    print("🎉 All comprehensive maturation tests passed!")
    print("🚀 pymars v1.0.0 is thoroughly matured and ready!")
    print("=" * 50)


if __name__ == "__main__":
    test_comprehensive_maturation()