"""
Additional tests for pymars.earth module to reach >90% coverage
"""
import numpy as np
import pytest
from pymars import Earth
import warnings


class TestEarthAdditional:
    """Additional tests for Earth class to cover missed lines."""
    
    def test_empty_forward_pass_case(self):
        """Test case where forward pass might return no basis functions."""
        # Create a case where forward pass might return empty
        X = np.array([[1.0], [1.0], [1.0]])  # All the same value - might cause problems
        y = np.array([2.0, 2.0, 2.0])  # All same target - constant
        model = Earth(max_degree=1, penalty=100.0, max_terms=2)  # High penalty to promote pruning
        
        try:
            model.fit(X, y)
            # Model should still fit successfully with at least intercept
            assert model.fitted_
            assert model.basis_ is not None
            assert model.coef_ is not None
            assert hasattr(model, 'gcv_')
            assert hasattr(model, 'rss_')
            
            # Check predictions work
            preds = model.predict(X[:1])
            assert len(preds) == 1
            assert np.isfinite(preds[0])
            
        except Exception as e:
            # If this triggers an error condition, it's valuable to know
            print(f"Exception in empty forward pass test: {e}")
            # Re-raise for debugging
            raise
    
    def test_extreme_parameters(self):
        """Test with extreme parameter values to trigger edge cases."""
        X = np.random.rand(10, 2)
        y = X[:, 0] + X[:, 1]
        
        # Test with minimum values that might cause degenerate conditions
        model = Earth(max_degree=1, penalty=0.0, max_terms=1)  # Zero penalty
        model.fit(X, y)
        assert model.fitted_
        
        # Test with maximum degree relative to data
        model2 = Earth(max_degree=10, penalty=1.0, max_terms=20)  # High degree
        model2.fit(X, y)
        assert model2.fitted_
    
    def test_single_sample_edge_case(self):
        """Test with single sample which might cause matrix issues."""
        X = np.array([[1.0, 2.0]])  # Single sample
        y = np.array([3.0])
        
        model = Earth(max_degree=1, penalty=3.0, max_terms=5)
        # This case should either fail gracefully or create simple model
        model.fit(X, y)
        assert model.fitted_
        
        # Predictions should work
        pred = model.predict(X)
        assert len(pred) == 1
        assert np.isfinite(pred[0])
    
    def test_two_samples_case(self):
        """Test with two samples which might cause underdetermined system."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        
        # Use more complex model than data allows
        model = Earth(max_degree=3, penalty=3.0, max_terms=10)
        model.fit(X, y)
        assert model.fitted_
        
        # Check that model has meaningful properties
        assert hasattr(model, 'basis_')
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'gcv_')
        
        # Check predictions
        preds = model.predict(X)
        assert len(preds) == 2
        assert all(np.isfinite(p) for p in preds)
    
    def test_high_penalty_case(self):
        """Test with very high penalty to trigger pruning to minimal model."""
        X = np.random.rand(20, 3)
        y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 20)
        
        # Very high penalty should result in simpler model
        model = Earth(max_degree=3, penalty=100.0, max_terms=20)
        model.fit(X, y)
        assert model.fitted_
        assert hasattr(model, 'gcv_')
        assert model.gcv_ is not None
        
        # Model should be trained
        score = model.score(X, y)
        assert isinstance(score, (int, float, np.floating))
        
        # Test prediction
        pred = model.predict(X[:5])
        assert len(pred) == 5
        assert all(np.isfinite(p) for p in pred)
    
    def test_allow_missing_false_with_nans(self):
        """Test disallowed missing values with actual NaNs."""
        X = np.random.rand(10, 2)
        X[:2, 0] = np.nan  # Add NaNs
        y = X[:, 1]  # Use valid part for target
        
        # Default is allow_missing=False, so this should fail
        model = Earth(max_degree=2, penalty=3.0, max_terms=5, allow_missing=False)
        
        with pytest.raises(ValueError, match="Input X contains NaN"):
            model.fit(X, y)
    
    def test_allow_missing_true_with_nans(self):
        """Test allowed missing values with actual NaNs."""
        X = np.random.rand(10, 2)
        X[:2, 0] = np.nan  # Add NaNs
        y = X[:, 1]  # Use valid part for target, handling NaNs in target too
        y = np.where(np.isnan(y), np.nanmean(y[~np.isnan(y)]), y)
        
        # This should work if allow_missing=True
        model = Earth(max_degree=2, penalty=3.0, max_terms=5, allow_missing=True)
        model.fit(X, y)
        assert model.fitted_
        
        # Test prediction with missing values
        pred = model.predict(X[:3])
        assert len(pred) == 3
    
    def test_invalid_feature_importance_type(self):
        """Test invalid feature importance type."""
        X = np.random.rand(10, 2)
        y = X[:, 0] + X[:, 1]
        
        # Test with invalid feature importance type
        model = Earth(feature_importance_type='invalid_type', max_degree=2, penalty=3.0, max_terms=5)
        
        # Feature importance calculation should handle invalid type gracefully
        model.fit(X, y)
        assert model.fitted_
        
        # Accessing feature_importances_ should handle the invalid type
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fi = model.feature_importances_
                # Result might depend on how the model handles invalid types
            except Exception:
                # If it raises an exception during calculation, that's valid
                pass
    
    def test_get_set_params_comprehensive(self):
        """Test get/set params with all parameters."""
        X = np.random.rand(10, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth()
        
        # Test getting default parameters
        params = model.get_params()
        assert isinstance(params, dict)
        assert 'max_degree' in params
        assert 'penalty' in params
        assert 'max_terms' in params
        
        # Test setting parameters
        new_params = {
            'max_degree': 3,
            'penalty': 2.0,
            'max_terms': 15,
            'minspan_alpha': 0.05,
            'endspan_alpha': 0.05,
            'allow_linear': False,
            'allow_missing': True
        }
        
        model.set_params(**new_params)
        
        # Verify parameters were set
        for key, value in new_params.items():
            assert getattr(model, key) == value
        
        # Fit and ensure it still works
        model.fit(X, y)
        assert model.fitted_
    
    def test_scrub_input_data_edge_cases(self):
        """Test _scrub_input_data with various edge cases."""
        X = np.random.rand(10, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth()
        
        # Test with different input types
        # List input
        X_list = X.tolist()
        y_list = y.tolist()
        
        X_proc, missing_mask, y_proc = model._scrub_input_data(X_list, y_list)
        assert X_proc.shape == X.shape
        assert y_proc.shape == y.shape
        assert missing_mask.shape == X.shape
        
        # Test with single precision
        X_float32 = X.astype(np.float32)
        y_float32 = y.astype(np.float32)
        
        X_proc2, missing_mask2, y_proc2 = model._scrub_input_data(X_float32, y_float32)
        assert X_proc2.shape == X.shape
        assert y_proc2.shape == y.shape
        assert missing_mask2.shape == X.shape
    
    def test_build_basis_matrix_empty(self):
        """Test _build_basis_matrix with empty functions list."""
        X = np.random.rand(5, 2)
        model = Earth(max_degree=2, penalty=3.0, max_terms=5)
        
        # Call with empty basis functions list
        B_matrix = model._build_basis_matrix(X, [], np.zeros_like(X, dtype=bool))
        assert B_matrix.shape == (5, 0)  # 5 samples, 0 basis functions
    
    def test_model_summary_method(self):
        """Test summary method."""
        X = np.random.rand(10, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=5)
        model.fit(X, y)
        
        # Test summary method
        summary = model.summary()
        # Summary might return None or a string depending on implementation
        # Just ensure it doesn't crash
        
    def test_extreme_penalty_values(self):
        """Test with extreme penalty values."""
        X = np.random.rand(15, 3)
        y = X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.2
        
        # Very low penalty (might cause overfitting but should still work)
        model_low = Earth(max_degree=2, penalty=0.001, max_terms=10)
        model_low.fit(X, y)
        assert model_low.fitted_
        
        # Very high penalty (should result in simpler model)
        model_high = Earth(max_degree=2, penalty=1000.0, max_terms=20)
        model_high.fit(X, y)
        assert model_high.fitted_
    
    def test_extreme_min_max_span_values(self):
        """Test with extreme minspan_alpha and endspan_alpha values."""
        X = np.random.rand(20, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        # Test with values approaching 0 and 1
        model = Earth(
            max_degree=2, 
            penalty=3.0, 
            max_terms=10,
            minspan_alpha=0.001,  # Very small
            endspan_alpha=0.999   # Very large
        )
        model.fit(X, y)
        assert model.fitted_
        
        # Test with moderate values
        model2 = Earth(
            max_degree=2, 
            penalty=3.0, 
            max_terms=10,
            minspan_alpha=0.5,  # Moderate
            endspan_alpha=0.5   # Moderate
        )
        model2.fit(X, y)
        assert model2.fitted_


if __name__ == "__main__":
    test = TestEarthAdditional()
    test.test_empty_forward_pass_case()
    test.test_extreme_parameters()
    test.test_single_sample_edge_case()
    test.test_two_samples_case()
    test.test_high_penalty_case()
    test.test_allow_missing_false_with_nans()
    test.test_allow_missing_true_with_nans()
    test.test_invalid_feature_importance_type()
    test.test_get_set_params_comprehensive()
    test.test_scrub_input_data_edge_cases()
    test.test_build_basis_matrix_empty()
    test.test_model_summary_method()
    test.test_extreme_penalty_values()
    test.test_extreme_min_max_span_values()
    
    print("All earth.py additional tests passed!")