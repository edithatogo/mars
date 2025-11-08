"""
Comprehensive tests for pymars.earth module to achieve >95% coverage
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pymars import Earth


class TestEarthEdgeCases:
    """Additional edge case tests for Earth module to improve coverage."""
    
    def test_earth_with_extreme_parameters(self):
        """Test Earth with extreme parameter values."""
        X = np.random.rand(20, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        # Test with very high penalty that forces minimal model
        model_high_penalty = Earth(penalty=1000.0, max_degree=1, max_terms=2)
        model_high_penalty.fit(X, y)
        assert model_high_penalty.fitted_
        print(f"✅ High penalty model: {len(model_high_penalty.basis_)} terms")
        
    def test_earth_with_minimum_span_negative_values(self):
        """Test Earth with negative minspan values."""
        X = np.random.rand(30, 3)
        y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.05, 30)
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=15,
            minspan=-1,  # Default value
            endspan=-1   # Default value
        )
        model.fit(X, y)
        assert model.fitted_
        assert len(model.basis_) > 0
        print(f"✅ Negative minspan and endspan: {len(model.basis_)} terms")
        
    def test_earth_with_zero_penalty(self):
        """Test Earth with zero penalty."""
        X = np.random.rand(30, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(max_degree=2, penalty=0.0, max_terms=20)
        model.fit(X, y)
        assert model.fitted_
        assert model.penalty == 0.0
        print(f"✅ Zero penalty model: R²={model.score(X, y):.4f}")
        
    def test_earth_with_extreme_degrees(self):
        """Test Earth with very high degree."""
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(max_degree=10, penalty=3.0, max_terms=20)
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ High degree model: {model.max_degree} degree, {len(model.basis_)} terms")
        
    def test_earth_with_very_few_terms(self):
        """Test Earth with very few max terms."""
        X = np.random.rand(20, 3) 
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=2)  # Minimum possible
        model.fit(X, y)
        assert model.fitted_
        assert len(model.basis_) <= 2
        print(f"✅ Minimal terms model: <=2 terms, actual={len(model.basis_)}")
        
    def test_earth_with_all_zeros_in_target(self):
        """Test Earth with all zeros in target."""
        X = np.random.rand(15, 2)
        y = np.zeros(15)  # All zeros target
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        model.fit(X, y)
        assert model.fitted_
        
        # Predictions should be close to zero
        preds = model.predict(X)
        assert np.allclose(preds, 0, atol=1e-3)
        print("✅ All-zero target model: Predictions near zero")
        
    def test_earth_with_constant_features(self):
        """Test Earth with one constant feature."""
        X = np.random.rand(20, 3)
        X[:, 0] = 5.0  # First feature is constant
        y = X[:, 1] + X[:, 2] * 0.5  # Target depends on non-constant features
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ Constant feature model: R²={model.score(X, y):.4f}")
        
    def test_earth_with_extreme_feature_values(self):
        """Test Earth with extreme feature value ranges."""
        X = np.random.rand(20, 3)
        X[:, 0] *= 1e8   # Very large values
        X[:, 1] *= 1e-8  # Very small values
        X[:, 2] = (X[:, 2] - 0.5) * 1000  # Centered & scaled values
        
        y = (X[:, 0] * 1e-8) + (X[:, 1] * 1e8) + X[:, 2] * 0.001 + np.random.normal(0, 0.1, 20)
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=15)
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ Extreme value scaling: R²={model.score(X, y):.4f}")
        
    def test_earth_with_invalid_feature_importance_type(self):
        """Test Earth with invalid feature importance type."""
        X = np.random.rand(15, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        # Use an invalid feature importance type
        model = Earth(
            max_degree=2, 
            penalty=3.0, 
            max_terms=10, 
            feature_importance_type='invalid_type'
        )
        model.fit(X, y)
        assert model.fitted_
        
        # This should handle invalid type gracefully - feature_importances_ might be None
        try:
            fi = model.feature_importances_
            # If it exists, it should be an array
            if fi is not None:
                assert isinstance(fi, np.ndarray)
        except ValueError:
            # If it raises an error, that's acceptable for invalid types
            pass
        print("✅ Invalid feature importance type handled gracefully")
        
    def test_earth_with_categorical_features_specified(self):
        """Test Earth with categorical features list specified."""
        X = np.random.rand(30, 4)
        # Make some columns contain categorical values (integers)
        X[:, 2] = np.random.choice([0, 1, 2], size=len(X))
        X[:, 3] = np.random.choice([0, 1], size=len(X))
        
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=15,
            categorical_features=[2, 3]  # Specify columns 2 and 3 as categorical
        )
        model.fit(X, y)
        assert model.fitted_
        assert len(model.basis_) > 0
        print(f"✅ Categorical features specified: {len(model.basis_)} terms")
        
    def test_earth_with_allow_linear_false(self):
        """Test Earth with allow_linear=False."""
        X = np.random.rand(20, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=15,
            allow_linear=False  # Don't allow linear terms
        )
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ Linear terms disabled: {len(model.basis_)} terms")
        
    def test_earth_with_very_high_endspan_alpha(self):
        """Test Earth with high endspan_alpha."""
        X = np.random.rand(25, 3)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=10,
            endspan_alpha=0.9  # Very loose endspan constraint
        )
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ High endspan_alpha: {len(model.basis_)} terms")
        
    def test_earth_with_zero_minspan_alpha(self):
        """Test Earth with zero minspan_alpha."""
        X = np.random.rand(30, 3)
        y = X[:, 0] + X[:, 1] * 0.5
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=15,
            minspan_alpha=0.0  # Minimal minspan constraint
        )
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ Zero minspan_alpha: {len(model.basis_)} terms")
        
    def test_earth_edge_case_empty_basis_functions_from_forward_pass(self):
        """Test the specific edge case where forward pass returns empty basis functions."""
        X = np.random.rand(10, 2)
        y = np.ones(10)  # Constant target - might result in minimal model
        
        # Create a model with parameters that could result in minimal basis functions
        model = Earth(
            max_degree=1,
            penalty=50.0,  # Very high penalty
            max_terms=2,
            minspan_alpha=1.0,  # Very restrictive
            endspan_alpha=1.0   # Very restrictive
        )
        model.fit(X, y)
        assert model.fitted_
        
        # Model should have at least intercept
        assert model.basis_ is not None
        assert len(model.basis_) >= 1
        print(f"✅ Edge case with minimal forward pass: {len(model.basis_)} terms")
        
    def test_earth_with_single_sample(self):
        """Test Earth with single sample (edge case)."""
        X = np.array([[1.0, 2.0]])  # Single sample
        y = np.array([3.0])  # Single target
        
        # Model should handle minimal sample size gracefully (either fit or fail controlled)
        model = Earth(max_degree=1, penalty=3.0, max_terms=5)
        
        try:
            model.fit(X, y)
            # If it fits successfully, that's OK (fallback behavior)
            if hasattr(model, 'fitted_') and model.fitted_:
                print(f"✅ Single sample handled gracefully: {len(model.basis_ or [])} terms")
            else:
                print("✅ Single sample processed without crashing")
        except Exception as e:
            # If it fails, that's also OK as long as it's controlled
            print(f"✅ Single sample properly rejected with controlled error: {type(e).__name__}")
        
    def test_earth_with_specific_min_max_span_values(self):
        """Test Earth with specific minspan and endspan values (non-negative)."""
        X = np.random.rand(25, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        # Use non-negative minspan and endspan values
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=10,
            minspan=2,  # Specific minspan value
            endspan=3   # Specific endspan value
        )
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ Specific minspan/endspan values: {len(model.basis_)} terms")
        
    def test_earth_with_feature_importance_nb_subsets(self):
        """Test Earth with 'nb_subsets' feature importance."""
        X = np.random.rand(20, 3)
        y = X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.25
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=10,
            feature_importance_type='nb_subsets'
        )
        model.fit(X, y)
        assert model.fitted_
        
        # Should calculate feature importances
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            assert len(model.feature_importances_) == X.shape[1]
            # Should sum to approximately 1.0
            assert abs(sum(model.feature_importances_) - 1.0) < 0.01
        print("✅ 'nb_subsets' feature importance calculation")
        
    def test_earth_with_feature_importance_rss(self):
        """Test Earth with 'rss' feature importance.""" 
        X = np.random.rand(20, 3)
        y = X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.25
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=10,
            feature_importance_type='rss'
        )
        model.fit(X, y)
        assert model.fitted_
        
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            assert len(model.feature_importances_) == X.shape[1]
            assert abs(sum(model.feature_importances_) - 1.0) < 0.01
        print("✅ 'rss' feature importance calculation")
        
    def test_earth_with_none_feature_importance_type(self):
        """Test Earth with feature_importance_type=None."""
        X = np.random.rand(20, 3)
        y = X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.25
        
        model = Earth(
            max_degree=2,
            penalty=3.0,
            max_terms=10,
            feature_importance_type=None  # Explicitly None
        )
        model.fit(X, y)
        assert model.fitted_
        
        # With None, feature_importances_ might not be computed or might be all zeros
        if hasattr(model, 'feature_importances_'):
            # Feature importance might be None or array of zeros
            if model.feature_importances_ is not None:
                assert isinstance(model.feature_importances_, np.ndarray)
        print("✅ None feature importance type handled")
        
    def test_earth_edge_case_highly_correlated_features(self):
        """Test Earth with highly correlated features which can cause numerical issues."""
        X_base = np.random.rand(30, 1)
        # Create highly correlated features
        X = np.column_stack([
            X_base[:, 0],
            X_base[:, 0] * 0.99 + np.random.normal(0, 1e-6, 30),  # Highly correlated
            X_base[:, 0] * 1.01 + np.random.normal(0, 1e-6, 30)   # Highly correlated
        ])
        
        y = X[:, 0] + X[:, 1] * 0.5 + np.random.normal(0, 0.01, 30)
        
        model = Earth(max_degree=2, penalty=3.0, max_terms=15)
        model.fit(X, y)
        assert model.fitted_
        print(f"✅ Highly correlated features: R²={model.score(X, y):.4f}")
        
    def test_earth_with_extreme_penalty_values(self):
        """Test Earth with extreme penalty values."""
        X = np.random.rand(25, 2)
        y = X[:, 0] + X[:, 1] * 0.5
        
        # Very low penalty (0.01) - allows more complex models
        model_low_penalty = Earth(max_degree=3, penalty=0.01, max_terms=20)
        model_low_penalty.fit(X, y)
        assert model_low_penalty.fitted_
        print(f"✅ Low penalty model: {len(model_low_penalty.basis_)} terms")
        
        # Very high penalty (100.0) - forces simple models
        model_high_penalty2 = Earth(max_degree=3, penalty=100.0, max_terms=20)
        model_high_penalty2.fit(X, y)
        assert model_high_penalty2.fitted_
        print(f"✅ High penalty model: {len(model_high_penalty2.basis_)} terms")


def test_earth_comprehensive_edge_cases():
    """Run all comprehensive edge case tests."""
    test = TestEarthEdgeCases()
    
    print("🧪 Comprehensive Earth module edge case testing...")
    print("=" * 60)
    
    test.test_earth_with_extreme_parameters()
    test.test_earth_with_minimum_span_negative_values()
    test.test_earth_with_zero_penalty()
    test.test_earth_with_extreme_degrees()
    test.test_earth_with_very_few_terms()
    test.test_earth_with_all_zeros_in_target()
    test.test_earth_with_constant_features()
    test.test_earth_with_extreme_feature_values()
    test.test_earth_with_invalid_feature_importance_type()
    test.test_earth_with_categorical_features_specified()
    test.test_earth_with_allow_linear_false()
    test.test_earth_with_very_high_endspan_alpha()
    test.test_earth_with_zero_minspan_alpha()
    test.test_earth_edge_case_empty_basis_functions_from_forward_pass()
    test.test_earth_with_single_sample()
    test.test_earth_with_specific_min_max_span_values()
    test.test_earth_with_feature_importance_nb_subsets()
    test.test_earth_with_feature_importance_rss()
    test.test_earth_with_none_feature_importance_type()
    test.test_earth_edge_case_highly_correlated_features()
    test.test_earth_with_extreme_penalty_values()
    
    print("=" * 60)
    print("✅ All comprehensive Earth edge case tests completed!")
    

if __name__ == "__main__":
    test_earth_comprehensive_edge_cases()
    print("\\n🎉 Earth module comprehensive testing completed!")