"""
Additional tests for pymars._categorical to achieve >90% coverage
"""
import numpy as np
from pymars._categorical import CategoricalImputer


def test_categorical_imputer_edge_cases():
    """Test categorical imputer with all edge cases."""
    
    # Test 1: All missing values in a categorical column
    imputer = CategoricalImputer()
    X = np.array([[None, 1.0], [None, 2.0], [None, 3.0]], dtype=object)  # Second column numeric
    categorical_features = [0]  # First column is categorical (but all missing)
    
    imputer.fit(X, categorical_features)
    
    # Transform should work even with all missing values
    X_transformed = imputer.transform(X)
    assert X_transformed.shape == X.shape
    assert np.isfinite(X_transformed).all()
    
    # Test 2: Single dimensional array
    X_1d = np.array(['A', 'B', 'A'], dtype=object)
    imputer_1d = CategoricalImputer()
    imputer_1d.fit(X_1d, [0])
    X_1d_transformed = imputer_1d.transform(X_1d)
    assert X_1d_transformed.ndim == 2  # Should become 2D after reshape
    assert X_1d_transformed.shape[0] == 3
    assert np.isfinite(X_1d_transformed).all()
    
    # Test 3: Float NaN values (testing _is_missing with np.nan)
    X_nan = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=object)  # Second column is numeric
    imputer_nan = CategoricalImputer()
    imputer_nan.fit(X_nan, [0])
    X_nan_transformed = imputer_nan.transform(X_nan)
    assert X_nan_transformed.shape == X_nan.shape
    assert np.isfinite(X_nan_transformed).all()
    
    # Test 4: Transform with unseen categories (should use most frequent)
    X_train = np.array([['cat', 1.0], ['dog', 2.0], ['cat', 3.0]], dtype=object)
    X_test = np.array([['bird', 4.0], ['fish', 5.0]], dtype=object)  # bird/fish not seen in training
    
    imputer_unseen = CategoricalImputer()
    imputer_unseen.fit(X_train, [0])  # Make first column categorical
    X_test_transformed = imputer_unseen.transform(X_test)
    
    # Should handle unseen categories by using most frequent from training
    assert X_test_transformed.shape == X_test.shape
    assert np.isfinite(X_test_transformed).all()
    
    # Test 5: Empty categorical feature list
    X_regular = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=object)  # Both numeric columns
    imputer_empty = CategoricalImputer()
    imputer_empty.fit(X_regular, [])  # No categorical features
    X_regular_transformed = imputer_empty.transform(X_regular)
    # Should still work - no transformation applied to any column
    assert X_regular_transformed.shape == X_regular.shape
    
    # Test 6: Fit_transform method
    X_fit_transform = np.array([['red', 1.0], ['blue', 2.0], ['red', 3.0]], dtype=object)
    imputer_fit_trans = CategoricalImputer()
    X_fit_transformed = imputer_fit_trans.fit_transform(X_fit_transform, [0])
    assert X_fit_transformed.shape == X_fit_transform.shape
    assert np.isfinite(X_fit_transformed).all()
    
    # Test 7: Column with mixed types including None
    X_mixed = np.array([[None, 1.0], ['value', 2.0], [None, 3.0]], dtype=object)
    imputer_mixed = CategoricalImputer()
    imputer_mixed.fit(X_mixed, [0])
    X_mixed_transformed = imputer_mixed.transform(X_mixed)
    assert X_mixed_transformed.shape == X_mixed.shape
    assert np.isfinite(X_mixed_transformed).all()
    
    print("All categorical imputer edge cases passed!")


def test_is_missing_function():
    """Test the _is_missing static method directly."""
    # Test None
    assert CategoricalImputer._is_missing(None)
    
    # Test NaN
    assert CategoricalImputer._is_missing(np.nan)
    
    # Test float NaN
    assert CategoricalImputer._is_missing(float('nan'))
    
    # Test non-missing values
    assert not CategoricalImputer._is_missing('A')
    assert not CategoricalImputer._is_missing(1)
    assert not CategoricalImputer._is_missing(1.0)
    assert not CategoricalImputer._is_missing('')
    
    print("_is_missing function tests passed!")


if __name__ == "__main__":
    test_is_missing_function()
    test_categorical_imputer_edge_cases()
    print("All _categorical.py tests passed!")