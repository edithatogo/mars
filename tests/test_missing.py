"""
Tests for pymars._missing module - missing value handling functionality
"""
import numpy as np
import pytest
from pymars._missing import handle_missing_X, handle_missing_y


class TestHandleMissingX:
    """Test handle_missing_X function."""
    
    def test_no_missing_values(self):
        """Test when there are no missing values."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = handle_missing_X(X)
        np.testing.assert_array_equal(result, X)
    
    def test_mean_strategy(self):
        """Test mean imputation strategy."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = handle_missing_X(X, strategy='mean')
        expected = np.array([[1.0, 4.0], [3.0, 4.0]])  # mean of [4.0] is 4.0
        np.testing.assert_array_equal(result, expected)
    
    def test_median_strategy(self):
        """Test median imputation strategy."""
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        result = handle_missing_X(X, strategy='median')
        expected = np.array([[1.0, 5.0], [3.0, 4.0], [5.0, 6.0]])  # median of [4.0, 6.0] is 5.0
        np.testing.assert_array_equal(result, expected)
    
    def test_most_frequent_strategy(self):
        """Test most frequent imputation strategy."""
        X = np.array([[1.0, 2.0], [1.0, np.nan], [3.0, 2.0]])
        result = handle_missing_X(X, strategy='most_frequent')
        # Most frequent value in second column is 2.0
        expected = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 2.0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_error_strategy_with_nans(self):
        """Test error strategy when NaNs are present."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Input X contains NaN values and strategy is 'error'"):
            handle_missing_X(X, strategy='error')
    
    def test_error_strategy_no_nans(self):
        """Test error strategy when no NaNs are present."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = handle_missing_X(X, strategy='error')
        np.testing.assert_array_equal(result, X)
    
    def test_pass_through_strategy_allowed(self):
        """Test pass through strategy when allowed."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = handle_missing_X(X, strategy='pass_through', allow_missing_for_some_strategies=True)
        np.testing.assert_array_equal(result, X)
    
    def test_pass_through_strategy_not_allowed(self):
        """Test pass through strategy when not allowed."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Strategy 'pass_through' for NaNs requires model to be configured to allow missing values"):
            handle_missing_X(X, strategy='pass_through', allow_missing_for_some_strategies=False)
    
    def test_unknown_strategy(self):
        """Test unknown strategy raises error."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Unknown missing value strategy: unknown"):
            handle_missing_X(X, strategy='unknown')
    
    def test_1d_array(self):
        """Test with 1D array input."""
        X = np.array([1.0, np.nan, 3.0])
        result = handle_missing_X(X, strategy='mean')
        expected = np.array([1.0, 2.0, 3.0])  # mean of [1.0, 3.0] is 2.0
        np.testing.assert_array_equal(result, expected)
    
    def test_all_nans_in_column_most_frequent(self):
        """Test most frequent strategy when all values in a column are NaN."""
        X = np.array([[np.nan, 2.0], [np.nan, 4.0]])
        result = handle_missing_X(X, strategy='most_frequent')
        # When all values in first column are NaN, replace with 0
        expected = np.array([[0.0, 2.0], [0.0, 4.0]])
        np.testing.assert_array_equal(result, expected)


class TestHandleMissingY:
    """Test handle_missing_y function."""
    
    def test_no_missing_values(self):
        """Test when there are no missing values in y."""
        y = np.array([1.0, 2.0, 3.0])
        result, mask = handle_missing_y(y)
        np.testing.assert_array_equal(result, y)
        np.testing.assert_array_equal(mask, np.array([False, False, False]))
    
    def test_mean_strategy_regression(self):
        """Test mean imputation for regression target."""
        y = np.array([1.0, np.nan, 3.0])
        result, mask = handle_missing_y(y, strategy='mean', problem_type='regression')
        expected = np.array([1.0, 2.0, 3.0])  # mean of [1.0, 3.0] is 2.0
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(mask, np.array([False, True, False]))
    
    def test_median_strategy_regression(self):
        """Test median imputation for regression target."""
        y = np.array([1.0, np.nan, 3.0, 4.0])
        result, mask = handle_missing_y(y, strategy='median', problem_type='regression')
        expected = np.array([1.0, 3.0, 3.0, 4.0])  # median of [1.0, 3.0, 4.0] is 3.0
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(mask, np.array([False, True, False, False]))
    
    def test_most_frequent_strategy(self):
        """Test most frequent imputation for target."""
        y = np.array([1.0, np.nan, 2.0, 1.0])
        result, mask = handle_missing_y(y, strategy='most_frequent')
        expected = np.array([1.0, 1.0, 2.0, 1.0])  # most frequent is 1.0
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(mask, np.array([False, True, False, False]))
    
    def test_error_strategy_with_nans(self):
        """Test error strategy when NaNs are present in y."""
        y = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Target y contains NaN values and strategy is 'error'"):
            handle_missing_y(y, strategy='error')
    
    def test_remove_samples_strategy(self):
        """Test remove samples strategy."""
        y = np.array([1.0, np.nan, 3.0, np.nan])
        result, mask = handle_missing_y(y, strategy='remove_samples')
        expected = np.array([1.0, 3.0])  # Remove NaN positions
        expected_mask = np.array([True, True, True, True])  # Original mask (all NaN positions)
        np.testing.assert_array_equal(result, expected)
        # The mask should indicate which samples were originally NaN
        np.testing.assert_array_equal(mask, np.array([False, True, False, True]))
    
    def test_mean_strategy_classification_error(self):
        """Test mean strategy raises error for classification."""
        y = np.array([1, np.nan, 0])
        with pytest.raises(ValueError, match="Cannot use 'mean' imputation for classification target"):
            handle_missing_y(y, strategy='mean', problem_type='classification')
    
    def test_median_strategy_classification_error(self):
        """Test median strategy raises error for classification."""
        y = np.array([1, np.nan, 0])
        with pytest.raises(ValueError, match="Cannot use 'median' imputation for classification target"):
            handle_missing_y(y, strategy='median', problem_type='classification')
    
    def test_unknown_strategy_y(self):
        """Test unknown strategy for y raises error."""
        y = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Unknown missing value strategy for y: unknown"):
            handle_missing_y(y, strategy='unknown')
    
    def test_most_frequent_all_nans(self):
        """Test most frequent strategy when all values are NaN."""
        y = np.array([np.nan, np.nan, np.nan])
        result, mask = handle_missing_y(y, strategy='most_frequent')
        # When all are NaN, should default to 0 for regression
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(mask, np.array([True, True, True]))
    
    def test_none_strategy_regression_default(self):
        """Test None strategy uses regression default (mean)."""
        y = np.array([1.0, np.nan, 3.0])
        result, mask = handle_missing_y(y, strategy=None, problem_type='regression')
        expected = np.array([1.0, 2.0, 3.0])  # mean of [1.0, 3.0] is 2.0
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    # Run tests manually if needed
    test = TestHandleMissingX()
    test.test_no_missing_values()
    test.test_mean_strategy()
    test.test_median_strategy()
    test.test_most_frequent_strategy()
    test.test_error_strategy_with_nans()
    test.test_error_strategy_no_nans()
    test.test_pass_through_strategy_allowed()
    test.test_pass_through_strategy_not_allowed()
    test.test_unknown_strategy()
    test.test_1d_array()
    test.test_all_nans_in_column_most_frequent()
    
    test_y = TestHandleMissingY()
    test_y.test_no_missing_values()
    test_y.test_mean_strategy_regression()
    test_y.test_median_strategy_regression()
    test_y.test_most_frequent_strategy()
    test_y.test_error_strategy_with_nans()
    test_y.test_remove_samples_strategy()
    test_y.test_mean_strategy_classification_error()
    test_y.test_median_strategy_classification_error()
    test_y.test_unknown_strategy_y()
    test_y.test_most_frequent_all_nans()
    test_y.test_none_strategy_regression_default()
    
    print("All _missing.py tests passed!")