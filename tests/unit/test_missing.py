"""Tests for the _missing.py module."""

import numpy as np
import pytest

from pymars._missing import handle_missing_X, handle_missing_y


def test_handle_missing_X_no_nans():
    """Test handle_missing_X_no_nans behavior."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_out = handle_missing_X(X)
    np.testing.assert_array_equal(X_out, X)


@pytest.mark.parametrize(
    "strategy, expected_fill",
    [
        ("mean", (1.0 + 3.0 + 1.0) / 3.0),
        ("median", 1.0),
        ("most_frequent", 1.0),
    ],
)
def test_handle_missing_X_1d(strategy, expected_fill):
    """Test handle_missing_X_1d behavior."""
    X = np.array([1.0, np.nan, 3.0, 1.0])
    X_out = handle_missing_X(X, strategy=strategy)
    expected = np.array([1.0, expected_fill, 3.0, 1.0])
    np.testing.assert_array_equal(X_out, expected)


def test_handle_missing_X_2d_mean():
    """Test handle_missing_X_2d_mean behavior."""
    X = np.array([[1.0, np.nan], [3.0, 4.0], [2.0, 2.0]])
    X_out = handle_missing_X(X, strategy="mean")
    expected = np.array([[1.0, 3.0], [3.0, 4.0], [2.0, 2.0]])
    np.testing.assert_array_equal(X_out, expected)


def test_handle_missing_X_most_frequent_all_nans():
    """Test handle_missing_X_most_frequent_all_nans behavior."""
    X = np.array([np.nan, np.nan])
    X_out = handle_missing_X(X, strategy="most_frequent")
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_equal(X_out, expected)


def test_handle_missing_X_error_strategy():
    """Test handle_missing_X_error_strategy behavior."""
    X = np.array([1.0, np.nan])
    with pytest.raises(
        ValueError, match=r"Input X contains NaN values and strategy is 'error'\."
    ):
        handle_missing_X(X, strategy="error")


def test_handle_missing_X_pass_through():
    """Test handle_missing_X_pass_through behavior."""
    X = np.array([1.0, np.nan])
    with pytest.raises(
        ValueError,
        match=r"Strategy 'pass_through' for NaNs requires model to be configured to allow missing values\.",
    ):
        handle_missing_X(
            X, strategy="pass_through", allow_missing_for_some_strategies=False
        )

    X_out = handle_missing_X(
        X, strategy="pass_through", allow_missing_for_some_strategies=True
    )
    np.testing.assert_array_equal(X_out, X)


def test_handle_missing_X_unknown_strategy():
    """Test handle_missing_X_unknown_strategy behavior."""
    X = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="Unknown missing value strategy: unknown"):
        handle_missing_X(X, strategy="unknown")


def test_handle_missing_y_no_nans():
    """Test handle_missing_y_no_nans behavior."""
    y = np.array([1.0, 2.0, 3.0])
    y_out, mask = handle_missing_y(y)
    np.testing.assert_array_equal(y_out, y)
    np.testing.assert_array_equal(mask, np.array([False, False, False]))


def test_handle_missing_y_default_regression():
    """Test handle_missing_y_default_regression behavior."""
    y = np.array([1.0, np.nan, 3.0])
    y_out, mask = handle_missing_y(y, strategy=None, problem_type="regression")
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y_out, expected)
    np.testing.assert_array_equal(mask, np.array([False, True, False]))


def test_handle_missing_y_default_classification():
    """Test handle_missing_y_default_classification behavior."""
    y = np.array([1.0, np.nan, 3.0])
    with pytest.raises(
        ValueError, match=r"Target y contains NaN values and strategy is 'error'\."
    ):
        handle_missing_y(y, strategy=None, problem_type="classification")


@pytest.mark.parametrize(
    "strategy, expected_fill",
    [
        ("mean", (1.0 + 3.0 + 1.0) / 3.0),
        ("median", 1.0),
        ("most_frequent", 1.0),
    ],
)
def test_handle_missing_y_impute(strategy, expected_fill):
    """Test handle_missing_y_impute behavior."""
    y = np.array([1.0, np.nan, 3.0, 1.0])
    y_out, mask = handle_missing_y(y, strategy=strategy)
    expected = np.array([1.0, expected_fill, 3.0, 1.0])
    np.testing.assert_array_equal(y_out, expected)
    np.testing.assert_array_equal(mask, np.array([False, True, False, False]))


def test_handle_missing_y_remove_samples():
    """Test handle_missing_y_remove_samples behavior."""
    y = np.array([1.0, np.nan, 3.0])
    y_out, mask = handle_missing_y(y, strategy="remove_samples")
    expected = np.array([1.0, 3.0])
    np.testing.assert_array_equal(y_out, expected)
    np.testing.assert_array_equal(mask, np.array([False, True, False]))


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_handle_missing_y_classification_error(strategy):
    """Test handle_missing_y_classification_error behavior."""
    y = np.array([1.0, np.nan, 3.0])
    with pytest.raises(
        ValueError,
        match=rf"Cannot use '{strategy}' imputation for classification target\.",
    ):
        handle_missing_y(y, strategy=strategy, problem_type="classification")


def test_handle_missing_y_most_frequent_all_nans_regression():
    """Test handle_missing_y_most_frequent_all_nans_regression behavior."""
    y = np.array([np.nan, np.nan])
    y_out, mask = handle_missing_y(
        y, strategy="most_frequent", problem_type="regression"
    )
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_equal(y_out, expected)
    np.testing.assert_array_equal(mask, np.array([True, True]))


def test_handle_missing_y_most_frequent_all_nans_classification():
    """Test handle_missing_y_most_frequent_all_nans_classification behavior."""
    y = np.array([np.nan, np.nan])
    y_out, mask = handle_missing_y(
        y, strategy="most_frequent", problem_type="classification"
    )
    expected = np.array([0.0, 0.0])
    np.testing.assert_array_equal(y_out, expected)
    np.testing.assert_array_equal(mask, np.array([True, True]))


def test_handle_missing_y_error_strategy():
    """Test handle_missing_y_error_strategy behavior."""
    y = np.array([1.0, np.nan])
    with pytest.raises(
        ValueError, match=r"Target y contains NaN values and strategy is 'error'\."
    ):
        handle_missing_y(y, strategy="error")


def test_handle_missing_y_unknown_strategy():
    """Test handle_missing_y_unknown_strategy behavior."""
    y = np.array([1.0, np.nan])
    with pytest.raises(
        ValueError, match="Unknown missing value strategy for y: unknown"
    ):
        handle_missing_y(y, strategy="unknown")
