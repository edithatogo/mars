# -*- coding: utf-8 -*-

"""
Utilities for handling missing values in pymars.

py-earth has specific strategies for handling missing data, such as
the Missing Data Imputation (MDI) method or treating missing as a distinct value.
This module will house such functionalities.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

def handle_missing_X(X, strategy='mean', allow_missing_for_some_strategies=False):
    """
    Handle missing values in the input feature matrix X.

    Parameters
    ----------
    X : numpy.ndarray or array-like
        Input data with potential NaNs.
    strategy : str, optional (default='mean')
        The strategy to handle missing values:
        - 'mean': Impute with mean of the column.
        - 'median': Impute with median of the column.
        - 'most_frequent': Impute with the most frequent value.
        - 'error': Raise an error if NaNs are present.
        - 'pass_through': (Use with caution) Allows NaNs if the basis functions
                          or downstream processes can handle them. `allow_missing_for_some_strategies`
                          should typically be True for this.
    allow_missing_for_some_strategies : bool, default=False
        If True, allows strategies like 'pass_through'. The main model's
        `allow_missing` parameter would typically control this.

    Returns
    -------
    X_processed : numpy.ndarray
        Data with NaNs handled.
    nan_map : numpy.ndarray of bool, optional
        A boolean array indicating positions of original NaNs. May be useful
        for some advanced handling but not returned by default by this basic version.
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if not np.issubdtype(X.dtype, np.number):
        # If X is not numeric (e.g. object array due to mixed types or strings)
        # and contains non-numeric NaNs like None, or actual strings.
        # This basic handler assumes numeric data primarily.
        # More sophisticated handling for mixed types would be needed.
        pass # For now, let it proceed, np.isnan will fail if not float.

    nan_present = np.isnan(X).any()

    if not nan_present:
        return X

    if strategy == 'error':
        raise ValueError("Input X contains NaN values and strategy is 'error'.")

    if strategy == 'pass_through':
        if allow_missing_for_some_strategies:
            return X # Basis functions must be able to handle NaNs
        else:
            raise ValueError("Strategy 'pass_through' for NaNs requires model to be configured to allow missing values.")

    X_processed = np.copy(X) # Work on a copy

    if X_processed.ndim == 1: # Handle 1D array case
      X_processed = X_processed.reshape(-1, 1) # Temporarily make it 2D for consistent processing
      was_1d = True
    else:
      was_1d = False

    for j in range(X_processed.shape[1]):
        col = X_processed[:, j]
        nan_mask_col = np.isnan(col)

        if not nan_mask_col.any():
            continue

        if strategy == 'mean':
            fill_value = np.nanmean(col)
        elif strategy == 'median':
            fill_value = np.nanmedian(col)
        elif strategy == 'most_frequent':
            # Simple approach for most_frequent with numbers
            # For categorical, a more robust method (e.g., scipy.stats.mode) is needed
            unique_vals, counts = np.unique(col[~nan_mask_col], return_counts=True)
            if unique_vals.size > 0:
                fill_value = unique_vals[np.argmax(counts)]
            else: # All values were NaN
                fill_value = 0 # Or some other default
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

        col[nan_mask_col] = fill_value

    if was_1d and X_processed.shape[1] == 1:
      X_processed = X_processed.ravel() # Convert back to 1D if original was 1D

    return X_processed


def handle_missing_y(y, strategy='mean', allow_missing_for_some_strategies=False, problem_type='regression'):
    """
    Handle missing values in the target variable y.

    Parameters
    ----------
    y : numpy.ndarray or array-like
        Target data with potential NaNs.
    strategy : str, optional (default='mean' for regression, 'error' for classification)
        The strategy to handle missing values.
        - 'mean': Impute with mean (for regression).
        - 'median': Impute with median (for regression).
        - 'most_frequent': Impute with most frequent (can be used for classification).
        - 'remove_samples': Remove samples (rows) where y is NaN.
        - 'error': Raise an error.
    allow_missing_for_some_strategies : bool, default=False
        (Currently not used much for y, but for consistency).
    problem_type : str, default='regression'
        'regression' or 'classification'. Affects default strategy if not given.

    Returns
    -------
    y_processed : numpy.ndarray
        Target data with NaNs handled.
    nan_mask : numpy.ndarray of bool
        Boolean mask indicating which samples were NaN (if strategy involves imputation).
        If 'remove_samples', this indicates removed samples.
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    nan_mask = np.isnan(y)
    if not nan_mask.any():
        return y, nan_mask # No NaNs

    if strategy is None: # Determine default based on problem type
        strategy = 'mean' if problem_type == 'regression' else 'error'

    if strategy == 'error':
        raise ValueError("Target y contains NaN values and strategy is 'error'.")

    if strategy == 'remove_samples':
        # This strategy implies X also needs to be filtered.
        # The function calling this should handle that synchronization.
        # Here, we just return the filtered y and the mask of what *was* NaN.
        return y[~nan_mask], nan_mask

    y_processed = np.copy(y)

    if strategy == 'mean':
        if problem_type == 'classification':
            raise ValueError("Cannot use 'mean' imputation for classification target.")
        fill_value = np.nanmean(y_processed)
    elif strategy == 'median':
        if problem_type == 'classification':
            raise ValueError("Cannot use 'median' imputation for classification target.")
        fill_value = np.nanmedian(y_processed)
    elif strategy == 'most_frequent':
        unique_vals, counts = np.unique(y_processed[~nan_mask], return_counts=True)
        if unique_vals.size > 0:
            fill_value = unique_vals[np.argmax(counts)]
        else: # All values were NaN
            fill_value = 0 if problem_type == 'regression' else (y_processed.dtype.type(0) if np.issubdtype(y_processed.dtype, np.integer) else 0.0) # Default
    else:
        raise ValueError(f"Unknown missing value strategy for y: {strategy}")

    y_processed[nan_mask] = fill_value
    return y_processed, nan_mask


if __name__ == '__main__':
    # Example for X
    X_test = np.array([[1, 2, np.nan], [4, np.nan, 6], [np.nan, 8, 9], [10, 11, 12]])
    logger.info("Original X:\n%s", X_test)

    X_mean_imputed = handle_missing_X(X_test, strategy='mean')
    logger.info("\nX mean imputed:\n%s", X_mean_imputed)

    X_median_imputed = handle_missing_X(X_test, strategy='median')
    logger.info("\nX median imputed:\n%s", X_median_imputed)

    X_mf_imputed = handle_missing_X(X_test, strategy='most_frequent')
    logger.info("\nX most_frequent imputed:\n%s", X_mf_imputed)

    try:
        handle_missing_X(X_test, strategy='error')
    except ValueError as e:
        logger.info("\nError caught as expected: %s", e)

    # Example for y
    y_test_reg = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    logger.info("\nOriginal y (regression):\n%s", y_test_reg)
    y_mean_imputed, _ = handle_missing_y(y_test_reg, strategy='mean')
    logger.info("\ny mean imputed (regression):\n%s", y_mean_imputed)

    y_removed, removed_mask = handle_missing_y(y_test_reg, strategy='remove_samples')
    logger.info("\ny with NaNs removed (regression):\n%s", y_removed)
    logger.info("Mask of removed samples:\n%s", removed_mask)


    y_test_clf = np.array([0, 1, np.nan, 0, 1, 1, np.nan]).astype(float)  # Use float for nan
    logger.info("\nOriginal y (classification):\n%s", y_test_clf)
    y_mf_imputed, _ = handle_missing_y(y_test_clf, strategy='most_frequent', problem_type='classification')
    logger.info("\ny most_frequent imputed (classification):\n%s", y_mf_imputed)

    try:
        handle_missing_y(y_test_clf, strategy='mean', problem_type='classification')
    except ValueError as e:
        logger.info("\nError caught for clf with mean: %s", e)
