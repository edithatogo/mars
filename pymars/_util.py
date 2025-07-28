# -*- coding: utf-8 -*-

"""
Utility functions for the pymars library.

This module can contain helper functions used across different modules,
such as input validation, specific calculations not tied to a class, etc.
"""

# Standard library imports
import logging
import numpy as np

logger = logging.getLogger(__name__)
# from ._types import XType, YType, NumericArray, BoolArray # If using custom types

# Scikit-learn like input validation (simplified examples)
# In a real scenario, we would use sklearn.utils.validation directly
# if scikit-learn is a hard dependency. If not, we might implement some checks.

def check_array(array, ensure_2d=False, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, allow_missing=False):
    """
    Rudimentary input validation for an array.
    Inspired by sklearn.utils.validation.check_array.
    """
    if not isinstance(array, np.ndarray):
        try:
            array = np.asarray(array)
        except Exception as e:
            raise ValueError(f"Input could not be converted to a NumPy array. Error: {e}")

    if not allow_missing and np.isnan(array).any():
        raise ValueError("Input contains NaN values.")

    if ensure_2d and array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead.")

    if not allow_nd and array.ndim > 2:
        raise ValueError(f"Expected 1D or 2D array, got {array.ndim}D array instead.")

    if array.shape[0] < ensure_min_samples:
        raise ValueError(f"Found array with {array.shape[0]} sample(s), but a minimum of {ensure_min_samples} is required.")

    if array.ndim > 1 and array.shape[1] < ensure_min_features:
        raise ValueError(f"Found array with {array.shape[1]} feature(s), but a minimum of {ensure_min_features} is required.")

    return array


def check_X_y(X, y, ensure_X_2d=True, allow_X_nd=False, ensure_y_1d=True, allow_missing_X=False, allow_missing_y=False):
    """
    Rudimentary input validation for X and y.
    Inspired by sklearn.utils.validation.check_X_y.
    """
    X = check_array(X, ensure_2d=ensure_X_2d, allow_nd=allow_X_nd, allow_missing=allow_missing_X)
    y = check_array(y, ensure_2d=False, allow_nd=True, allow_missing=allow_missing_y) # y can be 1D or 2D (multi-output)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: X has {X.shape[0]}, y has {y.shape[0]}")

    if ensure_y_1d and y.ndim != 1:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel() # Convert column vector to 1D array
        else:
            raise ValueError(f"Expected 1D y array, got {y.ndim}D array instead.")

    return X, y


def gcv_penalty_cost_effective_parameters(num_terms: int, num_hinge_terms: int, penalty: float, num_samples: int) -> float:
    """
    Calculate the effective number of parameters for GCV, aligning with py-earth's approach.
    Effective parameters = num_terms + penalty * num_hinge_terms.

    Args:
        num_terms: Total number of basis functions (including intercept if present).
        num_hinge_terms: Number of hinge basis functions (excluding linear, constant).
        penalty: The penalty factor for hinge terms (corresponds to 'd' in Friedman's paper,
                 and 'penalty' in py-earth).
        num_samples: Number of samples used for fitting. This is used to cap the
                     effective parameters to avoid GCV becoming undefined.

    Returns:
        The calculated effective number of parameters.
    """
    if num_terms == 0:
        return 0.0

    # py-earth style: effective_num_parameters = num_terms + penalty * num_knots
    # Here, num_knots corresponds to num_hinge_terms (as each hinge pair comes from one knot,
    # and linear terms are not penalized in this way).
    effective_params = float(num_terms + penalty * num_hinge_terms)

    # Cap effective parameters to be less than num_samples to keep GCV formula stable.
    # If effective_params == num_samples, denominator in GCV is 0.
    # If effective_params > num_samples, (1 - eff/N)^2 can be negative if not careful, or GCV is meaningless.
    # We ensure it's at most num_samples - EPSILON effectively by how calculate_gcv handles it.
    # For this function, just return the calculated value; calculate_gcv handles the num_samples check.
    return effective_params


def calculate_gcv(rss: float, num_samples: int, num_effective_params: float) -> float:
    """
    Calculate Generalized Cross-Validation score.

    GCV = RSS / (N * (1 - (C_effective / N))^2)
    where RSS is Residual Sum of Squares, N is num_samples,
    C_effective is the effective number of parameters.
    """
    if num_samples == 0:
        return np.inf
    if num_effective_params >= num_samples:
        return np.inf # Avoid division by zero or sqrt of negative

    denominator = (1.0 - num_effective_params / num_samples)**2
    if denominator < 1e-9: # Effectively zero
        return np.inf

    gcv = rss / (num_samples * denominator)
    return gcv


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Test check_array and check_X_y
    logger.info("--- Testing check_array & check_X_y ---")
    X_good = np.array([[1,2],[3,4],[5,6]])
    y_good = np.array([10,20,30])
    X_c, y_c = check_X_y(X_good, y_good)
    logger.info("Checked X shape: %s, Checked y shape: %s", X_c.shape, y_c.shape)

    try:
        X_bad_dim = np.array([1,2,3])
        check_X_y(X_bad_dim, y_good)
    except ValueError as e:
        logger.info("Caught expected error for X_bad_dim: %s", e)

    try:
        y_bad_dim = np.array([[10],[20],[30],[40]]) # Wrong samples
        check_X_y(X_good, y_bad_dim)
    except ValueError as e:
        logger.info("Caught expected error for y_bad_dim (shape mismatch): %s", e)

    try:
        y_bad_dim_2d = np.array([[10,11],[20,21],[30,31]]) # y not 1D
        check_X_y(X_good, y_bad_dim_2d, ensure_y_1d=True)
    except ValueError as e:
        logger.info("Caught expected error for y_bad_dim (not 1D): %s", e)

    y_col_vec = np.array([[10],[20],[30]])
    _, y_col_c = check_X_y(X_good, y_col_vec, ensure_y_1d=True)
    logger.info("y column vector converted to 1D, new shape: %s", y_col_c.shape)


    # Test GCV calculation components
    logger.info("\n--- Testing GCV Calculations ---")
    rss_example = 100.0
    n_samples_example = 50
    n_terms_example = 5
    penalty_d_example = 3.0 # Typical penalty for MARS GCV

    # Assume one intercept term so the number of hinge terms is ``n_terms_example - 1``.
    num_hinge_example = n_terms_example - 1
    effective_params = gcv_penalty_cost_effective_parameters(
        n_terms_example, num_hinge_example, penalty_d_example, n_samples_example
    )
    logger.info(
        "RSS=%s, N=%s, Terms=%s, Penalty=%s",
        rss_example,
        n_samples_example,
        n_terms_example,
        penalty_d_example,
    )
    logger.info("Effective parameters (C(M)): %s", effective_params)
    gcv_score = calculate_gcv(rss_example, n_samples_example, effective_params)
    logger.info("GCV Score: %s", gcv_score)

    # Edge case: more terms than samples (after penalty)
    n_terms_high = 40
    num_hinge_high = n_terms_high - 1
    effective_params_high = gcv_penalty_cost_effective_parameters(
        n_terms_high, num_hinge_high, penalty_d_example, n_samples_example
    )
    logger.info(
        "\nTerms=%s, Effective parameters (C(M)): %s",
        n_terms_high,
        effective_params_high,
    )
    gcv_score_high = calculate_gcv(
        rss_example, n_samples_example, effective_params_high
    )
    logger.info("GCV Score (high terms): %s", gcv_score_high)

    # Edge case: num_effective_params == num_samples
    # This should lead to inf GCV. Our C(M) caps at N-1.
    # If C(M) was allowed to be N, then (1 - N/N)^2 = 0.
    effective_params_eq_N = n_samples_example
    gcv_score_eq_N = calculate_gcv(
        rss_example, n_samples_example, effective_params_eq_N
    )
    logger.info("GCV Score (effective_params == N): %s", gcv_score_eq_N)

    effective_params_just_under_N = n_samples_example - 1
    gcv_score_just_under_N = calculate_gcv(
        rss_example, n_samples_example, effective_params_just_under_N
    )
    logger.info(
        "GCV Score (effective_params == N-1): %s", gcv_score_just_under_N
    )

    # Example block finished
    pass
