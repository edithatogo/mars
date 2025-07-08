# -*- coding: utf-8 -*-

"""
Utility functions for the pymars library.

This module can contain helper functions used across different modules,
such as input validation, specific calculations not tied to a class, etc.
"""

import numpy as np
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


def gcv_penalty_cost_effective_parameters(num_terms, num_samples, penalty_factor, has_intercept=True):
    """
    Calculate the effective number of parameters C(M) for GCV,
    as defined in Friedman's MARS paper (Section 3.6, equation 55).

    C(M) = Number of basis functions + penalty_factor * (Number of basis functions - 1) / 2
           (if an intercept is present and counted in num_terms)
    Or more generally, it's a cost associated with the model complexity.

    A common simplification for the penalty in GCV is:
    Effective parameters = num_terms * penalty_factor (where penalty_factor is 'd' from paper)

    Let's use the form from py-earth which is simpler:
    num_params_effective = num_nonzero_coeffs + penalty * (num_nonzero_coeffs - 1) / 2 (if intercept)
    Or if no intercept, num_params_effective = num_nonzero_coeffs + penalty * (num_nonzero_coeffs + 1) / 2

    This function provides a placeholder for such a calculation.
    The actual GCV formula uses this.
    """
    if num_terms == 0:
        return 0

    # This is a common form for the "cost" component in GCV.
    # The 'penalty_factor' here is the 'd' in Friedman's paper (often a value like 2 or 3).
    # py-earth's `penalty` parameter corresponds to this `d`.
    if has_intercept:
        # If intercept is one of the terms
        if num_terms <= 1: # Only intercept or one term
            cost = num_terms
        else:
            cost = num_terms + penalty_factor * (num_terms - 1) / 2.0
    else:
        # If no intercept term is forced into the model initially
        cost = num_terms + penalty_factor * (num_terms + 1) / 2.0

    # Ensure cost does not exceed number of samples, which would make GCV undefined or negative
    return min(cost, num_samples - 1) if num_samples > num_terms else num_terms


def calculate_gcv(rss, num_samples, num_effective_params):
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
    # Test check_array and check_X_y
    print("--- Testing check_array & check_X_y ---")
    X_good = np.array([[1,2],[3,4],[5,6]])
    y_good = np.array([10,20,30])
    X_c, y_c = check_X_y(X_good, y_good)
    print(f"Checked X shape: {X_c.shape}, Checked y shape: {y_c.shape}")

    try:
        X_bad_dim = np.array([1,2,3])
        check_X_y(X_bad_dim, y_good)
    except ValueError as e:
        print(f"Caught expected error for X_bad_dim: {e}")

    try:
        y_bad_dim = np.array([[10],[20],[30],[40]]) # Wrong samples
        check_X_y(X_good, y_bad_dim)
    except ValueError as e:
        print(f"Caught expected error for y_bad_dim (shape mismatch): {e}")

    try:
        y_bad_dim_2d = np.array([[10,11],[20,21],[30,31]]) # y not 1D
        check_X_y(X_good, y_bad_dim_2d, ensure_y_1d=True)
    except ValueError as e:
        print(f"Caught expected error for y_bad_dim (not 1D): {e}")

    y_col_vec = np.array([[10],[20],[30]])
    _, y_col_c = check_X_y(X_good, y_col_vec, ensure_y_1d=True)
    print(f"y column vector converted to 1D, new shape: {y_col_c.shape}")


    # Test GCV calculation components
    print("\n--- Testing GCV Calculations ---")
    rss_example = 100.0
    n_samples_example = 50
    n_terms_example = 5
    penalty_d_example = 3.0 # Typical penalty for MARS GCV

    effective_params = gcv_penalty_cost_effective_parameters(n_terms_example, n_samples_example, penalty_d_example, has_intercept=True)
    print(f"RSS={rss_example}, N={n_samples_example}, Terms={n_terms_example}, Penalty={penalty_d_example}")
    print(f"Effective parameters (C(M)): {effective_params}")
    gcv_score = calculate_gcv(rss_example, n_samples_example, effective_params)
    print(f"GCV Score: {gcv_score}")

    # Edge case: more terms than samples (after penalty)
    n_terms_high = 40
    effective_params_high = gcv_penalty_cost_effective_parameters(n_terms_high, n_samples_example, penalty_d_example, has_intercept=True)
    print(f"\nTerms={n_terms_high}, Effective parameters (C(M)): {effective_params_high}")
    gcv_score_high = calculate_gcv(rss_example, n_samples_example, effective_params_high)
    print(f"GCV Score (high terms): {gcv_score_high}")

    # Edge case: num_effective_params == num_samples
    # This should lead to inf GCV. Our C(M) caps at N-1.
    # If C(M) was allowed to be N, then (1 - N/N)^2 = 0.
    effective_params_eq_N = n_samples_example
    gcv_score_eq_N = calculate_gcv(rss_example, n_samples_example, effective_params_eq_N)
    print(f"GCV Score (effective_params == N): {gcv_score_eq_N}")

    effective_params_just_under_N = n_samples_example - 1
    gcv_score_just_under_N = calculate_gcv(rss_example, n_samples_example, effective_params_just_under_N)
    print(f"GCV Score (effective_params == N-1): {gcv_score_just_under_N}")

    # Test with no terms
    effective_params_no_terms = gcv_penalty_cost_effective_parameters(0, n_samples_example, penalty_d_example)
    print(f"\nTerms=0, Effective parameters (C(M)): {effective_params_no_terms}")
    gcv_score_no_terms = calculate_gcv(0, n_samples_example, effective_params_no_terms) # RSS would be sum((y-mean(y))^2)
    print(f"GCV Score (0 terms, RSS=0 for test): {gcv_score_no_terms}")
