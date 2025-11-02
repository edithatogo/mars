
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


# ``check_X_y`` was formerly implemented here as a lightweight alternative to
# :func:`sklearn.utils.validation.check_X_y`.  The project now depends on
# scikit-learn's implementation directly.  Import ``check_X_y`` from
# :mod:`sklearn.utils.validation` instead.


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

