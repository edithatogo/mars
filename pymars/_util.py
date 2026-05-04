"""
Utility functions for the pymars library.

This module can contain helper functions used across different modules,
such as input validation, specific calculations not tied to a class, etc.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


def check_array(
    array: Any,
    ensure_2d: bool = False,
    allow_nd: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
    allow_missing: bool = False,
) -> np.ndarray:
    """
    Rudimentary input validation for an array.
    Inspired by sklearn.utils.validation.check_array.
    """
    if not isinstance(array, np.ndarray):
        try:
            array = np.asarray(array)
        except Exception as e:
            raise ValueError(
                f"Input could not be converted to a NumPy array. Error: {e}"
            )

    if not allow_missing and np.isnan(array).any():
        raise ValueError("Input contains NaN values.")

    if ensure_2d and array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead.")

    if not allow_nd and array.ndim > 2:
        raise ValueError(f"Expected 1D or 2D array, got {array.ndim}D array instead.")

    if array.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {array.shape[0]} sample(s), but a minimum of {ensure_min_samples} is required."
        )

    if array.ndim > 1 and array.shape[1] < ensure_min_features:
        raise ValueError(
            f"Found array with {array.shape[1]} feature(s), but a minimum of {ensure_min_features} is required."
        )

    return cast("np.ndarray", array)

def gcv_penalty_cost_effective_parameters(
    num_terms: int, num_hinge_terms: int, penalty: float, num_samples: int | float
) -> float:
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
    del num_samples
    if num_terms == 0:
        return 0.0

    return float(num_terms + penalty * num_hinge_terms)


def calculate_gcv(
    rss: float, num_samples: int | float, num_effective_params: float
) -> float:
    """
    Calculate Generalized Cross-Validation score.

    GCV = RSS / (N * (1 - (C_effective / N))^2)
    where RSS is Residual Sum of Squares, N is num_samples,
    C_effective is the effective number of parameters.
    """
    if num_samples == 0:
        return np.inf
    if num_effective_params >= num_samples:
        return np.inf

    denominator = (1.0 - num_effective_params / num_samples) ** 2
    if denominator < 1e-9:
        return np.inf

    return rss / (num_samples * denominator)
