# -*- coding: utf-8 -*-

"""
Custom type definitions for pymars.

This file can be used to define common type aliases or custom types
used throughout the project, improving readability and maintainability,
especially when working with complex data structures or for type hinting.
"""

from typing import List, Union, Optional, Any, Tuple, Callable
import numpy as np
import numpy.typing as npt

# Generic array types
ArrayLike = Union[npt.NDArray[Any], List[Any], Tuple[Any, ...]] # More general than np.ndarray for inputs
NumericArray = npt.NDArray[Union[np.float_, np.int_]]
FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]
BoolArray = npt.NDArray[np.bool_]

# For features (X) and target (y)
# X can be 2D array of numbers
XType = NumericArray # Typically FloatArray after processing
# y can be 1D or 2D array of numbers
YType = NumericArray # Typically FloatArray or IntArray after processing

# BasisFunction related types (forward declaration, actual class in _basis.py)
# from ._basis import BasisFunction # This would cause circular import if _basis imports from _types
BasisFunctionType = Any # Placeholder for actual BasisFunction class

# List of basis functions
BasisFunctionList = List[BasisFunctionType]

# Coefficients
CoefficientType = FloatArray # Typically a 1D array of floats

# Could define types for specific parameters if they have complex structures
# e.g., PenaltyType = Union[float, Dict[str, float]]

# For knot selection or other strategies that might involve functions
# KnotSelectorCallable = Callable[[FloatArray], FloatArray] # Function that takes a variable's values and returns knots


if __name__ == '__main__':
    # Example usage of these types for clarity in function signatures (conceptual)

    def example_function(X: XType, y: YType) -> Tuple[BasisFunctionList, CoefficientType]:
        # X_processed: FloatArray = np.asarray(X, dtype=float)
        # y_processed: FloatArray = np.asarray(y, dtype=float)

        # dummy basis function and coeffs
        class DummyBasis:
            def __str__(self): return "dummy_basis"

        basis_functions: BasisFunctionList = [DummyBasis()]
        coefficients: CoefficientType = np.array([1.0, 2.0])

        print(f"X type: {type(X)}, y type: {type(y)}")
        print(f"Basis functions: {basis_functions}")
        print(f"Coefficients: {coefficients}")

        return basis_functions, coefficients

    # test_X: FloatArray = np.array([[1.0, 2.0], [3.0, 4.0]])
    # test_y: FloatArray = np.array([1.5, 3.5])
    # example_function(test_X, test_y)

    # test_X_list: ArrayLike = [[1,2],[3,4]]
    # test_y_list: ArrayLike = [1,3]
    # example_function(test_X_list, test_y_list)
    pass
