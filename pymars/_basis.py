# -*- coding: utf-8 -*-

"""
Basis functions used in the MARS model.

This module will define various types of basis functions, such as:
- Constant basis function (intercept)
- Hinge functions (ReLU-like)
- Linear functions
"""

import numpy as np

from abc import ABC, abstractmethod
# from ._types import XType, FloatArray # Assuming XType is np.ndarray for internal use

class BasisFunction(ABC):
    """
    Abstract base class for all basis functions in the MARS model.

    Each basis function must be able to transform input data and provide
    a string representation and its degree.
    """
    def __init__(self, name: str = "BasisFunction"):
        self._name = name
        # Attributes common to many basis functions, but not all will use them directly.
        # Subclasses will define how these are used.
        self.variable_idx: int = None # Index of the feature used by this basis function
        self.knot_val: float = None   # Knot value for hinge functions
        self.parent1: 'BasisFunction' = None # First parent for interaction terms
        self.parent2: 'BasisFunction' = None # Second parent for interaction terms
        self.is_linear_term: bool = False # Indicates if it's a simple linear term x_j
        self.is_hinge_term: bool = False  # Indicates if it's a hinge max(0, x-k) or max(0, k-x)

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the basis function transformation to the input data X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The transformed values.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the basis function.
        """
        pass

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def degree(self) -> int:
        """
        Return the degree of the basis function.
        For MARS, degree is often defined as the number of hinge components
        multiplied together.
        - A constant (intercept) term has degree 0.
        - A hinge term (e.g., max(0, x-k)) has degree 1.
        - A linear term (e.g., x) if treated as a special case, also degree 1.
        - An interaction term (product of two degree-1 terms) has degree 2.
        """
        pass

    def get_name(self) -> str:
        """Get the name of the basis function (can be auto-generated or custom)."""
        return self._name

    def _set_name(self, name: str):
        """Set the name, typically used by subclasses during init."""
        self._name = name

    # Optional: Methods to explicitly set properties if not done in init by subclasses
    def _set_properties(self, variable_idx: int = None, knot_val: float = None,
                        parent1: 'BasisFunction' = None, parent2: 'BasisFunction' = None,
                        is_linear: bool = False, is_hinge: bool = False):
        self.variable_idx = variable_idx
        self.knot_val = knot_val
        self.parent1 = parent1
        self.parent2 = parent2
        self.is_linear_term = is_linear
        self.is_hinge_term = is_hinge


class ConstantBasisFunction(BasisFunction):
    """
    Represents the intercept term (constant value of 1).
    This basis function always returns 1 for any input.
    Its degree is 0.
    """
    def __init__(self):
        super().__init__(name="Intercept")
        # Constant basis functions don't use these, but they are part of the base class structure
        self._set_properties(is_linear=False, is_hinge=False)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an array of ones, with the same number of samples as X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input data. Only n_samples (X.shape[0]) is used.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            An array containing ones.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array.")
        if X.ndim == 1: # If X is a 1D array (e.g. single feature, multiple samples)
            return np.ones(X.shape[0])
        elif X.ndim == 2: # If X is a 2D array (samples, features)
            return np.ones(X.shape[0])
        else:
            raise ValueError("Input X must be 1D or 2D.")


    def __str__(self) -> str:
        return self.get_name() # Returns "Intercept"

    def degree(self) -> int:
        """
        The degree of a constant basis function is 0.
        """
        return 0


class HingeBasisFunction(BasisFunction):
    """
    Represents a hinge function: max(0, x - knot) or max(0, knot - x).
    Its degree is 1.
    """
    def __init__(self, variable_idx: int, knot_val: float, is_right_hinge: bool = True, variable_name: str = None, parent_bf: 'BasisFunction' = None):
        # Determine name based on properties
        self.variable_name = variable_name if variable_name else f"x{variable_idx}"
        name_str = ""
        if parent_bf:
            name_str += f"({str(parent_bf)}) * "

        if is_right_hinge:
            name_str += f"max(0, {self.variable_name} - {knot_val:.2f})"
        else:
            name_str += f"max(0, {knot_val:.2f} - {self.variable_name})"

        super().__init__(name=name_str)

        self._set_properties(variable_idx=variable_idx, knot_val=knot_val, is_hinge=True, parent1=parent_bf)
        self.is_right_hinge = is_right_hinge # True for max(0, x-knot), False for max(0, knot-x)
        # self.variable_name is already set above for constructing the name

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the hinge transformation to the specified variable in X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The transformed values.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array.")
        if X.ndim not in [1, 2]:
             raise ValueError("Input X must be 1D or 2D.")
        if X.ndim == 1 and self.variable_idx != 0:
            raise ValueError("For 1D X input, variable_idx must be 0.")
        if X.ndim == 2 and self.variable_idx >= X.shape[1]:
            raise IndexError(f"variable_idx {self.variable_idx} is out of bounds for X with {X.shape[1]} features.")

        x_col = X if X.ndim == 1 else X[:, self.variable_idx]

        if self.parent1: # This is an interaction term
            parent_transformed = self.parent1.transform(X)
            if self.is_right_hinge:
                current_hinge = np.maximum(0, x_col - self.knot_val)
            else:
                current_hinge = np.maximum(0, self.knot_val - x_col)
            return parent_transformed * current_hinge
        else: # This is a simple hinge (degree 1)
            if self.is_right_hinge:
                return np.maximum(0, x_col - self.knot_val)
            else:
                return np.maximum(0, self.knot_val - x_col)

    def __str__(self) -> str:
        # The name is already constructed in __init__ to handle interactions properly.
        return self.get_name()

    def degree(self) -> int:
        """
        The degree of a HingeBasisFunction.
        If it's a simple hinge, degree is 1.
        If it's an interaction term (has a parent), its degree is parent.degree() + 1.
        """
        if self.parent1:
            return self.parent1.degree() + 1
        return 1


class LinearBasisFunction(BasisFunction):
    """
    Represents a linear term: x_i
    py-earth typically only introduces linear terms if they are part of an interaction,
    or if hinge functions on that variable are not effective (e.g., if the relationship
    is purely linear). Its degree is 1.
    """
    def __init__(self, variable_idx: int, variable_name: str = None, parent_bf: 'BasisFunction' = None):
        self.variable_name = variable_name if variable_name else f"x{variable_idx}"
        name_str = ""
        if parent_bf:
            name_str += f"({str(parent_bf)}) * "
        name_str += self.variable_name

        super().__init__(name=name_str)
        self._set_properties(variable_idx=variable_idx, is_linear=True, parent1=parent_bf)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the values of the specified variable in X.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The values of the specified feature.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array.")
        if X.ndim not in [1,2]:
             raise ValueError("Input X must be 1D or 2D.")
        if X.ndim == 1 and self.variable_idx != 0:
            raise ValueError("For 1D X input, variable_idx must be 0.")
        if X.ndim == 2 and self.variable_idx >= X.shape[1]:
            raise IndexError(f"variable_idx {self.variable_idx} is out of bounds for X with {X.shape[1]} features.")

        x_col = X if X.ndim == 1 else X[:, self.variable_idx]

        if self.parent1: # This is an interaction term
            parent_transformed = self.parent1.transform(X)
            return parent_transformed * x_col
        else: # This is a simple linear term (degree 1)
            return x_col

    def __str__(self) -> str:
        return self.get_name()

    def degree(self) -> int:
        """
        The degree of a LinearBasisFunction.
        If it's a simple linear term, degree is 1.
        If it's an interaction term (has a parent), its degree is parent.degree() + 1.
        """
        if self.parent1:
            return self.parent1.degree() + 1
        return 1


# TODO: Consider a more generic InteractionBasisFunction(bf1, bf2) if needed,
# or ensure HingeBasisFunction and LinearBasisFunction can always represent one part of an interaction.
# Current design: Hinge and Linear can have one parent, implying Parent * CurrentType.
# This covers interactions like Hinge*Hinge, Hinge*Linear, Linear*Linear (if parent is Linear).

# Example:
# parent_linear = LinearBasisFunction(0, "x0")
# interaction_linear_times_hinge = HingeBasisFunction(1, 5.0, parent_bf=parent_linear, variable_name="x1")
#   str: (x0) * max(0, x1 - 5.00)
#   degree: 1 (from parent_linear) + 1 (from hinge) = 2

# class InteractionBasisFunction(BasisFunction):
#     def __init__(self, bf1: BasisFunction, bf2: BasisFunction):
#         # Need to handle variable_idx, knot_val appropriately, or mark them as NA
#         super().__init__(is_hinge=False, is_linear=False, parent1=bf1, parent2=bf2)
#         if bf1 is None or bf2 is None:
#             raise ValueError("Parent basis functions for interaction term cannot be None.")
#
#     def transform(self, X):
#         return self.parent1.transform(X) * self.parent2.transform(X)
#
#     def __str__(self):
#         return f"({str(self.parent1)}) * ({str(self.parent2)})"
#
#     def degree(self):
#         return self.parent1.degree() + self.parent2.degree()

if __name__ == '__main__':
    # Example Usage
    X_sample = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    const_bf = ConstantBasisFunction()
    print(f"{str(const_bf)}: {const_bf.transform(X_sample)}")

    hinge_bf_right = HingeBasisFunction(variable_idx=0, knot_val=4.0, variable_name="FeatureA")
    print(f"{str(hinge_bf_right)}: {hinge_bf_right.transform(X_sample)}")

    hinge_bf_left = HingeBasisFunction(variable_idx=1, knot_val=5.0, is_right_hinge=False, variable_name="FeatureB")
    print(f"{str(hinge_bf_left)}: {hinge_bf_left.transform(X_sample)}")

    linear_bf = LinearBasisFunction(variable_idx=2, variable_name="FeatureC")
    print(f"{str(linear_bf)}: {linear_bf.transform(X_sample)}")

    # Example of how interaction might work (conceptual)
    # inter_bf = InteractionBasisFunction(hinge_bf_right, linear_bf)
    # print(f"{str(inter_bf)}: {inter_bf.transform(X_sample)}")
    # print(f"Degree of interaction: {inter_bf.degree()}")
    # print(f"Degree of hinge: {hinge_bf_right.degree()}")
    # print(f"Degree of const: {const_bf.degree()}")
