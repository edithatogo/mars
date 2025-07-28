# -*- coding: utf-8 -*-

"""
Basis functions used in the MARS model.

This module will define various types of basis functions, such as:
- Constant basis function (intercept)
- Hinge functions (ReLU-like)
- Linear functions
"""

import logging
import numpy as np

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
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
        self.variable_idx: int = None # Index of the feature used by this basis function (if applicable)
        self.knot_val: float = None   # Knot value for hinge functions (if applicable)
        self.parent1: 'BasisFunction' = None # First parent for interaction terms
        self.parent2: 'BasisFunction' = None # Second parent for interaction terms (not used in current 1-parent model)
        self.is_linear_term: bool = False # Indicates if the *newest* component is linear
        self.is_hinge_term: bool = False  # Indicates if the *newest* component is a hinge
        self._involved_variables: frozenset[int] = frozenset() # Variables involved in this basis function
        self.gcv_score_: float = 0.0 # GCV reduction contribution of this basis function (when added as part of a pair)
        self.rss_score_: float = 0.0 # RSS reduction contribution of this basis function (when added as part of a pair)

    def get_involved_variables(self) -> frozenset[int]:
        """
        Returns a frozenset of variable indices involved in this basis function,
        including those from its parents.
        """
        return self._involved_variables

    @abstractmethod
    def transform(self, X_processed: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Apply the basis function transformation to the input data X_processed,
        considering the missing_mask.

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
                        is_linear: bool = False, is_hinge: bool = False,
                        involved_variables: frozenset[int] = frozenset()):
        self.variable_idx = variable_idx
        self.knot_val = knot_val
        self.parent1 = parent1
        self.parent2 = parent2 # Not currently used for 2nd order interactions via parent2
        self.is_linear_term = is_linear
        self.is_hinge_term = is_hinge
        self._involved_variables = involved_variables

    @abstractmethod
    def is_constant(self) -> bool:
        """
        Return True if the basis function is a constant (intercept) term.
        """
        pass


class ConstantBasisFunction(BasisFunction):
    """
    Represents the intercept term (constant value of 1).
    This basis function always returns 1 for any input.
    Its degree is 0.
    """
    def __init__(self):
        super().__init__(name="Intercept")
        self._set_properties(is_linear=False, is_hinge=False, involved_variables=frozenset())

    def transform(self, X_processed: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Returns an array of ones, with the same number of samples as X_processed.
        Missing mask is ignored as this function is data-independent.

        Parameters
        ----------
        X_processed : numpy.ndarray of shape (n_samples, n_features)
            The input data. Only n_samples (X_processed.shape[0]) is used.
        missing_mask : numpy.ndarray
            Boolean mask, ignored by this function.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            An array containing ones.
        """
        if not isinstance(X_processed, np.ndarray): # Corrected: removed the erroneous check for 'X'
            raise TypeError("Input X_processed must be a numpy array.")
        if X_processed.ndim == 1:
            return np.ones(X_processed.shape[0])
        elif X_processed.ndim == 2:
            return np.ones(X_processed.shape[0])
        else:
            raise ValueError("Input X_processed must be 1D or 2D.")


    def __str__(self) -> str:
        return self.get_name() # Returns "Intercept"

    def degree(self) -> int:
        """
        The degree of a constant basis function is 0.
        """
        return 0

    def is_constant(self) -> bool:
        return True


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

        parent_involved_vars = frozenset()
        if parent_bf:
            parent_involved_vars = parent_bf.get_involved_variables()
        current_involved_vars = parent_involved_vars.union({variable_idx})

        self._set_properties(variable_idx=variable_idx, knot_val=knot_val,
                             is_hinge=True, parent1=parent_bf,
                             involved_variables=current_involved_vars)
        self.is_right_hinge = is_right_hinge
        # self.variable_name is already set above for constructing the name

    def transform(self, X_processed: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Applies the hinge transformation. NaNs propagate.
        """
        if not isinstance(X_processed, np.ndarray):
            raise TypeError("Input X_processed must be a numpy array.")
        if X_processed.ndim not in [1, 2]:
             raise ValueError("Input X_processed must be 1D or 2D.")
        if X_processed.ndim == 1 and self.variable_idx != 0:
            raise ValueError("For 1D X input, variable_idx must be 0.")
        if X_processed.ndim == 2 and self.variable_idx >= X_processed.shape[1]:
            raise IndexError(f"variable_idx {self.variable_idx} is out of bounds for X_processed with {X_processed.shape[1]} features.")

        x_col = X_processed if X_processed.ndim == 1 else X_processed[:, self.variable_idx]

        # Calculate current hinge term's values (on zero-filled data)
        if self.is_right_hinge:
            current_term_values = np.maximum(0, x_col - self.knot_val)
        else:
            current_term_values = np.maximum(0, self.knot_val - x_col)

        # Apply NaN where original variable was missing
        # missing_mask corresponds to original X.
        # If X_processed is 1D, missing_mask should also be 1D or (N,1)
        # If X_processed is 2D, missing_mask is (N, n_features)
        current_var_missing = missing_mask if X_processed.ndim == 1 else missing_mask[:, self.variable_idx]
        if current_term_values.dtype != float: # Ensure float type before assigning NaN
            current_term_values = current_term_values.astype(float)
        current_term_values[current_var_missing] = np.nan

        if self.parent1: # This is an interaction term
            parent_transformed = self.parent1.transform(X_processed, missing_mask) # Recursive call
            # NaN propagation happens if either parent_transformed or current_term_values is NaN
            return parent_transformed * current_term_values
        else: # This is a simple hinge (degree 1)
            return current_term_values

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

    def is_constant(self) -> bool:
        return False


class CategoricalBasisFunction(BasisFunction):
    """
    Represents a categorical variable indicator: 1 if variable_idx is a specific category, 0 otherwise.
    Its degree is 1.
    """
    def __init__(self, variable_idx: int, category: any, variable_name: str = None, parent_bf: 'BasisFunction' = None):
        self.category = category
        self.variable_name = variable_name if variable_name else f"x{variable_idx}"

        name_str = ""
        if parent_bf:
            name_str += f"({str(parent_bf)}) * "
        name_str += f"{self.variable_name}_is_{self.category}"

        super().__init__(name=name_str)

        parent_involved_vars = frozenset()
        if parent_bf:
            parent_involved_vars = parent_bf.get_involved_variables()
        current_involved_vars = parent_involved_vars.union({variable_idx})

        self._set_properties(variable_idx=variable_idx, is_hinge=False, is_linear=False,
                             parent1=parent_bf,
                             involved_variables=current_involved_vars)

    def transform(self, X_processed: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Returns 1 if the value for self.variable_idx matches the category, 0 otherwise.
        NaNs propagate.
        """
        if not isinstance(X_processed, np.ndarray):
            raise TypeError("Input X_processed must be a numpy array.")
        if X_processed.ndim != 2:
             raise ValueError("Input X_processed must be 2D (n_samples, n_features).")
        if self.variable_idx >= X_processed.shape[1]:
            raise IndexError(f"variable_idx {self.variable_idx} is out of bounds for X_processed with {X_processed.shape[1]} features.")

        x_col = X_processed[:, self.variable_idx]
        current_term_values = (x_col == self.category).astype(float)

        # Apply NaN where original variable was missing
        current_var_missing = missing_mask[:, self.variable_idx]
        current_term_values[current_var_missing] = np.nan

        if self.parent1: # This is an interaction term
            parent_transformed = self.parent1.transform(X_processed, missing_mask)
            return parent_transformed * current_term_values
        else:
            return current_term_values

    def __str__(self) -> str:
        return self.get_name()

    def degree(self) -> int:
        """
        The degree of a CategoricalBasisFunction.
        If it's a simple categorical term, degree is 1.
        If it's an interaction term (has a parent), its degree is parent.degree() + 1.
        """
        if self.parent1:
            return self.parent1.degree() + 1
        return 1

    def is_constant(self) -> bool:
        return False


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

        parent_involved_vars = frozenset()
        if parent_bf:
            parent_involved_vars = parent_bf.get_involved_variables()
        current_involved_vars = parent_involved_vars.union({variable_idx})

        self._set_properties(variable_idx=variable_idx, is_linear=True, parent1=parent_bf,
                             involved_variables=current_involved_vars)

    def transform(self, X_processed: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Applies the linear transformation. NaNs propagate.
        """
        if not isinstance(X_processed, np.ndarray):
            raise TypeError("Input X_processed must be a numpy array.")
        if X_processed.ndim not in [1,2]:
             raise ValueError("Input X_processed must be 1D or 2D.")
        if X_processed.ndim == 1 and self.variable_idx != 0:
            raise ValueError("For 1D X input, variable_idx must be 0.")
        if X_processed.ndim == 2 and self.variable_idx >= X_processed.shape[1]:
            raise IndexError(f"variable_idx {self.variable_idx} is out of bounds for X_processed with {X_processed.shape[1]} features.")

        current_term_values = X_processed if X_processed.ndim == 1 else X_processed[:, self.variable_idx].copy() # Use .copy() to avoid modifying X_processed

        # Apply NaN where original variable was missing
        current_var_missing = missing_mask if X_processed.ndim == 1 else missing_mask[:, self.variable_idx]
        if current_term_values.dtype != float: # Ensure float type before assigning NaN
            current_term_values = current_term_values.astype(float)
        current_term_values[current_var_missing] = np.nan

        if self.parent1: # This is an interaction term
            parent_transformed = self.parent1.transform(X_processed, missing_mask) # Recursive call
            return parent_transformed * current_term_values
        else: # This is a simple linear term (degree 1)
            return current_term_values

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

    def is_constant(self) -> bool:
        return False


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
    logger.info("%s: %s", str(const_bf), const_bf.transform(X_sample))

    hinge_bf_right = HingeBasisFunction(variable_idx=0, knot_val=4.0, variable_name="FeatureA")
    logger.info("%s: %s", str(hinge_bf_right), hinge_bf_right.transform(X_sample))

    hinge_bf_left = HingeBasisFunction(variable_idx=1, knot_val=5.0, is_right_hinge=False, variable_name="FeatureB")
    logger.info("%s: %s", str(hinge_bf_left), hinge_bf_left.transform(X_sample))

    linear_bf = LinearBasisFunction(variable_idx=2, variable_name="FeatureC")
    logger.info("%s: %s", str(linear_bf), linear_bf.transform(X_sample))

    # Example of how interaction might work (conceptual)
    # inter_bf = InteractionBasisFunction(hinge_bf_right, linear_bf)
    # print(f"{str(inter_bf)}: {inter_bf.transform(X_sample)}")
    # print(f"Degree of interaction: {inter_bf.degree()}")
    # print(f"Degree of hinge: {hinge_bf_right.degree()}")
    # print(f"Degree of const: {const_bf.degree()}")


class MissingnessBasisFunction(BasisFunction):
    """
    Represents a missingness indicator function: 1 if variable_idx was missing, 0 otherwise.
    Its degree is 1. It does not interact with parent functions for degree calculation.
    """
    def __init__(self, variable_idx: int, variable_name: str = None):
        self.variable_name = variable_name if variable_name else f"x{variable_idx}"
        name_str = f"is_missing({self.variable_name})"

        super().__init__(name=name_str)

        # Missingness terms are typically not considered "hinges" or "linear" in the MARS sense.
        # They are indicators.
        self._set_properties(variable_idx=variable_idx, is_hinge=False, is_linear=False,
                             parent1=None, # Missingness functions are usually additive or interact differently
                             involved_variables=frozenset({variable_idx}))

    def transform(self, X_processed: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        Returns 1 if the original value for self.variable_idx was missing, 0 otherwise.

        Parameters
        ----------
        X_processed : numpy.ndarray
            The processed input data (NaNs typically filled). Not directly used by this function.
        missing_mask : numpy.ndarray of shape (n_samples, n_features)
            Boolean mask indicating which original values were NaN.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            An array of 0s and 1s.
        """
        if not isinstance(missing_mask, np.ndarray):
            raise TypeError("Input missing_mask must be a numpy array.")
        if missing_mask.ndim != 2:
             raise ValueError("Input missing_mask must be 2D (n_samples, n_features).")
        if self.variable_idx >= missing_mask.shape[1]:
            raise IndexError(f"variable_idx {self.variable_idx} is out of bounds for missing_mask with {missing_mask.shape[1]} features.")

        # missing_mask[sample_idx, self.variable_idx] is True if original was NaN
        # So, we directly convert this boolean mask column to int (True->1, False->0)
        return missing_mask[:, self.variable_idx].astype(int)

    def __str__(self) -> str:
        return self.get_name()

    def degree(self) -> int:
        """
        The degree of a MissingnessBasisFunction is conventionally 1.
        It represents a condition on a single variable.
        """
        return 1

    def is_constant(self) -> bool:
        return False
