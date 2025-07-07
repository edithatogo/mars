# -*- coding: utf-8 -*-

"""
Basis functions used in the MARS model.

This module will define various types of basis functions, such as:
- Constant basis function (intercept)
- Hinge functions (ReLU-like)
- Linear functions
"""

import numpy as np

class BasisFunction:
    """
    Abstract base class for all basis functions.
    """
    def __init__(self, variable_idx=None, knot_val=None, is_hinge=False, is_linear=False, parent1=None, parent2=None):
        self.variable_idx = variable_idx  # Index of the variable this basis function operates on
        self.knot_val = knot_val          # Knot value for hinge functions
        self.is_hinge = is_hinge
        self.is_linear = is_linear
        self.parent1 = parent1            # For interaction terms, the first parent basis function
        self.parent2 = parent2            # For interaction terms, the second parent basis function
        # TODO: Add more properties like degree, name, etc.

    def transform(self, X):
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
        raise NotImplementedError("Subclasses must implement this method.")

    def __str__(self):
        """
        Return a string representation of the basis function.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __repr__(self):
        return self.__str__()

    def degree(self):
        """
        Return the degree of the basis function (1 for non-interaction, >1 for interaction).
        """
        if self.parent1 is None and self.parent2 is None:
            return 1 if (self.is_hinge or self.is_linear) else 0 # Constant is degree 0 effectively, but often counted as 1 term

        deg = 0
        if self.parent1:
            deg += self.parent1.degree()
        if self.parent2: # This logic assumes parent2 means it's an interaction of parent1 * something_new
            # Or if parent2 is also a basis function, it's parent1 * parent2
            # py-earth's definition of degree is number of hinge components
            pass # Needs refinement based on how interactions are structured.
        # For now, simplified:
        if self.parent1 and self.parent2: # Product of two basis functions
             return self.parent1.degree() + self.parent2.degree() # This might be too simple
        elif self.parent1: # e.g. Linear * Hinge
             return self.parent1.degree() + (1 if self.is_hinge else 0) # Needs more thought

        return 1 # Default for simple non-constant basis functions

class ConstantBasisFunction(BasisFunction):
    """
    Represents the intercept term (constant value of 1).
    """
    def __init__(self):
        super().__init__(is_hinge=False, is_linear=False)

    def transform(self, X):
        return np.ones(X.shape[0])

    def __str__(self):
        return "Intercept"

    def degree(self):
        return 0 # Or 1, depending on convention. py-earth counts it in num_terms.


class HingeBasisFunction(BasisFunction):
    """
    Represents a hinge function: max(0, x - knot) or max(0, knot - x).
    """
    def __init__(self, variable_idx, knot_val, is_right_hinge=True, variable_name=None):
        super().__init__(variable_idx=variable_idx, knot_val=knot_val, is_hinge=True)
        self.is_right_hinge = is_right_hinge # True for max(0, x-knot), False for max(0, knot-x)
        self.variable_name = variable_name if variable_name else f"x{self.variable_idx}"

    def transform(self, X):
        x_col = X[:, self.variable_idx]
        if self.is_right_hinge:
            return np.maximum(0, x_col - self.knot_val)
        else:
            return np.maximum(0, self.knot_val - x_col)

    def __str__(self):
        op = ">" if self.is_right_hinge else "<"
        # Example: h(x1 - 5.0) or h(5.0 - x1)
        # py-earth style: max(0, x1 - 5.0) or max(0, 5.0 - x1)
        if self.is_right_hinge:
            return f"max(0, {self.variable_name} - {self.knot_val:.2f})"
        else:
            return f"max(0, {self.knot_val:.2f} - {self.variable_name})"


class LinearBasisFunction(BasisFunction):
    """
    Represents a linear term: x_i
    py-earth typically only introduces linear terms if they are part of an interaction
    or if hinge functions on that variable are not effective.
    Often, linear terms are just hinge functions with knots at -inf or +inf.
    This class might be simplified or merged depending on final design.
    """
    def __init__(self, variable_idx, variable_name=None):
        super().__init__(variable_idx=variable_idx, is_linear=True)
        self.variable_name = variable_name if variable_name else f"x{self.variable_idx}"

    def transform(self, X):
        return X[:, self.variable_idx]

    def __str__(self):
        return f"{self.variable_name}"


# TODO: InteractionBasisFunction (product of two other basis functions)
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
