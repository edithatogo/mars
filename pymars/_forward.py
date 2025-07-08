# -*- coding: utf-8 -*-

"""
The forward pass of the MARS algorithm.

This module is responsible for iteratively adding basis functions to the model
to minimize a criterion (e.g., sum of squared errors).
"""

import numpy as np
from ._basis import BasisFunction, HingeBasisFunction, ConstantBasisFunction, LinearBasisFunction
from ._record import EarthRecord # Assuming EarthRecord is used by Earth model instance
from .earth import Earth # For type hinting

# Define a small constant for numerical stability if needed
EPSILON = np.finfo(float).eps

class ForwardPasser:
    """
    Manages the forward pass of the MARS algorithm.

    Identifies candidate basis functions by splitting existing basis functions
    on different variables and knots. It selects the pair of functions that
    results in the greatest reduction in sum-of-squares error.
    """
    def __init__(self, earth_model: Earth):
        """
        Initialize the ForwardPasser.

        Parameters
        ----------
        earth_model : Earth
            An instance of the main Earth model. This provides access to
            hyperparameters like `max_terms`, `max_degree`, `minspan_alpha`,
            `endspan_alpha`, and the `record_` object for logging.
        """
        self.model = earth_model

        # Internal state, initialized at the start of run()
        self.X_train = None
        self.y_train = None
        self.n_samples = 0
        self.n_features = 0

        self.current_basis_functions: list[BasisFunction] = []
        self.current_B_matrix = None # Basis matrix for current model
        self.current_coefficients = None
        self.current_rss = np.inf

        # To store information about the best candidate found in a pass
        self._best_candidate_pair = None # Stores tuple of (new_bf1, new_bf2) or (new_bf, None)
        self._best_new_B_matrix = None
        self._best_new_coeffs = None
        self._min_candidate_rss = np.inf # Minimize this value, which is the RSS with candidate

    def _calculate_rss_and_coeffs(self, B_matrix: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculates Residual Sum of Squares and coefficients for a given basis matrix and target y.
        Returns (rss, coeffs). Returns (np.inf, None) on error.
        """
        if B_matrix is None or B_matrix.shape[1] == 0: # No basis functions
            # RSS is variance of y * N if no model (or sum of squares of y - mean(y))
            mean_y = np.mean(y)
            rss = np.sum((y - mean_y)**2)
            # Return coefficients for mean if B_matrix was supposed to be intercept only
            # For truly empty B_matrix, coeffs concept is ill-defined, return None or mean as a single coeff
            return rss, np.array([mean_y]) if (B_matrix is not None and B_matrix.shape[1] == 0) else None

        try:
            # Ensure B_matrix is 2D
            if B_matrix.ndim == 1:
                B_matrix = B_matrix.reshape(-1, 1)

            coeffs, residuals_sum_sq, rank, s = np.linalg.lstsq(B_matrix, y, rcond=None)

            # If residuals_sum_sq is empty (e.g. perfect fit or underdetermined and m <= n)
            # residuals_sum_sq is sum of squares of residuals if m > n and B_matrix is full rank (rank == n)
            # for underdetermined or exact, it's empty or an array of zeros.
            if residuals_sum_sq.size == 0 or rank < B_matrix.shape[1]:
                # This means either perfect fit, or an underdetermined system where lstsq found one solution.
                # For MARS, we usually expect m > n. If rank < n, it's rank-deficient.
                # We need to calculate RSS manually.
                y_pred = B_matrix @ coeffs
                rss = np.sum((y - y_pred)**2)
            else: # When lstsq returns sum of squares of residuals directly
                rss = residuals_sum_sq[0]
            return rss, coeffs
        except np.linalg.LinAlgError:
            return np.inf, None # Penalize unstable fits heavily

    def _build_basis_matrix(self, X: np.ndarray, basis_functions: list[BasisFunction]) -> np.ndarray:
        """Constructs the basis matrix B from X and a list of basis functions."""
        if not basis_functions:
            # Return a column of ones if model is empty (for intercept calculation)
            # Or handle as per specific model requirements (e.g. py-earth starts with intercept)
            # For now, let's assume an empty list means an empty matrix for general purpose
            return np.empty((X.shape[0], 0))

        B_list = [bf.transform(X).reshape(-1, 1) for bf in basis_functions]
        return np.hstack(B_list)

    def run(self, X: np.ndarray, y: np.ndarray) -> tuple[list[BasisFunction], np.ndarray]:
        """
        Execute the forward pass.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training input data. (Should be validated by Earth model before passing)
        y : numpy.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Training target data. (Should be validated and be 1D for now by Earth model)

        Returns
        -------
        current_basis_functions : list of BasisFunction
            The list of selected basis functions from the forward pass.
        current_coefficients : numpy.ndarray
            The coefficients corresponding to the selected basis functions.
        """
        self.X_train = X
        self.y_train = y
        if y.ndim > 1 and y.shape[1] > 1:
            # TODO: Add support for multi-output targets later if needed.
            # For now, ForwardPasser assumes y is effectively 1D.
            raise ValueError("ForwardPasser currently supports only single-output targets (y should be 1D).")
        self.y_train = self.y_train.ravel() # Ensure y is 1D

        self.n_samples, self.n_features = X.shape

        # Initialize model with an intercept term
        intercept_bf = ConstantBasisFunction()
        self.current_basis_functions = [intercept_bf]
        self.current_B_matrix = self._build_basis_matrix(self.X_train, self.current_basis_functions)

        rss, coeffs = self._calculate_rss_and_coeffs(self.current_B_matrix, self.y_train)
        if coeffs is None: # Should not happen with just intercept unless y is problematic
            # This case implies an issue with initial RSS calculation, e.g. LinAlgError
            # Or if _calculate_rss_and_coeffs returns (rss, None) for empty B_matrix
            # For intercept only, B_matrix is (N,1) of ones. coeffs should be [mean(y)].
            print("Warning: Could not calculate initial coefficients for intercept model.")
            return [], np.array([])

        self.current_coefficients = coeffs
        self.current_rss = rss

        if self.model.record_ is not None and hasattr(self.model.record_, 'log_forward_pass_step'):
            self.model.record_.log_forward_pass_step(
                self.current_basis_functions, self.current_coefficients, self.current_rss
            )

        # Determine max_terms for the loop
        max_terms_for_loop = self.model.max_terms
        if max_terms_for_loop is None:
            # Heuristic from py-earth, but simplified: max(21, 2*num_features + 1)
            # The +1 is for the intercept. py-earth's max_terms includes the intercept.
            max_terms_for_loop = min(self.n_samples - 1, max(21, 2 * self.n_features + 1))

        # Main loop for adding terms
        # The loop should go up to max_terms_for_loop-1 because we already have one term (intercept)
        # and each iteration successfully adds one or two terms.
        # A typical MARS iteration adds a *pair* of basis functions.
        # So, if max_terms is M, we add (M-1)/2 pairs approx.
        # Let's adjust loop condition based on number of basis functions.

        # Iteratively add terms (placeholder for now)
        # Loop will go here in subsequent steps. For now, this step only sets up initial model.

        # Placeholder print for now, actual loop will be implemented later
        print(f"Initial model setup: {len(self.current_basis_functions)} term(s), RSS={self.current_rss:.4f}")
        if self.current_coefficients is not None:
             print(f"Initial coeffs: {self.current_coefficients}")

        # The rest of the forward pass loop will be implemented in subsequent steps.
        # For now, we return the state after initial setup.
        # This is not the final return of the forward pass, just for this step's completion.
        # The actual loop (steps 3,4,5) will modify self.current_basis_functions and self.current_coefficients

        # Main Forward Pass Loop
        while True:
            # Check stopping criterion: maximum number of terms
            # Note: Each MARS step adds a PAIR of basis functions.
            # So, if max_terms_for_loop is M, we can have at most M terms.
            # The intercept is 1 term. Each iteration adds 2 terms.
            # If current_basis_functions has L terms, we can add 2 more if L+2 <= M.
            if len(self.current_basis_functions) + 2 > max_terms_for_loop:
                # print(f"Stopping: Max terms ({max_terms_for_loop}) would be exceeded.")
                break

            self._find_best_candidate_pair()

            # Check stopping criterion: no improvement found or no valid candidates
            if self._best_candidate_pair is None or self._min_candidate_rss >= self.current_rss - EPSILON :
                # print(f"Stopping: No further RSS improvement or no valid candidates. Current RSS: {self.current_rss:.4f}, Best Candidate RSS: {self._min_candidate_rss:.4f}")
                break

            # Add the best pair to the model
            bf_left, bf_right = self._best_candidate_pair
            self.current_basis_functions.extend([bf_left, bf_right])
            self.current_B_matrix = self._best_new_B_matrix # Or rebuild: self._build_basis_matrix(self.X_train, self.current_basis_functions)
            self.current_coefficients = self._best_new_coeffs
            self.current_rss = self._min_candidate_rss

            # print(f"Added pair: {str(bf_left)} & {str(bf_right)}. New RSS: {self.current_rss:.4f}. Num terms: {len(self.current_basis_functions)}")

            if self.model.record_ is not None and hasattr(self.model.record_, 'log_forward_pass_step'):
                self.model.record_.log_forward_pass_step(
                    self.current_basis_functions, self.current_coefficients, self.current_rss
                )

            # Reset for next iteration's search
            self._best_candidate_pair = None
            self._best_new_B_matrix = None
            self._best_new_coeffs = None
            # self._min_candidate_rss is implicitly reset at the start of _find_best_candidate_pair

        # print(f"Forward pass complete. Final RSS: {self.current_rss:.4f}, Num terms: {len(self.current_basis_functions)}")
        return self.current_basis_functions, self.current_coefficients

    def _get_allowable_knot_values(self, X_col: np.ndarray, parent_bf: BasisFunction, var_idx: int) -> np.ndarray:
        """
        Get allowable knot values for a variable, considering min_span and end_span.
        Simplified version: returns unique values of X_col, excluding min/max if endspan is restrictive.
        A more complete version would use self.model.minspan_alpha and self.model.endspan_alpha.
        """
        # TODO: Implement full min_span and end_span logic.
        # min_span: Minimum number of observations between knots.
        # end_span: Minimum number of observations from data ends where knots can be placed.
        # These are complex and depend on sorted unique values of X_col and model parameters.

        # Simplified: use unique values from X_col.
        # Knots are typically placed at observed data points.
        # We must exclude values that would result in empty splits for min_span.
        # For a new hinge max(0, x-t) or max(0, t-x), the knot t cannot be min(X_col) or max(X_col)
        # because that would make one of the pair of hinges zero everywhere or identical to the other.
        # (e.g. if t = min(X_col), then max(0, X_col - t) is X_col - min(X_col), and max(0, t - X_col) is zero).

        unique_X_vals = np.unique(X_col)

        # Basic endspan: exclude min and max values as knots for splitting a variable for the first time.
        # If parent_bf is not just the intercept, more complex rules might apply based on parent's knots.
        # For now, a simple exclusion:
        if len(unique_X_vals) > 2: # Need at least 3 unique values to pick a knot in between
            # For interaction terms, py-earth uses all unique values as potential knots.
            # For additive terms (parent is intercept), it excludes the max value.
            # This was a simplification. Actual endspan and minspan are more complex.

            # Calculate end_span value based on alpha
            # endspan is the number of unique X values to prohibit at each end.
            # If endspan_alpha = 0, endspan = 0 (no prohibition based on alpha).
            # If endspan_alpha > 0, endspan = floor( (N_unique_X_col * endspan_alpha) / 2 )
            # py-earth has a hardcoded minimum endspan of 0, even if alpha is negative (which is not typical).
            # And a minimum of 1 if alpha > 0 and N_unique_X_col * alpha / 2 < 1
            # Let's use a simpler interpretation: endspan is a count of points.
            # Default endspan from py-earth is 0 if not specified.

            effective_endspan = 0
            if self.model.endspan_alpha > 0:
                # This is a simplified interpretation of py-earth's endspan.
                # py-earth's `endspan` parameter is derived from `endspan_alpha` if `endspan` itself is not set.
                # `endspan = max(0, floor(num_unique_values * endspan_alpha / 2))`
                # `endspan = max(1, endspan)` if endspan_alpha > 0 and endspan was 0.
                # Let's assume self.model.endspan is the direct count for now, or calculated in Earth class.
                # For now, a simpler fixed exclusion if parent is intercept:
                if parent_bf is not None and parent_bf.degree() == 0 : # Parent is intercept
                    # py-earth default: exclude max knot for additive, unless only 2 unique values then exclude none.
                    if len(unique_X_vals) > 2:
                        return unique_X_vals[:-1]
                    return unique_X_vals # Allow splitting if only two unique values

            # If not additive or specific endspan logic not fully active, return all unique values.
            # More refined logic would slice unique_X_vals based on calculated end_span count.
            # e.g., unique_X_vals[calculated_endspan : -calculated_endspan or None]
            # and then apply min_span between these.
            # For now, let's return all unique values if not the simple additive case above.
            return unique_X_vals

        # If len(unique_X_vals) <= 2, and not the special additive case above,
        # it implies not enough points to make a meaningful split after considering endspan.
        return np.array([])


    def _generate_candidates(self) -> list[tuple[BasisFunction, BasisFunction]]:
        """
        Generates candidate pairs of basis functions to add to the model.
        Each candidate is a pair of HingeBasisFunctions (left and right hinge)
        created by splitting an existing basis function on a new variable and knot.
        """
        candidate_basis_function_pairs = []

        for parent_bf in self.current_basis_functions:
            if parent_bf.degree() >= self.model.max_degree:
                continue # Cannot increase degree further for this parent

            parent_involved_vars = parent_bf.get_involved_variables()

            for var_idx in range(self.n_features):
                if var_idx in parent_involved_vars:
                    continue # Variable already used in this lineage, skip to prevent x_i * x_i type terms

                # Get potential knot values for this variable (X_train[:, var_idx])
                # This needs to respect min_span/end_span constraints.
                potential_knots = self._get_allowable_knot_values(self.X_train[:, var_idx], parent_bf, var_idx)

                for knot_val in potential_knots:
                    # Create candidate pair: one right hinge, one left hinge
                    # Both are children of parent_bf, operating on var_idx at knot_val

                    # Candidate 1: parent_bf * max(0, X_var - knot)
                    bf_right = HingeBasisFunction(
                        variable_idx=var_idx,
                        knot_val=knot_val,
                        is_right_hinge=True,
                        parent_bf=parent_bf,
                        # variable_name will be auto-generated if None
                    )

                    # Candidate 2: parent_bf * max(0, knot - X_var)
                    bf_left = HingeBasisFunction(
                        variable_idx=var_idx,
                        knot_val=knot_val,
                        is_right_hinge=False,
                        parent_bf=parent_bf,
                    )

                    # TODO: Add more sophisticated min_span check here.
                    # A simple check: if the knot is at an existing value, ensure the hinge isn't trivial.
                    # Example: if knot_val is min or max of X_train[:, var_idx], one of these might be all zeros.
                    # The _get_allowable_knot_values should ideally handle this.
                    # For now, we assume _get_allowable_knot_values provides valid knots.
                    candidate_basis_function_pairs.append((bf_left, bf_right))

        # TODO: Consider adding LinearBasisFunction candidates if allowed by model config (e.g. self.model.allow_linear)
        # This is usually done if no hinge on a variable provides improvement.
        # For now, focusing on hinge-based splitting.

        return candidate_basis_function_pairs

    def _find_best_candidate_pair(self):
        """
        Evaluates all candidate basis function pairs and finds the one that
        maximally reduces the RSS (Residual Sum of Squares).

        Updates internal attributes like `_best_candidate_pair`, `_min_candidate_rss`,
        `_best_new_B_matrix`, and `_best_new_coeffs`.
        """
        self._min_candidate_rss = self.current_rss
        self._best_candidate_pair = None
        self._best_new_B_matrix = None # Store the B matrix for the best pair
        self._best_new_coeffs = None   # Store the coeffs for the best pair

        candidate_pairs = self._generate_candidates()

        if not candidate_pairs:
            return # No candidates to evaluate

        for bf_left, bf_right in candidate_pairs:
            # Create temporary list of basis functions for this candidate pair
            # The new model will include current functions + bf_left + bf_right
            temp_basis_list = self.current_basis_functions + [bf_left, bf_right]

            # Build the new basis matrix B_candidate
            # We need to be careful if current_B_matrix is None (e.g. first step after intercept)
            # or if it's empty. _build_basis_matrix should handle a list.
            B_candidate = self._build_basis_matrix(self.X_train, temp_basis_list)

            if B_candidate.shape[1] == 0 : # Should not happen if we always have intercept
                continue

            # Check for rank deficiency before lstsq if possible, or handle error from it
            # A simple check: if number of columns > number of samples (after adding 2)
            if B_candidate.shape[1] >= self.n_samples: # Cannot be solved uniquely or reliably
                # This might happen with very few samples or many terms.
                # Pruning pass would typically handle overly complex models.
                # For forward pass, we might just ignore such candidates.
                continue

            rss_candidate, coeffs_candidate = self._calculate_rss_and_coeffs(B_candidate, self.y_train)

            if coeffs_candidate is None: # LinAlgError or other issue
                continue

            # Using a small epsilon to ensure actual improvement and handle floating point issues
            if rss_candidate < self._min_candidate_rss - EPSILON:
                self._min_candidate_rss = rss_candidate
                self._best_candidate_pair = (bf_left, bf_right)
                self._best_new_B_matrix = B_candidate
                self._best_new_coeffs = coeffs_candidate

        # After checking all candidates, self._best_candidate_pair will be None if no improvement,
        # or will hold the best pair.
        # self._min_candidate_rss will hold the RSS of that best model.


if __name__ == '__main__':
    # This is a placeholder for example usage or internal tests.
    # It would require a mock Earth model and data.
    class MockEarth:
        def __init__(self, max_degree=1, max_terms=10):
            self.max_degree = max_degree
            self.max_terms = max_terms
            # self.record_ = type('MockRecord', (), {'log_forward_pass_results': lambda *args: None})()


    # X_sample = np.random.rand(100, 3)
    # y_sample = X_sample[:, 0] * 2 - X_sample[:, 1] + np.random.randn(100) * 0.1
    # mock_model = MockEarth()
    # forward_passer = ForwardPasser(mock_model)
    # basis_functions, coefficients = forward_passer.run(X_sample, y_sample)
    # print(f"Selected {len(basis_functions)} basis functions.")
    pass
