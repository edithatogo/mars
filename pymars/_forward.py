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
from ._util import calculate_gcv, gcv_penalty_cost_effective_parameters # For GCV calculations

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
            # Determine how many terms would be added by the best candidate (1 for linear, 2 for hinge pair)
            # This is needed for the max_terms check.
            # We call _find_best_candidate_addition first, then check max_terms.

            self._find_best_candidate_addition()

            if self._best_candidate_addition is None or self._min_candidate_rss >= self.current_rss - EPSILON:
                # print(f"Stopping: No further RSS improvement or no valid candidates. Current RSS: {self.current_rss:.4f}, Best Candidate RSS: {self._min_candidate_rss:.4f}")
                break

            # An improvement was found. Now check max_terms.
            bf1_cand, bf2_cand_or_None = self._best_candidate_addition
            num_terms_to_add_this_step = 1
            if bf2_cand_or_None is not None:
                num_terms_to_add_this_step = 2

            if len(self.current_basis_functions) + num_terms_to_add_this_step > max_terms_for_loop:
                # print(f"Stopping: Max terms ({max_terms_for_loop}) would be exceeded by adding {num_terms_to_add_this_step} term(s).")
                break

            # --- Add the best found addition to the model ---
            # Calculate GCV reduction for this addition
            gcv_before, _ = self._calculate_gcv_for_basis_set(self.current_basis_functions)

            # GCV after adding the term(s) uses self._min_candidate_rss and self._best_new_B_matrix
            effective_params_with_addition = gcv_penalty_cost_effective_parameters(
                self._best_new_B_matrix.shape[1], self.model.penalty, self.n_samples
            )
            gcv_with_addition = calculate_gcv(self._min_candidate_rss, effective_params_with_addition, self.n_samples)

            gcv_reduction = None
            if gcv_before is not None and gcv_with_addition is not None:
                gcv_reduction = gcv_before - gcv_with_addition

            # Calculate RSS reduction
            rss_reduction = self.current_rss - self._min_candidate_rss

            # Assign scores to the added basis function(s)
            bf1_cand.gcv_score_ = gcv_reduction if gcv_reduction is not None else 0.0
            bf1_cand.rss_score_ = rss_reduction
            if bf2_cand_or_None is not None:
                bf2_cand_or_None.gcv_score_ = gcv_reduction if gcv_reduction is not None else 0.0
                bf2_cand_or_None.rss_score_ = rss_reduction

            # Update model state
            terms_added_this_iteration = [bf1_cand]
            if bf2_cand_or_None is not None:
                terms_added_this_iteration.append(bf2_cand_or_None)

            self.current_basis_functions.extend(terms_added_this_iteration)
            self.current_B_matrix = self._best_new_B_matrix
            self.current_coefficients = self._best_new_coeffs
            self.current_rss = self._min_candidate_rss

            # Logging, etc.
            # term_names = " & ".join([str(t) for t in terms_added_this_iteration])
            # print(f"Added: {term_names}. New RSS: {self.current_rss:.4f}. GCV reduction: {gcv_reduction if gcv_reduction is not None else 'N/A'}. Num terms: {len(self.current_basis_functions)}")

            if self.model.record_ is not None and hasattr(self.model.record_, 'log_forward_pass_step'):
                self.model.record_.log_forward_pass_step(
                    self.current_basis_functions, self.current_coefficients, self.current_rss
                )

            # Reset for next iteration's search
            self._best_candidate_addition = None
            self._best_new_B_matrix = None
            self._best_new_coeffs = None
            # _min_candidate_rss is implicitly reset at the start of _find_best_candidate_addition


        # print(f"Forward pass complete. Final RSS: {self.current_rss:.4f}, Num terms: {len(self.current_basis_functions)}")
        return self.current_basis_functions, self.current_coefficients

    def _calculate_gcv_for_basis_set(self, basis_functions: list[BasisFunction]) -> tuple[float | None, np.ndarray | None]:
        """
        Calculates GCV and coefficients for a given set of basis functions.
        Returns (gcv, coeffs) or (None, None) if GCV cannot be computed.
        """
        if not basis_functions: # Should not happen if intercept is always there
            return np.inf, None

        B_matrix = self._build_basis_matrix(self.X_train, basis_functions)
        if B_matrix.shape[1] == 0: # No terms in basis matrix
            # GCV for a model that predicts mean (intercept only, but B_matrix is empty)
            # This case needs careful handling. If basis_functions = [ConstantBasisFunction()],
            # B_matrix should be (N,1). If basis_functions is truly empty, it's different.
            # For now, assume if basis_functions is not empty, B_matrix won't be empty unless error.
            # If it *is* empty, GCV is essentially for a model predicting the mean.
            rss = np.sum((self.y_train - np.mean(self.y_train))**2)
            # Effective parameters for intercept-only model is typically 1.
            # However, gcv_penalty_cost_effective_parameters expects num_model_params (cols in B)
            # This path might indicate an issue if basis_functions is not empty but B_matrix is.
            effective_params = gcv_penalty_cost_effective_parameters(1, self.model.penalty, self.n_samples)
            gcv = calculate_gcv(rss, effective_params, self.n_samples)
            return gcv, np.array([np.mean(self.y_train)])


        rss, coeffs = self._calculate_rss_and_coeffs(B_matrix, self.y_train)

        if rss == np.inf or coeffs is None:
            return np.inf, None # Unstable fit or error

        # Calculate effective number of parameters for GCV
        # num_model_params = B_matrix.shape[1] (number of basis functions in current set)
        effective_num_params = gcv_penalty_cost_effective_parameters(
            B_matrix.shape[1], self.model.penalty, self.n_samples
        )

        gcv_score = calculate_gcv(rss, effective_num_params, self.n_samples)
        return gcv_score, coeffs

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

        if not unique_X_vals.size: # No unique values, no knots.
            return np.array([])

        # Determine endspan_count
        # print(f"DEBUG: In _get_allowable_knot_values for var_idx {var_idx}: model.endspan={self.model.endspan}, model.endspan_alpha={self.model.endspan_alpha}, n_features={self.n_features}, len(unique_X_vals)={len(unique_X_vals)}")
        endspan_count = 0
        if self.model.endspan >= 0: # Direct value provided
            endspan_count = self.model.endspan
            # print(f"DEBUG: Using direct endspan: {endspan_count}")
        elif self.model.endspan_alpha > 0: # Calculate from alpha
            if self.n_features > 0:
                # Handle potential log(0) or negative if endspan_alpha is not in (0,1), though it should be.
                # Clamp argument to log2 to be positive.
                log_arg = self.model.endspan_alpha / self.n_features
                if log_arg <= 0: # Should not happen with valid endspan_alpha and n_features > 0
                    val = 3.0 # Default to a value that might lead to endspan_count=1 or 0 after rounding
                else:
                    val = 3.0 - np.log2(log_arg)

                endspan_count = int(np.round(val))
                endspan_count = max(0, endspan_count)
                if endspan_count == 0: # py-earth ensures endspan is at least 1 if endspan_alpha > 0
                    endspan_count = 1
            else: # No features, no endspan applicable, though this case should be handled earlier
                endspan_count = 0
        # If endspan_alpha <= 0 and self.model.endspan < 0, endspan_count remains 0.

        if 2 * endspan_count >= len(unique_X_vals):
            return np.array([]) # Not enough unique values to satisfy endspan

        candidate_knots = unique_X_vals[endspan_count : len(unique_X_vals) - endspan_count]

        if not candidate_knots.size: # Check if slicing resulted in empty
             return np.array([])

        # Special handling for additive terms (parent is intercept):
        # py-earth often excludes the maximum value from the (endspan-filtered) candidate knots
        # to prevent creating a hinge that's non-zero for only the very last point region.
        # This is done unless only one candidate knot would remain after this exclusion.
        if parent_bf is not None and parent_bf.degree() == 0:
            if len(candidate_knots) > 1:
                candidate_knots = candidate_knots[:-1] # Exclude the largest remaining knot
            # If len(candidate_knots) == 1, we keep it. If it was empty already, still empty.

        if not candidate_knots.size:
            return np.array([])

        # Determine min_span_count
        min_span_count = 0
        if self.model.minspan >= 0: # Direct value provided
            min_span_count = max(0, self.model.minspan)
        elif self.model.minspan_alpha > 0: # Calculate from alpha
            # Calculate count_parent_nonzero: samples where parent_bf is non-zero
            if parent_bf.degree() == 0: # Intercept is always non-zero (value 1) for all samples
                count_parent_nonzero = self.n_samples
            else:
                parent_transform_values = parent_bf.transform(self.X_train)
                count_parent_nonzero = np.sum(parent_transform_values != 0)

            if count_parent_nonzero > 0 and self.n_features > 0 and \
               0 < self.model.minspan_alpha < 1: # Valid range for alpha for formula
                # Formula from py-earth's documentation (simplified for direct calculation)
                # Original: (int) -log2(-(1.0/(n*count))*log(1.0-minspan_alpha)) / 2.5
                # This formula is sensitive; direct translation to numpy logs:
                try:
                    log_val = np.log(1.0 - self.model.minspan_alpha)
                    # Ensure argument to outer log2 is positive
                    inner_term = -(1.0 / (self.n_features * count_parent_nonzero)) * log_val
                    if inner_term <= 0: # Avoid log of non-positive
                        min_span_float = 0.0
                    else:
                        min_span_float = -np.log2(inner_term) / 2.5
                    min_span_count = max(1, int(np.round(min_span_float)))
                except (ValueError, FloatingPointError): # Catch math errors from logs
                    min_span_count = 1 # Default to 1 if calculation fails
            elif self.model.minspan_alpha == 0: # No minspan constraint by alpha
                 min_span_count = 0
            else: # Default if alpha is invalid or other conditions not met for formula
                min_span_count = 1
        # If minspan_alpha <= 0 and self.model.minspan < 0, min_span_count remains 0.
        # ---- Start of new py-earth aligned implementation ----

        # 1. Determine n_vars_for_calc (for alpha calculations)
        # TODO: Refine n_vars_for_calc for closer py-earth parity.
        # py-earth uses n_effective_variables for minspan_alpha (n_features for additive,
        # max(1, n_features - parent.num_variables) for interactions) and
        # n_selected_variables (unique vars in current model) for endspan_alpha.
        # Using self.n_features is a simplification for now.
        n_vars_for_calc = self.n_features
        if n_vars_for_calc == 0: n_vars_for_calc = 1 # Avoid division by zero if n_features is 0

        # 2. Calculate endspan_abs (integer number of unique points to exclude from each end)
        endspan_abs = 0
        if self.model.endspan >= 0:
            endspan_abs = self.model.endspan
        elif self.model.endspan_alpha > 0:
            try:
                # Ensure argument to log2 is positive
                log_arg = self.model.endspan_alpha / n_vars_for_calc
                if log_arg <= 0: # Should not happen with valid endspan_alpha
                    val = 3.0
                else:
                    val = 3.0 - np.log2(log_arg) # Using np.log2

                endspan_abs = int(round(val))
                endspan_abs = max(0, endspan_abs)
                if endspan_abs == 0: # py-earth ensures endspan is at least 1 if endspan_alpha > 0
                    endspan_abs = 1
            except (ValueError, FloatingPointError): # Catch math errors
                endspan_abs = 1 # Default if calculation fails

        # 3. Prepare Active Data for the current variable X_col (original full column)
        if parent_bf.is_constant():
            p_parent_active = np.ones(self.n_samples, dtype=bool)
            count_parent_nonzero = self.n_samples
        else:
            parent_transform = parent_bf.transform(self.X_train)
            p_parent_active = (parent_transform != 0)
            count_parent_nonzero = np.sum(p_parent_active)

        if count_parent_nonzero == 0:
            return np.array([]) # No active region from parent

        X_values_for_knots = self.X_train[p_parent_active, var_idx]

        if X_values_for_knots.size == 0:
            return np.array([])

        # 4. Get unique sorted values from the active region
        unique_sorted_X_active = np.unique(X_values_for_knots) # Already sorted by np.unique

        # 5. Apply endspan_abs to unique sorted active values
        num_unique_active = len(unique_sorted_X_active)
        if 2 * endspan_abs >= num_unique_active:
            return np.array([])

        potential_knots_after_endspan = unique_sorted_X_active[endspan_abs : num_unique_active - endspan_abs]

        if not potential_knots_after_endspan.size:
            return np.array([])

        # 6. Special Rule for Additive Terms (parent is Intercept)
        if parent_bf.is_constant():
            if len(potential_knots_after_endspan) > 1:
                potential_knots_after_endspan = potential_knots_after_endspan[:-1] # Remove the largest knot

        if not potential_knots_after_endspan.size:
            return np.array([])

        # 7. Calculate minspan_abs (integer number of distinct subsequent knots to skip)
        minspan_abs = 0
        if self.model.minspan >= 0:
            minspan_abs = self.model.minspan
        elif self.model.minspan_alpha > 0:
            if count_parent_nonzero > 0 and n_vars_for_calc > 0 and \
               0 < self.model.minspan_alpha < 1:
                try:
                    # Formula from py-earth _knot_search.pyx
                    log_val = np.log(1.0 - self.model.minspan_alpha) # Natural log
                    inner_term = -(1.0 / (n_vars_for_calc * count_parent_nonzero)) * log_val
                    if inner_term <= 0:
                        minspan_float = 0.0
                    else:
                        minspan_float = -np.log2(inner_term) / 2.5 # np.log2
                    minspan_abs = int(round(minspan_float))
                    minspan_abs = max(0, minspan_abs) # py-earth seems to allow 0 here from formula
                                                      # but often defaults to 1 if alpha used and calc positive
                    # py-earth: minspan_ = <int> (...) then if minspan<0 (alpha used), minspan_ can be 0.
                    # Let's stick to max(0, result_from_formula).
                except (ValueError, FloatingPointError):
                    minspan_abs = 1 # Default if calc fails, similar to py-earth implicit behavior
            # If alpha is 0 or invalid, minspan_abs remains 0 (no constraint)

        # 8. Apply minspan_abs (cooldown logic)
        final_allowable_knots = []
        minspan_countdown = 0
        for knot_candidate_val in potential_knots_after_endspan: # These are already unique and sorted
            if minspan_countdown > 0:
                minspan_countdown -= 1
                continue

            final_allowable_knots.append(knot_candidate_val)
            minspan_countdown = max(0, minspan_abs - 1) # Skip (minspan_abs - 1) next distinct knots

        return np.array(final_allowable_knots)
        # ---- End of new py-earth aligned implementation ----


    def _generate_candidates(self) -> list[tuple[BasisFunction, BasisFunction | None]]:
        """
        Generates candidate additions to the model.

        Candidate additions can be:
        1. Single `LinearBasisFunction` terms (if `self.model.allow_linear` is True).
           These are formed by trying to add a linear effect for each variable,
           potentially as an interaction with an existing `parent_bf`.
           Stored as `(linear_bf, None)`.
        2. Pairs of `HingeBasisFunction` terms. These are formed by splitting
           an existing `parent_bf` on a new variable and knot.
           Stored as `(hinge_bf_left, hinge_bf_right)`.

        The generation order is linear candidates first, then hinge candidates.
        Degree constraints (`self.model.max_degree`) and rules against redundant
        variable usage are respected.

        Returns
        -------
        list[tuple[BasisFunction, BasisFunction | None]]
            A list of candidate additions.
        """
        candidate_additions: list[tuple[BasisFunction, BasisFunction | None]] = []

        # Generate Linear Term Candidates First (if allowed)
        if self.model.allow_linear:
            for parent_bf in self.current_basis_functions:
                # For a new linear term (degree 1 itself), the resulting model term's degree
                # will be parent_bf.degree() + 1. This must not exceed self.model.max_degree.
                if parent_bf.degree() + 1 > self.model.max_degree:
                    continue

                parent_involved_vars = parent_bf.get_involved_variables()
                for var_idx in range(self.n_features):
                    if var_idx in parent_involved_vars:
                        # Avoid using the same variable in an interaction like Parent(X_i) * Linear(X_i)
                        # if X_i is already part of Parent(X_i)'s definition.
                        continue

                    linear_candidate = LinearBasisFunction(variable_idx=var_idx, parent_bf=parent_bf)
                    candidate_additions.append((linear_candidate, None))

        # Generate Hinge Pair Candidates
        for parent_bf in self.current_basis_functions:
            # For a new hinge term (degree 1 itself), the resulting model term's degree
            # will be parent_bf.degree() + 1. This must not exceed self.model.max_degree.
            if parent_bf.degree() + 1 > self.model.max_degree:
                continue

            parent_involved_vars = parent_bf.get_involved_variables()

            for var_idx in range(self.n_features):
                if var_idx in parent_involved_vars:
                    # Cannot create a hinge on a variable already directly in the parent's lineage
                    # This prevents terms like (x0>1)*(x0>2) being formed from parent (x0>1) in one step.
                    # Interactions like (x0>1)*(x1>2) are allowed if x1 is not in parent's vars.
                    continue

                potential_knots = self._get_allowable_knot_values(self.X_train[:, var_idx], parent_bf, var_idx)

                for knot_val in potential_knots:
                    bf_right = HingeBasisFunction(
                        variable_idx=var_idx, knot_val=knot_val, is_right_hinge=True, parent_bf=parent_bf
                    )
                    bf_left = HingeBasisFunction(
                        variable_idx=var_idx, knot_val=knot_val, is_right_hinge=False, parent_bf=parent_bf
                    )
                    candidate_additions.append((bf_left, bf_right))

        # The Linear Term Candidates are already generated at the beginning of this method.
        # The duplicated block below was an error and is removed.

        return candidate_additions

    def _find_best_candidate_addition(self): # Renamed from _find_best_candidate_pair
        """
        Evaluates all candidate basis function additions (linear singles or hinge pairs)
        generated by `_generate_candidates` and identifies the one that results in the
        greatest reduction in the Residual Sum of Squares (RSS).

        This method updates the following internal attributes if a better model is found:
        - `_best_candidate_addition`: Stores the tuple `(bf1, bf2_or_None)` of the best addition.
        - `_min_candidate_rss`: The RSS value of the model with the best addition.
        - `_best_new_B_matrix`: The basis matrix corresponding to the model with the best addition.
        - `_best_new_coeffs`: The coefficients for the model with the best addition.

        If no candidate addition improves the RSS, `_best_candidate_addition` remains None.
        """
        self._min_candidate_rss = self.current_rss
        self._best_candidate_addition = None # Changed: self._best_candidate_pair to self._best_candidate_addition
        self._best_new_B_matrix = None
        self._best_new_coeffs = None

        candidate_additions = self._generate_candidates()

        if not candidate_additions:
            return

        # This loop needs to be updated in the next step to handle (bf1, None) for linear terms
        for bf_left, bf_right in candidate_additions: # bf_right will be None for linear terms
            terms_to_add = [bf_left]
            if bf_right is not None: # This check handles both pairs and singles
                terms_to_add.append(bf_right)

            temp_basis_list = self.current_basis_functions + terms_to_add

            B_candidate = self._build_basis_matrix(self.X_train, temp_basis_list)

            if B_candidate.shape[1] == 0 :
                continue

            if B_candidate.shape[1] >= self.n_samples:
                continue

            rss_candidate, coeffs_candidate = self._calculate_rss_and_coeffs(B_candidate, self.y_train)

            if coeffs_candidate is None:
                continue

            if rss_candidate < self._min_candidate_rss - EPSILON:
                self._min_candidate_rss = rss_candidate
                self._best_candidate_addition = (bf_left, bf_right) # Store the selected addition
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
