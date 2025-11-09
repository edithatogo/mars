
"""
The forward pass of the MARS algorithm.

This module is responsible for iteratively adding basis functions to the model
to minimize a criterion (e.g., sum of squared errors).
"""

import logging
from typing import Optional

import numpy as np

from ._basis import (
    BasisFunction,
    CategoricalBasisFunction,
    ConstantBasisFunction,
    HingeBasisFunction,
    LinearBasisFunction,
    MissingnessBasisFunction,
)
from ._util import (  # For GCV calculations
    calculate_gcv,
    gcv_penalty_cost_effective_parameters,
)
from .earth import Earth  # For type hinting

logger = logging.getLogger(__name__)

# Define a small constant for numerical stability if needed
EPSILON = np.finfo(float).eps

class ForwardPasser:
    """
    Manages the forward pass of the MARS algorithm.
    """
    def __init__(self, earth_model: Earth):
        self.model = earth_model
        self.X_train = None
        self.y_train = None
        self.n_samples = 0
        self.n_features = 0
        self.current_basis_functions: list[BasisFunction] = []
        self.current_B_matrix = None
        self.current_coefficients = None
        self.current_rss = np.inf
        self._best_candidate_addition = None
        self._best_new_B_matrix = None
        self._best_new_coeffs = None
        self._min_candidate_rss = np.inf
        self.X_fit_original = None
        self.missing_mask = None
        self.categorical_features = self.model.categorical_features

    def _calculate_rss_and_coeffs(
        self, B_matrix: np.ndarray, y: np.ndarray, *, drop_nan_rows: bool = True
    ) -> tuple[float, Optional[np.ndarray], int]:
        if B_matrix is None or B_matrix.shape[1] == 0:
            mean_y = np.mean(y)
            rss = np.sum((y - mean_y)**2)
            num_valid_rows = len(y)
            coeffs_for_mean = np.array([mean_y]) if (B_matrix is not None and B_matrix.shape[1] == 0) else None
            return rss, coeffs_for_mean, num_valid_rows

        if drop_nan_rows:
            # Drop rows with NaNs so RSS and GCV are based only on rows where all
            # involved basis functions are defined.  This mirrors the behaviour of
            # the original MARS algorithm when evaluating candidate terms.
            if np.isnan(B_matrix).any():
                valid_rows_mask = ~np.any(np.isnan(B_matrix), axis=1)
                B_complete = B_matrix[valid_rows_mask]
                y = y[valid_rows_mask]
            else:
                B_complete = B_matrix
        else:
            # Treat NaNs as zeros.  This is useful when evaluating missingness
            # candidates so that rows with NaNs are still considered.
            B_complete = np.nan_to_num(B_matrix, nan=0.0)
        num_valid_rows = B_complete.shape[0]

        if num_valid_rows == 0:
            return np.inf, None, 0

        try:
            if B_complete.ndim == 1:
                B_complete = B_complete.reshape(-1, 1)
            coeffs, residuals_sum_sq, rank, s = np.linalg.lstsq(B_complete, y, rcond=None)
            if residuals_sum_sq.size == 0 or rank < B_complete.shape[1]:
                y_pred_complete = B_complete @ coeffs
                rss = np.sum((y - y_pred_complete) ** 2)
            else:
                rss = residuals_sum_sq[0]
            return rss, coeffs, num_valid_rows
        except np.linalg.LinAlgError:
            return np.inf, None, num_valid_rows

    def _build_basis_matrix(self, X_processed: np.ndarray, basis_functions: list[BasisFunction]) -> np.ndarray:
        if not basis_functions:
            return np.empty((X_processed.shape[0], 0))

        # Preallocate basis matrix for efficiency instead of building a list of
        # arrays and hstacking them (which triggers many temporary allocations
        # and copies). This also allows us to avoid repeated dtype checks during
        # np.hstack.
        n_samples = X_processed.shape[0]
        B_matrix = np.empty((n_samples, len(basis_functions)), dtype=float)
        for idx, bf in enumerate(basis_functions):
            B_matrix[:, idx] = bf.transform(X_processed, self.missing_mask)
        return B_matrix

    def run(self, X_fit_processed: np.ndarray, y_fit: np.ndarray,
            missing_mask: np.ndarray, X_fit_original: np.ndarray) -> tuple[list[BasisFunction], np.ndarray]:
        self.X_train = X_fit_processed
        self.y_train = y_fit
        self.missing_mask = missing_mask
        self.X_fit_original = X_fit_original

        if self.y_train.ndim > 1 and self.y_train.shape[1] > 1:
            raise ValueError("ForwardPasser currently supports only single-output targets (y should be 1D).")
        self.y_train = self.y_train.ravel()

        self.n_samples, self.n_features = self.X_train.shape

        intercept_bf = ConstantBasisFunction()
        self.current_basis_functions = [intercept_bf]
        self.current_B_matrix = self._build_basis_matrix(self.X_train, self.current_basis_functions)

        rss, coeffs, _ = self._calculate_rss_and_coeffs(
            self.current_B_matrix, self.y_train
        )  # Unpack 3
        if coeffs is None:
            logger.warning(
                "Could not calculate initial coefficients for intercept model."
            )
            return [], np.array([])

        self.current_coefficients = coeffs
        self.current_rss = rss

        if self.model.record_ is not None and hasattr(self.model.record_, 'log_forward_pass_step'):
            self.model.record_.log_forward_pass_step(
                self.current_basis_functions, self.current_coefficients, self.current_rss
            )

        max_terms_for_loop = self.model.max_terms
        if max_terms_for_loop is None:
            max_terms_for_loop = min(self.n_samples - 1, max(21, 2 * self.n_features + 1))

        logger.info(
            "Initial model setup: %d term(s), RSS=%.4f",
            len(self.current_basis_functions),
            self.current_rss,
        )
        if self.current_coefficients is not None:
            logger.debug("Initial coeffs: %s", self.current_coefficients)

        while True:
            self._find_best_candidate_addition()

            if self._best_candidate_addition is None or self._min_candidate_rss >= self.current_rss - EPSILON:
                break

            bf1_cand, bf2_cand_or_None = self._best_candidate_addition
            num_terms_to_add_this_step = 1
            if bf2_cand_or_None is not None:
                num_terms_to_add_this_step = 2

            if len(self.current_basis_functions) + num_terms_to_add_this_step > max_terms_for_loop:
                break

            gcv_before, _ = self._calculate_gcv_for_basis_set(self.current_basis_functions)

            num_valid_rows_for_gcv_calc = self.n_samples # Default
            if self._best_new_B_matrix is not None:
                best_cand_valid_rows_mask = ~np.any(np.isnan(self._best_new_B_matrix), axis=1)
                num_valid_rows_for_best_candidate = np.sum(best_cand_valid_rows_mask)
                if num_valid_rows_for_best_candidate > 0 :
                     num_valid_rows_for_gcv_calc = num_valid_rows_for_best_candidate

            # Determine num_hinge_terms for the candidate model
            candidate_model_terms_list = self.current_basis_functions + ([bf1_cand] if bf2_cand_or_None is None else [bf1_cand, bf2_cand_or_None])
            num_hinge_terms_candidate = sum(isinstance(bf, HingeBasisFunction) for bf in candidate_model_terms_list)
            num_total_terms_candidate = self._best_new_B_matrix.shape[1] # Number of columns in the candidate basis matrix

            effective_params_for_addition = gcv_penalty_cost_effective_parameters(
                num_total_terms_candidate,
                num_hinge_terms_candidate,
                self.model.penalty,
                num_valid_rows_for_gcv_calc
            )
            gcv_with_addition = calculate_gcv(
                self._min_candidate_rss,
                num_valid_rows_for_gcv_calc,
                effective_params_for_addition
            )

            gcv_reduction = None
            if gcv_before is not None and gcv_with_addition is not None:
                gcv_reduction = gcv_before - gcv_with_addition

            rss_reduction = self.current_rss - self._min_candidate_rss

            bf1_cand.gcv_score_ = gcv_reduction if gcv_reduction is not None else 0.0
            bf1_cand.rss_score_ = rss_reduction
            if bf2_cand_or_None is not None:
                bf2_cand_or_None.gcv_score_ = gcv_reduction if gcv_reduction is not None else 0.0
                bf2_cand_or_None.rss_score_ = rss_reduction

            terms_added_this_iteration = [bf1_cand]
            if bf2_cand_or_None is not None:
                terms_added_this_iteration.append(bf2_cand_or_None)

            self.current_basis_functions.extend(terms_added_this_iteration)
            self.current_B_matrix = self._best_new_B_matrix
            self.current_coefficients = self._best_new_coeffs
            self.current_rss = self._min_candidate_rss

            if self.model.record_ is not None and hasattr(self.model.record_, 'log_forward_pass_step'):
                self.model.record_.log_forward_pass_step(
                    self.current_basis_functions, self.current_coefficients, self.current_rss
                )

            self._best_candidate_addition = None
            self._best_new_B_matrix = None
            self._best_new_coeffs = None

        return self.current_basis_functions, self.current_coefficients

    def _calculate_gcv_for_basis_set(self, basis_functions: list[BasisFunction]) -> tuple[Optional[float], Optional[np.ndarray]]:
        if not basis_functions:
            # This implies an intercept-only model for GCV calculation purposes
            rss_intercept_only = np.sum((self.y_train - np.mean(self.y_train))**2)
            num_terms_intercept = 1
            num_hinge_terms_intercept = 0
            effective_params_intercept = gcv_penalty_cost_effective_parameters(
                num_terms_intercept, num_hinge_terms_intercept, self.model.penalty, self.n_samples
            )
            gcv_intercept = calculate_gcv(rss_intercept_only, self.n_samples, effective_params_intercept)
            return gcv_intercept, np.array([np.mean(self.y_train)])

        B_matrix = self._build_basis_matrix(self.X_train, basis_functions)
        rss, coeffs, num_valid_rows = self._calculate_rss_and_coeffs(
            B_matrix, self.y_train
        )

        if rss == np.inf or coeffs is None or num_valid_rows == 0:
            return np.inf, None

        num_actual_coeffs = coeffs.shape[0]
        if num_actual_coeffs == 0 : # Should be caught by coeffs is None, but as a safeguard
             return np.inf, None

        # Fallback for singular matrix not resulting in None coeffs but zero coeffs (highly unlikely with lstsq)
        # or if B_matrix was non-empty but all columns were zero for the valid_rows.
        # This specific block might be redundant if _calculate_rss_and_coeffs robustly returns None for coeffs
        # or rss=inf in all singular/degenerate cases.
        num_terms_in_b_matrix = B_matrix.shape[1]
        if B_matrix.size > 0 and num_terms_in_b_matrix > 0 and num_actual_coeffs == 0 and num_valid_rows > 0:
            # This scenario (matrix had columns, but fit resulted in zero coeffs) is problematic.
            # Defaulting to a very high GCV.
            return np.inf, None


        num_hinge_terms = sum(isinstance(bf, HingeBasisFunction) for bf in basis_functions)
        effective_num_params = gcv_penalty_cost_effective_parameters(
            num_actual_coeffs, num_hinge_terms, self.model.penalty, num_valid_rows
        )

        gcv_score = calculate_gcv(rss, num_valid_rows, effective_num_params)
        return gcv_score, coeffs

    def _get_allowable_knot_values(self, X_col_original_for_var: np.ndarray, parent_bf: BasisFunction, var_idx: int) -> np.ndarray:
        n_vars_for_calc = self.n_features
        if n_vars_for_calc == 0: n_vars_for_calc = 1

        endspan_abs = 0
        if self.model.endspan >= 0:
            endspan_abs = self.model.endspan
        elif self.model.endspan_alpha > 0:
            try:
                log_arg = self.model.endspan_alpha / n_vars_for_calc
                if log_arg <= 0: val = 3.0
                else: val = 3.0 - np.log2(log_arg)
                endspan_abs = int(round(val))
                endspan_abs = max(0, endspan_abs)
                if endspan_abs == 0: endspan_abs = 1
            except (ValueError, FloatingPointError): endspan_abs = 1

        count_parent_nonzero_for_minspan = 0 # Initialize for minspan calculation
        if parent_bf.is_constant():
            p_parent_active = np.ones(self.n_samples, dtype=bool)
            count_parent_nonzero_for_minspan = self.n_samples
        else:
            parent_transform = parent_bf.transform(self.X_train, self.missing_mask) # Use self.X_train (X_processed) and self.missing_mask
            p_parent_active = (parent_transform != 0) & (~np.isnan(parent_transform))
            count_parent_nonzero_for_minspan = np.sum(p_parent_active)

        if count_parent_nonzero_for_minspan == 0:
            return np.array([])

        current_var_missing_mask = self.missing_mask[:, var_idx]
        truly_usable_for_knots_mask = p_parent_active & ~current_var_missing_mask
        X_values_for_knots = X_col_original_for_var[truly_usable_for_knots_mask]

        if X_values_for_knots.size == 0:
            return np.array([])

        unique_sorted_X_active = np.unique(X_values_for_knots)

        num_unique_active = len(unique_sorted_X_active)
        if 2 * endspan_abs >= num_unique_active:
            return np.array([])

        potential_knots_after_endspan = unique_sorted_X_active[endspan_abs : num_unique_active - endspan_abs]

        if not potential_knots_after_endspan.size:
            return np.array([])

        if parent_bf.is_constant(): # Additive rule
            if len(potential_knots_after_endspan) > 1:
                potential_knots_after_endspan = potential_knots_after_endspan[:-1]

        if not potential_knots_after_endspan.size:
            return np.array([])

        minspan_abs = 0 # This is the 'cooldown' count
        if self.model.minspan >= 0:
            minspan_abs = self.model.minspan
        elif self.model.minspan_alpha > 0:
            # count_parent_nonzero_for_minspan is already calculated above based on p_parent_active
            if count_parent_nonzero_for_minspan > 0 and n_vars_for_calc > 0 and \
               0 < self.model.minspan_alpha < 1:
                try:
                    log_val = np.log(1.0 - self.model.minspan_alpha)
                    inner_term = -(1.0 / (n_vars_for_calc * count_parent_nonzero_for_minspan)) * log_val
                    if inner_term <= 0: min_span_float = 0.0
                    else: min_span_float = -np.log2(inner_term) / 2.5
                    minspan_abs = int(round(min_span_float))
                    minspan_abs = max(0, minspan_abs)
                except (ValueError, FloatingPointError): minspan_abs = 1

        final_allowable_knots = []
        minspan_countdown = 0
        for knot_candidate_val in potential_knots_after_endspan:
            if minspan_countdown > 0:
                minspan_countdown -= 1
                continue
            final_allowable_knots.append(knot_candidate_val)
            minspan_countdown = max(0, minspan_abs - 1)
        return np.array(final_allowable_knots)

    def _generate_candidates(self) -> list[tuple[BasisFunction, Optional[BasisFunction]]]:
        candidate_additions: list[tuple[BasisFunction, Optional[BasisFunction]]] = []
        for parent_bf in self.current_basis_functions:
            if parent_bf.degree() + 1 > self.model.max_degree: continue
            parent_involved_vars = parent_bf.get_involved_variables()
            for var_idx in range(self.n_features):
                if var_idx in parent_involved_vars: continue

                # Handle continuous features (not in categorical list)
                if not self.categorical_features or var_idx not in self.categorical_features:
                    if self.model.allow_linear:
                        exists_linear = any(
                            isinstance(bf_existing, LinearBasisFunction)
                            and bf_existing.variable_idx == var_idx
                            and bf_existing.parent1 == parent_bf
                            for bf_existing in self.current_basis_functions
                        )
                        if not exists_linear:
                            linear_candidate = LinearBasisFunction(variable_idx=var_idx, parent_bf=parent_bf)
                            candidate_additions.append((linear_candidate, None))
                    potential_knots = self._get_allowable_knot_values(self.X_fit_original[:, var_idx], parent_bf, var_idx)
                    for knot_val in potential_knots:
                        bf_right = HingeBasisFunction(variable_idx=var_idx, knot_val=knot_val, is_right_hinge=True, parent_bf=parent_bf)
                        bf_left = HingeBasisFunction(variable_idx=var_idx, knot_val=knot_val, is_right_hinge=False, parent_bf=parent_bf)
                        candidate_additions.append((bf_left, bf_right))
                # Handle categorical features
                elif self.categorical_features and var_idx in self.categorical_features:
                    col_vals = self.X_fit_original[:, var_idx]
                    if self.missing_mask is not None:
                        col_vals = col_vals[~self.missing_mask[:, var_idx]]
                    unique_categories = np.unique(col_vals)
                    for category in unique_categories:
                        categorical_candidate = CategoricalBasisFunction(
                            variable_idx=var_idx, category=category, parent_bf=parent_bf
                        )
                        candidate_additions.append((categorical_candidate, None))

        # Generate MissingnessBasisFunction candidates if allow_missing is True
        if self.model.allow_missing and self.missing_mask is not None:
            # Check if any variable is already used by a MissingnessBasisFunction
            # This is to prevent adding duplicate is_missing(varX) terms.
            # A MissingnessBasisFunction does not interact with a parent for its base definition.
            # It's an indicator for a single variable's original missingness.
            # For it to be part of an interaction, it would become a parent_bf for a subsequent Hinge/Linear.

            current_missingness_vars = set()
            for bf_existing in self.current_basis_functions:
                if isinstance(bf_existing, MissingnessBasisFunction):
                    current_missingness_vars.add(bf_existing.variable_idx)

            for var_idx in range(self.n_features):
                if var_idx in current_missingness_vars: # Already have an is_missing term for this variable
                    continue

                if np.any(self.missing_mask[:, var_idx]): # If this feature has any missing values
                    var_name = None
                    if self.model.record_ and hasattr(self.model.record_, 'feature_names_in_') and \
                       self.model.record_.feature_names_in_ is not None and \
                       var_idx < len(self.model.record_.feature_names_in_):
                        var_name = self.model.record_.feature_names_in_[var_idx]
                    else:
                        var_name = f"x{var_idx}"

                    # Missingness functions are typically added additively initially.
                    # Their "parent" for generation is effectively the intercept.
                    # They don't take a parent_bf in their constructor for degree calculation.
                    missing_bf_candidate = MissingnessBasisFunction(variable_idx=var_idx, variable_name=var_name)
                    candidate_additions.append((missing_bf_candidate, None))

        return candidate_additions

    def _find_best_candidate_addition(self):
        self._min_candidate_rss = self.current_rss
        self._min_candidate_gcv = np.inf
        self._best_candidate_addition = None
        self._best_new_B_matrix = None
        self._best_new_coeffs = None

        candidate_additions = self._generate_candidates()
        if not candidate_additions: return

        max_terms_for_loop = self.model.max_terms
        if max_terms_for_loop is None:
            max_terms_for_loop = min(self.n_samples - 1, max(21, 2 * self.n_features + 1))

        remaining_capacity = max_terms_for_loop - len(self.current_basis_functions)

        for bf1, bf2_or_None in candidate_additions:
            required_terms = 1 + (1 if bf2_or_None is not None else 0)
            if required_terms > remaining_capacity:
                continue
            terms_to_add = [bf1]
            if bf2_or_None is not None:
                terms_to_add.append(bf2_or_None)
            temp_basis_list = self.current_basis_functions + terms_to_add
            B_candidate = self._build_basis_matrix(self.X_train, temp_basis_list)

            if B_candidate.shape[1] == 0 : continue
            # Use n_samples (from processed X) for this check, num_valid_rows from _calc_rss_... will handle actual fit data size
            if B_candidate.shape[1] >= self.n_samples: continue

            drop_nans = True
            if self.model.allow_missing and any(
                isinstance(bf, MissingnessBasisFunction) for bf in terms_to_add
            ):
                drop_nans = False

            rss_candidate, coeffs_candidate, num_valid_rows_candidate = self._calculate_rss_and_coeffs(
                B_candidate, self.y_train, drop_nan_rows=drop_nans
            )

            if coeffs_candidate is None or num_valid_rows_candidate == 0:
                continue

            num_terms_candidate = len(temp_basis_list)
            num_hinge_candidate = sum(isinstance(bf, HingeBasisFunction) for bf in temp_basis_list)
            eff_params = gcv_penalty_cost_effective_parameters(
                num_terms_candidate,
                num_hinge_candidate,
                self.model.penalty,
                num_valid_rows_candidate,
            )
            gcv_candidate = calculate_gcv(rss_candidate, num_valid_rows_candidate, eff_params)

            if (
                gcv_candidate < self._min_candidate_gcv - EPSILON
                or (
                    abs(gcv_candidate - self._min_candidate_gcv) <= EPSILON
                    and rss_candidate < self._min_candidate_rss - EPSILON
                )
            ):
                self._min_candidate_gcv = gcv_candidate
                self._min_candidate_rss = rss_candidate
                self._best_candidate_addition = (bf1, bf2_or_None)
                self._best_new_B_matrix = B_candidate
                self._best_new_coeffs = coeffs_candidate

