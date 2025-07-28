# -*- coding: utf-8 -*-

"""
The pruning pass of the MARS algorithm.

This module is responsible for removing basis functions from the model
(typically built by the forward pass) to improve generalization, often
using a criterion like Generalized Cross-Validation (GCV).
"""

import logging
import numpy as np
from .earth import Earth  # For type hinting
from ._basis import BasisFunction, ConstantBasisFunction, HingeBasisFunction
from ._util import calculate_gcv, gcv_penalty_cost_effective_parameters

logger = logging.getLogger(__name__)

EPSILON = np.finfo(float).eps

class PruningPasser:
    """
    Manages the pruning pass of the MARS algorithm.
    Iteratively removes basis functions to minimize GCV.
    """
    def __init__(self, earth_model: Earth):
        self.model = earth_model

        self.X_train = None # Will be X_fit_processed
        self.y_train = None # Will be y_fit
        self.n_samples = 0  # Will be based on X_fit_processed

        self.missing_mask = None
        self.X_fit_original = None

        self.best_gcv_so_far = np.inf
        self.best_basis_functions_so_far: list[BasisFunction] = []
        self.best_coeffs_so_far: np.ndarray = None

    def _calculate_rss_and_coeffs(self, B_matrix: np.ndarray, y_data: np.ndarray) -> tuple[float, np.ndarray | None, int]:
        """
        Calculates RSS, coefficients, and num_valid_rows, considering NaNs in B_matrix.
        y_data is assumed finite.
        Returns (rss, coeffs, num_valid_rows). (np.inf, None, 0) on error/no valid rows.
        """
        if B_matrix is None or B_matrix.shape[1] == 0:
            mean_y = np.mean(y_data)
            rss = np.sum((y_data - mean_y)**2)
            num_valid_rows = len(y_data)
            coeffs_for_mean = np.array([mean_y]) if (B_matrix is not None and B_matrix.shape[1] == 0) else None
            return rss, coeffs_for_mean, num_valid_rows

        # Identify rows with no NaNs in B_matrix (y_data is already finite)
        valid_rows_mask = ~np.any(np.isnan(B_matrix), axis=1)
        num_valid_rows = np.sum(valid_rows_mask)

        if num_valid_rows == 0 or num_valid_rows < B_matrix.shape[1]: # Not enough data or underdetermined
            return np.inf, None, 0

        B_complete = B_matrix[valid_rows_mask, :]
        y_complete = y_data[valid_rows_mask] # y_data is already 1D from PruningPasser.run

        try:
            # B_complete should be 2D here because B_matrix.shape[1] > 0
            # and num_valid_rows >= B_matrix.shape[1]
            coeffs, residuals_sum_sq, rank, s = np.linalg.lstsq(B_complete, y_complete, rcond=None)

            if residuals_sum_sq.size == 0 or rank < B_complete.shape[1]:
                y_pred_complete = B_complete @ coeffs
                rss = np.sum((y_complete - y_pred_complete)**2)
            else:
                rss = residuals_sum_sq[0]
            return rss, coeffs, num_valid_rows
        except np.linalg.LinAlgError as e:
            logger.warning(
                "LinAlgError in PruningPasser._calculate_rss_and_coeffs: %s", e
            )
            return np.inf, None, num_valid_rows

    def _build_basis_matrix(self, X_data: np.ndarray, basis_functions: list[BasisFunction],
                            missing_mask: np.ndarray) -> np.ndarray:
        """
        Constructs the basis matrix B from X_data (which is X_processed)
        and a list of basis functions, using the provided missing_mask.
        """
        if not basis_functions:
            return np.empty((X_data.shape[0], 0))
        B_list = [bf.transform(X_data, missing_mask).reshape(-1, 1) for bf in basis_functions]
        return np.hstack(B_list)

    def _compute_gcv_for_subset(self, X_fit_processed: np.ndarray, y_fit: np.ndarray,
                                missing_mask: np.ndarray, X_fit_original: np.ndarray,
                                basis_subset: list[BasisFunction]) -> tuple[float | None, float | None, np.ndarray | None]:
        """
        Computes GCV, RSS, and coefficients for a given subset of basis functions.
        Returns (gcv, rss, coeffs).
        """
        if not basis_subset:
            # This case implies an intercept-only model IF an intercept is implicitly assumed
            # or if _calculate_rss_and_coeffs handles B_matrix.shape[1] == 0 by returning intercept.
            # Current _calculate_rss_and_coeffs for empty B_matrix (shape[1]==0) returns mean(y) as coeff.
            rss_empty, coeffs_empty, n_valid_rows_for_empty = self._calculate_rss_and_coeffs(
                np.empty((len(y_fit), 0)), y_fit
            ) # B_matrix with 0 columns

            if coeffs_empty is None: # Should not happen with current _calc_rss_and_coeffs for empty B
                return np.inf, rss_empty, None

            num_terms_intercept_only = 1 # Intercept
            num_hinge_terms_intercept_only = 0

            effective_params_empty = gcv_penalty_cost_effective_parameters(
                num_terms_intercept_only,
                num_hinge_terms_intercept_only,
                self.model.penalty,
                n_valid_rows_for_empty
            )
            gcv_empty = calculate_gcv(rss_empty, n_valid_rows_for_empty, effective_params_empty)
            return gcv_empty, rss_empty, coeffs_empty # coeffs_empty is np.array([np.mean(y_fit)])

        # There is a duplicate line here, removing one instance.
        # B_subset = self._build_basis_matrix(X_fit_processed, basis_subset, missing_mask)
        B_subset = self._build_basis_matrix(X_fit_processed, basis_subset, missing_mask)

        rss, coeffs, num_valid_rows_for_fit = self._calculate_rss_and_coeffs(B_subset, y_fit)

        if coeffs is None or rss == np.inf or num_valid_rows_for_fit == 0:
            return np.inf, rss if rss is not None else np.inf, None

        num_model_terms = coeffs.shape[0] # Actual number of terms in the fitted model

        if num_model_terms == 0:
             # This case implies LSTSQ resulted in an empty coefficient array, which means no model.
            return np.inf, rss, None

        num_hinge_terms_in_subset = sum(isinstance(bf, HingeBasisFunction) for bf in basis_subset)

        effective_params = gcv_penalty_cost_effective_parameters(
            num_model_terms,
            num_hinge_terms_in_subset,
            self.model.penalty,
            num_valid_rows_for_fit  # This is n_samples for the GCV calculation
        )

        # The arguments for calculate_gcv are (rss, num_samples, num_effective_params)
        gcv_score = calculate_gcv(rss, num_valid_rows_for_fit, effective_params)
        return gcv_score, rss, coeffs

    def run(self, X_fit_processed: np.ndarray, y_fit: np.ndarray,
            missing_mask: np.ndarray, X_fit_original: np.ndarray,
            initial_basis_functions: list[BasisFunction],
            initial_coefficients: np.ndarray) -> tuple[list[BasisFunction], np.ndarray, float]:

        self.X_train = X_fit_processed
        self.y_train = y_fit.ravel()
        self.missing_mask = missing_mask
        self.X_fit_original = X_fit_original
        self.n_samples = self.X_train.shape[0]

        if not initial_basis_functions:
            self.best_gcv_so_far = np.inf
            return [], np.array([]), np.inf

        current_pruning_sequence_bfs = list(initial_basis_functions)

        initial_gcv, initial_rss, initial_coeffs_refit = self._compute_gcv_for_subset(
            self.X_train, self.y_train, self.missing_mask, self.X_fit_original,
            current_pruning_sequence_bfs
        )

        if initial_coeffs_refit is None:
            logger.warning(
                "Could not compute GCV for the initial full model from forward pass in PruningPasser."
            )
            self.best_gcv_so_far = np.inf
            return initial_basis_functions, initial_coefficients, np.inf

        self.best_gcv_so_far = initial_gcv
        self.best_basis_functions_so_far = list(current_pruning_sequence_bfs)
        self.best_coeffs_so_far = initial_coeffs_refit

        if self.model.record_ is not None and hasattr(self.model.record_, 'log_pruning_step'):
            self.model.record_.log_pruning_step(
                self.best_basis_functions_so_far,
                self.best_coeffs_so_far,
                self.best_gcv_so_far,
                initial_rss
            )

        active_bfs_for_loop = list(current_pruning_sequence_bfs)

        min_allowable_terms = 1 if any(isinstance(bf, ConstantBasisFunction) for bf in active_bfs_for_loop) else 0
        num_iterations = len(active_bfs_for_loop) - min_allowable_terms

        for _ in range(num_iterations):
            if len(active_bfs_for_loop) <= min_allowable_terms:
                break

            gcv_for_removal_candidates = []

            for i in range(len(active_bfs_for_loop)):
                bf_to_test_removal = active_bfs_for_loop[i]

                if isinstance(bf_to_test_removal, ConstantBasisFunction) and len(active_bfs_for_loop) == min_allowable_terms:
                    gcv_for_removal_candidates.append((np.inf, np.inf, None, i))
                    continue

                temp_basis_subset = [bf for j, bf in enumerate(active_bfs_for_loop) if j != i]

                if not temp_basis_subset and min_allowable_terms > 0 :
                     gcv_for_removal_candidates.append((np.inf, np.inf, None, i))
                     continue

                gcv, rss, coeffs = self._compute_gcv_for_subset(
                    self.X_train, self.y_train, self.missing_mask, self.X_fit_original,
                    temp_basis_subset
                )
                gcv_for_removal_candidates.append((gcv, rss, coeffs, i))

            if not gcv_for_removal_candidates:
                break

            best_removal_this_step = min(gcv_for_removal_candidates, key=lambda x: x[0])
            gcv_after_removal, rss_after_removal, coeffs_after_removal, idx_removed = best_removal_this_step

            if coeffs_after_removal is None:
                break

            active_bfs_for_loop.pop(idx_removed)

            if gcv_after_removal < self.best_gcv_so_far:
                self.best_gcv_so_far = gcv_after_removal
                self.best_basis_functions_so_far = list(active_bfs_for_loop)
                self.best_coeffs_so_far = coeffs_after_removal

            if self.model.record_ is not None and hasattr(self.model.record_, 'log_pruning_step'):
                self.model.record_.log_pruning_step(
                    list(active_bfs_for_loop),
                    coeffs_after_removal,
                    gcv_after_removal,
                    rss_after_removal
                )

        return self.best_basis_functions_so_far, self.best_coeffs_so_far, self.best_gcv_so_far

if __name__ == '__main__':
    pass
