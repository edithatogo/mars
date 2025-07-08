# -*- coding: utf-8 -*-

"""
The pruning pass of the MARS algorithm.

This module is responsible for removing basis functions from the model
(typically built by the forward pass) to improve generalization, often
using a criterion like Generalized Cross-Validation (GCV).
"""

import numpy as np
from .earth import Earth # For type hinting
from ._basis import BasisFunction, ConstantBasisFunction
# from ._record import EarthRecord # If EarthRecord is used by Earth model instance
from ._util import calculate_gcv, gcv_penalty_cost_effective_parameters # Import GCV helpers

# Define a small constant for numerical stability if needed
EPSILON = np.finfo(float).eps

class PruningPasser:
    """
    Manages the pruning pass of the MARS algorithm.

    Iteratively removes basis functions from the model (obtained from the
    forward pass) to minimize the Generalized Cross-Validation (GCV) score.
    """
    def __init__(self, earth_model: Earth):
        """
        Initialize the PruningPasser.

        Parameters
        ----------
        earth_model : Earth
            An instance of the main Earth model. This provides access to
            hyperparameters like `penalty` and the `record_` object for logging.
        """
        self.model = earth_model # Earth model instance

        # Internal state, initialized at the start of run()
        self.X_train = None
        self.y_train = None
        self.n_samples = 0

        self.best_gcv_so_far = np.inf
        self.best_basis_functions_so_far: list[BasisFunction] = []
        self.best_coeffs_so_far: np.ndarray = None
        self.best_B_matrix_so_far: np.ndarray = None # Basis matrix for the best model

    def _calculate_rss_and_coeffs(self, B_matrix: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculates Residual Sum of Squares and coefficients for a given basis matrix and target y.
        Returns (rss, coeffs). Returns (np.inf, None) on error.
        (This can be shared with ForwardPasser, perhaps in _util or a common base)
        """
        if B_matrix is None or B_matrix.shape[1] == 0:
            mean_y = np.mean(y)
            rss = np.sum((y - mean_y)**2)
            return rss, np.array([mean_y]) if (B_matrix is not None and B_matrix.shape[1] == 0) else None

        try:
            if B_matrix.ndim == 1: B_matrix = B_matrix.reshape(-1, 1)
            coeffs, residuals_sum_sq, rank, s = np.linalg.lstsq(B_matrix, y, rcond=None)
            if residuals_sum_sq.size == 0 or rank < B_matrix.shape[1]:
                y_pred = B_matrix @ coeffs
                rss = np.sum((y - y_pred)**2)
            else:
                rss = residuals_sum_sq[0]
            return rss, coeffs
        except np.linalg.LinAlgError:
            return np.inf, None

    def _build_basis_matrix(self, X: np.ndarray, basis_functions: list[BasisFunction]) -> np.ndarray:
        """
        Constructs the basis matrix B from X and a list of basis functions.
        (This can be shared with ForwardPasser, perhaps in _util or a common base)
        """
        if not basis_functions:
            return np.empty((X.shape[0], 0))
        B_list = [bf.transform(X).reshape(-1, 1) for bf in basis_functions]
        return np.hstack(B_list)

    def _compute_gcv_for_subset(self, current_basis_subset: list[BasisFunction]) -> tuple[float, float, np.ndarray]:
        """
        Computes GCV, RSS, and coefficients for a given subset of basis functions.
        Returns (gcv, rss, coeffs).
        """
        if not current_basis_subset: # Should not happen if we protect intercept
            return np.inf, np.inf, None

        B_current = self._build_basis_matrix(self.X_train, current_basis_subset)
        if B_current.shape[1] == 0: # No terms, should have at least intercept.
             return np.inf, np.inf, None

        rss, coeffs = self._calculate_rss_and_coeffs(B_current, self.y_train)

        if coeffs is None: # Error during LSTSQ
            return np.inf, rss, None # rss might be inf too

        num_terms = B_current.shape[1]
        # Check if intercept is present for penalty calculation
        has_intercept = any(isinstance(bf, ConstantBasisFunction) for bf in current_basis_subset)

        # Effective number of parameters C(M) for GCV
        # Using the gcv_penalty_cost_effective_parameters from _util, which uses Friedman's formula
        effective_params = gcv_penalty_cost_effective_parameters(
            num_terms, self.n_samples, self.model.penalty, has_intercept=has_intercept
        )

        gcv = calculate_gcv(rss, self.n_samples, effective_params)
        return gcv, rss, coeffs

    def run(self, X: np.ndarray, y: np.ndarray,
            initial_basis_functions: list[BasisFunction],
            initial_coefficients: np.ndarray) -> tuple[list[BasisFunction], np.ndarray, float]:
        """
        Execute the pruning pass.

        Parameters
        ----------
        X : numpy.ndarray
            Training input data.
        y : numpy.ndarray
            Training target data.
        initial_basis_functions : list of BasisFunction
            The full set of basis functions from the forward pass.
        initial_coefficients : numpy.ndarray
            Coefficients corresponding to the initial_basis_functions.

        Returns
        -------
        pruned_basis_functions : list of BasisFunction
            The list of basis functions after pruning.
        pruned_coefficients : numpy.ndarray
            The coefficients corresponding to the pruned basis functions.
        """
        # current_basis = list(initial_basis_functions)
        # current_coeffs = np.array(initial_coefficients)

        # if not current_basis:
        #     return [], []

        # best_gcv = self._calculate_gcv(X, y, current_basis, current_coeffs)
        # self.model.record_.log_pruning_step(current_basis, current_coeffs, best_gcv) # Log initial state

        # # Pruning loop
        # while len(current_basis) > 1: # Keep at least one (e.g., intercept if present)
        #     gcv_if_removed_list = []
        #     for i in range(len(current_basis)):
        #         # Cannot remove intercept if it's the only one and we want to keep it
        #         # if isinstance(current_basis[i], ConstantBasisFunction) and len(current_basis) == 1:
        #         #    gcv_if_removed_list.append(np.inf)
        #         #    continue

        #         temp_basis = current_basis[:i] + current_basis[i+1:]
        #         if not temp_basis: # Should not happen if we ensure len > 1
        #             gcv_if_removed_list.append(np.inf)
        #             continue

        #         # Refit model with temp_basis
        #         # B_temp = self.model._transform_X_to_basis_matrix(X, temp_basis)
        #         # try:
        #         #     coeffs_temp, _, _, _ = np.linalg.lstsq(B_temp, y, rcond=None)
        #         # except np.linalg.LinAlgError:
        #         #     gcv_if_removed_list.append(np.inf) # Penalize if model becomes unstable
        #         #     continue

        #         # gcv_val = self._calculate_gcv(X, y, temp_basis, coeffs_temp)
        #         # gcv_if_removed_list.append(gcv_val)

        #     min_gcv_candidate = min(gcv_if_removed_list)
        #     idx_to_remove = np.argmin(gcv_if_removed_list)

        #     if min_gcv_candidate < best_gcv:
        #         best_gcv = min_gcv_candidate
        #         current_basis.pop(idx_to_remove)

        #         # Refit with the truly removed basis function for new current_coeffs
        #         # B_current = self.model._transform_X_to_basis_matrix(X, current_basis)
        #         # current_coeffs, _, _, _ = np.linalg.lstsq(B_current, y, rcond=None)
        #         # self.model.record_.log_pruning_step(current_basis, current_coeffs, best_gcv)
        #     else:
        #         # No removal improves GCV, so stop
        #         break

        # Final refit for the selected basis functions
        # if not current_basis: # Should mean initial_basis_functions was empty
        #     return [], []

        # B_final = self.model._transform_X_to_basis_matrix(X, current_basis)
        # final_coeffs, _, _, _ = np.linalg.lstsq(B_final, y, rcond=None)

        # self.model.bwd_gcv_ = best_gcv # Store best GCV from pruning
        # self.model.bwd_mse_ = self._calculate_mse(y, B_final @ final_coeffs) # Helper for MSE

        # return current_basis, final_coeffs
        print(f"PruningPasser.run called with {len(initial_basis_functions)} initial basis functions.")
        # Placeholder: return initial set for now. The main loop will modify these.
        # The actual best_basis_functions_so_far etc. will be determined by the loop.

        self.X_train = X
        self.y_train = y
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError("PruningPasser currently supports only single-output targets (y should be 1D).")
        self.y_train = self.y_train.ravel()
        self.n_samples = X.shape[0]

        if not initial_basis_functions:
            # No terms from forward pass, nothing to prune.
            # This case should ideally be handled by the Earth class before calling.
            self.best_gcv_so_far = np.inf
            return [], np.array([]), np.inf

        # The initial model (full model from forward pass) is the first one in our pruning sequence.
        # Its GCV needs to be calculated and stored as the current best.
        current_pruning_sequence_bfs = list(initial_basis_functions) # Work with a copy
        # The initial_coefficients are for the initial_basis_functions on the training data.
        # We can use these to calculate initial RSS, or re-calculate.
        # For consistency, let _compute_gcv_for_subset re-calculate coeffs.

        initial_gcv, initial_rss, initial_coeffs_refit = self._compute_gcv_for_subset(current_pruning_sequence_bfs)

        if initial_coeffs_refit is None: # Should not happen if forward pass produced a valid model
            print("Warning: Could not compute GCV for the initial full model from forward pass.")
            # Fallback: return the model as is, or handle error appropriately.
            self.best_gcv_so_far = np.inf
            return initial_basis_functions, initial_coefficients, np.inf # Or raise error

        self.best_gcv_so_far = initial_gcv
        self.best_basis_functions_so_far = list(current_pruning_sequence_bfs) # Store a copy
        self.best_coeffs_so_far = initial_coeffs_refit
        # self.best_B_matrix_so_far could be stored if needed for debugging or other metrics.

        if self.model.record_ is not None and hasattr(self.model.record_, 'log_pruning_step'):
            self.model.record_.log_pruning_step(
                self.best_basis_functions_so_far,
                self.best_coeffs_so_far,
                self.best_gcv_so_far,
                initial_rss # Log initial RSS as well
            )

        # active_bfs_for_loop will be used in the main pruning loop (next step)
        active_bfs_for_loop = current_pruning_sequence_bfs

        # Placeholder for the main pruning loop (to be implemented in step 4 & 5)
        # For now, this function will return the initial state as the "best"
        # This will be replaced by the loop that populates best_basis_functions_so_far etc.

        print(f"Initial pruning state: {len(self.best_basis_functions_so_far)} terms, GCV={self.best_gcv_so_far:.4f}")

        # The main pruning loop will determine the actual best pruned model.
        # For this step, we are just setting up. The return will be updated after loop implementation.
        # --- Main Pruning Loop (following refined strategy from plan step 5) ---

        # active_bfs_for_loop was initialized with initial_basis_functions
        # The initial full model is already considered and stored as best_so_far if its GCV is lowest initially.

        # We need to determine the minimum number of terms to keep.
        # Often, this is 1 (the intercept, if present and protected).
        min_allowable_terms = 0
        has_intercept_in_initial_set = any(isinstance(bf, ConstantBasisFunction) for bf in active_bfs_for_loop)
        if has_intercept_in_initial_set:
            min_allowable_terms = 1 # Protect the intercept by default if it's there

        num_iterations = len(active_bfs_for_loop) - min_allowable_terms

        for _ in range(num_iterations):
            if len(active_bfs_for_loop) <= min_allowable_terms:
                break # Should not happen if num_iterations is correct, but safe check.

            gcv_for_removal_candidates = [] # Stores (gcv, rss, coeffs, index_of_removed_bf)

            # Iterate through each basis function in the current active set to consider removing it
            for i in range(len(active_bfs_for_loop)):
                bf_to_test_removal = active_bfs_for_loop[i]

                # Protection for intercept: do not remove it if it's the only term left
                # or if it's configured to be always included.
                # For now, simple protection: if it's ConstantBasisFunction and only min_allowable_terms would remain.
                if isinstance(bf_to_test_removal, ConstantBasisFunction) and len(active_bfs_for_loop) <= min_allowable_terms:
                    gcv_for_removal_candidates.append((np.inf, np.inf, None, i)) # Effectively don't remove
                    continue

                temp_basis_subset = [bf for j, bf in enumerate(active_bfs_for_loop) if j != i]

                if not temp_basis_subset and min_allowable_terms > 0 : # e.g. trying to remove the intercept when it's the last term
                     gcv_for_removal_candidates.append((np.inf, np.inf, None, i))
                     continue


                gcv, rss, coeffs = self._compute_gcv_for_subset(temp_basis_subset)
                gcv_for_removal_candidates.append((gcv, rss, coeffs, i))

            if not gcv_for_removal_candidates: # No terms were removable or considered
                break

            # Find the removal that results in the lowest GCV for the smaller model
            best_removal_this_step = min(gcv_for_removal_candidates, key=lambda x: x[0])
            gcv_after_removal, rss_after_removal, coeffs_after_removal, idx_removed = best_removal_this_step

            if coeffs_after_removal is None: # All removals led to errors or invalid models
                # print("Warning: All potential term removals led to invalid models in pruning.")
                break

            # Update active_bfs_for_loop by removing the chosen term
            removed_bf_object = active_bfs_for_loop.pop(idx_removed)
            # print(f"Pruning iteration: removed '{str(removed_bf_object)}'. New model size: {len(active_bfs_for_loop)}. GCV: {gcv_after_removal:.4f}")


            # Check if this new (smaller) model is better than the overall best found so far
            if gcv_after_removal < self.best_gcv_so_far:
                self.best_gcv_so_far = gcv_after_removal
                self.best_basis_functions_so_far = list(active_bfs_for_loop) # Store copy
                self.best_coeffs_so_far = coeffs_after_removal
                # Potentially store self.best_B_matrix_so_far if needed later

            if self.model.record_ is not None and hasattr(self.model.record_, 'log_pruning_step'):
                self.model.record_.log_pruning_step(
                    list(active_bfs_for_loop), # Current active set after removal
                    coeffs_after_removal,
                    gcv_after_removal,
                    rss_after_removal
                )

        # print(f"Pruning complete. Best GCV: {self.best_gcv_so_far:.4f} with {len(self.best_basis_functions_so_far)} terms.")
        return self.best_basis_functions_so_far, self.best_coeffs_so_far, self.best_gcv_so_far


    def _calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)


if __name__ == '__main__':
    # Placeholder for example usage or internal tests.
    # Requires a mock Earth model, data, and a set of basis functions from a forward pass.
    class MockEarth:
        def __init__(self, penalty=3.0):
            self.penalty = penalty
            # self.record_ = type('MockRecord', (), {'log_pruning_step': lambda *args: None})()
            # self._transform_X_to_basis_matrix = lambda X, bfs: np.hstack([bf.transform(X).reshape(-1,1) for bf in bfs]) if bfs else np.empty((X.shape[0],0))


    # from ._basis import ConstantBasisFunction, HingeBasisFunction
    # X_sample = np.random.rand(100, 1)
    # y_sample = 2 * X_sample[:,0] + np.sin(3 * X_sample[:,0]) + np.random.randn(100)*0.1

    # bf_const = ConstantBasisFunction()
    # bf_hinge1 = HingeBasisFunction(0, 0.5)
    # bf_hinge2 = HingeBasisFunction(0, 0.3, is_right_hinge=False)
    # bf_extra = HingeBasisFunction(0, 0.8) # Assume this one is less useful

    # initial_bfs = [bf_const, bf_hinge1, bf_hinge2, bf_extra]
    # B_initial = np.hstack([bf.transform(X_sample).reshape(-1,1) for bf in initial_bfs])
    # initial_coeffs_dummy, _, _, _ = np.linalg.lstsq(B_initial, y_sample, rcond=None)

    # mock_model = MockEarth()
    # pruner = PruningPasser(mock_model)

    # pruned_bfs, pruned_coeffs = pruner.run(X_sample, y_sample, initial_bfs, initial_coeffs_dummy)
    # print(f"Pruned from {len(initial_bfs)} to {len(pruned_bfs)} basis functions.")
    # for bf in pruned_bfs:
    #    print(str(bf))
    pass
