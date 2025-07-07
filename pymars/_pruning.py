# -*- coding: utf-8 -*-

"""
The pruning pass of the MARS algorithm.

This module is responsible for removing basis functions from the model
(typically built by the forward pass) to improve generalization, often
using a criterion like Generalized Cross-Validation (GCV).
"""

import numpy as np
# from ._record import PruningPassRecord # If we log details of pruning pass

class PruningPasser:
    """
    Manages the pruning pass of the MARS algorithm.

    Iteratively removes basis functions from a given model to minimize GCV
    or another pruning criterion.
    """
    def __init__(self, earth_model_instance):
        """
        Initialize the PruningPasser.

        Parameters
        ----------
        earth_model_instance : Earth
            An instance of the main Earth model, providing access to
            hyperparameters like penalty.
        """
        self.model = earth_model_instance
        # self.record = PruningPassRecord() # For logging

    def _calculate_gcv(self, X, y, basis_functions, coefficients):
        """
        Calculate the Generalized Cross-Validation (GCV) score.

        GCV = MSE / (1 - (effective_num_params / n_samples))^2

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Target data.
        basis_functions : list of BasisFunction
            The current set of basis functions in the model.
        coefficients : numpy.ndarray
            The coefficients for the current basis functions.

        Returns
        -------
        float
            The GCV score.
        """
        n_samples = X.shape[0]
        if not basis_functions or coefficients is None or coefficients.size == 0:
            # Undefined or very large GCV if no terms or problematic coefficients
            return np.inf

        # 1. Construct Basis Matrix B from basis_functions and X
        # B = self.model._transform_X_to_basis_matrix(X, basis_functions)
        # if B.shape[1] == 0: # No basis functions effectively
        #    return np.inf

        # 2. Calculate predictions and MSE
        # y_pred = B @ coefficients
        # mse = np.mean((y - y_pred)**2)

        # 3. Calculate effective number of parameters
        #    This often includes a penalty per basis function term.
        #    A common form: C(M) = number_of_terms + penalty * (number_of_terms - 1) / 2 (if intercept is term 1)
        #    Or more simply, as in py-earth, related to trace(B (B^T B)^-1 B^T) which simplifies
        #    under OLS to number of independent columns in B times a penalty factor.
        #    For basic GCV, effective_num_params = number of basis functions (columns in B)
        #    The 'penalty' in Earth model (Friedman's d) is part of the GCV formula.
        #    GCV = RSS / (N * (1 - (C_effective / N))^2)
        #    where C_effective = num_nonzero_coeffs + self.model.penalty * (num_nonzero_coeffs - 1)/2 (if intercept is present and counted)
        #    Let's use a simpler form first: Number of terms.
        # num_terms = B.shape[1]
        # if num_terms == 0:
        #     return np.inf # Avoid division by zero if no terms

        # effective_num_params = num_terms # Simplistic, py-earth uses a more complex calculation for C(M)
                                        # that depends on self.model.penalty.
                                        # py-earth GCV: rss / (n_samples * (1 - num_params_effective / n_samples)**2)
                                        # where num_params_effective is not just num_terms.

        # if n_samples <= effective_num_params:
        #     return np.inf  # Avoid division by zero or negative in denominator

        # gcv_denominator = (1 - (effective_num_params / n_samples))**2
        # if gcv_denominator <= 1e-9: # Effectively zero
        #     return np.inf

        # gcv = mse / gcv_denominator
        # return gcv
        pass # Placeholder


    def run(self, X, y, initial_basis_functions, initial_coefficients):
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
        # Placeholder: return initial set for now
        return initial_basis_functions, initial_coefficients

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
