# -*- coding: utf-8 -*-

"""
The main Earth class, coordinating the model fitting process.
"""
import numpy as np
from ._basis import ConstantBasisFunction # Used in fallbacks
# from ._forward import ForwardPasser # Imported locally in fit
# from ._pruning import PruningPasser # Imported locally in fit
# from ._record import EarthRecord
# from ._util import check_X_y_docs # Example, will need proper sklearn later

# For scikit-learn compatibility
# from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class Earth: # Add (BaseEstimator, RegressorMixin) later
    """
    Multivariate Adaptive Regression Splines (MARS) model.

    Parameters
    ----------
    max_degree : int, optional (default=1)
        The maximum degree of interaction terms. A value of 1 builds an additive
        model (no interaction terms). A value of 2 allows for two-way
        interactions, and so on.

    penalty : float, optional (default=3.0)
        The penalty parameter used in the General Cross-Validation (GCV)
        criterion for the pruning pass. Higher values lead to simpler models.

    max_terms : int, optional (default=None)
        The maximum number of basis functions to be included in the model after
        the forward pass. If None, it's determined by a heuristic based on
        the number of samples.

    minspan_alpha : float, optional (default=0.0)
        Parameter controlling the minimum span of a basis function.
        Not yet fully implemented in this pure Python version.

    endspan_alpha : float, optional (default=0.0)
        Parameter controlling the end span of a basis function.
        Not yet fully implemented in this pure Python version.

    # Add other py-earth parameters as needed

    Attributes
    ----------
    basis_ : list of BasisFunction
        The selected basis functions after the pruning pass.

    coef_ : array of shape (n_basis_functions, n_outputs)
        The coefficients of the selected basis functions.

    record_ : EarthRecord
        An object storing information about the fitting process.

    mse_ : float
        Mean Squared Error of the final model on the training data.

    gcv_ : float
        Generalized Cross-Validation score of the final model.

    Notes
    -----
    This is a pure Python implementation inspired by py-earth.
    Some features and optimizations from the original Cython-based py-earth
    may not be present or may behave differently.
    """

    def __init__(self, max_degree=1, penalty=3.0, max_terms=None,
                 minspan_alpha: float = 0.0, endspan_alpha: float = 0.0,
                 allow_linear: bool = True, # From py-earth, controls if linear terms can be added for variables that are not part of hinges
                 # TODO: Consider other py-earth params like:
                 # min_search_points, endspan, minspan (direct specification vs alpha based)
                 # smooth, use_fast, fast_K, fast_h,
                 # allow_missing, zero_tol, check_every, thresh, verbose
                 ):
        # Core MARS algorithm parameters
        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.minspan_alpha = minspan_alpha
        self.endspan_alpha = endspan_alpha
        self.allow_linear = allow_linear # If True, ForwardPasser might try adding pure linear terms

        # Attributes that will be learned during fit
        self.basis_: list = None  # List of selected BasisFunction objects
        self.coef_: np.ndarray = None    # Coefficients for the basis functions
        self.record_ = None       # EarthRecord object storing fitting history

        self.rss_: float = None          # Residual Sum of Squares on training data
        self.mse_: float = None          # Mean Squared Error on training data
        self.gcv_: float = None          # Generalized Cross-Validation score of the final model

        # py-earth also stores gcv and rss for each model in the pruning sequence.
        # self.bwd_gcv_seq_ = None # GCV sequence during pruning
        # self.bwd_rss_seq_ = None # RSS sequence during pruning
        # self.bwd_basis_counts_ = None # Number of terms at each pruning step
        # These could be part of a more detailed record_ or stored directly if desired.
        # For now, just storing the final GCV of the selected model.

        self.fitted_ = False # Flag to indicate if model has been fitted

    def _build_basis_matrix(self, X: np.ndarray, basis_functions: list) -> np.ndarray:
        """
        Constructs the basis matrix B from X and a list of basis functions.
        Centralized helper method.
        """
        if not basis_functions:
            return np.empty((X.shape[0], 0))

        # Ensure X is a numpy array if it's not already (e.g. list of lists from user)
        # This basic check might be expanded or rely on sklearn's check_array later
        if not isinstance(X, np.ndarray):
            X_arr = np.asarray(X)
        else:
            X_arr = X

        B_list = [bf.transform(X_arr).reshape(-1, 1) for bf in basis_functions]
        return np.hstack(B_list)

    def fit(self, X, y):
        """
        Fit the Earth model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values. Multi-output y is not currently supported.

        Returns
        -------
        self : Earth
            The fitted model.
        """
        # Import necessary modules here to avoid circular dependencies at module level
        # and to keep them local to fit if they are only used here.
        import numpy as np
        from ._forward import ForwardPasser
        from ._pruning import PruningPasser
        from ._record import EarthRecord # If using detailed recording
        from ._util import check_X_y # Basic input validation

        # Input validation
        # TODO: Later, replace with sklearn.utils.validation.check_X_y for full compatibility
        #       This would also handle converting X, y to float64, ensuring 2D X, 1D y etc.
        # For now, use our basic check_X_y from _util.
        # X_checked, y_checked = check_X_y(X, y, ensure_y_1d=True) # ensure_y_1d for current single output

        # Temporary direct conversion and checks (to be replaced by robust validation)
        if not isinstance(X, np.ndarray): X = np.asarray(X, dtype=float)
        if not isinstance(y, np.ndarray): y = np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1,1)
        if y.ndim > 1 and y.shape[1] == 1: y = y.ravel()
        if y.ndim > 1 : raise ValueError("Target y must be 1-dimensional.")
        if X.shape[0] != y.shape[0]: raise ValueError("X and y have inconsistent numbers of samples.")


        # Initialize record object (optional, depends on verbosity/debugging needs)
        # For now, let's assume self.record_ might be set up if verbose, etc.
        # If EarthRecord is to be used, it should be initialized:
        # self.record_ = EarthRecord(X_checked, y_checked, self)
        self.record_ = EarthRecord(X, y, self) # Using original X,y for record for now

        # Forward Pass
        forward_passer = ForwardPasser(self)
        fwd_basis_functions, fwd_coefficients = forward_passer.run(X, y)

        if not fwd_basis_functions:
            # This might happen if only an intercept was fit and it's deemed invalid,
            # or if an error occurred. For now, assume forward pass always gives something.
            # If it's just an intercept, pruning might still occur or confirm it.
            # If truly empty (error), we might need to raise an error or set a degenerate model.
            print("Warning: Forward pass returned no basis functions.")
            # Set a model that predicts mean of y, or handle as error
            self.basis_ = [ConstantBasisFunction()] if ConstantBasisFunction not in [type(bf) for bf in fwd_basis_functions] else fwd_basis_functions
            if not self.basis_ : # if fwd_basis_functions was also empty
                 self.basis_ = [ConstantBasisFunction()] # Ensure at least an intercept for predict

            B_final = self._build_basis_matrix(X, self.basis_)
            if B_final.shape[1] > 0:
                self.coef_, _, _, _ = np.linalg.lstsq(B_final, y, rcond=None)
            else: # Should not happen if we force intercept
                self.coef_ = np.array([np.mean(y)])
                self.basis_ = [ConstantBasisFunction()] # ensure basis list matches coef
                B_final = self._build_basis_matrix(X,self.basis_)


            self.gcv_ = np.inf # Or calculate for intercept only
            self.fitted_ = True
            # Calculate RSS and MSE for this simple model
            if self.coef_ is not None and B_final.shape[1] > 0 :
                y_pred_train = B_final @ self.coef_
                self.rss_ = np.sum((y - y_pred_train)**2)
                self.mse_ = self.rss_ / len(y)
            else: # Should not be reached if intercept is forced
                self.rss_ = np.sum((y - np.mean(y))**2)
                self.mse_ = self.rss_ / len(y)

            return self

        # Pruning Pass
        pruning_passer = PruningPasser(self)
        pruned_bfs, pruned_coeffs, best_gcv = pruning_passer.run(X, y,
                                                                 fwd_basis_functions,
                                                                 fwd_coefficients)
        self.basis_ = pruned_bfs
        self.coef_ = pruned_coeffs
        self.gcv_ = best_gcv

        # Calculate final RSS and MSE on training data using the pruned model
        if self.basis_ and self.coef_ is not None:
            B_final = self._build_basis_matrix(X, self.basis_)
            if B_final.shape[1] == len(self.coef_): # Ensure dimensions match
                y_pred_train = B_final @ self.coef_
                self.rss_ = np.sum((y - y_pred_train)**2)
                self.mse_ = self.rss_ / len(y)
            else: # Mismatch, likely means no terms left after pruning or error
                # This might happen if pruning removes all terms including intercept (if not protected well)
                # Fallback to mean prediction if all terms pruned
                self.basis_ = [ConstantBasisFunction()]
                self.coef_ = np.array([np.mean(y)])
                B_final = self._build_basis_matrix(X, self.basis_)
                y_pred_train = B_final @ self.coef_
                self.rss_ = np.sum((y - y_pred_train)**2)
                self.mse_ = self.rss_ / len(y)
                # Use X, y, len(y) from the current fit scope
                self.gcv_ = pruning_passer._compute_gcv_for_subset(X, y, len(y), self.basis_)[0] if self.basis_ else np.inf


        else: # No basis functions selected after pruning (should ideally not happen if intercept protected)
            self.basis_ = [ConstantBasisFunction()] # Default to intercept model
            self.coef_ = np.array([np.mean(y)])
            B_final = self._build_basis_matrix(X, self.basis_)
            y_pred_train = B_final @ self.coef_
            self.rss_ = np.sum((y - y_pred_train)**2)
            self.mse_ = self.rss_ / len(y)
             # Recalculate GCV for intercept-only model
            if hasattr(pruning_passer, '_compute_gcv_for_subset'): # Check if method exists
                 self.gcv_ = pruning_passer._compute_gcv_for_subset(X, y, len(y), self.basis_)[0]
            else: # Fallback if method not found (e.g. during testing with mocks)
                 self.gcv_ = np.inf


        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        # if not self.fitted_:
        #     raise NotFittedError("This Earth instance is not fitted yet.")
        # X = check_array_docs(X, allow_missing=self.allow_missing) # Placeholder
        import numpy as np # Local import for predict
        from ._util import check_array # Basic input validation for X
        # from sklearn.utils.validation import check_is_fitted # If using sklearn exceptions

        # Check if fit has been called FIRST
        if not self.fitted_:
            # TODO: Use sklearn.exceptions.NotFittedError if adopting sklearn base classes
            raise RuntimeError("This Earth instance is not fitted yet. Call 'fit' with "
                               "appropriate arguments before using this estimator.")

        # Then check for consistent state if fitted (these should ideally not be None if fitted_ is True)
        # This check is important because self.fitted_ could be True but fit() might have failed to set them.
        if self.basis_ is None or self.coef_ is None:
             # This implies fit might have completed but left an inconsistent state, which is an issue.
            raise ValueError("Model basis or coefficients are not available despite model being marked as fitted.")

        # Input validation for X
        # TODO: Replace with sklearn.utils.validation.check_array for full compatibility
        if not isinstance(X, np.ndarray): X_arr = np.asarray(X, dtype=float)
        else: X_arr = X
        if X_arr.ndim == 1: X_arr = X_arr.reshape(-1,1)
        # Add check for number of features matching training if possible (e.g. store self.n_features_in_)
        # n_features_in_ = self.basis_[0]._involved_variables ... (this is complex if basis is empty)
        # For now, _build_basis_matrix will rely on basis functions to handle X correctly.

        if not self.basis_: # Should be caught by above, but as a safeguard if only intercept was "pruned" to nothing
             # Predict mean of y_train if model is empty (or was just an intercept that got removed)
             # This requires storing y_train_mean or similar during fit.
             # For now, if this happens, it's an issue. The fit method tries to ensure at least an intercept.
             # If self.coef_ is just [mean_y], and self.basis_ is [ConstantBasisFunction], it works.
             # If self.basis_ is truly empty, this is problematic.
            if hasattr(self, '_y_train_mean'): # Check if mean was stored
                 return np.full(X_arr.shape[0], self._y_train_mean)
            else: # Fallback, though ideally fit() ensures a valid model or raises error.
                 raise ValueError("Predict called on an empty model with no fallback mean.")


        B_pred = self._build_basis_matrix(X_arr, self.basis_)

        if B_pred.shape[1] != len(self.coef_):
            # This indicates a mismatch, possibly an empty B_pred if X was incompatible with basis functions
            # Or if self.basis_ is empty but coef_ is not (inconsistent state)
            # If basis_ is empty, B_pred will have 0 columns.
            # If coef_ is for an intercept, but basis_ somehow became empty.
            if not self.basis_ and len(self.coef_) == 1: # Intercept only model, but basis list is empty
                 # This state should be fixed by fit() ensuring self.basis_ has ConstantBasisFunction
                 # For now, assume coef_[0] is the intercept value
                 return np.full(X_arr.shape[0], self.coef_[0])
            raise ValueError(f"Shape mismatch between basis matrix ({B_pred.shape}) and coefficients ({self.coef_.shape}).")

        y_pred = B_pred @ self.coef_
        return y_pred

    # Remove the duplicate _transform_X_to_basis_matrix, as _build_basis_matrix is now the one.
    # def _transform_X_to_basis_matrix(self, X, basis_functions):
    #     """
    #     Transform input X into a matrix where columns are evaluated basis functions.
    #     """
    #     # n_samples = X.shape[0]
    #     # n_basis = len(basis_functions)
    #     # B = np.empty((n_samples, n_basis))
    #     # for i, bf in enumerate(basis_functions):
    #     #     B[:, i] = bf.transform(X)
    #     # return B
    #     pass # Placeholder

    def summary(self):
        """
        Print a summary of the fitted model.
        """
        if self.basis_ is None:
            print("Model not yet fitted.")
            return

        # print("pymars Model Summary")
        # print("--------------------")
        # print(f"Number of basis functions: {len(self.basis_)}")
        # print(f"MSE: {self.mse_:.4f}")
        # print(f"GCV: {self.gcv_:.4f}")
        # print("\nBasis Functions and Coefficients:")
        # for bf, coef in zip(self.basis_, self.coef_):
        #     # This assumes coef_ is 1D for now
        #     print(f"  {str(bf):<50} Coef: {coef:.4f}")
        # Placeholder for a more structured summary
        import numpy as np # Local import

        if not self.fitted_:
            print("Model not yet fitted.")
            return

        print("pymars Earth Model Summary")
        print("==========================")
        print(f"Number of samples: {self.record_.n_samples if self.record_ else 'N/A'}")
        print(f"Number of features: {self.record_.n_features if self.record_ else 'N/A'}")
        print("--------------------------")
        print(f"Selected Basis Functions: {len(self.basis_)}")
        print(f"GCV (final model): {self.gcv_:.4f}" if self.gcv_ is not None else "GCV: N/A")
        print(f"RSS (training): {self.rss_:.4f}" if self.rss_ is not None else "RSS: N/A")
        print(f"MSE (training): {self.mse_:.4f}" if self.mse_ is not None else "MSE: N/A")
        print("--------------------------")

        if self.basis_ and self.coef_ is not None:
            print("\nBasis Functions and Coefficients:")
            # Determine max length of basis function string for alignment
            max_bf_str_len = 0
            if self.basis_: # Ensure basis_ is not empty
                max_bf_str_len = max(len(str(bf)) for bf in self.basis_)

            for i, bf in enumerate(self.basis_):
                coef_val = self.coef_[i]
                # Format coefficient nicely, handling potential arrays from multi-output later
                coef_str = f"{coef_val:.4f}"
                if isinstance(coef_val, np.ndarray): # Should not happen with current 1D y
                    coef_str = ", ".join([f"{c:.4f}" for c in coef_val])

                print(f"  {str(bf):<{max_bf_str_len + 2}} Coef: {coef_str}")
        else:
            print("No basis functions or coefficients available.")

        print("==========================")


if __name__ == '__main__':
    # Example usage (will require actual data and other modules)
    # import numpy as np
    # X_train = np.random.rand(100, 3)
    # y_train = X_train[:, 0] * 2 - X_train[:, 1] + np.random.randn(100) * 0.1

    # model = Earth(max_degree=1, max_terms=10)
    # model.fit(X_train, y_train)
    # model.summary()

    # X_test = np.random.rand(20, 3)
    # y_pred = model.predict(X_test)
    # print("\nPredictions on new data:", y_pred)
    pass
