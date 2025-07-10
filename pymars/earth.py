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

    minspan : int or float, optional
        Controls the minimum separation between knots.
        If `minspan >= 0`, it's the direct minimum number of unique predictor values
        to skip between chosen knots (a "cooldown" period). `minspan=0` or `minspan=1`
        typically means no skipping of distinct available knots.
        If `minspan < 0` (default is -1), `minspan_alpha` is used.
        This behavior is aligned with `py-earth`.

    minspan_alpha : float, optional (default=0.0)
        If `minspan < 0`, this alpha parameter is used to calculate `minspan`.
        The formula (from `py-earth`) is approximately:
        `int(round(-log2(-(1/(N_eff * C_parent)) * log(1-alpha)) / 2.5))`
        where `N_eff` relates to number of features and `C_parent` to active
        parent samples. A higher alpha leads to a larger `minspan`.
        If `minspan_alpha <= 0`, `minspan` becomes 0 (no cooldown).

    endspan : int or float, optional
        Controls how close knots can be to the data boundaries.
        If `endspan >= 0`, it's the number of unique predictor values to exclude
        from each end of the sorted unique values (in the active region defined
        by the parent basis function) when considering knot locations.
        If `endspan < 0` (default is -1), `endspan_alpha` is used.

    endspan_alpha : float, optional (default=0.0)
        If `endspan < 0`, this alpha parameter is used to calculate `endspan`.
        The formula (from `py-earth`) is approximately:
        `int(round(3.0 - log2(alpha / N_eff)))`, then `max(0, val)`, and
        if result is 0 and alpha > 0, it becomes 1.
        A higher alpha leads to a larger `endspan`.
        If `endspan_alpha <= 0`, `endspan` becomes 0.

    allow_linear : bool, optional (default=True)
        Whether to allow linear basis functions to be considered.
        Note: Current implementation primarily focuses on hinge functions;
        full linear term consideration might be limited.

    feature_importance_type : {'nb_subsets', 'gcv', 'rss', None}, optional (default=None)
        If not None, specifies the method to calculate feature importances,
        which are then available in the `feature_importances_` attribute.
        - 'nb_subsets': Importance is the number of times each feature
          appears in a basis function in the models considered during
          the pruning pass, normalized.
        - 'gcv': Importance is based on the sum of GCV improvements
          when terms involving a feature are added during the forward pass.
          Only terms that survive pruning contribute. Normalized.
        - 'rss': Importance is based on the sum of RSS reductions
          when terms involving a feature are added during the forward pass.
          Only terms that survive pruning contribute. Normalized.
        If None, feature importances are not computed.

    Attributes
    ----------
    basis_ : list of BasisFunction
        The selected basis functions after the pruning pass.

    coef_ : array of shape (n_basis_functions,)
        The coefficients of the selected basis functions. (Assumes single output target)

    record_ : EarthRecord
        An object storing information about the fitting process.

    rss_ : float
        Residual Sum of Squares of the final model on the training data.

    mse_ : float
        Mean Squared Error of the final model on the training data.

    gcv_ : float
        Generalized Cross-Validation score of the final model.

    feature_importances_ : numpy.ndarray of shape (n_features,) or None
        Normalized feature importances if `feature_importance_type` was set.
        The calculation method depends on the `feature_importance_type` parameter.
        None if feature importances were not computed.

    Notes
    -----
    This is a pure Python implementation inspired by py-earth.
    Some features and optimizations from the original Cython-based py-earth
    may not be present or may behave differently.
    """

    def __init__(self, max_degree: int = 1, penalty: float = 3.0, max_terms: int = None,
                 minspan_alpha: float = 0.0, endspan_alpha: float = 0.0,
                 minspan: int = -1, endspan: int = -1,
                 allow_linear: bool = True,
                 feature_importance_type: str = None
                 # TODO: Consider other py-earth params
                 ):
        # Core MARS algorithm parameters
        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.minspan_alpha = minspan_alpha
        self.endspan_alpha = endspan_alpha
        self.minspan = minspan
        self.endspan = endspan
        self.allow_linear = allow_linear
        self.feature_importance_type = feature_importance_type

        # Attributes that will be learned during fit
        self.basis_: list = None
        self.coef_: np.ndarray = None
        self.record_ = None

        self.rss_: float = None
        self.mse_: float = None
        self.gcv_: float = None

        self.feature_importances_: np.ndarray = None # Or dict if multiple types

        self.fitted_ = False

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

        # Calculate feature importances if requested
        if self.feature_importance_type is not None:
            self._calculate_feature_importances(X) # Pass X for n_features, or store n_features during fit

        self.fitted_ = True
        return self

    def _calculate_feature_importances(self, X_fit):
        """
        Placeholder for feature importance calculation.
        This will be implemented based on self.feature_importance_type.
        # For 'nb_subsets', it uses the pruning trace stored in self.record_.
        """
        import numpy as np # Ensure numpy is available

        if not hasattr(self.record_, 'n_features'):
             # Fallback if record somehow doesn't have n_features, though it should.
             # This might happen if fit failed very early or record not initialized.
            if hasattr(X_fit, 'shape') and X_fit.ndim == 2:
                 num_features = X_fit.shape[1]
            else: # Cannot determine num_features
                print("Warning: Cannot determine number of features for importance calculation.")
                self.feature_importances_ = np.array([])
                return
        else:
            num_features = self.record_.n_features

        if self.feature_importance_type == 'nb_subsets':
            if not (hasattr(self.record_, 'pruning_trace_basis_functions_') and \
                    self.record_.pruning_trace_basis_functions_):
                print("Warning: Pruning trace not available in record. "
                      "Cannot calculate 'nb_subsets' feature importance. Returning zeros.")
                self.feature_importances_ = np.zeros(num_features)
                return

            importances = np.zeros(num_features)
            for basis_set in self.record_.pruning_trace_basis_functions_:
                # For each model in the pruning sequence
                model_variables = set()
                for bf in basis_set:
                    model_variables.update(bf.get_involved_variables())

                for var_idx in model_variables:
                    if 0 <= var_idx < num_features: # Ensure var_idx is valid
                        importances[var_idx] += 1

            if np.sum(importances) > 0:
                self.feature_importances_ = importances / np.sum(importances)
            else:
                # All zeros if no features were ever used or trace was empty in a weird way
                self.feature_importances_ = importances

        elif self.feature_importance_type == 'gcv':
            importances = np.zeros(num_features)
            if not self.basis_: # No basis functions in the final model
                self.feature_importances_ = importances
                return

            for bf in self.basis_:
                if bf.is_constant(): # Skip intercept
                    continue

                # gcv_score_ should have been set during the forward pass
                # for basis functions that were part of a selected pair.
                score_to_add = getattr(bf, 'gcv_score_', 0.0)

                # py-earth uses max(0, gcv_reduction_score) when summing for feature importance.
                # This ensures that terms that actually improved GCV (or didn't worsen it)
                # contribute positively to the importance score.
                actual_contribution = max(0.0, score_to_add)

                if actual_contribution > 0: # Only add if there's a non-negative contribution
                    for var_idx in bf.get_involved_variables():
                        if 0 <= var_idx < num_features:
                            importances[var_idx] += actual_contribution

            # Normalize importances
            total_importance = np.sum(importances)
            if total_importance > 0:
                self.feature_importances_ = importances / total_importance
            else:
                self.feature_importances_ = importances # All zeros if no positive contributions or all contributions were <=0

        elif self.feature_importance_type == 'rss':
            importances = np.zeros(num_features)
            if not self.basis_: # No basis functions in the final model
                self.feature_importances_ = importances
                return

            for bf in self.basis_:
                if bf.is_constant(): # Skip intercept
                    continue

                # rss_score_ should have been set during the forward pass
                score_to_add = getattr(bf, 'rss_score_', 0.0)

                # py-earth uses max(0, rss_reduction_score) for consistency with GCV logic
                actual_contribution = max(0.0, score_to_add)

                if actual_contribution > 0:
                    for var_idx in bf.get_involved_variables():
                        if 0 <= var_idx < num_features:
                            importances[var_idx] += actual_contribution

            # Normalize importances
            total_importance = np.sum(importances)
            if total_importance > 0:
                self.feature_importances_ = importances / total_importance
            else:
                self.feature_importances_ = importances # All zeros

        elif self.feature_importance_type is not None:
            # Placeholder for other types or warning for unknown types
            print(f"Warning: feature_importance_type '{self.feature_importance_type}' "
                  "is not yet fully implemented. Returning zeros for importances.")
            self.feature_importances_ = np.zeros(num_features)
        else:
            # feature_importance_type is None, so do nothing, self.feature_importances_ remains None
            pass


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

    def summary_feature_importances(self, sort_by_importance: bool = True) -> str:
        """
        Return a string describing the estimated feature importances.

        Parameters
        ----------
        sort_by_importance : bool, optional (default=True)
            If True, the features in the summary will be sorted by their
            importance values in descending order.

        Returns
        -------
        str
            A summary string of feature importances.
            Returns a message if importances have not been computed.
        """
        if not self.fitted_:
            return "Model not yet fitted. Feature importances not available."
        if self.feature_importances_ is None:
            return (f"Feature importances not computed. "
                    f"Set `feature_importance_type` (e.g., 'nb_subsets') "
                    f"during model initialization and call `fit()`.")

        # Assuming self.feature_importances_ is a 1D numpy array of scores
        # and self.record_ contains feature names if available, or we use generic names.

        num_features = len(self.feature_importances_)
        if hasattr(self.record_, 'n_features') and self.record_.n_features != num_features:
            # This case should ideally not happen if X_fit was used correctly
            feature_names = [f"x{i}" for i in range(num_features)]
        elif hasattr(self.record_, 'feature_names_in_'): # If sklearn wrapper set this in record
             feature_names = self.record_.feature_names_in_
        elif hasattr(self.record_, 'model_params') and \
             self.record_.model_params.get('feature_names_in_') is not None: # Check if stored from wrapper
             feature_names = self.record_.model_params['feature_names_in_']
        else: # Fallback to generic names
            feature_names = [f"x{i}" for i in range(num_features)]

        if len(feature_names) != num_features: # Fallback if names mismatch
            feature_names = [f"x{i}" for i in range(num_features)]

        importances = self.feature_importances_

        output = ["Feature Importances ({type})".format(type=self.feature_importance_type if self.feature_importance_type else "N/A")]
        output.append("-------------------------------------")

        if importances.size == 0:
            output.append("No features or importances available.")
            return "\n".join(output)

        indices = np.argsort(importances)[::-1] if sort_by_importance else np.arange(num_features)

        # Determine max length of feature name for alignment
        max_name_len = max(len(name) for name in feature_names) if feature_names else 10

        for i in indices:
            output.append(f"  {feature_names[i]:<{max_name_len + 2}} : {importances[i]:.4f}")

        output.append("-------------------------------------")
        return "\n".join(output)


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
