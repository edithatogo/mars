# -*- coding: utf-8 -*-

"""
Scikit-learn compatibility layer for pymars.

This module will contain classes like EarthRegressor and EarthClassifier
that wrap the core Earth model to make it fully compliant with
scikit-learn's Estimator API.
"""

# from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from sklearn.utils.multiclass import unique_labels # For classifiers
# from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted # Will use these in fit/predict
# from sklearn.utils.multiclass import unique_labels # For classifiers
from .earth import Earth as CoreEarth # Rename to avoid clash if EarthRegressor inherits it directly
# from ._types import XType, YType # Custom types

import logging
import numpy as np

logger = logging.getLogger(__name__)


class EarthRegressor(RegressorMixin, BaseEstimator): # Corrected Mixin Order
    """
    Pymars Earth model for regression tasks, scikit-learn compatible.

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
        Interpreted relative to the number of unique values in a variable.

    endspan_alpha : float, optional (default=0.0)
        Parameter controlling the end span of a basis function.
        Interpreted relative to the number of unique values in a variable.

    allow_linear : bool, optional (default=True)
        Whether to allow the forward pass to consider adding purely linear terms
        for variables, in addition to hinge functions.

    Attributes
    ----------
    earth_ : pymars.earth.Earth
        The underlying core Earth model instance.

    basis_ : list of BasisFunction
        The selected basis functions after fitting. Copied from `earth_.basis_`.

    coef_ : np.ndarray
        The coefficients of the selected basis functions. Copied from `earth_.coef_`.

    gcv_ : float
        The GCV of the final selected model. Copied from `earth_.gcv_`.

    rss_ : float
        The RSS of the final selected model on the training data. Copied from `earth_.rss_`.

    mse_ : float
        The MSE of the final selected model on the training data. Copied from `earth_.mse_`.

    n_features_in_ : int
        Number of features seen during `fit`.

    feature_names_in_ : np.ndarray
        Names of features seen during `fit`. Defined if `X` has feature names.

    is_fitted_ : bool
        Flag indicating whether the model has been fitted.

    See Also
    --------
    pymars.earth.Earth : The core implementation of the MARS algorithm.

    Examples
    --------
    >>> from pymars._sklearn_compat import EarthRegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.sin(X[:,0]/10) + X[:,0]/50
    >>> model = EarthRegressor(max_degree=1, max_terms=10)
    >>> model.fit(X, y)
    EarthRegressor(...)
    >>> preds = model.predict(X)
    """
    def __init__(self, max_degree: int = 1, penalty: float = 3.0, max_terms: int = None,
                 minspan_alpha: float = 0.0, endspan_alpha: float = 0.0,
                 allow_linear: bool = True):

        super().__init__() # Recommended for BaseEstimator subclasses

        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.minspan_alpha = minspan_alpha
        self.endspan_alpha = endspan_alpha
        self.allow_linear = allow_linear

        # Internal Earth model instance will be created in fit or lazily
        # For get_params/set_params to work seamlessly with nested models,
        # it's common to instantiate it here.
        # However, to pass check_estimator's check_no_attributes_set_in_init,
        # we should not instantiate it here if 'earth_' is not an __init__ param.
        # self.earth_ = CoreEarth(
        #     max_degree=self.max_degree,
        #     penalty=self.penalty,
        #     max_terms=self.max_terms,
        #     minspan_alpha=self.minspan_alpha,
        #     endspan_alpha=self.endspan_alpha,
        #     allow_linear=self.allow_linear
        # )
        # self.earth_ will be created in fit()

        # Fitted attributes (like self.coef_, self.basis_, self.n_features_in_, self.is_fitted_)
        # will be initialized in the fit() method, as per scikit-learn convention
        # to pass check_no_attributes_set_in_init from check_estimator.


    def fit(self, X, y):
        """
        Fit the Earth regressor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values (continuous).

        Returns
        -------
        self : EarthRegressor
            The fitted regressor.
        """
        from sklearn.utils.validation import check_X_y, check_array  # For validation

        # Validate X and y using scikit-learn utilities
        # This ensures X is 2D, y is 1D, they have consistent lengths,
        # are numeric, finite, and converts them to float64 by default.
        X_validated, y_validated = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=False)

        # Store number of features and feature names (if X was a DataFrame)
        self.n_features_in_ = X_validated.shape[1]
        if hasattr(X, 'columns'): # Check if original X was a DataFrame
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            # Create generic feature names if not provided
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

        # Re-initialize the core Earth model with current hyperparameters
        # This ensures that if set_params was called, the new params are used.
        self.earth_ = CoreEarth(
            max_degree=self.max_degree,
            penalty=self.penalty,
            max_terms=self.max_terms,
            minspan_alpha=self.minspan_alpha,
            endspan_alpha=self.endspan_alpha,
            allow_linear=self.allow_linear
        )

        # Fit the internal Earth model
        self.earth_.fit(X_validated, y_validated)

        # Copy fitted attributes from the core model to the wrapper
        self.basis_ = self.earth_.basis_
        self.coef_ = self.earth_.coef_
        self.gcv_ = self.earth_.gcv_
        self.rss_ = self.earth_.rss_
        self.mse_ = self.earth_.mse_
        # self.record_ could also be copied if desired for inspection via the wrapper

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X : XType
            Input samples.

        Returns
        -------
        y_pred : YType
            Predicted continuous values.
        """
        from sklearn.utils.validation import check_is_fitted, check_array
        import numpy as np

        # Check if fit has been called
        # Attributes that signify it's fitted: coef_ and basis_ must exist.
        # n_features_in_ is also essential for validating X.
        check_is_fitted(self, ["coef_", "basis_", "n_features_in_"])

        # Validate X
        # ensure_min_samples=0 allows prediction on a single sample if desired by user.
        # ensure_2d=False because check_array handles 1D X (e.g. single sample, multiple features if reshaped before)
        # but typically X for predict is 2D. check_array will convert 1D to 2D if it's a single sample.
        X_validated = check_array(X, accept_sparse=False, ensure_2d=False)
        if X_validated.ndim == 1:
            # If X is 1D (e.g. a single sample like [f1, f2, f3]), reshape to (1, n_features)
            # This check is important because our CoreEarth.predict expects 2D X.
             X_validated = X_validated.reshape(1, -1)


        # Check that the number of features is the same as during fit
        if X_validated.shape[1] != self.n_features_in_:
            # Message pattern for check_estimator
            raise ValueError(
                f"X has {X_validated.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

        # Delegate prediction to the internal Earth model
        return self.earth_.predict(X_validated)

    # score method is inherited from RegressorMixin, which calculates R^2 score.
    # No need to override unless a different default scoring behavior is desired.

    # get_params and set_params are inherited from BaseEstimator.
    # The default implementations should work correctly because:
    # 1. All __init__ parameters are stored as public attributes with the same names.
    # 2. The `fit` method re-instantiates `self.earth_` using these attributes,
    #    so changes made by `set_params` to `EarthRegressor`'s attributes
    #    will be reflected in the `CoreEarth` instance used during the next `fit`.

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Using LogisticRegression as a default internal classifier for now.
# Could be made configurable.

class EarthClassifier(ClassifierMixin, BaseEstimator): # Corrected Mixin Order
    """
    Pymars Earth model for classification tasks, scikit-learn compatible.

    This estimator uses the MARS algorithm (via the core Earth model) to generate
    a set of basis functions from the input features X. These basis functions
    then serve as transformed features for an internal scikit-learn classifier
    (defaulting to LogisticRegression).

    Parameters
    ----------
    max_degree : int, optional (default=1)
        Maximum degree of interaction terms for the MARS basis function generation.
    penalty : float, optional (default=3.0)
        Penalty for model complexity in GCV for MARS basis function selection.
    max_terms : int, optional (default=None)
        Maximum number of basis functions generated by MARS.
    minspan_alpha : float, optional (default=0.0)
        MARS parameter for minimum span.
    endspan_alpha : float, optional (default=0.0)
        MARS parameter for end span.
    allow_linear : bool, optional (default=True)
        MARS parameter to allow linear terms.

    classifier : estimator object, optional (default=None)
        The scikit-learn compatible classifier to be used on top of the MARS
        basis functions. If None, `LogisticRegression(solver='lbfgs', random_state=0)`
        is used by default. The `random_state` is set for reproducibility.
        Common choices include `LogisticRegression`, `SVC`, `RandomForestClassifier`.

    # TODO: Add other relevant Earth params if they should be exposed directly.
    # TODO: Add relevant params of the default LogisticRegression (e.g., C, class_weight)
    #       or allow passing a fully configured classifier instance.

    Attributes
    ----------
    earth_ : pymars.earth.Earth
        The fitted underlying core Earth model instance used for feature transformation.

    classifier_ : estimator object
        The fitted internal scikit-learn classifier.

    basis_ : list of BasisFunction
        The selected basis functions from the Earth model. Copied from `earth_.basis_`.

    classes_ : np.ndarray of shape (n_classes,)
        The unique class labels seen during `fit`.

    n_features_in_ : int
        Number of features seen during `fit`.

    feature_names_in_ : np.ndarray
        Names of features seen during `fit`. Defined if `X` has feature names.

    is_fitted_ : bool
        Flag indicating whether the model has been fitted.
    """
    def __init__(self, max_degree: int = 1, penalty: float = 3.0, max_terms: int = None,
                 minspan_alpha: float = 0.0, endspan_alpha: float = 0.0,
                 allow_linear: bool = True,
                 classifier=None # Allow user to pass a classifier instance
                 # TODO: Add specific classifier params like C for LogisticRegression if classifier is None
                ):

        super().__init__()

        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.minspan_alpha = minspan_alpha
        self.endspan_alpha = endspan_alpha
        self.allow_linear = allow_linear
        self.classifier = classifier # User-provided or None

        # Internal Earth model instance
        # Instantiated here so get_params can find its params if deep=True
        # and set_params can potentially set them (though our fit re-instantiates earth_).
        # To pass check_estimator, self.earth_ should not be set in __init__ if it's not an init param.
        # It will be created in fit().
        # self.earth_ = CoreEarth(
        #     max_degree=self.max_degree,
        #     penalty=self.penalty,
        #     max_terms=self.max_terms,
        #     minspan_alpha=self.minspan_alpha,
        #     endspan_alpha=self.endspan_alpha,
        #     allow_linear=self.allow_linear
        # )

        # Internal classifier instance `self.classifier_` will be created in `fit`
        # based on `self.classifier`.

        # Fitted attributes (like self.basis_, self.classes_, self.n_features_in_,
        # self.is_fitted_, self.classifier_) will be initialized in the fit() method.


    def fit(self, X, y):
        """
        Fit the Earth classifier to the training data.

        This involves first fitting the internal MARS model (Earth) to generate
        basis functions from X (potentially using y for supervised selection),
        then transforming X into this new basis function space, and finally
        fitting the internal scikit-learn classifier on these transformed features
        and the original y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values (discrete classes).

        Returns
        -------
        self : EarthClassifier
            The fitted classifier.
        """
        from sklearn.utils.validation import check_X_y
        from sklearn.utils.multiclass import unique_labels
        from sklearn.base import clone # To clone the classifier if user provided an instance
        import numpy as np

        # Validate X and y
        # For classification, y doesn't strictly need to be numeric for check_X_y,
        # but our CoreEarth model currently expects numeric y.
        # This might need adjustment if CoreEarth cannot handle non-numeric y in its GCV.
        # For now, assume y will be label encoded or numeric.
        X_validated, y_original_labels = check_X_y(X, y, accept_sparse=False, multi_output=False)

        self.classes_ = unique_labels(y_original_labels)
        self.n_features_in_ = X_validated.shape[1]

        # CoreEarth.fit expects a numeric y for its internal RSS/GCV calculations.
        # We use LabelEncoder to convert y to numeric for CoreEarth,
        # but the final classifier self.classifier_ will be fit on original (or validated) y labels.
        from sklearn.preprocessing import LabelEncoder
        self._label_encoder = LabelEncoder()
        y_numeric_for_earth = self._label_encoder.fit_transform(y_original_labels)

        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

        # Re-initialize the core Earth model
        self.earth_ = CoreEarth(
            max_degree=self.max_degree,
            penalty=self.penalty,
            max_terms=self.max_terms,
            minspan_alpha=self.minspan_alpha,
            endspan_alpha=self.endspan_alpha,
            allow_linear=self.allow_linear
        )

        # Fit the Earth model to generate basis functions.
        # The CoreEarth.fit method uses y for calculating RSS for term selection and GCV for pruning.
        # For classification, this y might not be ideal for RSS/GCV if it's categorical.
        # A common strategy is to use a modified y (e.g., one-hot encoded means, or use X only
        # if MARS had an unsupervised mode).
        # For now, we pass y_validated. If it's label-encoded (0, 1, 2...), it might work for GCV.
        # This is an area that might need refinement based on how well CoreEarth handles categorical-like y.
        self.earth_.fit(X_validated, y_numeric_for_earth)
        self.basis_ = self.earth_.basis_

        if not self.basis_:
            # If earth model fitting failed to produce any basis functions (even intercept)
            # This implies an issue, or a very degenerate case.
            # We should probably fit the classifier on original X or raise error.
            # For now, let's assume earth_ always produces at least an intercept.
            # If self.basis_ is empty, _build_basis_matrix will return empty B.
             logger.warning(
                 "Earth model fitting resulted in no basis functions. Classifier will be fit on original X (if applicable) or might fail."
             )
             # Fallback: use original features, or handle as error.
             # If we use original features, the "Earth" part isn't doing much.
             # For simplicity now, if no basis functions, the classifier step might fail or be trivial.
             # A robust solution would be to ensure earth_ at least has an intercept or handle this.
             X_transformed = X_validated # Fallback, though not ideal.
             if not X_transformed.any(): # If X_validated was also empty (should be caught by check_X_y)
                 self.is_fitted_ = True # Mark as fitted, but it's a degenerate model
                 return self
        else:
            # Transform X using the fitted Earth model's basis functions
            # Since X_validated is assumed to be NaN-free by this point (due to check_X_y),
            # an all-False missing_mask is appropriate for the core Earth's _build_basis_matrix.
            missing_mask_for_transform = np.zeros_like(X_validated, dtype=bool)
            X_transformed = self.earth_._build_basis_matrix(X_validated, self.basis_, missing_mask_for_transform)

        # Initialize/Clone and Fit the internal classifier
        if self.classifier is None:  # Use default
            self.classifier_ = LogisticRegression(solver='lbfgs', random_state=0)
        else: # User provided a classifier instance
            self.classifier_ = clone(self.classifier) # Clone to ensure fresh state

        self.classifier_.fit(X_transformed, y_original_labels) # Fit classifier on original (validated) labels

        self.is_fitted_ = True
        return self

    def _transform_X_for_classifier(self, X):
        """Helper to transform X using fitted Earth basis functions."""
        from sklearn.utils.validation import check_is_fitted, check_array
        import numpy as np

        # Attributes that signify it's fitted: classes_, classifier_, earth_ (which implies basis_ is also set up or handled)
        check_is_fitted(self, ["classes_", "classifier_", "earth_"])
        X_validated = check_array(X, accept_sparse=False, ensure_2d=False)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)

        # n_features_in_ should be set by fit. If not, check_is_fitted should have caught it.
        if self.n_features_in_ is not None and X_validated.shape[1] != self.n_features_in_: # check self.n_features_in_ existence
            # Message pattern for check_estimator
            raise ValueError(
                f"X has {X_validated.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

        if not self.basis_: # Earth model resulted in no basis functions
            # This case should be handled based on how fit decided to proceed.
            # If fit defaulted to using original X for classifier:
            # print("Warning: Earth model has no basis functions. Using original X for classifier prediction.")
            # return X_validated
            # For now, assume if basis_ is empty, it's an issue or should have at least an intercept.
            # If core Earth ensures an intercept, self.basis_ won't be empty.
            # If it could be empty, this indicates a state where the classifier might not be meaningfully fit.
            # Let's assume self.earth_._build_basis_matrix handles empty self.basis_ by returning empty matrix,
            # and the classifier must have been fit on something (e.g. original X or failed).
            # If self.basis_ is empty, the transform should reflect that (e.g. empty feature set for classifier).
            # This path implies the classifier was fit on X_validated directly in `fit` method's fallback.
             logger.warning(
                 "EarthClassifier.basis_ is empty. Predictions might be based on original features if fit handled this."
             )
             return X_validated # Fallback if fit decided to use original X

        # Since X_validated is assumed to be NaN-free by this point (due to check_array),
        # an all-False missing_mask is appropriate.
        missing_mask_for_transform = np.zeros_like(X_validated, dtype=bool)
        X_transformed = self.earth_._build_basis_matrix(X_validated, self.basis_, missing_mask_for_transform)

        # Ensure X_transformed is not empty if basis functions exist, otherwise it's an issue.
        # (e.g. if all basis functions evaluate to constants that get removed, or some error)
        # For now, assume _build_basis_matrix and basis functions are well-behaved.
        if X_transformed.shape[1] == 0 and self.basis_:
            # This means basis functions existed but produced an empty matrix (e.g. all zero columns).
            # This is an edge case, may need a more robust handling in _build_basis_matrix or here.
            # For now, if classifier was fit on this, it might expect 0 features.
            # Or, more likely, we should ensure B_matrix is not empty if basis_ is not.
            # A common fallback is to return an array of zeros or means if this happens.
            # Let's assume for now that if self.basis_ is not empty, X_transformed will have columns.
             logger.warning(
                 "Transformed features matrix is empty despite basis functions existing."
             )

        return X_transformed

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X_transformed = self._transform_X_for_classifier(X)
        return self.classifier_.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        p : np.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        from sklearn.utils.validation import check_is_fitted # Local import

        # Attributes that signify it's fitted for classifier context
        check_is_fitted(self, ["classes_", "classifier_", "n_features_in_"])

        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError(
                f"The internal classifier {self.classifier_.__class__.__name__} "
                "does not support predict_proba."
            )
        X_transformed = self._transform_X_for_classifier(X) # This also calls check_is_fitted
        return self.classifier_.predict_proba(X_transformed)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : XType
            Test samples.
        y : YType
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        # Relies on ClassifierMixin or implement directly
        return accuracy_score(y, self.predict(X))

    # def get_params(self, deep=True):
    #     # From BaseEstimator - should work due to __init__ params being public attributes
    #     # and self.classifier being an __init__ param.
    #     pass

    # def set_params(self, **params):
    #     # From BaseEstimator - should work. It will update self.classifier
    #     # or its sub-parameters. `fit` then uses the updated `self.classifier`.
    #     # Similarly for MARS params, they are set on self, and `fit` re-creates self.earth_
    #     pass

