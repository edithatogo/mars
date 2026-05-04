"""
Scikit-learn compatibility layer for pymars.

This module will contain classes like EarthRegressor and EarthClassifier
that wrap the core Earth model to make it fully compliant with
scikit-learn's Estimator API.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import has_fit_parameter

from .earth import (
    Earth as CoreEarth,
)

logger = logging.getLogger(__name__)


class EarthRegressor(RegressorMixin, BaseEstimator):
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

    def __init__(
        self,
        max_degree: int = 1,
        penalty: float = 3.0,
        max_terms: int | None = None,
        minspan_alpha: float = 0.0,
        endspan_alpha: float = 0.0,
        allow_linear: bool = True,
    ):

        super().__init__()  # Recommended for BaseEstimator subclasses

        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.minspan_alpha = minspan_alpha
        self.endspan_alpha = endspan_alpha
        self.allow_linear = allow_linear

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> EarthRegressor:
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
        from sklearn.utils.validation import check_X_y  # For validation

        X_validated, y_validated = check_X_y(
            X, y, accept_sparse=False, y_numeric=True, multi_output=False
        )

        self.n_features_in_ = X_validated.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )

        earth: Any = CoreEarth(
            max_degree=self.max_degree,
            penalty=self.penalty,
            max_terms=self.max_terms,
            minspan_alpha=self.minspan_alpha,
            endspan_alpha=self.endspan_alpha,
            allow_linear=self.allow_linear,
        )

        earth.fit(X_validated, y_validated, sample_weight=sample_weight)
        self.earth_ = earth

        self.basis_ = self.earth_.basis_
        self.coef_ = self.earth_.coef_
        self.gcv_ = self.earth_.gcv_
        self.rss_ = self.earth_.rss_
        self.mse_ = self.earth_.mse_

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
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
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self, ["coef_", "basis_", "n_features_in_"])

        X_validated = check_array(X, accept_sparse=False, ensure_2d=False)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)

        if X_validated.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_validated.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

        earth: Any = self.earth_
        return cast("np.ndarray", earth.predict(X_validated))
class EarthClassifier(ClassifierMixin, BaseEstimator):
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

    def __init__(
        self,
        max_degree: int = 1,
        penalty: float = 3.0,
        max_terms: int | None = None,
        minspan_alpha: float = 0.0,
        endspan_alpha: float = 0.0,
        allow_linear: bool = True,
        classifier: Any | None = None,
    ):

        super().__init__()

        self.max_degree = max_degree
        self.penalty = penalty
        self.max_terms = max_terms
        self.minspan_alpha = minspan_alpha
        self.endspan_alpha = endspan_alpha
        self.allow_linear = allow_linear
        self.classifier = classifier

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None) -> EarthClassifier:
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
        import numpy as np
        from sklearn.base import (
            clone,  # To clone the classifier if user provided an instance
        )
        from sklearn.utils.multiclass import unique_labels
        from sklearn.utils.validation import check_X_y

        X_validated, y_original_labels = check_X_y(
            X, y, accept_sparse=False, multi_output=False
        )

        self.classes_ = unique_labels(y_original_labels)
        self.n_features_in_ = X_validated.shape[1]

        from sklearn.preprocessing import LabelEncoder

        self._label_encoder = LabelEncoder()
        y_numeric_for_earth = self._label_encoder.fit_transform(y_original_labels)

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )

        earth: Any = CoreEarth(
            max_degree=self.max_degree,
            penalty=self.penalty,
            max_terms=self.max_terms,
            minspan_alpha=self.minspan_alpha,
            endspan_alpha=self.endspan_alpha,
            allow_linear=self.allow_linear,
        )

        earth.fit(X_validated, y_numeric_for_earth, sample_weight=sample_weight)
        self.earth_ = earth
        self.basis_ = self.earth_.basis_

        if not self.basis_:
            logger.warning(
                "Earth model fitting resulted in no basis functions. Classifier will be fit on original X (if applicable) or might fail."
            )
            X_transformed = X_validated
            if not X_transformed.any():
                self.is_fitted_ = True
                return self
        else:
            missing_mask_for_transform = np.zeros_like(X_validated, dtype=bool)
            X_transformed = self.earth_._build_basis_matrix(
                X_validated, self.basis_, missing_mask_for_transform
            )

        if self.classifier is None:
            classifier: Any = LogisticRegression(solver="lbfgs", random_state=0)
        else:
            classifier = clone(self.classifier)

        if sample_weight is not None and has_fit_parameter(classifier, "sample_weight"):
            classifier.fit(
                X_transformed,
                y_original_labels,
                sample_weight=sample_weight,
            )
        else:
            if sample_weight is not None:
                logger.warning(
                    "Internal classifier %s does not accept sample_weight; weights were ignored at the classification layer.",
                    classifier.__class__.__name__,
                )
            classifier.fit(X_transformed, y_original_labels)
        self.classifier_ = classifier

        self.is_fitted_ = True
        return self

    def _transform_X_for_classifier(self, X: Any) -> np.ndarray:
        """Helper to transform X using fitted Earth basis functions."""
        import numpy as np
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self, ["classes_", "classifier_", "earth_"])
        X_validated = check_array(X, accept_sparse=False, ensure_2d=False)
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(1, -1)

        if self.n_features_in_ is not None and X_validated.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_validated.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

        if not self.basis_:  # Earth model resulted in no basis functions
            logger.warning(
                "EarthClassifier.basis_ is empty. Predictions might be based on original features if fit handled this."
            )
            return cast("np.ndarray", X_validated)

        missing_mask_for_transform = np.zeros_like(X_validated, dtype=bool)
        earth: Any = self.earth_
        X_transformed = earth._build_basis_matrix(
            X_validated, self.basis_, missing_mask_for_transform
        )

        if X_transformed.shape[1] == 0 and self.basis_:
            logger.warning(
                "Transformed features matrix is empty despite basis functions existing."
            )

        return cast("np.ndarray", X_transformed)

    def predict(self, X: Any) -> np.ndarray:
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
        classifier: Any = self.classifier_
        return cast("np.ndarray", classifier.predict(X_transformed))

    def predict_proba(self, X: Any) -> np.ndarray:
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
        from sklearn.utils.validation import check_is_fitted  # Local import

        check_is_fitted(self, ["classes_", "classifier_", "n_features_in_"])

        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError(
                f"The internal classifier {self.classifier_.__class__.__name__} "
                "does not support predict_proba."
            )
        X_transformed = self._transform_X_for_classifier(X)
        classifier: Any = self.classifier_
        return cast("np.ndarray", classifier.predict_proba(X_transformed))

    def score(self, X: Any, y: Any, sample_weight: Any | None = None) -> float:
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
        return float(accuracy_score(y, self.predict(X), sample_weight=sample_weight))
