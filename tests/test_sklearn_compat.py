"""
Unit tests for the scikit-learn compatibility layer in pymars._sklearn_compat
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from pymars._sklearn_compat import (
    EarthRegressor,  # EarthClassifier will be tested separately
)


@pytest.fixture
def reg_data():
    """Return deterministic regression data."""
    np.random.seed(0)
    X = np.random.rand(50, 3) * 10
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] ** 2 + np.random.randn(50) * 2
    return X, y


def test_sklearn_compat_module_importable():
    """The scikit-learn compatibility module should import cleanly."""
    from pymars._sklearn_compat import EarthRegressor as Regressor

    assert Regressor is not None


def test_earth_regressor_instantiation_defaults():
    """EarthRegressor should expose its default constructor values."""
    reg = EarthRegressor()
    assert reg.max_degree == 1
    assert reg.penalty == 3.0
    assert reg.max_terms is None
    assert reg.minspan_alpha == 0.0
    assert reg.endspan_alpha == 0.0
    assert reg.allow_linear is True
    assert not hasattr(reg, "is_fitted_")
    assert not hasattr(reg, "coef_")
    assert not hasattr(reg, "basis_")
    assert not hasattr(reg, "n_features_in_")


def test_earth_regressor_instantiation_custom():
    """EarthRegressor should store custom constructor values unchanged."""
    reg = EarthRegressor(
        max_degree=2,
        penalty=2.5,
        max_terms=20,
        minspan_alpha=0.1,
        endspan_alpha=0.1,
        allow_linear=False,
    )
    assert reg.max_degree == 2
    assert reg.penalty == 2.5
    assert reg.max_terms == 20
    assert reg.minspan_alpha == 0.1
    assert reg.endspan_alpha == 0.1
    assert reg.allow_linear is False


def test_earth_regressor_fit(reg_data):
    """Fitting should populate the learned regression attributes."""
    X, y = reg_data
    reg = EarthRegressor(max_terms=5, penalty=0)
    reg.fit(X, y)

    assert reg.is_fitted_ is True
    assert hasattr(reg, "n_features_in_")
    assert reg.n_features_in_ == X.shape[1]
    assert hasattr(reg, "feature_names_in_")  # Will be generic names

    assert reg.basis_ is not None
    assert reg.coef_ is not None
    assert len(reg.basis_) == len(reg.coef_)
    assert len(reg.basis_) > 0
    assert len(reg.basis_) <= 5  # Intercept + up to 2 pairs

    assert reg.gcv_ is not None
    assert reg.rss_ is not None
    assert reg.mse_ is not None


def test_earth_regressor_fit_accepts_sample_weight(reg_data):
    """Fit should accept sample weights and complete successfully."""
    X, y = reg_data
    sample_weight = np.ones_like(y)
    sample_weight[-10:] = 25.0
    reg = EarthRegressor(max_terms=7, penalty=0)
    reg.fit(X, y, sample_weight=sample_weight)

    assert reg.is_fitted_ is True
    assert reg.coef_ is not None


def test_earth_regressor_sample_weight_changes_fit():
    """Heavier weights should shift predictions toward the weighted region."""
    X = np.array(
        [[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]],
        dtype=float,
    )
    y = np.array([0.0, 0.2, 0.1, 0.0, 20.0, 20.2, 19.8, 20.0], dtype=float)
    sample_weight = np.array([1.0, 1.0, 1.0, 1.0, 40.0, 40.0, 40.0, 40.0])

    unweighted = EarthRegressor(max_terms=3, penalty=0).fit(X, y)
    weighted = EarthRegressor(max_terms=3, penalty=0).fit(
        X, y, sample_weight=sample_weight
    )

    probe = np.array([[10.5]], dtype=float)
    assert weighted.predict(probe)[0] > unweighted.predict(probe)[0]


def test_earth_regressor_predict_before_fit(reg_data):
    """Predicting before fit should raise NotFittedError."""
    X, _ = reg_data
    reg = EarthRegressor()
    with pytest.raises(NotFittedError):
        reg.predict(X)


def test_earth_regressor_predict_after_fit(reg_data):
    """Predict should return arrays with the expected shape."""
    X, y = reg_data
    reg = EarthRegressor(max_terms=7)
    reg.fit(X, y)

    predictions = reg.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0],)

    # Predict on a single sample (1D array)
    single_sample_1d = X[0, :]
    pred_single_1d = reg.predict(single_sample_1d)
    assert pred_single_1d.shape == (1,)

    # Predict on a single sample (2D array)
    single_sample_2d = X[0, :].reshape(1, -1)
    pred_single_2d = reg.predict(single_sample_2d)
    assert pred_single_2d.shape == (1,)
    assert np.isclose(pred_single_1d[0], pred_single_2d[0])

    X_wrong_features = X[:, :-1]
    expected_msg_regex = (
        r"X has \d+ features, but EarthRegressor is expecting \d+ features as input."
    )
    with pytest.raises(ValueError, match=expected_msg_regex):
        reg.predict(X_wrong_features)


def test_earth_regressor_score(reg_data):
    """Score should behave like the regressor mixin's R^2 implementation."""
    X, y = reg_data
    reg = EarthRegressor(max_terms=5)
    reg.fit(X, y)

    score = reg.score(X, y)
    assert isinstance(score, float)
    assert -np.inf < score <= 1.0


def test_earth_regressor_get_set_params(reg_data):
    """Parameter getters and setters should round-trip cleanly."""
    reg = EarthRegressor(max_degree=1, penalty=3.0)
    params = reg.get_params()

    assert params["max_degree"] == 1
    assert params["penalty"] == 3.0

    reg.set_params(max_degree=2, penalty=5.0)
    assert reg.max_degree == 2
    assert reg.penalty == 5.0

    X, y = reg_data
    reg.fit(X, y)
    assert hasattr(reg, "earth_")
    assert reg.earth_.max_degree == 2
    assert reg.earth_.penalty == 5.0


def test_earth_regressor_check_estimator():
    """Run scikit-learn's estimator checks for the regressor."""
    estimator = EarthRegressor()
    expected_failures = {
        "check_fit2d_predict1d": "Estimator fit enforces 1D y.",
        "check_estimators_pickle": "BasisFunction objects may not pickle correctly yet.",
        "check_complex_data": "MARS does not support complex data.",
        "check_regressors_train": "Specific data value/type checks in this test might require deeper investigation.",
        "check_regressor_multioutput": "Multioutput regression not supported.",
        "check_fit_score_takes_y_दानी": "Sparse y_दानी not supported.",
    }
    check_estimator(estimator, expected_failed_checks=expected_failures)


from sklearn.svm import SVC

from pymars._sklearn_compat import EarthClassifier


@pytest.fixture
def clf_data():
    """Return deterministic classification data."""
    np.random.seed(1)
    X = np.random.rand(60, 3) * 10
    y = ((X[:, 0] > 5) & (X[:, 1] < 13) | (X[:, 2] > 7)).astype(int)
    return X, y


def test_earth_classifier_module_importable():
    """The classifier wrapper should import cleanly."""
    from pymars._sklearn_compat import EarthClassifier as Classifier

    assert Classifier is not None


def test_earth_classifier_instantiation_defaults():
    """EarthClassifier should expose its default constructor values."""
    clf = EarthClassifier()
    assert clf.max_degree == 1
    assert clf.penalty == 3.0
    assert clf.classifier is None  # User did not provide one
    assert not hasattr(clf, "is_fitted_")
    assert not hasattr(clf, "classifier_")  # Should not exist before fit
    assert not hasattr(clf, "earth_")  # Should not exist before fit


def test_earth_classifier_instantiation_custom_mars_params():
    """Custom MARS parameters should be stored as given."""
    clf = EarthClassifier(max_degree=2, penalty=2.0, max_terms=15)
    assert clf.max_degree == 2


def test_earth_classifier_instantiation_custom_classifier():
    """A custom classifier instance should be preserved on the wrapper."""
    my_svc = SVC(probability=True, random_state=0)
    clf = EarthClassifier(classifier=my_svc)
    assert clf.classifier is my_svc
    assert not hasattr(clf, "classifier_")


def test_earth_classifier_fit_accepts_sample_weight(clf_data):
    """Fit should accept sample weights for the classifier wrapper."""
    X, y = clf_data
    sample_weight = np.ones_like(y, dtype=float)
    sample_weight[:10] = 10.0
    clf = EarthClassifier(max_terms=5)
    clf.fit(X, y, sample_weight=sample_weight)

    assert clf.is_fitted_ is True
    assert hasattr(clf, "classifier_")


def test_earth_classifier_fit(clf_data):
    """Fitting should populate the learned classifier attributes."""
    X, y = clf_data
    clf = EarthClassifier(max_terms=6)
    clf.fit(X, y)

    assert clf.is_fitted_
    assert hasattr(clf, "classes_")
    assert np.array_equal(clf.classes_, np.array([0, 1]))
    assert hasattr(clf, "n_features_in_")
    assert clf.n_features_in_ == X.shape[1]

    assert clf.basis_ is not None
    assert len(clf.basis_) > 0
    assert len(clf.basis_) <= 6

    assert hasattr(clf, "classifier_")
    assert hasattr(clf.classifier_, "predict")  # Check if it's a fitted classifier


def test_earth_classifier_predict_before_fit(clf_data):
    """Predicting before fit should raise NotFittedError."""
    X, _ = clf_data
    clf = EarthClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_earth_classifier_predict_predict_proba(clf_data):
    """Predict and predict_proba should behave consistently after fit."""
    X, y = clf_data
    clf = EarthClassifier(max_terms=6)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(np.isin(y_pred, clf.classes_))
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (X.shape[0], len(clf.classes_))
    assert np.all((y_proba >= 0) & (y_proba <= 1))
    assert np.allclose(np.sum(y_proba, axis=1), 1.0)

    clf_no_proba = EarthClassifier(classifier=SVC(random_state=0))
    clf_no_proba.fit(X, y)
    with pytest.raises(AttributeError, match="does not support predict_proba"):
        clf_no_proba.predict_proba(X)


def test_earth_classifier_score(clf_data):
    """Score should stay within the classifier accuracy range."""
    X, y = clf_data
    clf = EarthClassifier(max_terms=6)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_earth_classifier_get_set_params(clf_data):
    """Parameter getters and setters should round-trip cleanly."""
    clf = EarthClassifier(
        max_degree=1, penalty=4.0, classifier=SVC(C=1.0, random_state=0)
    )
    params = clf.get_params()

    assert params["max_degree"] == 1
    assert params["penalty"] == 4.0
    assert isinstance(params["classifier"], SVC)
    assert params["classifier__C"] == 1.0

    clf.set_params(max_degree=2, classifier__C=5.0, classifier__kernel="poly")
    assert clf.max_degree == 2
    assert clf.get_params()["classifier__C"] == 5.0
    assert clf.get_params()["classifier__kernel"] == "poly"

    X, y = clf_data
    clf.fit(X, y)
    assert clf.earth_.max_degree == 2
    assert clf.classifier_.C == 5.0
    assert clf.classifier_.kernel == "poly"


def test_earth_classifier_check_estimator():
    """Run scikit-learn's estimator checks for the classifier wrapper."""
    estimator = EarthClassifier(max_terms=5)
    expected_failures_clf = {
        "check_estimators_pickle": "BasisFunction objects may not pickle correctly yet.",
        "check_complex_data": "MARS does not support complex data.",
        "check_classifiers_predictions": "Exact prediction matching might be tricky due to two-stage nature.",
        "check_classifiers_train": "Specific data/type checks.",
        "check_supervised_y_2d": "CoreEarth fit currently expects 1D y.",
        "check_fit2d_predict1d": "Estimator fit enforces 1D y for CoreEarth part.",
        "check_fit_score_takes_y_दानी": "Sparse y_दानी not supported.",
        "check_classifier_multioutput": "Multioutput classification not supported.",
        "check_n_features_in_after_fitting": "Feature consistency for score method.",
    }
    check_estimator(estimator, expected_failed_checks=expected_failures_clf)
