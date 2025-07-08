# -*- coding: utf-8 -*-

"""
Unit tests for the scikit-learn compatibility layer in pymars._sklearn_compat
"""

import pytest
# import numpy as np
# from pymars._sklearn_compat import EarthRegressor, EarthClassifier

# Attempt to use sklearn's estimator checker if sklearn is available
try:
    # from sklearn.utils.estimator_checks import check_estimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def test_sklearn_compat_module_importable():
    """Test that the _sklearn_compat module can be imported."""
    try:
        from pymars import _sklearn_compat
        assert _sklearn_compat is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pymars._sklearn_compat: {e}")

def test_earth_regressor_instantiation():
    """Test basic instantiation of EarthRegressor."""
    # from pymars._sklearn_compat import EarthRegressor
    # reg = EarthRegressor(max_degree=2, penalty=2.5, max_terms=20)
    # assert reg.max_degree == 2
    # assert reg.penalty == 2.5
    # assert reg.max_terms == 20
    # # assert hasattr(reg, 'earth_') # Check if underlying Earth model is created
    print("Placeholder: test_earth_regressor_instantiation")
    pass

def test_earth_classifier_instantiation():
    """Test basic instantiation of EarthClassifier."""
    # from pymars._sklearn_compat import EarthClassifier
    # clf = EarthClassifier(max_degree=1, penalty=3.5)
    # assert clf.max_degree == 1
    # assert clf.penalty == 3.5
    # # assert hasattr(clf, 'earth_')
    # # assert hasattr(clf, 'classifier_') # Check if internal sklearn classifier is set up
    print("Placeholder: test_earth_classifier_instantiation")
    pass

# The following tests would use sklearn.utils.estimator_checks.check_estimator
# which runs a comprehensive suite of tests to ensure scikit-learn compatibility.
# These will be more relevant once the core Earth model and the compatibility
# layer are more fully implemented.

# @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available for check_estimator")
# def test_earth_regressor_sklearn_compatibility():
#     """Check EarthRegressor for scikit-learn compatibility."""
#     from pymars._sklearn_compat import EarthRegressor
#     # This check is very thorough and will likely fail until EarthRegressor is fully implemented.
#     # It tests for things like get_params, set_params, fit, predict, score, input validation etc.
#     # Errors from check_estimator are usually quite descriptive.
#     # return check_estimator(EarthRegressor())
#     print("Placeholder: test_earth_regressor_sklearn_compatibility (requires full implementation and sklearn)")
#     # As a very basic preliminary check:
#     # estimator = EarthRegressor()
#     # assert hasattr(estimator, "get_params")
#     # assert hasattr(estimator, "set_params")
#     # assert hasattr(estimator, "fit")
#     # assert hasattr(estimator, "predict")
#     # assert hasattr(estimator, "score")
#     pass


# @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available for check_estimator")
# def test_earth_classifier_sklearn_compatibility():
#     """Check EarthClassifier for scikit-learn compatibility."""
#     from pymars._sklearn_compat import EarthClassifier
#     # return check_estimator(EarthClassifier())
#     print("Placeholder: test_earth_classifier_sklearn_compatibility (requires full implementation and sklearn)")
#     # estimator = EarthClassifier()
#     # assert hasattr(estimator, "get_params")
#     # assert hasattr(estimator, "set_params")
#     # assert hasattr(estimator, "fit")
#     # assert hasattr(estimator, "predict")
#     # assert hasattr(estimator, "predict_proba")
#     # assert hasattr(estimator, "score")
#     pass


# Example of more specific fit/predict tests for the wrappers
# def test_earth_regressor_fit_predict():
#     """Test fit and predict methods of EarthRegressor."""
#     # from pymars._sklearn_compat import EarthRegressor
#     # import numpy as np
#     # X = np.random.rand(10,2)
#     # y = X[:,0] * 2 + X[:,1] * 0.5 + np.random.randn(10) * 0.1
#     # reg = EarthRegressor(max_terms=5) # Keep it simple
#     # reg.fit(X,y)
#     # assert hasattr(reg, 'fitted_') and reg.fitted_
#     # predictions = reg.predict(X)
#     # assert predictions.shape == y.shape
#     # score = reg.score(X,y)
#     # assert isinstance(score, float)
#     print("Placeholder: test_earth_regressor_fit_predict")
#     pass

if __name__ == '__main__':
    # pytest.main([__file__])
    print("Run tests using 'pytest tests/test_sklearn_compat.py'")
