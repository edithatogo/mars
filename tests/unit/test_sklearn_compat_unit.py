"""Unit tests for scikit-learn compatibility."""

import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score

pytest.importorskip("pymars")


@pytest.mark.unit
class TestEarthRegressorSklearn:
    """Test EarthRegressor sklearn compatibility."""

    def test_earth_regressor_basic_sklearn_compatibility(self):
        """Test EarthRegressor basic sklearn compatibility."""
        from pymars import EarthRegressor
        from sklearn.base import BaseEstimator, RegressorMixin

        # Verify inheritance
        assert issubclass(EarthRegressor, BaseEstimator)
        assert issubclass(EarthRegressor, RegressorMixin)

        # Test basic functionality
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = EarthRegressor(max_terms=10, max_degree=1)
        model.fit(X, y)

        # Verify sklearn-compatible interface
        assert hasattr(model, 'predict')
        assert hasattr(model, 'score')
        assert hasattr(model, 'get_params')
        assert hasattr(model, 'set_params')
        assert hasattr(model, 'coef_')

    def test_earth_regressor_fit_returns_self(self):
        """Test that fit returns self."""
        from pymars import EarthRegressor

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = EarthRegressor(max_terms=10, max_degree=1)
        result = model.fit(X, y)

        assert result is model

    def test_earth_regressor_predict_shape(self):
        """Test predict returns correct shape."""
        from pymars import EarthRegressor

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = EarthRegressor(max_terms=10, max_degree=1)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == (50,)

    def test_earth_regressor_score(self):
        """Test score returns R² score."""
        from pymars import EarthRegressor

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = EarthRegressor(max_terms=10, max_degree=1)
        model.fit(X, y)
        score = model.score(X, y)

        assert isinstance(score, float)
        assert score > 0.95

    def test_earth_regressor_get_params(self):
        """Test get_params returns hyperparameters."""
        from pymars import EarthRegressor

        model = EarthRegressor(max_terms=15, max_degree=2, penalty=3.0)
        params = model.get_params()

        assert params['max_terms'] == 15
        assert params['max_degree'] == 2
        assert params['penalty'] == 3.0

    def test_earth_regressor_set_params(self):
        """Test set_params updates hyperparameters."""
        from pymars import EarthRegressor

        model = EarthRegressor()
        model.set_params(max_terms=20, penalty=2.0)

        assert model.max_terms == 20
        assert model.penalty == 2.0

    def test_earth_regressor_in_pipeline(self):
        """Test EarthRegressor works in sklearn Pipeline."""
        from pymars import EarthRegressor

        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = np.sin(X[:, 0]) + X[:, 1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mars', EarthRegressor(max_terms=15, max_degree=1)),
        ])

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == (100,)
        assert not np.any(np.isnan(predictions))

    def test_earth_regressor_cross_validation(self):
        """Test EarthRegressor works with cross_val_score."""
        from pymars import EarthRegressor

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = EarthRegressor(max_terms=10, max_degree=1)
        scores = cross_val_score(model, X, y, cv=3)

        assert len(scores) == 3
        assert np.all(np.isfinite(scores))


@pytest.mark.unit
class TestEarthClassifierSklearn:
    """Test EarthClassifier sklearn compatibility."""

    def test_earth_classifier_fit_returns_self(self):
        """Test that fit returns self."""
        from pymars import EarthClassifier

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        model = EarthClassifier(max_terms=10, max_degree=1)
        result = model.fit(X, y)

        assert result is model

    def test_earth_classifier_predict_shape(self):
        """Test predict returns correct shape."""
        from pymars import EarthClassifier

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        model = EarthClassifier(max_terms=10, max_degree=1)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == (50,)
        assert set(predictions).issubset({0, 1})

    def test_earth_classifier_score(self):
        """Test score returns accuracy."""
        from pymars import EarthClassifier

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        model = EarthClassifier(max_terms=10, max_degree=1)
        model.fit(X, y)
        score = model.score(X, y)

        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0.85

    def test_earth_classifier_get_set_params(self):
        """Test get_params and set_params."""
        from pymars import EarthClassifier

        model = EarthClassifier(max_terms=15, max_degree=2)
        params = model.get_params()

        assert params['max_terms'] == 15
        assert params['max_degree'] == 2

        model.set_params(max_terms=25)
        assert model.max_terms == 25


@pytest.mark.unit
class TestGLMEarthSklearn:
    """Test GLMEarth sklearn compatibility."""

    @pytest.mark.skip(reason="GLMEarth API differs from sklearn interface")
    def test_glm_earth_fit_returns_self(self):
        """Test that fit returns self."""
        pass

    @pytest.mark.skip(reason="GLMEarth API differs from sklearn interface")
    def test_glm_earth_predict_shape(self):
        """Test predict returns correct shape."""
        pass

    @pytest.mark.skip(reason="GLMEarth API differs from sklearn interface")
    def test_glm_earth_get_set_params(self):
        """Test get_params and set_params."""
        pass
