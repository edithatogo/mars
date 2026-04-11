"""Integration tests for sklearn pipeline compatibility."""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

pytest.importorskip("pymars")


@pytest.mark.integration
class TestSklearnPipelineIntegration:
    """Test mars integration with sklearn pipelines."""

    def test_earth_in_pipeline_with_scaler(self):
        """Test EarthRegressor works in a pipeline with StandardScaler."""
        from pymars import Earth

        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = np.sin(X[:, 0]) + X[:, 1] + 0.1 * np.random.randn(100)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mars', Earth(max_degree=1, penalty=3.0)),
        ])

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == (100,)
        assert not np.any(np.isnan(predictions))

    def test_earth_cross_validation(self):
        """Test EarthRegressor works with sklearn cross_val_score."""
        from pymars import Earth

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = Earth(max_degree=1, penalty=3.0)

        scores = cross_val_score(model, X, y, cv=3)

        assert len(scores) == 3
        assert np.all(np.isfinite(scores))
        assert np.mean(scores) > 0.5

    def test_earth_classifier_in_pipeline(self):
        """Test EarthClassifier works in a pipeline."""
        from pymars import EarthClassifier

        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mars', EarthClassifier(max_degree=1, penalty=3.0)),
        ])

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == (100,)
        assert set(predictions).issubset({0, 1})


@pytest.mark.integration
class TestFileIOIntegration:
    """Test file I/O operations."""

    def test_model_save_load(self, tmp_path):
        """Test model serialization and deserialization."""
        import pickle
        from pymars import Earth

        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]

        model = Earth(max_degree=1, penalty=3.0)
        model.fit(X, y)

        model_path = tmp_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        predictions_original = model.predict(X)
        predictions_loaded = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(
            predictions_original, predictions_loaded
        )
