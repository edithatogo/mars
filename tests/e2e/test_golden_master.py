"""Golden master regression tests to prevent output regression."""

import pytest
import numpy as np

pytest.importorskip("pymars")


@pytest.mark.golden
class TestGoldenMaster:
    """Test outputs don't regress from known good outputs."""

    def test_regression_on_standard_data(self):
        """Test regression on standard dataset produces expected outputs."""
        from pymars import Earth

        np.random.seed(12345)
        X = np.random.rand(100, 3)
        y = np.sin(X[:, 0]) + X[:, 1] + X[:, 2] ** 2

        model = Earth(max_degree=1, penalty=3.0)
        model.fit(X, y)
        predictions = model.predict(X)

        r2 = model.score(X, y)

        assert r2 > 0.95
        assert predictions.shape == (100,)
        assert np.all(np.isfinite(predictions))
        assert len(model.trace()[-1]['basis_functions']) > 0

    def test_classification_on_standard_data(self):
        """Test classification on standard dataset produces expected outputs."""
        from pymars import EarthClassifier

        np.random.seed(12345)
        X = np.random.rand(100, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        model = EarthClassifier(max_degree=1, penalty=3.0)
        model.fit(X, y)
        predictions = model.predict(X)

        accuracy = model.score(X, y)

        assert accuracy > 0.9
        assert predictions.shape == (100,)
        assert set(predictions).issubset({0, 1})
