"""End-to-end tests for complete mars workflows."""

import pytest
import numpy as np

pytest.importorskip("pymars")


@pytest.mark.e2e
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    def test_regression_workflow(self):
        """Test complete regression model training → prediction → evaluation."""
        from pymars import Earth

        np.random.seed(42)
        X_train = np.random.rand(200, 5)
        y_train = (
            np.sin(X_train[:, 0])
            + X_train[:, 1] ** 2
            + 0.1 * np.random.randn(200)
        )

        X_test = np.random.rand(50, 5)
        y_test = (
            np.sin(X_test[:, 0])
            + X_test[:, 1] ** 2
            + 0.1 * np.random.randn(50)
        )

        model = Earth(max_degree=2, penalty=3.0, minspan_alpha=0.5)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        r2 = model.score(X_test, y_test)

        assert predictions.shape == (50,)
        assert r2 > 0.8
        assert len(model.basis_) > 0

    def test_classification_workflow(self):
        """Test complete classification workflow."""
        from pymars import EarthClassifier

        np.random.seed(42)
        X_train = np.random.rand(150, 4)
        y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)

        X_test = np.random.rand(30, 4)
        y_test = (X_test[:, 0] + X_test[:, 1] > 1).astype(int)

        model = EarthClassifier(max_degree=2, penalty=3.0)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        assert predictions.shape == (30,)
        assert accuracy > 0.85
        assert set(predictions).issubset({0, 1})


@pytest.mark.e2e
class TestDemoScripts:
    """Test demo scripts execute without errors."""

    def test_basic_regression_demo(self):
        """Test basic regression demo runs."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pymars.demos.basic_regression_demo"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Demo failed: {result.stderr}"

    def test_basic_classification_demo(self):
        """Test basic classification demo runs."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pymars.demos.basic_classification_demo"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Demo failed: {result.stderr}"
