"""Basic regression demonstration using :class:`EarthRegressor`."""

import logging

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pymars._sklearn_compat import EarthRegressor

logger = logging.getLogger(__name__)


def main() -> None:
    """Run a simple regression demo with synthetic data."""
    logging.basicConfig(level=logging.INFO)
    logger.info("--- pymars Basic Regression Demo ---")

    np.random.seed(42)
    n_samples = 200
    X = np.random.rand(n_samples, 5)
    y = (
        2 * X[:, 0]
        + np.sin(3 * X[:, 1])
        - 1.5 * X[:, 2] ** 2
        + np.random.normal(0, 0.5, n_samples)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    logger.info("Generated data: X shape %s, y shape %s", X.shape, y.shape)

    model = EarthRegressor(max_degree=1, penalty=3.0, max_terms=20)
    model.fit(X_train, y_train)
    logger.info("Model fitting complete.")

    if hasattr(model, "earth_") and hasattr(model.earth_, "summary"):
        model.earth_.summary()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    logger.info("Evaluation on Test Set:")
    logger.info("  Mean Squared Error (MSE): %.4f", mse)
    logger.info("  R-squared (R2): %.4f", r2)


if __name__ == "__main__":
    main()
