"""Basic classification demonstration using :class:`EarthClassifier`."""

import logging

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pymars._sklearn_compat import EarthClassifier

logger = logging.getLogger(__name__)


def main() -> None:
    """Run a simple classification demo with synthetic data."""
    logging.basicConfig(level=logging.INFO)
    logger.info("--- pymars Basic Classification Demo ---")

    np.random.seed(123)
    n_samples = 200
    X = np.random.rand(n_samples, 5)
    latent_y = (
        2 * X[:, 0]
        + np.sin(3 * X[:, 1])
        - 1.5 * X[:, 2] ** 2
        + np.random.normal(0, 1.0, n_samples)
    )
    y = (latent_y > np.median(latent_y)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123, stratify=y
    )
    logger.info("Generated data: X shape %s, y shape %s", X.shape, y.shape)

    model = EarthClassifier(max_degree=1, penalty=3.0, max_terms=15)
    model.fit(X_train, y_train)
    logger.info("Model fitting complete.")

    if hasattr(model, "earth_") and hasattr(model.earth_, "summary"):
        model.earth_.summary()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logger.info("Evaluation on Test Set:")
    logger.info("  Accuracy: %.4f", acc)
    logger.info("\nClassification Report:\n%s", report)


if __name__ == "__main__":
    main()
