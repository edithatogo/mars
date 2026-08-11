"""GLM, cross-validation, and plotting tests."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pymars._basis import HingeBasisFunction, InteractionBasisFunction
from pymars._sklearn_compat import EarthRegressor
from pymars.cv import EarthCV
from pymars.glm import GLMEarth
from pymars.plot import plot_basis_functions, plot_residuals


def test_interaction_basis_function():
    """Check interaction terms multiply their parent basis functions."""
    X = np.array([[2.0, 3.0], [0.0, 5.0]])
    mask = np.zeros_like(X, dtype=bool)
    bf1 = HingeBasisFunction(0, 1.0)
    bf2 = HingeBasisFunction(1, 2.0, is_right_hinge=False)
    inter = InteractionBasisFunction(bf1, bf2)
    expected = bf1.transform(X, mask) * bf2.transform(X, mask)
    assert np.allclose(inter.transform(X, mask), expected)
    assert inter.degree() == bf1.degree() + bf2.degree()


def test_glmearth_logistic():
    """Check GLMEarth fits and predicts a binary target."""
    rng = np.random.RandomState(0)
    X = rng.rand(50, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = GLMEarth(family="logistic", max_terms=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_glmearth_poisson():
    """Check GLMEarth fits and predicts a count target."""
    rng = np.random.RandomState(0)
    X = rng.rand(50, 2)
    y = rng.poisson(2 + X[:, 0])
    model = GLMEarth(family="poisson", max_terms=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.min() >= 0


def test_earthcv_and_plotting():
    """Check cross-validation scoring and plotting helpers."""
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = rng.rand(30)
    cv = EarthCV(EarthRegressor(max_terms=2), cv=3)
    scores = cv.score(X, y)
    assert len(scores) == 3
    model = GLMEarth(family="poisson", max_terms=2)
    model.fit(X, y + 1)
    plot_basis_functions(model, X)
    plot_residuals(model, X, y + 1)


def test_earthcv_rejects_single_fold() -> None:
    """Cross-validation should reject fewer than two folds."""
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = rng.rand(30)

    with pytest.raises(ValueError):
        EarthCV(cv=1).score(X, y)


def test_earthcv_rejects_unknown_scoring_rule() -> None:
    """Cross-validation should reject unknown scorer names."""
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = rng.rand(30)

    with pytest.raises(ValueError, match="unknown_scoring_rule"):
        EarthCV(cv=3, scoring="unknown_scoring_rule").score(X, y)


def test_earthcv_rejects_mismatched_sample_counts() -> None:
    """Cross-validation should reject mismatched feature and target rows."""
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = rng.rand(20)

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        EarthCV(cv=3).score(X, y)
