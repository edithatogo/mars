import numpy as np

from pymars import CategoricalImputer, Earth
from pymars._basis import CategoricalBasisFunction


def test_categorical_feature_handling():
    """Test that the Earth model can handle categorical features."""
    # Create a synthetic dataset with a categorical feature
    np.random.seed(0)
    X = np.random.rand(100, 3)
    # Create a categorical feature with 3 categories
    X[:, 1] = np.random.randint(0, 3, 100)
    y = X[:, 0] + (X[:, 1] == 0) * 5 + (X[:, 1] == 1) * 10 + (X[:, 1] == 2) * 15 + X[:, 2]**2

    # Fit an Earth model with the categorical feature specified
    model = Earth(max_degree=1, categorical_features=[1])
    model.fit(X, y)

    # Check that the model has selected at least one categorical basis function
    assert any(isinstance(bf, CategoricalBasisFunction) for bf in model.basis_)

    # Check that the predictions are reasonable
    y_pred = model.predict(X)
    assert np.mean((y - y_pred)**2) < 1.0

def test_categorical_interaction():
    """Test that the Earth model can handle interactions with categorical features."""
    # Create a synthetic dataset with an interaction between a continuous and a categorical feature
    np.random.seed(0)
    X = np.random.rand(100, 3)
    # Create a categorical feature with 2 categories
    X[:, 1] = np.random.randint(0, 2, 100)
    y = X[:, 0] * (X[:, 1] == 0) * 5 + X[:, 0] * (X[:, 1] == 1) * 10

    # Fit an Earth model with the categorical feature specified
    model = Earth(max_degree=2, categorical_features=[1])
    model.fit(X, y)

    # Check that the model has selected at least one categorical basis function with a parent
    assert any(isinstance(bf, CategoricalBasisFunction) and bf.parent1 is not None for bf in model.basis_)

    # Check that the predictions are reasonable
    y_pred = model.predict(X)
    assert np.mean((y - y_pred)**2) < 1.0


def test_categorical_imputer():
    X = np.array([
        ['a', 'b'],
        ['b', None],
        ['b', 'b'],
        ['a', 'a'],
    ], dtype=object)
    imputer = CategoricalImputer()
    X_trans = imputer.fit_transform(X, [0, 1])
    assert X_trans.shape == (4, 2)
    assert not np.isnan(X_trans).any()


def test_earth_categorical_with_strings_and_missing():
    X = np.array([
        [1.0, 'a'],
        [2.0, None],
        [3.0, 'b'],
        [4.0, 'a'],
    ], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = Earth(max_degree=1, categorical_features=[1], allow_missing=True)
    model.fit(X, y)
    assert model.fitted_
    preds = model.predict(X)
    assert preds.shape == y.shape
