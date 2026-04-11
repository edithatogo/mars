import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from ._sklearn_compat import EarthRegressor


class GLMEarth(EarthRegressor):
    """Earth model that fits GLM coefficients."""

    def __init__(self, family: str = "logistic", **kwargs):
        super().__init__(**kwargs)
        self.family = family

    def fit(self, X, y):
        super().fit(X, y)
        X_proc, mask, _ = self.earth_._scrub_input_data(X, y)
        B = self.earth_._build_basis_matrix(X_proc, self.basis_, mask)
        if self.family == "logistic":
            self.glm_ = LogisticRegression(max_iter=1000).fit(B, y)
        elif self.family == "poisson":
            self.glm_ = PoissonRegressor().fit(B, y)
        else:
            raise ValueError("family must be 'logistic' or 'poisson'")
        return self

    def predict(self, X):
        X_proc, mask, _ = self.earth_._scrub_input_data(X, np.zeros(len(X)))
        B = self.earth_._build_basis_matrix(X_proc, self.basis_, mask)
        pred = self.glm_.predict(B)
        if self.family == "logistic":
            return (pred > 0.5).astype(int)
        return pred
