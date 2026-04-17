from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from ._sklearn_compat import EarthRegressor


class GLMEarth(EarthRegressor):
    """Earth model that fits GLM coefficients."""

    def __init__(self, family: str = "logistic", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.family = family

    def fit(self, X: Any, y: Any) -> GLMEarth:
        super().fit(X, y)
        X_proc, mask, _ = self.earth_._scrub_input_data(X, y)
        basis = self.basis_
        if basis is None:
            raise ValueError("GLMEarth.fit requires a fitted Earth basis.")
        B = self.earth_._build_basis_matrix(X_proc, basis, mask)
        if self.family == "logistic":
            self.glm_ = LogisticRegression(max_iter=1000).fit(B, y)
        elif self.family == "poisson":
            self.glm_ = PoissonRegressor().fit(B, y)
        else:
            raise ValueError("family must be 'logistic' or 'poisson'")
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_proc, mask, _ = self.earth_._scrub_input_data(X, np.zeros(len(X)))
        basis = self.basis_
        if basis is None:
            raise ValueError("GLMEarth.predict requires a fitted Earth basis.")
        B = self.earth_._build_basis_matrix(X_proc, basis, mask)
        pred = self.glm_.predict(B)
        if self.family == "logistic":
            return cast(np.ndarray, (pred > 0.5).astype(int))
        return cast(np.ndarray, pred)
