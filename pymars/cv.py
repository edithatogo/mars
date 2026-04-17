from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.model_selection import cross_val_score

from .earth import Earth


class EarthCV:
    """Helper for cross-validation with Earth estimators."""

    def __init__(
        self, estimator: Earth | None = None, cv: int = 5, scoring: str | None = None
    ) -> None:
        self.estimator = estimator or Earth()
        self.cv = cv
        self.scoring = scoring

    def score(self, X: Any, y: Any) -> np.ndarray:
        return cast(
            np.ndarray,
            cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring),
        )
