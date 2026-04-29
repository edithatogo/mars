from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sklearn.model_selection import cross_val_score

from .earth import Earth

if TYPE_CHECKING:
    import numpy as np


class EarthCV:
    """Helper for cross-validation with Earth estimators."""

    def __init__(
        self, estimator: Earth | None = None, cv: int = 5, scoring: str | None = None
    ) -> None:
        self.estimator = estimator or Earth()
        self.cv = cv
        self.scoring = scoring

    def score(self, X: Any, y: Any) -> np.ndarray:
        """Return cross-validated scores for the configured Earth model."""
        return cast(
            "np.ndarray",
            cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring),
        )
