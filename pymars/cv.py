from sklearn.model_selection import cross_val_score

class EarthCV:
    """Helper for cross-validation with Earth estimators."""

    def __init__(self, estimator, cv=5, scoring=None):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring

    def score(self, X, y):
        return cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
