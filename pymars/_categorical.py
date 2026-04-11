"""Utilities for handling categorical features."""

from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder


class CategoricalImputer:
    """Simple imputer and label encoder for categorical features."""

    def __init__(self):
        self.encoders = {}
        self.most_frequent_ = {}

    def fit(self, X, categorical_features):
        X_arr = np.asarray(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        for idx in categorical_features:
            col = X_arr[:, idx]
            mask = np.array([self._is_missing(v) for v in col])
            values = [v for v in col if not self._is_missing(v)]
            if values:
                counts = Counter(values)
                most_freq = counts.most_common(1)[0][0]
            else:
                most_freq = None
            self.most_frequent_[idx] = most_freq
            le = LabelEncoder()
            if values:
                le.fit(values)
            else:
                le.fit([most_freq])
            self.encoders[idx] = le
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=object).copy()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        for idx, le in self.encoders.items():
            col = X_arr[:, idx]
            new_col = []
            for val in col:
                if self._is_missing(val):
                    val = self.most_frequent_[idx]
                try:
                    enc = le.transform([val])[0]
                except ValueError:
                    enc = le.transform([self.most_frequent_[idx]])[0]
                new_col.append(float(enc))
            X_arr[:, idx] = np.array(new_col, dtype=float)
        return np.asarray(X_arr, dtype=float)

    def fit_transform(self, X, categorical_features):
        return self.fit(X, categorical_features).transform(X)

    @staticmethod
    def _is_missing(value):
        return value is None or (isinstance(value, float) and np.isnan(value))
