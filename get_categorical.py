from pymars._categorical import CategoricalImputer
import numpy as np

X = np.array(
    [
        ["a", "b"],
        ["b", None],
        ["b", "b"],
        ["a", "a"],
    ],
    dtype=object,
)
imputer = CategoricalImputer()
X_trans = imputer.fit_transform(X, [0, 1])
print(X_trans)
