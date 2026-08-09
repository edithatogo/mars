import numpy as np

def _is_missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value))

def check_missing(X):
    # Vectorized check for strings/objects
    if not np.issubdtype(X.dtype, np.number):
        mask = np.vectorize(_is_missing)(X)
        return mask.any(), mask
    mask = np.isnan(X)
    return mask.any(), mask

X = np.array([['a', 'b'], ['c', None], [1.0, np.nan]], dtype=object)
print(check_missing(X))
