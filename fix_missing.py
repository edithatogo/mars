import numpy as np

def _is_missing(val):
    return val is None or (isinstance(val, float) and np.isnan(val))

def _get_nan_mask(X: np.ndarray) -> np.ndarray:
    if not np.issubdtype(X.dtype, np.number):
        return np.vectorize(_is_missing)(X)
    return np.isnan(X)

X = np.array([['a', 'b'], ['c', None], [1.0, np.nan]], dtype=object)
mask = _get_nan_mask(X)
print(mask)

col = X[:, 0]
nan_mask_col = mask[:, 0]
col_no_missing = col[~nan_mask_col]
print(col_no_missing)
