import numpy as np
X = np.array([[1, 2], [3, 4]], dtype=object)
X[0, 0] = np.nan
X_processed = np.copy(X)
col = X_processed[:, 0]
nan_mask_col = np.isnan(col.astype(float))
fill_value = 0.0
col = col.astype(float, copy=False)
col[nan_mask_col] = fill_value
print(X_processed)
