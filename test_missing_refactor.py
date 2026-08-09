import numpy as np

# Test code just to ensure I can import pymars._missing
import pymars._missing as _missing

X = np.array([[1.0, np.nan], [2.0, 3.0], [1.0, 3.0]])
print(_missing.handle_missing_X(X, strategy='mean'))
print(_missing.handle_missing_X(X, strategy='most_frequent'))
