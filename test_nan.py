import numpy as np
X = np.array(['a', 'b', 'c'])
try:
    print(np.isnan(X))
except Exception as e:
    print("Exception:", e)
