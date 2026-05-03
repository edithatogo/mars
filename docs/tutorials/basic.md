# Basic Tutorial

Use `Earth` for regression-style mars fitting:

```python
import numpy as np
import pymars as earth

X = np.random.rand(100, 3)
y = np.sin(X[:, 0]) + X[:, 1]

model = earth.Earth(max_degree=1, penalty=3.0)
model.fit(X, y)
predictions = model.predict(X)
```
