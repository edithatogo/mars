# Basic Tutorial

Use `Earth` for regression-style mars fitting, prediction, and model
inspection.

## Fit and predict

```python
import numpy as np
import pymars as earth

X = np.array(
    [
        [-1.0, 0.0, 0.5],
        [-0.2, 0.3, 0.0],
        [0.1, -0.4, 0.2],
        [0.5, 0.2, -0.3],
        [1.0, -0.7, 0.8],
        [1.4, -1.1, 1.1],
    ],
    dtype=float,
)
y = np.array([0.8, 1.3, 2.0, 2.6, 4.9, 6.2], dtype=float)

model = earth.Earth(max_degree=1, penalty=3.0)
model.fit(X, y)
predictions = model.predict(X)
score = model.score(X, y)
```

## Save and reload the portable spec

```python
from pathlib import Path

spec_path = Path("model.json")
earth.save_model(model, spec_path)
validated_spec = earth.validate(spec_path)
restored_model = earth.load_model(validated_spec)
restored_predictions = restored_model.predict(X)
```

## Inspect the fitted model

```python
summary = earth.inspect(spec_path)
design_matrix = earth.design_matrix(validated_spec, X)
```

The basic tutorial is meant to show the shortest path from import to a fitted
and replayable model.
