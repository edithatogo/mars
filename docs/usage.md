# Usage

Install the package as `mars-earth`, then import it as `pymars`. The supported
compatibility style remains:

```python
import pymars as earth
```

Stable estimators are `Earth`, `EarthRegressor`, and `EarthClassifier`.
`EarthCV` and `GLMEarth` remain experimental.

## Basic regression example

```python
import numpy as np
import pymars as earth

X = np.random.rand(100, 3)
y = np.sin(X[:, 0]) + X[:, 1]

model = earth.Earth(max_degree=1, penalty=3.0)
model.fit(X, y)
predictions = model.predict(X)
```

## Portable runtime helpers

```python
earth.save_model(model, "model.json")

validated = earth.validate("model.json")
spec = earth.load_model_spec("model.json")
portable_model = earth.load_model("model.json")

features = earth.design_matrix("model.json", X)
predictions = portable_model.predict(X)
runtime_predictions = earth.predict("model.json", X)
```

For the schema contract and runtime compatibility guarantees, see
[Model Spec](model_spec.md).
