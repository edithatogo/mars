# Usage

The supported compatibility style remains:

```python
import pymars as earth

model = earth.Earth(max_degree=1, penalty=3.0)
```

Stable estimators are `Earth`, `EarthRegressor`, and `EarthClassifier`.
`EarthCV` and `GLMEarth` remain experimental.

See [Bindings](bindings.md) for the current cross-language release surface.
