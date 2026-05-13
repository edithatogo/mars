# Usage

The supported compatibility style remains:

```python
import pymars as earth

model = earth.Earth(max_degree=1, penalty=3.0)
```

Stable estimators are `Earth`, `EarthRegressor`, and `EarthClassifier`.
`EarthCV` and `GLMEarth` remain experimental.

The tutorial landing page is [Tutorials](tutorials/index.md), which walks
through fit, predict, spec export, validation, and interpretability examples.
The runnable example hub is [Examples](examples/index.md), which collects the
canonical notebook and binding quickstarts.

Binding entry points are summarized in [Bindings](bindings.md).

See [Bindings](bindings.md) for the current cross-language release surface.
