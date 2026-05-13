# Python Quickstart

The canonical Python workflow is documented in the main tutorials and mirrored
in the runnable notebook.

## Core Workflow

```python
import numpy as np
import pymars as earth

X = np.array([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4], [0.4, 0.1, 0.3]])
y = np.array([1.0, 1.8, 2.4])

model = earth.Earth(max_terms=6, max_degree=1)
model.fit(X, y)

predictions = model.predict(X)
design = earth.design_matrix(model.export_model(), X)
```

## Validation

- Use [`tests/test_tutorial_smoke.py`](/Users/doughnut/GitHub/pymars/tests/test_tutorial_smoke.py)
  for tutorial-level smoke coverage.
- Use [`tests/test_accelerator_validation.py`](/Users/doughnut/GitHub/pymars/tests/test_accelerator_validation.py)
  for H3 contract validation scaffolding.
