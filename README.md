# pymars: Pure Python Earth (Multivariate Adaptive Regression Splines)

**pymars** is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), inspired by the popular `py-earth` library by Jason Friedman and an R package `earth` by Stephen Milborrow. The goal of `pymars` is to provide an easy-to-install, scikit-learn compatible version of the MARS algorithm without C/Cython dependencies.

## Key Features

- **Pure Python:** Easy to install and use across different platforms.
- **Scikit-learn Compatible:** Integrates with the scikit-learn ecosystem (estimators, pipelines, model selection tools).
- **MARS Algorithm:** Implements the core MARS fitting procedure, including:
  - Forward pass to select basis functions (both hinge and linear terms).
  - Pruning pass using Generalized Cross-Validation (GCV) to prevent overfitting.
  - Support for interaction terms (including interactions involving linear terms).
  - Refined `minspan` and `endspan` controls for knot placement, aligning more closely with `py-earth` behavior (e.g., `minspan` as a cooldown period).
- **Feature Importance:** Calculation of feature importances using methods like 'nb_subsets' (number of subsets in pruning trace), 'gcv' (GCV improvement), and 'rss' (RSS reduction).
- **Regression and Classification:** Provides `EarthRegressor` and `EarthClassifier` classes.
- **Generalized Linear Models:** The `GLMEarth` subclass fits logistic or Poisson models.
- **Cross‑Validation Helper:** The `EarthCV` class integrates with scikit‑learn's model selection utilities.
- **Plotting Utilities:** Simple diagnostics built on `matplotlib`.

## Project Status

This project is currently in the initial development phase. The core algorithm and scikit-learn compatibility are being built. See `ROADMAP.md` and `TODO.md` for more details on the development plan and progress.

## Installation

`pymars` can be installed from TestPyPI:

```bash
pip install pymars
```

To work with the latest source, clone the repository and install it in editable mode:

```bash
git clone https://your-repository-url/pymars.git
cd pymars
pip install -e .
```

After installation you can check the installed version:

```bash
pymars --version
```
```

## Running Tests

Install the dependencies listed in `requirements.txt` before running the test
suite. A small helper script is provided:

```bash
# Option 1: directly with pip
pip install -r requirements.txt

# Option 2: using the helper script
bash scripts/setup_tests.sh
```

After the dependencies are installed, run the tests with:

```bash
pytest

```

## Basic Usage Example

```python
import numpy as np
import pymars as earth # Target import style
# from pymars._sklearn_compat import EarthRegressor, EarthClassifier # Or direct import

# --- Regression Example ---
# Generate sample data
# X_reg = np.random.rand(100, 3)
# y_reg = 2 * X_reg[:, 0] + np.sin(np.pi * X_reg[:, 1]) - X_reg[:, 2]**2 + np.random.randn(100) * 0.1

# Initialize and fit the Earth Regressor
# model_reg = earth.Earth(max_degree=1) # Using the alias
# OR
# model_reg = EarthRegressor(max_degree=1, penalty=3.0) # Assuming EarthRegressor is imported
# model_reg.fit(X_reg, y_reg)

# Make predictions
# y_pred_reg = model_reg.predict(X_reg)

# Print model summary (if available)
# print(model_reg.summary())

# Access feature importances (if calculated)
# if hasattr(model_reg, 'feature_importances_') and model_reg.feature_importances_ is not None:
#     print("Feature Importances:", model_reg.feature_importances_)
#     # Or use the summary method:
#     # print(model_reg.summary_feature_importances())


# --- Classification Example (Conceptual) ---
# y_clf = (y_reg > np.median(y_reg)).astype(int)

# model_clf = EarthClassifier(max_degree=1) # Assuming EarthClassifier is imported
# model_clf.fit(X_reg, y_clf)
# y_pred_clf = model_clf.predict(X_reg)
# y_proba_clf = model_clf.predict_proba(X_reg)

# print(f"Classification accuracy: {model_clf.score(X_reg, y_clf)}")

# --- GLM Example ---
glm_model = GLMEarth(family='logistic', max_terms=3)
glm_model.fit(X_reg, y_clf)
glm_pred = glm_model.predict(X_reg)

# --- Cross-validation Example ---
cv = EarthCV(EarthRegressor(max_terms=2), cv=5)
scores = cv.score(X_reg, y_reg)
print("CV scores:", scores)

# --- Plotting Diagnostics ---
from pymars.plot import plot_basis_functions, plot_residuals
import matplotlib.pyplot as plt
plot_basis_functions(glm_model, X_reg)
plot_residuals(glm_model, X_reg, y_reg)
plt.show()
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` (to be created) and `AGENTS.md` for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Based on the work of Jerome H. Friedman on MARS.
- Inspired by the `py-earth` library (scikit-learn-contrib).
- Inspired by the R `earth` package.
