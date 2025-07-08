# pymars: Pure Python Earth (Multivariate Adaptive Regression Splines)

**pymars** is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), inspired by the popular `py-earth` library by Jason Friedman and an R package `earth` by Stephen Milborrow. The goal of `pymars` is to provide an easy-to-install, scikit-learn compatible version of the MARS algorithm without C/Cython dependencies.

## Key Features (Planned)

*   **Pure Python:** Easy to install and use across different platforms.
*   **Scikit-learn Compatible:** Integrates with the scikit-learn ecosystem (estimators, pipelines, model selection tools).
*   **MARS Algorithm:** Implements the core MARS fitting procedure, including:
    *   Forward pass to select basis functions (hinge functions).
    *   Pruning pass using Generalized Cross-Validation (GCV) to prevent overfitting.
    *   Support for interaction terms.
*   **Regression and Classification:** Provides `EarthRegressor` and `EarthClassifier` classes.

## Project Status

This project is currently in the initial development phase. The core algorithm and scikit-learn compatibility are being built. See `ROADMAP.md` and `TODO.md` for more details on the development plan and progress.

## Installation (Planned)

Once released, `pymars` will be installable via pip:

```bash
pip install pymars
```

For now, to use the development version, you can clone this repository:

```bash
git clone https://your-repository-url/pymars.git
cd pymars
# Potentially: pip install -e . (once setup.py or pyproject.toml is ready)
```

## Basic Usage (Planned Example)

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


# --- Classification Example (Conceptual) ---
# y_clf = (y_reg > np.median(y_reg)).astype(int)

# model_clf = EarthClassifier(max_degree=1) # Assuming EarthClassifier is imported
# model_clf.fit(X_reg, y_clf)
# y_pred_clf = model_clf.predict(X_reg)
# y_proba_clf = model_clf.predict_proba(X_reg)

# print(f"Classification accuracy: {model_clf.score(X_reg, y_clf)}")
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` (to be created) and `AGENTS.md` for guidelines.

## License

This project is licensed under the [MIT License](LICENSE) (to be created).

## Acknowledgements

*   Based on the work of Jerome H. Friedman on MARS.
*   Inspired by the `py-earth` library (scikit-learn-contrib).
*   Inspired by the R `earth` package.
```
