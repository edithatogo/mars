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

**Note:** Install `pymars` inside a Python virtual environment to silence `pip` warnings about running as root.

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

If you need to run scikit-learn's full estimator checks, install the optional
pandas dependency:

```bash
pip install "pymars[pandas]"
```

After installation you can check the installed version:

```bash
pymars --version
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

## Usage

### Quick demos

Run the included demo scripts to see `pymars` in action:

```bash
python -m pymars.demos.basic_regression_demo
python -m pymars.demos.basic_classification_demo
```

### Basic API

```python
import numpy as np
import pymars as earth

X = np.random.rand(100, 3)
y = np.sin(X[:, 0]) + X[:, 1]

model = earth.Earth(max_degree=1, penalty=3.0)
model.fit(X, y)
predictions = model.predict(X)
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` and `AGENTS.md` for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Based on the work of Jerome H. Friedman on MARS.
- Inspired by the `py-earth` library (scikit-learn-contrib).
- Inspired by the R `earth` package.
