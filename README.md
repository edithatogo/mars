# mars: A Pure Python Implementation of Multivariate Adaptive Regression Splines (formerly pymars)

![CI](https://github.com/edithatogo/mars/actions/workflows/ci.yml/badge.svg)
![Security](https://github.com/edithatogo/mars/actions/workflows/security.yml/badge.svg)
![Code Quality](https://github.com/edithatogo/mars/actions/workflows/code-quality.yml/badge.svg)
![Documentation](https://github.com/edithatogo/mars/actions/workflows/docs.yml/badge.svg)
![Code Coverage](https://codecov.io/gh/edithatogo/mars/branch/main/graph/badge.svg)
![PyPI](https://img.shields.io/pypi/v/mars-earth.svg)
![Python Version](https://img.shields.io/pypi/pyversions/mars-earth.svg)
![License](https://img.shields.io/github/license/edithatogo/mars.svg)

**mars** (formerly **pymars**) is a MARS implementation project moving from a
pure Python training implementation toward a shared Rust computational core
surfaced through Python, R, Julia, Rust, C#, Go, and TypeScript APIs. It is
inspired by Jerome H. Friedman's MARS work and by prior open-source APIs such as
`py-earth` and R `earth`, but it does not depend on those packages for
implementation or validation.

## Documentation

Complete documentation is published at: [https://edithatogo.github.io/mars/](https://edithatogo.github.io/mars/)

## Key Features

- **Python API Today:** Easy to install and use with a scikit-learn-compatible Python interface.
- **Rust Core Direction:** Shared fitting and runtime evaluation logic is moving toward a Rust core.
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
- **Advanced Interpretability:** Partial dependence plots, Individual Conditional Expectation (ICE) plots, and model explanation tools.
- **Portable Model Export:** Export fitted `Earth` models to a versioned JSON model spec and replay them without relying on pickle or Python object reconstruction.
- **Cross-Runtime Fixture Coverage:** Checked-in portable fixtures are validated by the Python runtime and a Rust reference runtime.
- **Binding Roadmap:** The target surface spans Python, R, Julia, Rust, C#, Go, and TypeScript.
- **Comprehensive CLI:** Command-line interface for model fitting, prediction, and evaluation.

## Project Status

The project is in active beta. The current Python MARS implementation,
sklearn-compatible estimators, portable `ModelSpec` contract, and Rust reference
runtime fixture validation are in place. Current roadmap work should keep the
Python API stable while moving shared computation toward a Rust core that can
power Python, R, Julia, Rust, C#, Go, and TypeScript bindings.

Near-term priorities are:

- stabilizing the portable JSON model contract
- hardening persistence and numerical-regression testing with checked-in fixtures
- growing the Rust replay prototype into the shared Rust core
- defining shared conformance tests for Python, R, Julia, Rust, C#, Go, and TypeScript bindings
- widening platform bindings and long-term API stability guarantees

## Installation

**Note:** Install `mars` inside a Python virtual environment to silence `pip` warnings about running as root.

Install the published distribution from PyPI:

```bash
pip install mars-earth
```

The distribution name is `mars-earth`, but the import name remains `pymars`.
The supported compatibility style is:

```python
import pymars as earth
```

To work with the latest source, clone the repository and install it in editable mode:

```bash
git clone https://github.com/edithatogo/mars.git
cd mars
pip install -e .
```

If you need DataFrame-aware estimator checks, install the optional pandas dependency:

```bash
pip install "mars-earth[pandas]"
```

After installation you can check the installed version:

```bash
mars --version
```

## Running Tests

Install the dependencies listed in `requirements.txt` before running the test
suite. A small helper script is provided:

```bash
# Option 1: directly with pip
pip install -r requirements.txt

# To run the full scikit-learn estimator checks, install the optional pandas
# dependency as well:
pip install "mars[pandas]"

# Option 2: using the helper script
bash scripts/setup_tests.sh
```

After the dependencies are installed, run the tests with:

```bash
pytest

```

## Usage

### Quick demos

Run the included demo scripts to see `mars` in action:

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

### Portable runtime API

```python
import pymars as earth

model = earth.Earth().fit(X, y)
earth.save_model(model, "model.json")

validated = earth.validate("model.json")
spec = earth.load_model_spec("model.json")
portable = earth.load_model("model.json")
features = earth.design_matrix("model.json", X)
predictions = portable.predict(X)
runtime_predictions = earth.predict("model.json", X)
```

## Public API

Stable estimators:

- `pymars.Earth`
- `pymars.EarthRegressor`
- `pymars.EarthClassifier`

Stable portability/runtime helpers:

- `pymars.validate`
- `pymars.load_model_spec`
- `pymars.load_model`
- `pymars.save_model`
- `pymars.predict`
- `pymars.design_matrix`
- `pymars.inspect`

Stable utility exports:

- `pymars.CategoricalImputer`
- Plotting helpers re-exported from `pymars.visualization`

Experimental APIs:

- `pymars.EarthCV`
- `pymars.GLMEarth`

Undocumented internal imports are not part of the supported public API.

## Documentation

The repository ships source documentation for MkDocs in [`docs/`](docs/) and publishes the built site through GitHub Pages. For usage examples and tutorials, check the [examples directory](examples/) and the [`pymars/demos/`](pymars/demos/) modules.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` and `AGENTS.md` for guidelines.

## Citing mars

If you use mars in your research, please cite it as follows:

```bibtex
@software{mars2026,
  author = {Mordaunt, Dylan A.},
  title = {mars: A Pure Python Implementation of Multivariate Adaptive Regression Splines},
  year = {2026},
  url = {https://github.com/edithatogo/mars},
  version = {1.0.4},
  license = {Apache-2.0},
}
```

A `CITATION.cff` file is also included in the repository for easy citation management.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Acknowledgements

- Based on the work of Jerome H. Friedman on MARS.
- API ideas were informed by `py-earth` and R `earth`, but mars does not depend
  on either package for implementation or validation.
