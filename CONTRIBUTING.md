# Contributing to pymars

Thank you for your interest in improving **pymars**! The project is a pure Python adaptation of the `py-earth` library and must remain fully compatible with scikit-learn. This document summarizes key guidelines from `AGENTS.md`, how to run tests, and the recommended workflow for feature development or bug fixes.

## Coding Standards

- **Pure Python only.** No C or Cython extensions are accepted.
- Focus on correctness and clarity first; optimize Python code only when needed.
- **Scikit-learn compatibility** is mandatory. Estimators should:
  - inherit from `sklearn.base.BaseEstimator` (and `RegressorMixin` or `ClassifierMixin` as appropriate),
  - implement `fit`, `predict`, `score`, `get_params`, and `set_params`,
  - validate inputs with `sklearn.utils.validation.check_X_y` and `check_array`,
  - return `self` from `fit`,
  - store learned attributes with trailing underscores (e.g., `coef_`).
- Keep the class structure close to `py-earth`. The main model class is `Earth` in `pymars/earth.py` and should be usable as:
  ```python
  import pymars as earth
  model = earth.Earth()
  ```
- Follow **PEP 8** and use type hints. Document modules, classes, functions, and methods with clear docstrings explaining *why* things are done.
- Use a linter such as **flake8** or **pylint** to keep style consistent.
- Write comments that focus on reasoning and motivation rather than line-by-line explanations.
- Aim for minimal dependencies (typically NumPy and SciPy). List them in `requirements.txt` and `pyproject.toml`.

## Running Tests

- Tests reside in the `tests/` directory and use `pytest`.
- **Test-Driven Development** is encouraged. Provide tests for new features, edge cases, and error handling. Include scikit-learn compatibility checks (for example with `sklearn.utils.estimator_checks.check_estimator`).
- Install dependencies and run the test suite with:
  ```bash
  pip install -r requirements.txt && pytest
  ```

## Workflow for New Features or Bug Fixes

1. Review `ROADMAP.md` and `TODO.md` to plan your work and update them when tasks change.
2. Create a feature branch from `main`.
3. Make small, atomic commits with clear messages (e.g., `feat: add hinge basis function`, `fix: resolve pruning error`).
4. Keep code and tests in the same branch and ensure `pytest` passes locally.
5. Update documentation (`ROADMAP.md`, `TODO.md`, `SESSION_LOGS.md`, or user docs) as needed.
6. Open a pull request for review when the feature or fix is complete and tested.

Following these guidelines helps keep the project consistent and maintainable. Happy coding!
