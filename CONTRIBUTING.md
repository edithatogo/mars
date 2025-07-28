# Contributing to pymars

Thank you for your interest in improving **pymars**! This repository aims to be a pure Python version of `py-earth` that integrates smoothly with scikit-learn. The following guidelines summarize the most important points from `AGENTS.md` and describe how to run tests and contribute new features or bug fixes.

## Coding Standards

- **Pure Python only.** No C or Cython extensions are used.
- **Scikit-learn compatibility** is essential: estimators must inherit from `sklearn.base.BaseEstimator`, implement the standard API (`fit`, `predict`, `score`, `get_params`, `set_params`), and rely on `check_X_y` / `check_array` for validation.
- Keep the class structure similar to `py-earth`. The main model class is `Earth` in `pymars/earth.py` and should be usable via:
  ```python
  import pymars as earth
  model = earth.Earth()
  ```
- Follow **PEP 8**, use type hints, and provide clear docstrings. Comments should explain *why* something is done.
- **Test-Driven Development** is encouraged. Tests live in `tests/` and should cover normal behaviour, edge cases, error handling, and scikit-learn compatibility.
- Minimize dependencies (NumPy, SciPy, scikit-learn) and list them in `requirements.txt` / `pyproject.toml`.

## Running Tests

Install the dependencies and run the test suite with:
```bash
pip install -r requirements.txt && pytest
```
All new code must pass the tests before submission, and new features or fixes should include appropriate tests.

## Workflow for New Features or Bug Fixes

1. Review `ROADMAP.md` and `TODO.md` to plan your work. Update them if you add or change tasks.
2. Create a feature branch from `main`.
3. Make small, atomic commits with clear messages (e.g., `feat: add hinge basis function` or `fix: handle NaN in forward pass`).
4. Keep code and tests together. Ensure `pip install -r requirements.txt && pytest` runs successfully before opening a pull request.
5. Submit a pull request for review and include any relevant documentation updates.

Following these steps keeps the codebase consistent and the development process smooth. Happy coding!
