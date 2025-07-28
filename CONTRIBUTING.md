# Contributing to pymars

Thank you for your interest in contributing to **pymars**! This project aims to provide a pure Python version of `py-earth` that works seamlessly with scikit-learn. The following guidelines summarize the key points from `AGENTS.md` and describe how to run tests and submit contributions.

## Coding Standards

- **Pure Python only.** No C or Cython extensions are accepted for current contributions. While Cython may be considered for future performance optimizations, the present focus is on a pure Python implementation.
- **Scikit-learn compatibility** is mandatory: estimators must inherit from `sklearn.base.BaseEstimator` and implement the usual scikit-learn API (`fit`, `predict`, `score`, `get_params`, `set_params`).
- Keep the class structure similar to `py-earth`. The main model class is `Earth` in `pymars/earth.py` and should be usable via:
  ```python
  import pymars as earth
  model = earth.Earth()
  ```
- Follow **PEP 8** and use type hints. Document all public classes, functions, and methods with clear docstrings. Comments should explain *why* something is done.
- Aim for minimal dependencies (NumPy/scikit-learn). List them in `requirements.txt` and `pyproject.toml`.

## Running Tests

- Tests live in the `tests/` directory and use `pytest`.
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  # or
  bash scripts/setup_tests.sh
  ```
- Run the entire suite with:
  ```bash
  pytest
  ```
- New features or bug fixes must include relevant tests covering expected behaviour, edge cases, and scikit-learn compatibility.

## Workflow for New Features or Bug Fixes

1. **Plan** your work using `ROADMAP.md` and `TODO.md`. Update them if you add or change tasks.
2. Create a **feature branch** from `main`.
3. Make **small, atomic commits** with descriptive messages (e.g., `feat: add hinge basis function` or `fix: handle NaN in forward pass`).
4. Keep code and tests together in your branch. Ensure `pytest` passes before opening a pull request.
5. Submit a **pull request** for review. Include relevant documentation updates (e.g., `ROADMAP.md`, `TODO.md`, or the user guide when applicable).

Following these guidelines will help maintain a consistent codebase and make collaboration smooth. Happy coding!
