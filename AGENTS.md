# Guidelines for AI Agents Working on `pymars`

Welcome, fellow AI! This document provides guidelines for contributing to the `pymars` library. Your adherence to these guidelines will help ensure the project's success, maintainability, and consistency.

## 1. Core Objective

The primary goal is to create a **pure Python** adaptation of the `py-earth` library, ensuring it is **fully compatible with scikit-learn** while retaining the **original class structure and import conventions** (e.g., `import pymars as earth` followed by `model = earth.Earth()`).

## 2. Scikit-learn Compatibility

This is a critical requirement. All estimators developed must:

*   Inherit from `sklearn.base.BaseEstimator`.
*   Implement `sklearn.base.RegressorMixin` for regression tasks and `sklearn.base.ClassifierMixin` for classification tasks.
*   Have a `fit(X, y, **kwargs)` method.
*   Have a `predict(X)` method.
*   Have a `score(X, y)` method (or rely on the mixin's default).
*   Implement `get_params(deep=True)` and `set_params(**params)`.
*   Use `sklearn.utils.validation.check_X_y` for input validation in `fit`.
*   Use `sklearn.utils.validation.check_array` for input validation in `predict`, `score`, etc.
*   Ensure that `fit` returns `self`.
*   Ensure that hyperparameters passed to the constructor are stored as public attributes with the same names.
*   Avoid storing data or state that is not a hyperparameter directly on the instance if it's learned during `fit`. These should have a trailing underscore (e.g., `self.coef_`).

Refer to the [scikit-learn developer documentation](https://scikit-learn.org/stable/developers/develop.html) for detailed guidelines.

## 3. `py-earth` Class Structure and API

*   The main model class should be named `Earth` and reside in `pymars/earth.py`.
*   It should be possible to import and use the library as follows:
    ```python
    import pymars as earth
    model = earth.Earth(max_degree=1, penalty=3.0)
    # ... further usage ...
    ```
*   Strive to keep method names and parameters consistent with `py-earth` where it doesn't conflict with scikit-learn compatibility or pure Python implementation constraints. When conflicts arise, scikit-learn compatibility takes precedence.

## 4. Pure Python Implementation

*   All code must be written in Python. **No Cython or C extensions.**
*   This means that some performance optimizations present in `py-earth` might need to be re-thought or might result in slower execution. Focus on correctness and clarity first, then optimize Python code if necessary.
*   Standard Python libraries and NumPy/SciPy are acceptable dependencies.

## 5. Coding Style and Conventions

*   Follow **PEP 8** for code style. Use a linter like Flake8 or Pylint if possible.
*   Use type hints (Python 3.6+).
*   Write clear and concise docstrings for all modules, classes, functions, and methods (Google style or NumPy style).
*   Comments should explain *why* something is done, not *what* is being done (if the code is self-explanatory).

## 6. Testing

*   **Test-Driven Development (TDD)** is encouraged. Write tests before or alongside your code.
*   Use the `pytest` framework for testing.
*   Tests should reside in the `tests/` directory.
*   Aim for high test coverage. Every new feature or bug fix should be accompanied by tests.
*   Include tests for:
    *   Correct output values.
    *   Edge cases.
    *   Error handling and expected exceptions.
    *   Scikit-learn compatibility (e.g., using `check_estimator` from `sklearn.utils.estimator_checks`).

## 7. Documentation

*   Maintain and update `ROADMAP.md` and `TODO.md` as development progresses.
*   Keep `SESSION_LOGS.md` for significant changes, decisions, or complex tool outputs.
*   User-facing documentation (User Guide, API reference) will be developed as per the `ROADMAP.md`.

## 8. Version Control (Git)

*   Make small, atomic commits with clear and descriptive messages.
*   Follow conventional commit message formats if possible (e.g., `feat: add hinge basis function`, `fix: resolve pruning error`).
*   Work on feature branches and create pull requests for review (if applicable in the development setup).

## 9. Dependencies

*   Minimize external dependencies. NumPy and SciPy are expected.
*   List all dependencies in `requirements.txt` and/or `setup.py` (or `pyproject.toml`).

## 10. Communication and Planning

*   Before starting a complex task, refer to `ROADMAP.md` and `TODO.md`.
*   If you need to make significant changes to the plan or encounter major roadblocks, update the plan using the `set_plan` tool and communicate this (e.g., via `message_user`).
*   When you complete a plan step, use `plan_step_complete()`.

## 11. Specific Tool Usage

*   **`create_file_with_block` / `overwrite_file_with_block`**: Use for new files or complete rewrites.
*   **`run_in_bash_session`**: Use for installing dependencies, running tests, etc.
*   **`set_plan`**: Use to set or update the development plan.
*   **`submit`**: Use when a feature is complete, tested, and ready for integration.

By following these guidelines, we can build a robust and user-friendly `pymars` library. Happy coding!
