# Product Guidelines: mars

## 1. Code Style & Conventions
- **PEP 8 Compliance:** All code must follow PEP 8. Use `ruff` for linting and `black` for formatting.
- **Type Hints:** Use type hints for all public APIs. Private functions may omit them for brevity.
- **Docstrings:** All modules, classes, and public functions must have Google-style docstrings.
- **Naming:** Use descriptive, snake_case names for functions/variables. Class names use PascalCase.

## 2. API Design Principles
- **Scikit-learn Compatibility:** All estimators must inherit from `sklearn.base.BaseEstimator` and appropriate mixins. Implement `fit()`, `predict()`, `score()`, `get_params()`, `set_params()`.
- **Immutable Hyperparameters:** Constructor arguments are hyperparameters. Learned attributes get trailing underscores (e.g., `coef_`, `basis_functions_`).
- **Input Validation:** Use `sklearn.utils.validation.check_X_y` in `fit()` and `check_array` in `predict()`.

## 3. Testing Standards
- **Test Framework:** Use `pytest`. Tests reside in `tests/`.
- **Coverage:** Target >80% coverage. Use `pytest-cov` for reporting.
- **Property-Based Testing:** Use `hypothesis` for edge-case generation and invariants.
- **Mutation Testing:** Run `mutmut` periodically to assess test quality.
- **Test Structure:** Each test function tests one behavior. Name tests descriptively: `test_<scenario>_should_<expected_behavior>`.

## 4. Documentation Guidelines
- **User-Facing Docs:** Written in clear, concise English. Assume intermediate Python/ML knowledge.
- **API Reference:** Auto-generated from docstrings via MkDocs.
- **Examples:** Include runnable examples in `examples/` and demo scripts in `pymars/demos/`.

## 5. Commit & PR Practices
- **Atomic Commits:** Each commit addresses one logical change.
- **Commit Messages:** Use conventional commits (e.g., `feat: add hinge basis function`, `fix: resolve pruning error`).
- **PR Reviews:** All PRs require review. CI must pass before merge.

## 6. Security & Dependencies
- **Dependency Minimization:** Prefer standard library, NumPy, SciPy, scikit-learn. Avoid unnecessary additions.
- **Security Audits:** Run `bandit` and `safety` in CI. No secrets in code or commits.

## 7. Release Process
- **Versioning:** Semantic versioning (MAJOR.MINOR.PATCH). Update `pyproject.toml` and `CHANGELOG.md`.
- **Pre-release:** Test on TestPyPI before publishing to PyPI.
