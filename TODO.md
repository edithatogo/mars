# pymars Project TODO List

This file tracks the tasks needed to complete the `pymars` library. It is a more granular breakdown of the `ROADMAP.md`.

## Phase 1: Core `Earth` Model Implementation

### `pymars/_basis.py` (Basis Functions)
- [ ] Define `BasisFunction` base class (or protocol).
- [ ] Implement `ConstantBasisFunction`.
- [ ] Implement `HingeBasisFunction` (and its variants for left/right).
- [ ] Implement `LinearBasisFunction`.
- [ ] Write unit tests for all basis functions.
    - [ ] Test evaluation of basis functions.
    *   [ ] Test string representation.
    *   [ ] Test properties (knot, variable index, etc.).

### `pymars/_forward.py` (Forward Pass)
- [ ] Implement `ForwardPasser` class.
- [ ] Method to generate candidate basis functions.
- [ ] Method to find the best basis function to add to the model (minimizing RSS or other criteria).
- [ ] Stopping criteria for the forward pass (max terms, min improvement).
- [ ] Mechanism to handle constraints (max_degree, interaction penalties).
- [ ] Write unit tests for the forward pass.
    - [ ] Test with simple, known datasets.
    - [ ] Test stopping criteria.
    - [ ] Test constraint handling.

### `pymars/_pruning.py` (Pruning Pass)
- [ ] Implement `PruningPasser` class.
- [ ] Method to calculate Generalized Cross-Validation (GCV) or other pruning criteria.
- [ ] Iteratively remove basis functions to find the model with the best GCV.
- [ ] Write unit tests for the pruning pass.
    - [ ] Test GCV calculation.
    - [ ] Test that pruning selects the correct subset of terms on a known example.

### `pymars/earth.py` (Core `Earth` Class)
- [ ] Define `Earth` class structure.
- [ ] Constructor (`__init__`) with hyperparameters (e.g., `max_degree`, `penalty`, `minspan_alpha`, `endspan_alpha`, `max_terms`).
- [ ] `fit(X, y)` method:
    - [ ] Orchestrate the forward pass.
    - [ ] Orchestrate the pruning pass.
    - [ ] Store the final set of basis functions.
    - [ ] Calculate model coefficients (e.g., via linear regression on the basis function outputs).
- [ ] `predict(X)` method:
    - [ ] Evaluate the selected basis functions for new `X`.
    - [ ] Compute predictions using stored coefficients.
- [ ] `summary()` method (or similar) to display model information.
- [ ] Write integration tests for the `Earth` class.
    - [ ] Test `fit` and `predict` on a simple dataset.
    - [ ] Compare with expected results if possible.

## Phase 2: Scikit-learn Compatibility Layer

### `pymars/earth.py` or `pymars/_sklearn_compat.py`
- [ ] Ensure `Earth` inherits from `sklearn.base.BaseEstimator`.
- [ ] Implement `sklearn.base.RegressorMixin` for `EarthRegressor`.
    - [ ] Create `EarthRegressor` class, possibly inheriting from `Earth`.
    - [ ] Ensure `fit(X,y)` in `EarthRegressor` calls the core `Earth.fit`.
    - [ ] Ensure `predict(X)` in `EarthRegressor` calls the core `Earth.predict`.
    - [ ] Implement `score(X,y)` or rely on mixin.
- [ ] Implement `sklearn.base.ClassifierMixin` for `EarthClassifier`.
    - [ ] Create `EarthClassifier` class. This might wrap `Earth` and use logistic regression or similar on top of basis functions.
    - [ ] `fit(X,y)` for classifier.
    - [ ] `predict(X)` for classifier.
    - [ ] `predict_proba(X)` for classifier.
    - [ ] `score(X,y)` for classifier.
- [ ] Implement `get_params(deep=True)` and `set_params(**params)` for all estimators.
- [ ] Use `sklearn.utils.validation.check_X_y` in `fit`.
- [ ] Use `sklearn.utils.validation.check_array` in `predict`, `score`, etc.
- [ ] Ensure `fit` returns `self`.
- [ ] Run `sklearn.utils.estimator_checks.check_estimator` on `EarthRegressor` and `EarthClassifier`.
    - [ ] Address any failures from `check_estimator`.

## Phase 3: Advanced Features & Refinements

- [ ] **Interaction Terms:**
    - [ ] Modify forward pass to consider products of existing basis functions (respecting `max_degree`).
- [ ] **Custom Basis Functions:**
    - [ ] Design API for users to provide their own basis function classes.
- [ ] **Categorical Feature Handling:**
    - [ ] Research and implement strategies (e.g., one-hot encoding, specialized basis functions).
- [ ] **Missing Value Handling:**
    - [ ] Define and implement strategies.
- [ ] **Feature Importance:**
    - [ ] Implement method(s) to calculate/display feature importance (e.g., based on GCV reduction, coefficient magnitude, number of times a variable is used).
- [ ] **Optimization:**
    - [ ] Profile code and identify bottlenecks.
    - [ ] Optimize critical sections if necessary.

## Phase 4: Comprehensive Testing

- [ ] Increase unit test coverage for all modules.
- [ ] Add more integration tests.
- [ ] **Comparison with `py-earth`:**
    - [ ] Set up test environment to run `py-earth`.
    - [ ] Create test scripts to compare `pymars` output (coefficients, predictions) with `py-earth` on benchmark datasets. Document any intentional deviations.
- [ ] Achieve target test coverage (e.g., >90%).
- [ ] Set up Continuous Integration (e.g., GitHub Actions).

## Phase 5: Documentation & Examples

- [ ] **API Reference:**
    - [ ] Set up Sphinx or other documentation generator.
    - [ ] Ensure all public classes, methods, functions have good docstrings for auto-generation.
- [ ] **User Guide:**
    - [ ] Write introduction to MARS.
    - [ ] Explain how to use `pymars` for regression.
    - [ ] Explain how to use `pymars` for classification.
    - [ ] Discuss hyperparameters and their impact.
- [ ] **Examples/Tutorials:**
    - [ ] Create Jupyter notebooks or Python scripts for:
        - [ ] Basic regression example.
        - [ ] Basic classification example.
        - [ ] Hyperparameter tuning with scikit-learn (e.g., `GridSearchCV`).
        - [ ] Visualizing basis functions or model components (if feasible).
- [ ] **Installation Instructions:**
    - [ ] Write clear `README.md` installation steps.
    - [ ] Create `setup.py` or `pyproject.toml`.
- [ ] **Contribution Guidelines:**
    - [ ] Refine `AGENTS.md` if needed.
    - [ ] Add a section for human contributors.

## General/Ongoing

- [ ] Refactor code for clarity and efficiency as needed.
- [ ] Keep dependencies updated.
- [ ] Address bugs as they are found.
- [ ] Update `ROADMAP.md`, `AGENTS.md`, `GEMINI.md`, `TODO.md` as project evolves.
- [ ] Maintain `SESSION_LOGS.md`.

This list will be updated as tasks are completed and new tasks are identified.
