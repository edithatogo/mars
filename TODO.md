# pymars Project TODO List

This file tracks the tasks needed to complete the `pymars` library. It is a more granular breakdown of the `ROADMAP.md`.

## Phase 1: Core `Earth` Model Implementation

### `pymars/_basis.py` (Basis Functions)
- [x] Define `BasisFunction` base class (or protocol).
- [x] Implement `ConstantBasisFunction`.
- [x] Implement `HingeBasisFunction` (and its variants for left/right).
- [x] Implement `LinearBasisFunction`.
- [x] Write unit tests for all basis functions.
    - [x] Test evaluation of basis functions.
    - [x] Test string representation.
    - [x] Test properties (knot, variable index, etc.).

### `pymars/_forward.py` (Forward Pass)
- [x] Implement `ForwardPasser` class.
- [x] Method to generate candidate basis functions.
- [x] Method to find the best basis function to add to the model (minimizing RSS or other criteria).
- [x] Stopping criteria for the forward pass (max terms, min improvement).
- [x] Mechanism to handle constraints (max_degree, interaction penalties, minspan/endspan).
- [x] Write unit tests for the forward pass.
    - [x] Test with simple, known datasets.
    - [x] Test stopping criteria.
    - [x] Test constraint handling.

### `pymars/_pruning.py` (Pruning Pass)
- [x] Implement `PruningPasser` class.
- [x] Method to calculate Generalized Cross-Validation (GCV) or other pruning criteria.
- [x] Iteratively remove basis functions to find the model with the best GCV.
- [x] Write unit tests for the pruning pass.
    - [x] Test GCV calculation.
    - [x] Test that pruning selects the correct subset of terms on a known example.

### `pymars/earth.py` (Core `Earth` Class)
- [x] Define `Earth` class structure.
- [x] Constructor (`__init__`) with hyperparameters (e.g., `max_degree`, `penalty`, `minspan_alpha`, `endspan_alpha`, `max_terms`).
- [x] `fit(X, y)` method:
    - [x] Orchestrate the forward pass.
    - [x] Orchestrate the pruning pass.
    - [x] Store the final set of basis functions.
    - [x] Calculate model coefficients (e.g., via linear regression on the basis function outputs).
- [x] `predict(X)` method:
    - [x] Evaluate the selected basis functions for new `X`.
    - [x] Compute predictions using stored coefficients.
- [x] `summary()` method (or similar) to display model information.
- [x] Write integration tests for the `Earth` class.
    - [x] Test `fit` and `predict` on a simple dataset.
    - [p] Compare with expected results if possible. (Basic checks done, more rigorous comparison pending)

## Phase 2: Scikit-learn Compatibility Layer

### `pymars/earth.py` or `pymars/_sklearn_compat.py`
- [x] Ensure `Earth` inherits from `sklearn.base.BaseEstimator`. (Done for wrappers `EarthRegressor`, `EarthClassifier`)
- [x] Implement `sklearn.base.RegressorMixin` for `EarthRegressor`.
    - [x] Create `EarthRegressor` class, possibly inheriting from `Earth`.
    - [x] Ensure `fit(X,y)` in `EarthRegressor` calls the core `Earth.fit`.
    - [x] Ensure `predict(X)` in `EarthRegressor` calls the core `Earth.predict`.
    - [x] Implement `score(X,y)` or rely on mixin.
- [x] Implement `sklearn.base.ClassifierMixin` for `EarthClassifier`.
    - [x] Create `EarthClassifier` class. This might wrap `Earth` and use logistic regression or similar on top of basis functions.
    - [x] `fit(X,y)` for classifier.
    - [x] `predict(X)` for classifier.
    - [x] `predict_proba(X)` for classifier.
    - [x] `score(X,y)` for classifier.
- [x] Implement `get_params(deep=True)` and `set_params(**params)` for all estimators.
- [x] Use `sklearn.utils.validation.check_X_y` in `fit`.
- [x] Use `sklearn.utils.validation.check_array` in `predict`, `score`, etc.
- [x] Ensure `fit` returns `self`.
- [p] Run `sklearn.utils.estimator_checks.check_estimator` on `EarthRegressor` and `EarthClassifier`.
    - [p] Address any failures from `check_estimator`. (Partially addressed, known skips/failures remain)

## Phase 3: Advanced Features & Refinements

- [x] **Interaction Terms:**
    - [x] Modify forward pass to consider products of existing basis functions (respecting `max_degree`).
- [ ] **Custom Basis Functions:**
    - [ ] Design API for users to provide their own basis function classes.
- [ ] **Categorical Feature Handling:**
    - [ ] Research and implement strategies (e.g., one-hot encoding, specialized basis functions).
- [p] **Missing Value Handling:**
    - [p] Define and implement strategies. (Basic `allow_missing` for NaN scrubbing and propagation implemented.)
    - [ ] Implement `MissingnessBasisFunction`.
    - [ ] Integrate `MissingnessBasisFunction` into forward pass.
    - [ ] Refine model behavior with `MissingnessBasisFunction`.
- [x] **Feature Importance:**
    - [x] Implement method(s) to calculate/display feature importance (`nb_subsets`, `gcv`, `rss` implemented).
- [ ] **Optimization:**
    - [ ] Profile code and identify bottlenecks.
    - [ ] Optimize critical sections if necessary.

## Phase 4: Comprehensive Testing

- [p] Increase unit test coverage for all modules. (Ongoing)
- [p] Add more integration tests. (Ongoing)
- [ ] **Comparison with `py-earth`:**
    - [ ] Set up test environment to run `py-earth`.
    - [ ] Create test scripts to compare `pymars` output (coefficients, predictions) with `py-earth` on benchmark datasets. Document any intentional deviations.
- [ ] Achieve target test coverage (e.g., >90%). (Not formally measured)
- [ ] Set up Continuous Integration (e.g., GitHub Actions). (Not set up by agent)

## Phase 5: Documentation & Examples

- [p] **API Reference:**
    - [p] Set up Sphinx or other documentation generator. (Docstrings exist, generation not set up)
    - [p] Ensure all public classes, methods, functions have good docstrings for auto-generation.
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
- [p] **Installation Instructions:**
    - [p] Write clear `README.md` installation steps.
    - [ ] Create `setup.py` or `pyproject.toml`. (Agent does not manage this directly)
- [p] **Contribution Guidelines:**
    - [p] Refine `AGENTS.md` if needed.
    - [ ] Add a section for human contributors.

## General/Ongoing

- [p] Refactor code for clarity and efficiency as needed. (Ongoing)
- [p] Keep dependencies updated. (Managed by environment)
- [p] Address bugs as they are found. (Ongoing)
- [x] Update `ROADMAP.md`, `AGENTS.md`, `GEMINI.md`, `TODO.md` as project evolves. (Updating now)
- [x] Maintain `SESSION_LOGS.md`.

This list will be updated as tasks are completed and new tasks are identified.
