# TODO

This document lists the tasks that need to be completed for the `pymars` project.

## Phase 1: Core Functionality and sklearn Compatibility

*   [x] Implement categorical feature support.
*   [x] Implement missing value support.

## Phase 2: Advanced Features

### Interaction Terms
*   [x] Define internal representation for interaction basis functions.
*   [x] Update `ForwardPasser` to generate and evaluate interaction candidates.
*   [x] Extend pruning logic so interaction terms are properly removed or kept.
*   [x] Create unit tests covering interaction selection and pruning behavior.

### Generalized Linear Models
*   [x] Create `GLMEarth` subclass inheriting from `Earth`.
*   [x] Support logistic and Poisson families with canonical link functions.
*   [x] Adapt scoring and prediction to handle GLM outputs.
*   [x] Provide tests comparing results with scikit‑learn GLM estimators.

### Cross‑Validation Helper
*   [x] Implement an `EarthCV` class using `sklearn.model_selection` utilities.
*   [x] Allow grid search over hyperparameters such as `penalty` and `max_degree`.
*   [x] Document usage examples for common validation workflows.
*   [x] Add tests verifying that splits and scoring behave as expected.

### Plotting Utilities
*   [x] Add a plotting module built on `matplotlib`.
*   [x] Implement basic plots for basis functions and residuals.
*   [x] Integrate plotting entry points with the `Earth` model.
*   [x] Provide example notebooks demonstrating the visualizations.

These items form the bulk of the work for Phase&nbsp;2 and will bring `pymars` closer to feature parity with other implementations.

## Phase 3: Performance and Optimization

*   [ ] Profile the code.
*   [ ] Optimize the code for speed using Python-level improvements and algorithmic optimizations; avoid Cython or native extensions.
*   [ ] Optimize the code for memory usage.
