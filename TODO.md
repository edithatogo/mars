# TODO

This document lists the tasks that need to be completed for the `pymars` project.

## Phase 1: Core Functionality and sklearn Compatibility

*   [x] Implement categorical feature support.
*   [x] Implement missing value support.

## Phase 2: Advanced Features

### Interaction Terms
*   [ ] Define internal representation for interaction basis functions.
*   [ ] Update `ForwardPasser` to generate and evaluate interaction candidates.
*   [ ] Extend pruning logic so interaction terms are properly removed or kept.
*   [ ] Create unit tests covering interaction selection and pruning behavior.

### Generalized Linear Models
*   [ ] Create `GLMEarth` subclass inheriting from `Earth`.
*   [ ] Support logistic and Poisson families with canonical link functions.
*   [ ] Adapt scoring and prediction to handle GLM outputs.
*   [ ] Provide tests comparing results with scikit‑learn GLM estimators.

### Cross‑Validation Helper
*   [ ] Implement an `EarthCV` class using `sklearn.model_selection` utilities.
*   [ ] Allow grid search over hyperparameters such as `penalty` and `max_degree`.
*   [ ] Document usage examples for common validation workflows.
*   [ ] Add tests verifying that splits and scoring behave as expected.

### Plotting Utilities
*   [ ] Add a plotting module built on `matplotlib`.
*   [ ] Implement basic plots for basis functions and residuals.
*   [ ] Integrate plotting entry points with the `Earth` model.
*   [ ] Provide example notebooks demonstrating the visualizations.

These items form the bulk of the work for Phase&nbsp;2 and will bring `pymars` closer to feature parity with other implementations.

## Phase 3: Performance and Optimization

*   [ ] Profile the code.
*   [ ] Optimize the code for speed.
*   [ ] Optimize the code for memory usage.
