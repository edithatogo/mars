# Roadmap

This document outlines the development roadmap for `pymars`.

## Phase 1: Core Functionality and sklearn Compatibility

The goal of this phase is to implement the core functionality of the MARS algorithm and ensure that `pymars` is fully compatible with the `sklearn` ecosystem.

*   [x] Implement the forward pass of the MARS algorithm.
*   [x] Implement the backward pass (pruning) of the MARS algorithm.
*   [x] Implement a `scikit-learn` compatible `fit` method.
*   [x] Implement a `scikit-learn` compatible `predict` method.
*   [x] Implement a `scikit-learn` compatible `score` method.
*   [x] Implement a `scikit-learn` compatible `get_params` and `set_params` methods.
*   [x] Add support for `sample_weight` in the `fit` method.
*   [x] Implement categorical feature support. This added a `CategoricalImputer` class and modified the `Earth` class to handle categorical features.
*   [x] Implement missing value support. This added a `MissingValuesImputer` class and modified the `Earth` class to handle missing values.

With these core tasks finished, the project moves on to advanced features such as interaction terms, generalized linear model (GLM) support, and cross-validation.

## Phase 2: Advanced Features

The goal of this phase is to implement advanced features that are available in the R `earth` package.

*   [ ] Implement interaction terms by extending the `Earth` class to handle interactions.
*   [ ] Implement generalized linear models (GLMs) through a new `GLMEarth` class.
*   [ ] Implement cross-validation with an `EarthCV` class.
*   [ ] Implement plotting utilities for model diagnostics.

## Phase 3: Performance and Optimization

The goal of this phase is to improve the performance of `pymars` and optimize the code for speed and memory usage.

*   **Profile the code.** This will involve using a profiler to identify performance bottlenecks.
*   **Optimize the code for speed.** This will involve using Cython to optimize the code for speed.
*   **Optimize the code for memory usage.** This will involve using memory profiling tools to identify memory usage issues.

## Feature Matrix

| Feature | pymars | py-earth | R-earth |
| --- | --- | --- | --- |
| **Core** | | | |
| Forward Pass | ✔️ | ✔️ | ✔️ |
| Backward Pass (Pruning) | ✔️ | ✔️ | ✔️ |
| `fit` | ✔️ | ✔️ | ✔️ |
| `predict` | ✔️ | ✔️ | ✔️ |
| `score` | ✔️ | ✔️ | ✔️ |
| `get_params` | ✔️ | ✔️ | ✔️ |
| `set_params` | ✔️ | ✔️ | ✔️ |
| `sample_weight` | ✔️ | ✔️ | ✔️ |
| Categorical Features | ✔️ | ✔️ | ✔️ |
| Missing Values | ✔️ | ✔️ | ✔️ |
| **Advanced** | | | |
| Interaction Terms | ❌ | ✔️ | ✔️ |
| GLMs | ❌ | ❌ | ✔️ |
| Cross-Validation | ❌ | ✔️ | ✔️ |
| Plotting | ❌ | ✔️ | ✔️ |
| **Performance** | | | |
| Cython | ❌ | ✔️ | N/A |
| Fortran | ❌ | ❌ | ✔️ |
| C | ❌ | ❌ | ✔️ |
