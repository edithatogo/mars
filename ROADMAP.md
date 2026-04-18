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
*   [x] Implement categorical feature support via `CategoricalImputer` and updates to `Earth`.
*   [x] Implement missing value support including imputation and `MissingnessBasisFunction` handling.

## Phase 2: Advanced Features

The goal of this phase is to implement advanced features that are available in the R `earth` package.

* [x] Interaction terms: extend `ForwardPasser`, update pruning logic, and add tests.
* [x] Generalized linear models (GLMs): implement a `GLMEarth` subclass with logistic and Poisson support.
* [x] Cross-validation helper: provide an `EarthCV` class using scikit-learn utilities for k-fold evaluation and hyperparameter search.
* [x] Plotting utilities: add matplotlib-based visualisations and integrate a `plot()` helper on the `Earth` model.

These items constitute the major goals for Phase&nbsp;2 and will extend `pymars` beyond basic fitting and prediction.

## Phase 3: Performance and Optimization

The goal of this phase is to improve the performance of `pymars` and optimize the code for speed and memory usage.

*   **Profile the code.** This will involve using a profiler to identify performance bottlenecks.
*   **Optimize the code for speed.** Apply pure Python algorithmic improvements guided by profiling results.
*   **Optimize the code for memory usage.** This will involve using memory profiling tools to identify memory usage issues.
*   **Profiling results so far:** Using `cProfile` on the forward and pruning passes revealed that repeatedly constructing basis matrices with `np.hstack` dominated runtime and memory usage. The implementation now preallocates the basis matrix and fills it column‑wise, reducing allocations and improving speed.
*   **Latest profiling follow-up:** A fresh profiling pass on 2026-04-18 shows the remaining dominant cost is in forward-pass candidate evaluation, specifically repeated basis-matrix reconstruction in `_build_basis_matrix` together with repeated `numpy.linalg.lstsq` solves in `_calculate_rss_and_coeffs`. No additional low-risk optimization was merged from that pass because the explored memoization approach did not produce a clear end-to-end runtime improvement.

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
| Interaction Terms | ✔️ | ✔️ | ✔️ |
| GLMs | ✔️ | ❌ | ✔️ |
| Cross-Validation | ✔️ | ✔️ | ✔️ |
| Plotting | ✔️ | ✔️ | ✔️ |
| **Performance** | | | |
| Fortran | ❌ | ❌ | ✔️ |
| C | ❌ | ❌ | ✔️ |
