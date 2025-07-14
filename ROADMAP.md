# Project Roadmap: pymars (Pure Python Earth)

This document outlines the development roadmap for `pymars`, a pure Python implementation of Multivariate Adaptive Regression Splines, inspired by the `py-earth` library.

## Guiding Principles

*   **Pure Python:** No C/Cython dependencies for easier installation and wider compatibility.
*   **Scikit-learn Compatibility:** Adhere to scikit-learn's API and conventions (Estimator, RegressorMixin, ClassifierMixin, etc.).
*   **`py-earth` Class Structure:** Maintain the familiar class structure and import style (e.g., `import pymars as earth`).
*   **Test-Driven Development:** Write comprehensive tests for all components.
*   **Clear Documentation:** Provide user-friendly documentation and examples.

## Phase 1: Core `Earth` Model Implementation

*   **Objective:** Implement the fundamental MARS algorithm components in pure Python.
*   **Key Tasks:**
    *   [x] **Basis Functions:** Implement different types of basis functions (linear, hinge, etc.).
        *   File: `pymars/_basis.py`
    *   [x] **Forward Pass:** Implement the forward pass algorithm to select basis functions and build an initial model.
        *   File: `pymars/_forward.py`
    *   [x] **Pruning Pass:** Implement the pruning pass algorithm (e.g., using GCV) to remove less important basis functions and prevent overfitting.
        *   File: `pymars/_pruning.py`
    *   [x] **Core `Earth` Class:** Integrate the above components into a central `Earth` class.
        *   File: `pymars/earth.py`
    *   [x] **Model Representation:** Define how the MARS model (coefficients, basis functions) is stored.
    *   [x] **Prediction Logic:** Implement the `predict` method based on the selected basis functions and coefficients.

## Phase 2: Scikit-learn Compatibility Layer

*   **Objective:** Make `pymars` fully compatible with the scikit-learn ecosystem.
*   **Key Tasks:**
    *   [x] **`BaseEstimator` Integration:** Ensure the `Earth` model inherits from `sklearn.base.BaseEstimator`. (Done for wrappers `EarthRegressor`, `EarthClassifier`)
    *   [x] **Mixins:** Implement `RegressorMixin` and `ClassifierMixin` for regression and classification tasks. (Done for wrappers)
        *   Files: `pymars/_sklearn_compat.py`
    *   [x] **`fit` Method:** Adapt the `fit` method to scikit-learn's `(X, y)` signature. (Done for wrappers)
    *   [x] **`predict_proba` (for classification):** Implement probability predictions. (Done)
    *   [x] **`score` Method:** Implement a default scoring method. (Defaults from mixins used)
    *   [x] **`get_params` and `set_params`:** Ensure these methods work correctly for hyperparameter tuning. (Done for wrappers)
    *   [x] **Input Validation:** Use scikit-learn's `check_X_y` and `check_array` utilities. (Used in wrappers)

## Phase 3: Advanced Features & Refinements

*   **Objective:** Incorporate advanced features from `py-earth` and improve the model.
*   **Key Tasks:**
    *   [x] **Interaction Terms:** Allow for the creation of interaction terms between variables. (Handled by `max_degree`)
    *   [ ] **Custom Basis Functions:** Allow users to define and use their own basis functions.
    *   [ ] **Categorical Feature Handling:** Implement strategies for handling categorical predictors.
    *   [p] **Missing Value Handling:** Define and implement strategies for missing data. (Basic `allow_missing` for NaN scrubbing and propagation implemented; advanced like `MissingnessBasisFunction` pending)
    *   [ ] **More Pruning Criteria:** Explore and implement alternative pruning criteria beyond GCV.
    *   [x] **Feature Importance:** Implement methods to assess feature importance. (`nb_subsets`, `gcv`, `rss` implemented)
    *   [ ] **Optimization:** Profile and optimize performance-critical sections of the code.

## Phase 4: Comprehensive Testing

*   **Objective:** Ensure the library is robust, reliable, and correct.
*   **Key Tasks:**
    *   [p] **Unit Tests:** Write unit tests for all classes, methods, and functions. (Many written, ongoing effort)
        *   Directory: `tests/`
    *   [p] **Integration Tests:** Test the interaction between different components. (Many written, ongoing effort)
    *   [ ] **Comparison with `py-earth`:** Where feasible, compare results with `py-earth` on benchmark datasets (this will require `py-earth` to be installed in the test environment).
    *   [ ] **Test Coverage:** Aim for high test coverage (e.g., >90%). (Not formally measured)
    *   [ ] **Continuous Integration (CI):** Set up CI (e.g., GitHub Actions). (Not set up by agent)

## Phase 5: Documentation & Examples

*   **Objective:** Provide clear, comprehensive documentation and practical examples.
*   **Key Tasks:**
    *   [p] **API Reference:** Generate an API reference using tools like Sphinx. (Docstrings exist, generation not set up)
    *   [ ] **User Guide:** Write a user guide explaining the concepts of MARS and how to use `pymars`.
    *   [ ] **Examples/Tutorials:** Create example notebooks or scripts demonstrating various use cases.
        *   Regression examples.
        *   Classification examples.
        *   Hyperparameter tuning.
    *   [p] **Installation Instructions:** Clear instructions on how to install the library. (`README.md` has basic info)
    *   [p] **Contribution Guidelines:** If open to contributions. (`AGENTS.md` exists)

## Future Considerations (Post v1.0)

*   Support for sparse matrices.
*   More advanced visualization tools for model interpretation.
*   Integration with other scikit-learn-contrib projects.

This roadmap is a living document and may be updated as the project progresses.
