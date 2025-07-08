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
    *   [ ] **Basis Functions:** Implement different types of basis functions (linear, hinge, etc.).
        *   File: `pymars/_basis.py`
    *   [ ] **Forward Pass:** Implement the forward pass algorithm to select basis functions and build an initial model.
        *   File: `pymars/_forward.py`
    *   [ ] **Pruning Pass:** Implement the pruning pass algorithm (e.g., using GCV) to remove less important basis functions and prevent overfitting.
        *   File: `pymars/_pruning.py`
    *   [ ] **Core `Earth` Class:** Integrate the above components into a central `Earth` class.
        *   File: `pymars/earth.py`
    *   [ ] **Model Representation:** Define how the MARS model (coefficients, basis functions) is stored.
    *   [ ] **Prediction Logic:** Implement the `predict` method based on the selected basis functions and coefficients.

## Phase 2: Scikit-learn Compatibility Layer

*   **Objective:** Make `pymars` fully compatible with the scikit-learn ecosystem.
*   **Key Tasks:**
    *   [ ] **`BaseEstimator` Integration:** Ensure the `Earth` model inherits from `sklearn.base.BaseEstimator`.
    *   [ ] **Mixins:** Implement `RegressorMixin` and `ClassifierMixin` for regression and classification tasks.
        *   Files: `pymars/_sklearn_compat.py` (or directly in `pymars/earth.py`)
    *   [ ] **`fit` Method:** Adapt the `fit` method to scikit-learn's `(X, y)` signature.
    *   [ ] **`predict_proba` (for classification):** Implement probability predictions.
    *   [ ] **`score` Method:** Implement a default scoring method.
    *   [ ] **`get_params` and `set_params`:** Ensure these methods work correctly for hyperparameter tuning.
    *   [ ] **Input Validation:** Use scikit-learn's `check_X_y` and `check_array` utilities.

## Phase 3: Advanced Features & Refinements

*   **Objective:** Incorporate advanced features from `py-earth` and improve the model.
*   **Key Tasks:**
    *   [ ] **Interaction Terms:** Allow for the creation of interaction terms between variables.
    *   [ ] **Custom Basis Functions:** Allow users to define and use their own basis functions.
    *   [ ] **Categorical Feature Handling:** Implement strategies for handling categorical predictors.
    *   [ ] **Missing Value Handling:** Define and implement strategies for missing data.
    *   [ ] **More Pruning Criteria:** Explore and implement alternative pruning criteria beyond GCV.
    *   [ ] **Feature Importance:** Implement methods to assess feature importance.
    *   [ ] **Optimization:** Profile and optimize performance-critical sections of the code.

## Phase 4: Comprehensive Testing

*   **Objective:** Ensure the library is robust, reliable, and correct.
*   **Key Tasks:**
    *   [ ] **Unit Tests:** Write unit tests for all classes, methods, and functions.
        *   Directory: `tests/`
    *   [ ] **Integration Tests:** Test the interaction between different components.
    *   [ ] **Comparison with `py-earth`:** Where feasible, compare results with `py-earth` on benchmark datasets (this will require `py-earth` to be installed in the test environment).
    *   [ ] **Test Coverage:** Aim for high test coverage (e.g., >90%).
    *   [ ] **Continuous Integration (CI):** Set up CI (e.g., GitHub Actions) to run tests automatically.

## Phase 5: Documentation & Examples

*   **Objective:** Provide clear, comprehensive documentation and practical examples.
*   **Key Tasks:**
    *   [ ] **API Reference:** Generate an API reference using tools like Sphinx.
    *   [ ] **User Guide:** Write a user guide explaining the concepts of MARS and how to use `pymars`.
    *   [ ] **Examples/Tutorials:** Create example notebooks or scripts demonstrating various use cases.
        *   Regression examples.
        *   Classification examples.
        *   Hyperparameter tuning.
    *   [ ] **Installation Instructions:** Clear instructions on how to install the library.
    *   [ ] **Contribution Guidelines:** If open to contributions.

## Future Considerations (Post v1.0)

*   Support for sparse matrices.
*   More advanced visualization tools for model interpretation.
*   Integration with other scikit-learn-contrib projects.

This roadmap is a living document and may be updated as the project progresses.
