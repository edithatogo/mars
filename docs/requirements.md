# Requirements for mars: Pure Python Earth (Multivariate Adaptive Regression Splines)

## Overview

mars is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), designed as a drop-in replacement for the `py-earth` library while maintaining full compatibility with the scikit-learn ecosystem. This document outlines the functional, performance, and compatibility requirements for the library.

## Core Functional Requirements

### 1. Core MARS Algorithm Implementation

**REQ-1.1: Forward Pass**
- [x] Must implement the forward selection phase of MARS
- [x] Must select basis functions (hinge and linear terms) based on reduction in residual sum of squares (RSS)
- [x] Must support interaction terms with maximum degree control
- [x] Must implement `minspan` and `endspan` parameters for knot placement control
- [x] Must handle both categorical and continuous features

**REQ-1.2: Pruning Pass**
- [x] Must implement backward elimination using Generalized Cross-Validation (GCV)
- [x] Must support penalty parameter for regularization
- [x] Must provide pruning trace for model selection
- [x] Must handle interaction terms during pruning

**REQ-1.3: Basis Functions**
- [x] Must implement hinge functions: max(0, x - const) and max(0, const - x)
- [x] Must implement linear terms
- [x] Must support interaction terms between basis functions
- [ ] Must support tensor products for multi-dimensional basis functions
- [x] Must handle missing values in basis function computation

### 2. Model Fitting and Prediction

**REQ-2.1: Model Fitting**
- [x] Must implement `fit(X, y)` method compatible with scikit-learn
- [x] Must support sample weights
- [x] Must handle missing values in both X and y
- [x] Must provide options for imputation strategies
- [x] Must support categorical feature encoding

**REQ-2.2: Prediction**
- [x] Must implement `predict(X)` method compatible with scikit-learn
- [x] Must handle missing values in prediction data
- [x] Must return consistent predictions across multiple calls
- [x] Must validate input dimensions match training data

**REQ-2.3: Model Evaluation**
- [x] Must implement `score(X, y)` method returning R² score
- [x] Must provide GCV score for model selection
- [x] Must compute RSS, MSE, and other key metrics
- [x] Must support custom scoring functions

### 3. Feature Importances

**REQ-3.1: Feature Importance Calculation**
- [x] Must support 'nb_subsets' importance calculation (number of subsets in pruning trace)
- [x] Must support 'gcv' importance calculation (GCV improvement)
- [x] Must support 'rss' importance calculation (RSS reduction)
- [x] Must normalize importance scores to sum to 1.0
- [x] Must handle interaction terms in importance calculations

## Scikit-learn Compatibility Requirements

### 4. Estimator Interface

**REQ-4.1: Base Estimator Compliance**
- [x] Must inherit from `sklearn.base.BaseEstimator`
- [x] Must implement `get_params(deep=True)` and `set_params(**params)`
- [x] Must follow scikit-learn parameter naming conventions
- [x] Must store all initialization parameters as public attributes
- [x] Must store learned attributes with trailing underscores

**REQ-4.2: Regressor Interface**
- [x] Must inherit from `sklearn.base.RegressorMixin`
- [x] Must implement `fit(X, y)`, `predict(X)`, and `score(X, y)`
- [x] Must validate inputs using `check_X_y` and `check_array`
- [x] Must return `self` from `fit` method
- [x] Must check if fitted before prediction using `check_is_fitted`

**REQ-4.3: Classifier Interface**
- [x] Must inherit from `sklearn.base.ClassifierMixin`
- [x] Must implement `fit(X, y)`, `predict(X)`, and `score(X, y)`
- [x] Must provide `predict_proba(X)` if the underlying classifier supports it
- [x] Must store class labels in `classes_` attribute
- [x] Must handle multi-class classification tasks

### 5. Pipeline Integration

**REQ-5.1: Pipeline Compatibility**
- [x] Must work seamlessly within scikit-learn pipelines
- [x] Must support feature selection and preprocessing steps
- [x] Must maintain state consistency across pipeline steps
- [x] Must handle different data types appropriately

**REQ-5.2: Model Selection Integration**
- [x] Must work with `GridSearchCV` and `RandomizedSearchCV`
- [x] Must support cross-validation using `cross_val_score`
- [x] Must be compatible with `learning_curve` and other validation utilities
- [x] Must support nested cross-validation

## Advanced Features

### 6. Generalized Linear Models

**REQ-6.1: GLM Support**
- [x] Must implement `GLMEarth` subclass for generalized linear models
- [x] Must support binomial family (logistic regression)
- [x] Must support Poisson family
- [x] Must implement canonical link functions
- [x] Must provide appropriate scoring for GLMs

### 7. Cross-Validation Helper

**REQ-7.1: EarthCV Class**
- [x] Must implement `EarthCV` for hyperparameter tuning
- [x] Must support grid search over MARS hyperparameters
- [x] Must allow tuning of `max_degree`, `penalty`, `max_terms`, `minspan_alpha`, `endspan_alpha`
- [x] Must provide `best_estimator_`, `best_params_`, and `best_score_`
- [x] Must be compatible with scikit-learn's model selection utilities

### 8. Visualization and Diagnostics

**REQ-8.1: Plotting Utilities**
- [x] Must provide `plot_basis_functions` for visualizing selected basis functions
- [x] Must provide `plot_residuals` for model diagnostics
- [x] Must support customizable plot parameters
- [x] Must integrate with matplotlib's plotting system
- [x] Must handle missing values in plotting functions

### 9. Data Preprocessing

**REQ-9.1: Categorical Feature Handling**
- [x] Must implement categorical feature imputation
- [x] Must support different encoding strategies
- [x] Must handle unseen categories in prediction
- [x] Must preserve categorical information during transformation

**REQ-9.2: Missing Value Handling**
- [x] Must support missing values in training data
- [x] Must support missing values in prediction data
- [x] Must implement appropriate imputation strategies
- [x] Must handle missing values during basis function computation
- [x] Must provide control over missing value handling via `allow_missing` parameter

## Performance Requirements

### 10. Algorithmic Efficiency

**REQ-10.1: Time Complexity**
- [x] Forward pass must scale reasonably with sample size and feature count
- [x] Pruning pass must be efficient for large numbers of basis functions
- [x] Must avoid excessive memory allocations during basis matrix construction
- [x] Must implement memory-efficient basis matrix building with preallocation
- [ ] Must provide performance benchmarks vs. py-earth when available

**REQ-10.2: Memory Usage**
- [x] Must avoid creating unnecessary copies of input data
- [x] Must use memory-efficient algorithms for large datasets
- [x] Must handle basis matrix construction without excessive memory usage
- [x] Must clean up temporary variables after computation

### 11. Numerical Stability

**REQ-11.1: Numerical Precision**
- [x] Must handle near-duplicate values robustly
- [x] Must handle extreme values without numerical overflow
- [x] Must provide stable linear algebra computations
- [x] Must handle rank-deficient cases gracefully

## Compatibility Requirements

### 12. API Compatibility with py-earth

**REQ-12.1: Parameter Compatibility**
- [x] Must support equivalent parameters to py-earth: `max_degree`, `penalty`, `max_terms`, `minspan_alpha`, `endspan_alpha`
- [x] Must implement `minspan` and `endspan` with cooldown behavior matching py-earth
- [x] Must support `allow_linear` parameter for linear term inclusion
- [x] Must provide equivalent feature importance types
- [ ] Must maintain same parameter defaults when possible

**REQ-12.2: Method Compatibility**
- [x] Must provide similar method signatures to py-earth
- [x] Must support `summary()` method for model inspection
- [x] Must provide access to basis functions and coefficients
- [x] Must support equivalent prediction interface
- [x] Must maintain consistent return types for metrics

### 13. Python Environment Compatibility

**REQ-13.1: Python Version Support**
- [x] Must support Python 3.8+
- [x] Must use Python features available in target versions
- [x] Must provide clear version compatibility information
- [x] Must handle version-specific language features appropriately

**REQ-13.2: Dependency Management**
- [x] Must depend only on standard Python libraries where possible
- [x] Must specify minimal required versions of NumPy and scikit-learn
- [x] Must avoid using deprecated library features
- [x] Must maintain compatibility with common Python environments
- [x] Must not use C/Cython extensions (pure Python only)

## Quality Assurance Requirements

### 14. Testing

**REQ-14.1: Test Coverage**
- [x] Must provide comprehensive unit tests for all core functionality
- [x] Must include scikit-learn estimator compatibility tests
- [x] Must test edge cases and error conditions
- [x] Must verify numerical correctness against reference implementations
- [x] Must include regression tests for bug fixes

**REQ-14.2: Test Quality**
- [x] Must follow pytest best practices
- [x] Must use appropriate test fixtures and parametrization
- [x] Must test API consistency and error handling
- [x] Must include property-based tests where appropriate
- [x] Must provide clear test documentation

### 15. Documentation

**REQ-15.1: API Documentation**
- [x] Must provide comprehensive docstrings for all public methods
- [x] Must document all parameters with types and descriptions
- [x] Must include usage examples in docstrings
- [x] Must document return types and exceptions
- [x] Must follow NumPy/SciPy documentation standards

**REQ-15.2: User Documentation**
- [x] Must provide getting started guides
- [x] Must include advanced usage examples
- [x] Must document scikit-learn integration
- [x] Must provide comparison with py-earth features
- [x] Must include troubleshooting guides

## Non-functional Requirements

### 16. Maintainability

**REQ-16.1: Code Quality**
- [x] Must follow PEP 8 coding standards
- [x] Must include type hints for public interfaces
- [x] Must maintain clean, readable code structure
- [x] Must follow consistent naming conventions
- [x] Must include appropriate comments explaining complex logic

**REQ-16.2: Architecture**
- [x] Must maintain clear separation of concerns
- [x] Must support modular testing of components
- [x] Must provide clean interfaces between modules
- [x] Must avoid tight coupling between components
- [x] Must support future extensibility

### 17. Usability

**REQ-17.1: User Experience**
- [x] Must provide intuitive API matching scikit-learn conventions
- [x] Must provide clear error messages
- [x] Must include helpful warnings for common issues
- [x] Must maintain consistency with scikit-learn behavior
- [x] Must provide meaningful progress indicators for long operations

### 18. Deployment

**REQ-18.1: Installation**
- [x] Must be installable via `pip install`
- [x] Must work with virtual environments
- [x] Must include proper metadata in `pyproject.toml`
- [x] Must support installation from source
- [x] Must provide clear installation instructions

---

*This requirements document is a living document and will be updated as the project evolves.*