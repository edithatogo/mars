# Tasks for pymars: Pure Python Earth (Multivariate Adaptive Regression Splines)

## 1. Core MARS Algorithm Implementation

### 1.1 Forward Pass Implementation
- [x] Implement forward selection phase of MARS algorithm
- [x] Implement basis function selection based on RSS reduction
- [x] Support for hinge and linear terms
- [x] Support for interaction terms with maximum degree control
- [x] Implement minspan and endspan parameters for knot placement
- [x] Handle categorical features in forward pass
- [x] Efficient basis matrix construction with preallocation
- [x] Optimize forward pass for very large datasets
- [x] Add advanced knot selection heuristics

### 1.2 Pruning Pass Implementation
- [x] Implement backward elimination using Generalized Cross-Validation (GCV)
- [x] Support penalty parameter for regularization
- [x] Provide pruning trace for model selection
- [x] Handle interaction terms during pruning
- [x] Add alternative pruning criteria beyond GCV

### 1.3 Basis Functions Implementation
- [x] Implement hinge functions max(0, x - const) and max(0, const - x)
- [x] Implement linear terms
- [x] Support interaction terms between basis functions
- [x] Implement tensor products for multi-dimensional basis functions
- [x] Handle missing values in basis function computation
- [x] Ensure proper tracking of involved variables for feature importance

## 2. Model Fitting and Prediction

### 2.1 Model Fitting
- [x] Implement fit(X, y) method compatible with scikit-learn
- [x] Support sample weights in fitting process
- [x] Handle missing values in both X and y during fitting
- [x] Provide options for imputation strategies
- [x] Support categorical feature encoding during fitting

### 2.2 Prediction
- [x] Implement predict(X) method compatible with scikit-learn
- [x] Handle missing values in prediction data
- [x] Return consistent predictions across multiple calls
- [x] Validate input dimensions match training data

### 2.3 Model Evaluation
- [x] Implement score(X, y) method returning R² score
- [x] Provide GCV score for model selection
- [x] Compute RSS, MSE, and other key metrics
- [x] Support custom scoring functions

## 3. Feature Importances

### 3.1 Feature Importance Calculation
- [x] Support 'nb_subsets' importance calculation (number of subsets in pruning trace)
- [x] Support 'gcv' importance calculation (GCV improvement)
- [x] Support 'rss' importance calculation (RSS reduction)
- [x] Normalize importance scores to sum to 1.0
- [x] Handle interaction terms in importance calculations

## 4. Scikit-learn Compatibility

### 4.1 Base Estimator Compliance
- [x] Inherit from sklearn.base.BaseEstimator
- [x] Implement get_params(deep=True) and set_params(**params)
- [x] Follow scikit-learn parameter naming conventions
- [x] Store all initialization parameters as public attributes
- [x] Store learned attributes with trailing underscores

### 4.2 Regressor Interface
- [x] Inherit from sklearn.base.RegressorMixin
- [x] Implement fit(X, y), predict(X), and score(X, y)
- [x] Validate inputs using check_X_y and check_array
- [x] Return self from fit method
- [x] Check if fitted before prediction using check_is_fitted

### 4.3 Classifier Interface
- [x] Inherit from sklearn.base.ClassifierMixin
- [x] Implement fit(X, y), predict(X), and score(X, y)
- [x] Provide predict_proba(X) if the underlying classifier supports it
- [x] Store class labels in classes_ attribute
- [x] Handle multi-class classification tasks

### 4.4 Pipeline Integration
- [x] Work seamlessly within scikit-learn pipelines
- [x] Support feature selection and preprocessing steps
- [x] Maintain state consistency across pipeline steps
- [x] Handle different data types appropriately

### 4.5 Model Selection Integration
- [x] Work with GridSearchCV and RandomizedSearchCV
- [x] Support cross-validation using cross_val_score
- [x] Compatible with learning_curve and other validation utilities
- [x] Support nested cross-validation

## 5. Generalized Linear Models

### 5.1 GLM Support
- [x] Implement GLMEarth subclass for generalized linear models
- [x] Support binomial family (logistic regression)
- [x] Support Poisson family
- [x] Implement canonical link functions
- [x] Provide appropriate scoring for GLMs

## 6. Cross-Validation Helper

### 6.1 EarthCV Class
- [x] Implement EarthCV for hyperparameter tuning
- [x] Support grid search over MARS hyperparameters
- [x] Allow tuning of max_degree, penalty, max_terms, minspan_alpha, endspan_alpha
- [x] Provide best_estimator_, best_params_, and best_score_
- [x] Compatible with scikit-learn's model selection utilities

## 7. Visualization and Diagnostics

### 7.1 Plotting Utilities
- [x] Provide plot_basis_functions for visualizing selected basis functions
- [x] Provide plot_residuals for model diagnostics
- [x] Support customizable plot parameters
- [x] Integrate with matplotlib's plotting system
- [x] Handle missing values in plotting functions

## 8. Data Preprocessing

### 8.1 Categorical Feature Handling
- [x] Implement categorical feature imputation
- [x] Support different encoding strategies
- [x] Handle unseen categories in prediction
- [x] Preserve categorical information during transformation

### 8.2 Missing Value Handling
- [x] Support missing values in training data
- [x] Support missing values in prediction data
- [x] Implement appropriate imputation strategies
- [x] Handle missing values during basis function computation
- [x] Provide control over missing value handling via allow_missing parameter

## 9. Performance Optimization

### 9.1 Algorithmic Efficiency
- [x] Forward pass scales reasonably with sample size and feature count
- [x] Pruning pass is efficient for large numbers of basis functions
- [x] Avoid excessive memory allocations during basis matrix construction
- [x] Implement memory-efficient basis matrix building with preallocation
- [ ] Provide performance benchmarks vs. py-earth when available

### 9.2 Memory Usage
- [x] Avoid creating unnecessary copies of input data
- [x] Use memory-efficient algorithms for large datasets
- [x] Handle basis matrix construction without excessive memory usage
- [x] Clean up temporary variables after computation

### 9.3 Numerical Stability
- [x] Handle near-duplicate values robustly
- [x] Handle extreme values without numerical overflow
- [x] Provide stable linear algebra computations
- [x] Handle rank-deficient cases gracefully

## 10. API Compatibility with py-earth

### 10.1 Parameter Compatibility
- [x] Support equivalent parameters to py-earth: max_degree, penalty, max_terms, minspan_alpha, endspan_alpha
- [x] Implement minspan and endspan with cooldown behavior matching py-earth
- [x] Support allow_linear parameter for linear term inclusion
- [x] Provide equivalent feature importance types
- [ ] Maintain same parameter defaults when possible

### 10.2 Method Compatibility
- [x] Provide similar method signatures to py-earth
- [x] Support summary() method for model inspection
- [x] Provide access to basis functions and coefficients
- [x] Support equivalent prediction interface
- [x] Maintain consistent return types for metrics

## 11. Python Environment Compatibility

### 11.1 Python Version Support
- [x] Support Python 3.8+
- [x] Use Python features available in target versions
- [x] Provide clear version compatibility information
- [x] Handle version-specific language features appropriately

### 11.2 Dependency Management
- [x] Depend only on standard Python libraries where possible
- [x] Specify minimal required versions of NumPy and scikit-learn
- [x] Avoid using deprecated library features
- [x] Maintain compatibility with common Python environments
- [x] Do not use C/Cython extensions (pure Python only)

## 12. Testing

### 12.1 Test Coverage
- [x] Provide comprehensive unit tests for all core functionality
- [x] Include scikit-learn estimator compatibility tests
- [x] Test edge cases and error conditions
- [x] Verify numerical correctness against reference implementations
- [x] Include regression tests for bug fixes

### 12.2 Test Quality
- [x] Follow pytest best practices
- [x] Use appropriate test fixtures and parametrization
- [x] Test API consistency and error handling
- [x] Include property-based tests using Hypothesis
- [x] Provide clear test documentation

## 13. Documentation

### 13.1 API Documentation
- [x] Provide comprehensive docstrings for all public methods
- [x] Document all parameters with types and descriptions
- [x] Include usage examples in docstrings
- [x] Document return types and exceptions
- [x] Follow NumPy/SciPy documentation standards

### 13.2 User Documentation
- [x] Provide getting started guides
- [x] Include advanced usage examples
- [x] Document scikit-learn integration
- [x] Provide comparison with py-earth features
- [x] Include troubleshooting guides

## 14. Maintainability

### 14.1 Code Quality
- [x] Follow PEP 8 coding standards
- [x] Include type hints for public interfaces
- [x] Maintain clean, readable code structure
- [x] Follow consistent naming conventions
- [x] Include appropriate comments explaining complex logic

### 14.2 Architecture
- [x] Maintain clear separation of concerns
- [x] Support modular testing of components
- [x] Provide clean interfaces between modules
- [x] Avoid tight coupling between components
- [x] Support future extensibility

## 15. Usability

### 15.1 User Experience
- [x] Provide intuitive API matching scikit-learn conventions
- [x] Provide clear error messages
- [x] Include helpful warnings for common issues
- [x] Maintain consistency with scikit-learn behavior
- [x] Provide meaningful progress indicators for long operations

## 16. Deployment

### 16.1 Installation
- [x] Be installable via pip install
- [x] Work with virtual environments
- [x] Include proper metadata in pyproject.toml
- [x] Support installation from source
- [x] Provide clear installation instructions

## 17. Demos and Examples

### 17.1 Demo Scripts
- [x] Create basic regression demo
- [x] Create basic classification demo
- [x] Demonstrate cross-validation usage
- [x] Show feature importance calculation
- [x] Include plotting examples
- [x] Add more complex, real-world examples

## 18. Command Line Interface

### 18.1 CLI Implementation
- [x] Implement basic CLI that reports version
- [x] Add more advanced CLI functionality for model fitting and evaluation
- [x] Support for loading/saving models via CLI
- [x] Batch prediction capabilities via CLI

## 19. Advanced Features

### 19.1 Feature Engineering
- [x] Support for categorical feature encoding
- [x] Missing value imputation strategies
- [ ] Advanced feature selection methods
- [ ] Feature scaling and normalization options

### 19.2 Model Interpretability
- [x] Feature importance calculation
- [x] Partial dependence plots
- [x] Individual conditional expectation plots
- [x] Model explanation tools

## 20. CI/CD and Automation

### 20.1 Code Quality Tools
- [x] Set up Ruff for formatting and linting
- [x] Configure MyPy for type checking
- [x] Set up pre-commit hooks for automated checks
- [x] Configure tox for multi-Python testing
- [x] Set up comprehensive pytest configuration

### 20.2 Testing Infrastructure
- [x] Implement test coverage checking with >90% requirement
- [x] Set up coverage reporting and analysis tools
- [x] Create scripts for coverage analysis by file
- [x] Integrate with CI/CD for automated coverage checks
- [x] Add property-based testing with Hypothesis
- [x] Add mutation testing setup with Mutmut
- [x] Add fuzz testing framework

### 20.3 CI/CD Pipelines
- [x] Set up GitHub Actions for CI/CD
- [x] Configure testing across multiple Python versions
- [x] Set up automated linting and formatting checks
- [x] Implement type checking in CI pipeline
- [x] Set up security scanning workflow
- [x] Configure performance benchmarking workflow
- [x] Set up documentation building and deployment
- [x] Add code quality checks with reviewdog
- [x] Add automated dependency updates with Dependabot
- [x] Add release workflow with GitHub Releases

### 20.4 Release Management
- [x] Configure automated PyPI releases
- [x] Set up release notes generation
- [x] Configure semantic versioning automation
- [x] Add beta release to TestPyPI
- [x] Add GitHub release creation

### 20.5 Project Management
- [x] Set up issue templates (bug reports, feature requests)
- [x] Create pull request templates
- [x] Configure standard labels for issues
- [x] Set up CODEOWNERS for automated review assignment
- [x] Create comprehensive development guidelines document
- [x] Create changelog file to track releases
- [x] Add commit message conventions
- [x] Add pull request checklist

### 20.6 Security and Compliance
- [x] Implement security scanning with Bandit
- [x] Set up dependency security checking with Safety
- [x] Configure automated dependency updates with Dependabot
- [x] Add vulnerability scanning in CI pipeline

## 21. Future Enhancements

### 21.1 Performance
- [x] Implement performance benchmarking with pytest-benchmark
- [ ] Potential caching mechanisms for repeated computations
- [ ] Parallel processing for basis function evaluation
- [ ] Sparse matrix support for large datasets

### 21.2 Features
- [ ] Additional feature importance methods
- [ ] Model interpretability tools
- [ ] Advanced cross-validation strategies
- [ ] Support for additional GLM families

---

*This tasks document tracks all requirements and features for the pymars project, with completed items marked as [x]. The document will be updated as development progresses.*