# Design for mars: Pure Python Earth (Multivariate Adaptive Regression Splines)

## Overview

This document describes the design of mars, a pure Python implementation of Multivariate Adaptive Regression Splines (MARS) that maintains full compatibility with the scikit-learn ecosystem. The design follows the requirements specified in requirements.md and implements a scikit-learn compliant API while providing feature parity with the py-earth library.

## System Architecture

### 2.1 Component Structure

The mars library is organized into the following components:

```
mars/
├── earth.py              # Core Earth model implementation
├── __init__.py           # Package interface
├── _basis.py             # Basis function definitions
├── _forward.py           # Forward pass implementation
├── _pruning.py           # Pruning pass implementation
├── _record.py            # Recording and logging utilities
├── _sklearn_compat.py    # Scikit-learn compatibility layer
├── _categorical.py       # Categorical feature handling
├── _missing.py           # Missing value handling
├── _util.py              # Utility functions
├── glm.py                # Generalized linear models
├── cv.py                 # Cross-validation utilities
├── plot.py               # Plotting and visualization
├── cli.py                # Command-line interface
└── demos/                # Example implementations
```

### 2.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Public Interface                        │
├─────────────────────────────────────────────────────────────┤
│ earth.Earth        ←  Core MARS Implementation             │
│ _sklearn_compat.   ←  Scikit-learn Adapters               │
│   EarthRegressor, EarthClassifier                           │
│ glm.GLMEarth       ←  Generalized Linear Models            │
│ cv.EarthCV         ←  Cross-Validation Helper              │
│ plot.*             ←  Visualization Tools                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core Algorithm Layer                      │
├─────────────────────────────────────────────────────────────┤
│ _forward.ForwardPasser    ← Forward Selection               │
│ _pruning.PruningPasser    ← Backward Elimination            │
│ _basis.*                  ← Basis Function System          │
│ _record.EarthRecord       ← Training Process Tracking       │
│ _util.*                   ← Supporting Utilities           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Preprocessing Layer                  │
├─────────────────────────────────────────────────────────────┤
│ _categorical.CategoricalImputer  ← Categorical Features     │
│ _missing.*                       ← Missing Value Handling   │
└─────────────────────────────────────────────────────────────┘
```

## Core Component Design

### 3.1 Earth Class Design

**Purpose**: The main MARS algorithm implementation that coordinates the forward and backward passes.

**Signature**:
```python
class Earth(BaseEstimator, RegressorMixin):
    def __init__(self, max_degree=1, penalty=3.0, max_terms=None,
                 minspan_alpha=0.0, endspan_alpha=0.0,
                 minspan=-1, endspan=-1,
                 allow_linear=True, allow_missing=False,
                 feature_importance_type=None,
                 categorical_features=None):
```

**Design Details**:
- Inherit from `sklearn.base.BaseEstimator` and `RegressorMixin` for scikit-learn compatibility
- Store all parameters as public attributes (with trailing underscore for fitted attributes)
- Implement `fit`, `predict`, `score` methods as required by scikit-learn
- Use `_scrub_input_data` helper for preprocessing and validation
- Maintain `record_`, `basis_`, and `coef_` as fitted attributes

**Dependencies**:
- `_forward.ForwardPasser` for forward selection
- `_pruning.PruningPasser` for backward elimination
- `_basis.ConstantBasisFunction` for intercept handling
- `_util` for GCV calculations

### 3.2 Forward Pass Design

**Component**: `_forward.py` → `ForwardPasser` class

**Purpose**: Implements the forward selection phase that builds basis functions incrementally.

**Algorithm**:
```
1. Start with intercept-only model
2. For each iteration:
   a. Evaluate all possible next basis functions
   b. Select the one that gives maximum RSS reduction
   c. Add to the model
   d. Check stopping conditions:
      - Maximum number of terms reached
      - No significant RSS improvement
      - Maximum degree constraints
3. Return set of selected basis functions with coefficients
```

**Key Features**:
- Support for hinge and linear terms
- Interaction term generation up to `max_degree`
- Minspan and endspan controls for knot placement
- Efficient basis matrix construction using preallocation

### 3.3 Pruning Pass Design

**Component**: `_pruning.py` → `PruningPasser` class

**Purpose**: Implements the backward elimination phase using GCV criteria.

**Algorithm**:
```
1. Start with full model from forward pass
2. While model contains more than intercept:
   a. For each basis function, compute GCV if removed
   b. Find function whose removal minimizes GCV
   c. Remove that function
   d. Record pruned model and GCV score
3. Select model with minimum GCV across pruning path
```

**Key Features**:
- Generalized Cross-Validation for model selection
- Support for interaction terms during pruning
- Efficient GCV computation without matrix inversions

### 3.4 Basis Function System

**Component**: `_basis.py` → Multiple classes

**Hierarchy**:
```
BasisFunction (abstract)
├── ConstantBasisFunction (intercept)
├── LinearBasisFunction
├── HingeBasisFunction (left and right)
├── ProductBasisFunction (for interactions)
└── MissingnessBasisFunction
```

**Design Considerations**:
- Each basis function implements `transform(X, missing_mask)` method
- Support for missing value awareness through `missing_mask` parameter
- Efficient computation for large datasets
- Proper tracking of involved variables for feature importance

## Scikit-learn Compatibility Layer

### 4.1 EarthRegressor Design

**Purpose**: Scikit-learn compatible regressor wrapper for the core Earth model.

**Design**:
```python
class EarthRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, max_degree=1, penalty=3.0, ...):
        # Store parameters as public attributes
        self.max_degree = max_degree
        # ...
        
    def fit(self, X, y):
        # Validate inputs using sklearn utilities
        X, y = check_X_y(X, y)
        # Store feature information
        self.n_features_in_ = X.shape[1]
        # Create and fit core model
        self.earth_ = CoreEarth(**self.get_params())
        self.earth_.fit(X, y)
        # Copy fitted attributes
        self.basis_ = self.earth_.basis_
        self.coef_ = self.earth_.coef_
        # ...
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.earth_.predict(X)
```

### 4.2 EarthClassifier Design

**Purpose**: Scikit-learn compatible classifier that uses MARS basis functions as features for a secondary classifier.

**Design**:
```python
class EarthClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, ..., classifier=None):
        # If classifier is None, default to LogisticRegression
        self.classifier = classifier
        
    def fit(self, X, y):
        # Fit core Earth model using numeric version of y
        y_numeric = LabelEncoder().fit_transform(y)
        self.earth_.fit(X, y_numeric)
        
        # Transform X using basis functions
        X_transformed = self.earth_._build_basis_matrix(X, self.basis_, missing_mask)
        
        # Fit internal classifier on transformed features
        self.classifier_.fit(X_transformed, y)
        
    def predict(self, X):
        # Transform X using fitted basis functions
        X_transformed = self._transform_X_for_classifier(X)
        # Predict using internal classifier
        return self.classifier_.predict(X_transformed)
```

**Key Design Elements**:
- Uses MARS basis functions as feature engineering step
- Wraps scikit-learn classifier for final prediction
- Handles class label encoding/decoding internally
- Supports probability prediction if internal classifier supports it

## Advanced Features Design

### 5.1 Generalized Linear Models (GLM)

**Component**: `glm.py` → `GLMEarth` class

**Design**:
```python
class GLMEarth(Earth):
    def __init__(self, family="gaussian", ...):
        self.family = family
        super().__init__(...)
        
    def fit(self, X, y):
        # Override fit to use GLM-specific algorithm
        # Use iterative reweighted least squares (IRLS) approach
        # Adapt forward and pruning passes for GLM criteria
```

**Features**:
- Support for binomial (logistic) and Poisson families
- Canonical link functions for each family
- GLM-specific scoring and prediction methods

### 5.2 Cross-Validation Helper

**Component**: `cv.py` → `EarthCV` class

**Design**:
```python
class EarthCV(BaseEstimator):
    def __init__(self, ..., cv=5, scoring=None):
        # Inherits parameters from Earth
        # Add cross-validation specific parameters
        self.cv = cv
        self.scoring = scoring
        
    def fit(self, X, y):
        # Use sklearn.model_selection.GridSearchCV internally
        # Grid search over MARS hyperparameters
        grid_search = GridSearchCV(
            Earth(), 
            param_grid=self._get_param_grid(),
            cv=self.cv,
            scoring=self.scoring
        )
        grid_search.fit(X, y)
        # Store best estimator and parameters
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
```

### 5.3 Visualization System

**Component**: `plot.py` → Multiple plotting functions

**Design**:
```
plot_basis_functions(model, X, y=None)  # Visualize selected basis functions
plot_residuals(model, X, y)           # Model diagnostic plots
# Additional plot types as needed
```

**Features**:
- Built on matplotlib for consistency
- Support for various diagnostic visualizations
- Integration with model inspection methods

## Data Preprocessing Design

### 6.1 Categorical Feature Handling

**Component**: `_categorical.py` → `CategoricalImputer`

**Design**:
```python
class CategoricalImputer:
    def fit(self, X, categorical_features):
        # Identify and store categorical columns
        # Learn encoding strategies
        return self
        
    def transform(self, X):
        # Apply learned encoding to transform categorical features
        # Return numeric array suitable for Earth model
```

### 6.2 Missing Value Handling

**Component**: `_missing.py` and integrated into core classes

**Design**:
- Missing mask creation and storage
- Zero-filling strategy for missing values during basis computation
- Support controlled via `allow_missing` parameter
- Integration with basis function transforms

## Implementation Strategy

### 7.1 Core Algorithm Implementation

**Phase 1**: Forward pass implementation
- Implement basis function generation
- Implement RSS-based selection criterion
- Add interaction term support
- Add minspan/endspan controls

**Phase 2**: Pruning pass implementation
- Implement GCV-based model selection
- Develop efficient pruning algorithm
- Add support for interaction term removal

**Phase 3**: Integration and optimization
- Combine forward and pruning passes
- Optimize basis matrix construction
- Add feature importance calculations

### 7.2 Scikit-learn Compatibility

**Implementation Order**:
1. Core Earth class with scikit-learn inheritance
2. EarthRegressor wrapper
3. EarthClassifier wrapper
4. Cross-validation integration
5. Pipeline compatibility verification

### 7.3 Performance Optimizations

**Memory Management**:
- Preallocate basis matrices instead of using `np.hstack`
- Reuse temporary arrays where possible
- Minimize data copying during transformations

**Computational Efficiency**:
- Use vectorized NumPy operations
- Efficient GCV computation without matrix inversions
- Optimized basis function evaluation

## Quality Assurance Design

### 8.1 Testing Strategy

**Unit Tests**:
- Core algorithm components (_forward, _pruning, _basis)
- Scikit-learn compatibility methods
- Feature importance calculations
- Missing value and categorical handling

**Integration Tests**:
- End-to-end model fitting and prediction
- Scikit-learn pipeline integration
- Cross-validation helper functionality

**Compatibility Tests**:
- Scikit-learn estimator checks using `check_estimator`
- API compatibility with py-earth
- Feature parity verification

### 8.2 Validation Approach

**Numerical Correctness**:
- Compare results with reference implementations where possible
- Unit tests for specific algorithmic components
- Regression tests for bug fixes

**API Consistency**:
- Scikit-learn compliance verification
- Parameter validation consistency
- Error handling verification

## Configuration and Parameters

### 9.1 Key Configuration Parameters

**Model Complexity**:
- `max_degree`: Maximum interaction degree (int, default=1)
- `penalty`: GCV penalty parameter (float, default=3.0)
- `max_terms`: Maximum basis functions after forward pass (int or None)

**Knot Placement**:
- `minspan_alpha`, `endspan_alpha`: Control parameters for knot spacing
- `minspan`, `endspan`: Direct control parameters (when >=0)

**Algorithm Options**:
- `allow_linear`: Whether to include linear terms (bool, default=True)
- `allow_missing`: Whether to allow missing values (bool, default=False)
- `feature_importance_type`: Type of feature importance calculation

### 9.2 Default Values

**Design Philosophy**:
- Default values similar to py-earth where possible
- Conservative settings to prevent overfitting by default
- Parameters that balance model complexity and interpretability

## Extension Points

### 10.1 Plugin Architecture

**Visualization Extensions**:
- Plugin system for additional plotting functions
- Custom diagnostic plots based on model properties

**Model Extensions**:
- Support for additional GLM families
- Custom basis function types
- Alternative pruning criteria

### 10.2 Future Enhancements

**Performance**:
- Potential caching mechanisms for repeated computations
- Parallel processing for basis function evaluation
- Sparse matrix support for large datasets

**Features**:
- Additional feature importance methods
- Model interpretability tools
- Advanced cross-validation strategies

## Dependencies and Requirements

### 11.1 Core Dependencies

**Required**:
- NumPy for numerical computations
- Scikit-learn for base classes and utilities
- Matplotlib for visualization

**Optional**:
- Pandas for enhanced data handling (for estimator checks)

### 11.2 Version Compatibility

**Python**: 3.8+ (to support type hints and other modern features)
**NumPy**: Minimum version that supports required functionality
**Scikit-learn**: Compatible with recent stable versions

## Performance Characteristics

### 12.1 Time Complexity

**Forward Pass**: O(n_terms × n_samples × n_features) where n_terms is the number of terms in final model
**Pruning Pass**: O(n_terms²) for full pruning path
**Prediction**: O(n_samples × n_basis_functions) for new predictions

### 12.2 Space Complexity

**Model Storage**: O(n_basis_functions × n_features) for basis function definitions
**Intermediate Matrices**: O(n_samples × n_basis_functions) for basis matrices
**Memory Efficiency**: Preallocation prevents repeated memory allocation

---

*This design document serves as a blueprint for the mars implementation, ensuring all requirements from the requirements.md file are properly addressed while maintaining the pure Python approach.*