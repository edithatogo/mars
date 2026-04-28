# Training Orchestration Inventory

## Overview

This document maps Python training orchestration to Rust equivalents and identifies the Rust training API boundary for the MARS training core migration.

---

## Python Forward-Pass Orchestration

### Main Entry Point
- **File**: `pymars/earth.py`
- **Class**: `Earth` (inherits `sklearn.base.BaseEstimator`, `sklearn.base.RegressorMixin`)
- **Method**: `Earth.fit(X, y, sample_weight=None) -> Earth`
  - Input validation via `sklearn.utils.validation.check_X_y`
  - Data preprocessing via `_scrub_input_data()`
  - Initializes `EarthRecord` for logging
  - Runs forward pass via `ForwardPasser.run()`
  - Runs pruning via `PruningPasser.run()`
  - Computes final metrics (RSS, MSE, GCV)

### ForwardPasser Class
- **File**: `pymars/_forward.py`
- **Class**: `ForwardPasser`

| Python Method | Purpose | Rust Equivalent |
|--------------|---------|-----------------|
| `run(X_fit_processed, y_fit, missing_mask, X_fit_original, sample_weight)` | Orchestrates forward pass, adds terms iteratively | ❌ Not yet implemented |
| `_generate_candidates()` | Generates hinge pairs, linear terms, categorical indicators, missingness indicators | ❌ Not yet implemented |
| `_find_best_candidate_addition()` | Evaluates candidates, selects best by GCV/RSS | ❌ Not yet implemented |
| `_get_allowable_knot_values()` | Computes valid knot positions with minspan/endspan filtering | ❌ Not yet implemented |
| `_calculate_rss_and_coeffs()` | Computes coefficients via `np.linalg.lstsq`, calculates RSS | ✅ `fit_least_squares` in `training.rs` |
| `_build_basis_matrix()` | Evaluates all basis functions on input data | ✅ `design_matrix` in `runtime.rs` |
| `_calculate_gcv_for_basis_set()` | Computes GCV using utility functions | ✅ `calculate_gcv` in `training.rs` |

### Hinge Function Implementation
- **File**: `pymars/_basis.py`
- **Class**: `HingeBasisFunction(BasisFunction)`
  - `transform(X_processed, missing_mask)`: Evaluates `max(0, x - knot)` or `max(0, knot - x)`
  - Handles missing value propagation via missing_mask

---

## Python Pruning Orchestration

### PruningPasser Class
- **File**: `pymars/_pruning.py`
- **Class**: `PruningPasser`

| Python Method | Purpose | Rust Equivalent |
|--------------|---------|-----------------|
| `run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_basis_functions, initial_coefficients, sample_weight)` | Orchestrates pruning, removes terms iteratively | ❌ Not yet implemented |
| `_compute_gcv_for_subset()` | Computes GCV, pruning RSS, coefficient refit for subset | ✅ `score_pruning_subset` in `training.rs` |
| `_calculate_rss_and_coeffs()` | Computes coefficients via least squares, pruning RSS, coefficient refit | ✅ `fit_least_squares` in `training.rs` |
| `_build_basis_matrix()` | Constructs basis matrix for pruning | ✅ `design_matrix` in `runtime.rs` |

---

## Coefficient Fitting

### Python Implementation
- **Location**: `pymars/_forward.py` and `pymars/_pruning.py`
- **Method**: `np.linalg.lstsq` for ordinary least squares
- **Weighted fitting**: Scale B and y by sqrt(sample_weight)
- **Intercept-only**: Weighted mean of y

### Rust Equivalent
- **File**: `rust-runtime/src/training.rs`
- **Function**: `fit_least_squares(design, y, sample_weight, drop_nan_rows) -> LeastSquaresFit`
  - Gaussian elimination with partial pivoting
  - Handles NaN values (drop or zero)
  - Returns RSS, coefficients, effective_n_samples

---

## Export Code (ModelSpec)

### Python Implementation
- **File**: `pymars/_model_spec.py`
- **Version**: `1.0` (MODEL_SPEC_VERSION)

| Python Function/Method | Purpose | Rust Equivalent |
|------------------------|---------|-----------------|
| `model_to_spec(model) -> dict` | Converts Earth model to portable dict | ❌ Not yet implemented |
| `basis_function_to_spec(bf) -> BasisTermSpec` | Serializes basis function | ❌ Not yet implemented |
| `spec_to_json(spec) -> str` | Converts spec dict to JSON | ✅ `serde_json` serialization |
| `spec_to_model(payload, earth_cls) -> Earth` | Rebuilds Earth from spec | ❌ Not yet implemented |
| `Earth.get_model_spec()` | Returns model spec dict | ❌ Not yet implemented |
| `Earth.export_model(format)` | Returns dict or JSON | ✅ `ModelSpec` serializable |

### Earth Class Export Methods
- `get_model_spec()`: Calls `model_to_spec(self)`
- `export_model(format="json")`: Returns spec as dict or JSON string
- `from_model(payload)`: Class method calling `spec_to_model(payload, cls)`

---

## Rust Primitives (Currently Implemented)

### Module Structure
- **Location**: `rust-runtime/src/`
- **Modules**: `lib.rs`, `model_spec.rs`, `runtime.rs`, `training.rs`, `errors.rs`

### Training Primitives in `training.rs`

| Rust Function | Purpose | Python Equivalent |
|--------------|---------|-------------------|
| `fit_least_squares(design, y, sample_weight, drop_nan_rows)` | Weighted least squares with NaN handling | `_calculate_rss_and_coeffs()` |
| `calculate_gcv(rss, num_samples, num_effective_params)` | GCV score computation | `calculate_gcv()` in `_util.py` |
| `effective_parameters(num_terms, num_hinge_terms, penalty)` | Effective parameter count | `gcv_penalty_cost_effective_parameters()` in `_util.py` |
| `score_candidate(design, candidate_columns, y, sample_weight, penalty, num_hinge_terms)` | Scores candidate term addition | `_find_best_candidate_addition()` |
| `score_pruning_subset(design, columns, y, sample_weight, penalty, num_hinge_terms)` | Scores pruning subset | `_compute_gcv_for_subset()` |

### Return Structures
```rust
pub struct LeastSquaresFit {
    pub rss: f64,
    pub coefficients: Vec<f64>,
    pub effective_n_samples: f64,
}

pub struct CandidateScore {
    pub rss: f64,
    pub gcv: f64,
    pub coefficients: Vec<f64>,
    pub effective_n_samples: f64,
}
```

---

## Rust ModelSpec Structure

### Defined in `model_spec.rs`

```rust
pub struct ModelSpec {
    pub spec_version: String,           // "1.0"
    pub params: Value,                  // serde_json::Value (hyperparameters)
    pub feature_schema: FeatureSchema,  // feature metadata
    pub basis_terms: Vec<BasisTermSpec>, // basis functions
    pub coefficients: Vec<f64>,         // model coefficients
}

pub struct FeatureSchema {
    pub n_features: Option<usize>,
    pub feature_names: Vec<String>,
}

pub struct BasisTermSpec {
    pub kind: String,                   // "constant", "linear", "hinge", etc.
    pub variable_idx: Option<usize>,
    pub variable_name: Option<String>,
    pub knot_val: Option<f64>,
    pub is_right_hinge: Option<bool>,
    pub category: Option<Value>,
    pub gcv_score: Option<f64>,
    pub rss_score: Option<f64>,
    pub parent1: Option<Box<BasisTermSpec>>,
    pub parent2: Option<Box<BasisTermSpec>>,
}
```

### Supported Basis Term Kinds
- `"constant"` - intercept term
- `"linear"` - linear term
- `"hinge"` - hinge function
- `"categorical"` - categorical indicator
- `"interaction"` - product of two parents
- `"missingness"` - missing value indicator

---

## Runtime API (Already Implemented in Rust)

### Public Functions in `runtime.rs`

```rust
pub fn load_model_spec_str(raw: &str) -> MarsResult<ModelSpec>
pub fn load_model_spec_path(path: impl AsRef<Path>) -> MarsResult<ModelSpec>
pub fn validate_model_spec(spec: &ModelSpec) -> MarsResult<()>
pub fn design_matrix(spec: &ModelSpec, rows: &[Vec<f64>]) -> MarsResult<Vec<Vec<f64>>>
pub fn predict(spec: &ModelSpec, rows: &[Vec<f64>]) -> MarsResult<Vec<f64>>
pub fn evaluate_basis(basis: &BasisTermSpec, row: &[f64]) -> MarsResult<f64>
```

---

## Unsupported Edge Cases and Gaps

### Python Features Not Yet in Rust

1. **Forward-Pass Orchestration**
   - ❌ Candidate generation with interaction terms
   - ❌ Knot selection with minspan/endspan constraints
   - ❌ Deterministic tie handling
   - ❌ Stopping rules (max_terms, threshold)

2. **Pruning Orchestration**
   - ❌ Iterative term removal loop
   - ❌ GCV-based subset selection
   - ❌ Final model selection logic

3. **Missing Value Handling**
   - Python: Uses missing_mask to propagate NaN through basis evaluation
   - Rust: `design_matrix` doesn't handle missingness indicators yet
   - Gap: Need to add missingness basis evaluation to Rust

4. **Categorical Feature Handling**
   - Python: `CategoricalImputer` in `_categorical.py`
   - Rust: No categorical encoding in Rust runtime
   - Gap: Categorical support is Python-side only

5. **Feature Importance**
   - Python: `_calculate_feature_importances()` in `earth.py`
   - Rust: Not implemented
   - Gap: Post-training analysis is Python-only

6. **Sample Weight Edge Cases**
   - Python: Handles uniform weights when None
   - Rust: Requires explicit `sample_weight` parameter
   - Gap: Need default weight handling in Rust

---

## Selected Rust Training API Boundary

### Proposed Public API for Rust Training Core

The following functions will be added to `rust-runtime/src/training.rs` and re-exported in `lib.rs`:

```rust
/// Fits a MARS model end-to-end and returns a ModelSpec
pub fn fit_model(
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    params: &TrainingParams,
) -> MarsResult<ModelSpec>

/// Parameters for training
pub struct TrainingParams {
    pub max_terms: usize,
    pub max_degree: usize,
    pub penalty: f64,
    pub minspan: f64,
    pub endspan: f64,
    pub threshold: f64,
    pub feature_names: Option<Vec<String>>,
}

/// Forward-pass orchestration (internal, but exposed for testing)
pub fn forward_pass(
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    params: &TrainingParams,
) -> MarsResult<ForwardPassResult>

/// Pruning orchestration (internal, but exposed for testing)
pub fn prune_model(
    x: &[Vec<f64>],
    y: &[f64],
    sample_weight: Option<&[f64]>,
    basis_terms: &[BasisTermSpec],
    coefficients: &[f64],
    params: &TrainingParams,
) -> MarsResult<PruningResult>

pub struct ForwardPassResult {
    pub basis_terms: Vec<BasisTermSpec>,
    pub coefficients: Vec<f64>,
    pub rss: f64,
    pub gcv: f64,
}

pub struct PruningResult {
    pub basis_terms: Vec<BasisTermSpec>,
    pub coefficients: Vec<f64>,
    pub rss: f64,
    pub gcv: f64,
}
```

### API Boundary Decision

The Rust training core will:
1. Accept normalized training inputs (f64 matrices)
2. Accept hyperparameters via `TrainingParams`
3. Accept optional sample weights
4. Return a `ModelSpec` compatible with existing runtime
5. Preserve deterministic behavior for identical inputs

The Python `Earth.fit()` will:
1. Continue to handle input validation and preprocessing
2. Optionally route to Rust `fit_model()` via feature gate
3. Fall back to Python implementation if Rust is unavailable
4. Maintain sklearn compatibility

---

## Files Referenced

| Component | File Path |
|-----------|-----------|
| Earth Estimator | `pymars/earth.py` |
| Forward Pass | `pymars/_forward.py` |
| Pruning Pass | `pymars/_pruning.py` |
| Basis Functions | `pymars/_basis.py` |
| Model Spec Export | `pymars/_model_spec.py` |
| Utilities | `pymars/_util.py` |
| Training Record | `pymars/_record.py` |
| Categorical Imputer | `pymars/_categorical.py` |
| Rust Runtime Lib | `rust-runtime/src/lib.rs` |
| Rust Model Spec | `rust-runtime/src/model_spec.rs` |
| Rust Runtime API | `rust-runtime/src/runtime.rs` |
| Rust Training Primitives | `rust-runtime/src/training.rs` |
| Rust Errors | `rust-runtime/src/errors.rs` |
| Rust Training Tests | `rust-runtime/tests/training_tests.rs` |
| Training Fixture | `tests/fixtures/training_core_fixture_v1.json` |
