# Specification

## Objective

Define the stable portability boundary for serialized MARS models so the Rust
core and downstream language bindings can be built against a deliberate
contract instead of an implementation accident.

## Scope

- Define schema migration and versioning rules for future spec revisions.
- Define the smallest stable runtime surface needed for the Rust core and language bindings.
- Define a minimal C-ABI-neutral core boundary suitable for embedded or foreign-language consumers.
- Specify a language-neutral core API centered on `ModelSpec` and basis-matrix evaluation.
- Decide whether the runtime remains in `pymars` or moves to a split runtime package.

## Out of Scope

- Shipping production-quality language bindings.
- Performance benchmarking of every target language.
- ONNX or interchange implementation work beyond contract-level prerequisites.

## Current State Constraints

The repository already serializes fitted models into a JSON-friendly `ModelSpec`,
but portable evaluation is still defined by Python implementation details rather
than by an independent contract.

### Python-Coupled Assumptions That Must Be Designed Away

1. Runtime replay reconstructs a Python `Earth` estimator instead of evaluating
   directly from a language-neutral artifact.
2. Basis terms are reified as Python classes (`BasisFunction`,
   `HingeBasisFunction`, categorical variants, and companions) and evaluated
   through Python method dispatch.
3. Public runtime helpers such as `predict(...)` and `design_matrix(...)`
   depend on Python-private estimator methods like `_prepare_prediction_data`
   and `_build_basis_matrix`.
4. Replay semantics assume NumPy array behavior:
   coefficients are restored as NumPy arrays, feature names are restored as an
   object array, and basis transforms expect `np.ndarray` inputs.
5. Categorical preprocessing replay depends on Python and scikit-learn
   components (`CategoricalImputer`, `LabelEncoder`, and object-array coercion
   rules) rather than on a standalone categorical contract.
6. Validation and prediction flow still assume scikit-learn estimator behavior,
   including Python-side shape normalization and error handling.
7. The serialized payload includes enough information to reconstruct the Python
   model, but it does not yet specify language-neutral evaluation semantics for
   basis activation, categorical handling, dtype coercion, or failure behavior.

### Implication for This Track

This track must define the portability boundary in terms of stable artifact
semantics, runtime inputs/outputs, and compatibility rules. It must not treat
the current Python replay path as the contract.

## Versioning and Compatibility Rules

### Contract Authority

`spec_version` is the compatibility contract for portable model artifacts. It
is authoritative for readers and writers across languages. Package versions such
as `pymars.__version__` or any future foreign-runtime package version are not
the artifact compatibility authority.

`module_version`, if present, is informational provenance only. Readers must not
use it to decide whether an artifact is compatible.

### Version Format

Portable artifacts must encode `spec_version` as a string in `<major>.<minor>`
format.

Examples:

- valid: `1.0`
- valid: `1.3`
- invalid: `1`
- invalid: `v1.0`
- invalid: `1.0.0`

Artifacts with malformed version strings fail validation immediately.

### Current Compatibility Line

The current portable artifact line is `1.x`.

- Writers in the current implementation emit `spec_version = "1.0"`.
- Readers may accept any well-formed `1.x` artifact only if all required fields
  and semantic invariants are satisfied.
- Readers must reject `2.x` and all other unsupported major versions unless an
  explicit migration or compatibility implementation is added.

This defines a fail-closed major-version boundary.

### Major and Minor Semantics

Major versions define structural or semantic compatibility boundaries.

- A major bump is required for any change that would cause an existing
  conforming `1.x` reader to misinterpret the artifact, produce different
  predictions for the same semantics, or require new mandatory behavior.
- Minor versions are reserved for additive, backward-compatible changes within a
  major line.

Allowed `1.x` minor-line evolution:

- adding optional top-level fields
- adding optional nested fields
- tightening documentation without changing artifact meaning
- clarifying error handling where existing valid artifacts remain valid

Disallowed within `1.x`:

- removing required fields
- renaming required fields
- changing the meaning of an existing field
- changing coefficient ordering or basis-term ordering semantics
- introducing new required basis-term behavior that existing `1.x` readers
  cannot safely ignore

### Required Fields and Invariants

Every conforming portable artifact must contain the following top-level fields:

- `params`
- `feature_schema`
- `basis_terms`
- `coefficients`

Current `1.x` validation invariants:

- the artifact payload is an object/map
- `params` is an object/map
- `feature_schema` is an object/map
- `basis_terms` is an array
- `coefficients` is an array
- `feature_schema.n_features`, when present, is null or a non-negative integer
- each basis term has a non-empty `kind`
- there is exactly one coefficient per basis term

Any artifact missing required fields or violating these invariants fails
validation immediately.

### Unknown Fields

Readers must ignore unknown optional fields within a supported major version
unless those fields are declared mandatory by a future major-version contract.

This rule exists to allow additive evolution in `1.x` without forcing lockstep
reader upgrades.

### Failure Semantics

Readers must fail deterministically and before evaluation when:

- `spec_version` is missing
- `spec_version` is malformed
- the artifact major version is unsupported
- a required field is missing
- a required field has the wrong shape or type
- basis-term and coefficient cardinalities do not match

These failures are contract-level validation failures, not runtime
prediction-time behavior.

## Runtime Boundary Definition

### Stable Runtime Entry Points

Portable evaluation is defined in terms of three stable operations:

1. `validate(artifact) -> validated_artifact`
2. `design_matrix(validated_artifact, X) -> matrix`
3. `predict(validated_artifact, X) -> y_pred`

These operations define the portable contract. The Rust core and any conforming
language binding must be able to implement them directly from the serialized
artifact.

A conforming runtime must not require:

- importing `pymars`
- instantiating `Earth`
- replaying Python basis-function classes
- calling Python-private estimator helpers such as `_prepare_prediction_data`
  or `_build_basis_matrix`

The current Python replay path may continue to exist as one implementation
strategy inside `pymars`, but it is not the portability contract.

### Input Data Contract

Runtime consumers must interpret prediction input `X` as a rank-2 row-major
table with shape `(n_samples, n_features)`.

`1.x` contract rules:

- `X` must provide exactly one value per feature position for every sample row.
- If `feature_schema.n_features` is not null, it must equal `X.shape[1]`.
- Zero samples are allowed if the feature count is valid.
- Feature-count mismatch is a validation failure.
- Feature order is authoritative in `1.x`; feature names are metadata and must
  not be used to reorder input columns automatically.

### Dtype and Value Semantics

The portable contract distinguishes three semantic value classes:

1. continuous or numeric values
2. categorical or string-like values
3. missing values

`1.x` runtimes must follow these rules:

- Continuous features must be coercible to the runtime's finite numeric
  representation before basis evaluation.
- Categorical features must be interpreted according to serialized categorical
  state, not according to host-language defaults or scikit-learn objects.
- Missing values must be handled according to serialized basis semantics and
  missingness terms, not according to incidental host-language coercion rules.
- A runtime may choose its own internal scalar representation, but that choice
  must not change artifact meaning.

This track does not require a specific wire encoding for missing values beyond
the existing structured artifact model. It does require that portable runtimes
treat missingness as explicit contract semantics rather than as Python/NumPy
implementation behavior.

### Output Contract

`design_matrix(validated_artifact, X)` must return a dense matrix with shape
`(n_samples, n_basis_terms)`.

`1.x` output rules:

- Basis columns are ordered exactly as serialized in `basis_terms`.
- There is exactly one design-matrix column per serialized basis term.
- `predict(validated_artifact, X)` returns a length-`n_samples` prediction
  vector.
- Predictions are computed by applying `coefficients` in serialized order to
  the evaluated design matrix.
- No additional estimator-side postprocessing is part of the `1.x` portable
  contract unless introduced by a future major version.

### Embedded and Foreign-Language Boundary Constraints

The portability boundary must remain embeddable and C-ABI-neutral.

A conforming portable artifact and runtime contract must not require:

- Python object graphs
- pickled state
- NumPy-specific container identities
- scikit-learn transformers or estimator instances
- callback-based evaluation into Python for every basis term

Artifacts must remain representable as plain structured data:

- objects/maps
- arrays/lists
- primitive scalars
- nulls

This ensures that the Rust core can act as the shared implementation and can be
surfaced through Python, R, Julia, Rust, C#, Go, TypeScript, or a C-compatible
host without needing Python reconstruction as an execution dependency.

## Packaging Decision

For the `1.x` contract line, the portability contract remains authored and
published from `pymars`.

A separate runtime package is not required to define or stabilize the `1.x`
contract. Future Rust-core-backed bindings may be shipped as separate packages,
but they must consume the same `ModelSpec` contract defined here.

### Rationale

1. `pymars` is currently the only writer and the only complete implementation
   of artifact production semantics.
2. The immediate problem is semantic portability, not distribution topology.
3. A split Rust core package remains viable later, but creating all binding
   packages now would add packaging surface area before the core contract is
   fully exercised.

### Maintenance Implication

`pymars` remains the contract authority for `1.x`.

Any future split Rust core package is a downstream consumer of the contract, not
a co-equal source of artifact truth. If a future split package becomes
necessary, that packaging decision can be revisited without changing `1.x`
artifact semantics.

## Portable Contract Summary

Downstream `1.x` implementers may rely on the following:

1. Portable artifacts are read as plain structured data.
2. Readers validate `spec_version`, required fields, feature-count invariants,
   and basis-term/coefficient cardinality before evaluation.
3. Runtime inputs are interpreted positionally as `(n_samples, n_features)`.
4. Basis terms are evaluated directly into a dense design matrix ordered exactly
   as `basis_terms`.
5. Predictions are produced by applying `coefficients` in serialized order to
   that design matrix.
6. Unsupported major versions and malformed artifacts fail deterministically.
7. Portable consumers and Rust-core bindings do not depend on Python reconstruction, NumPy container
   identity, or scikit-learn preprocessing objects.

## Acceptance Criteria

- Portability/versioning rules are written down and testable.
- The minimum runtime contract is expressed in terms of stable inputs, outputs, and error semantics.
- The package-boundary decision is documented with explicit tradeoffs.
- The resulting contract is sufficient to guide prototype consumers in later tracks.
