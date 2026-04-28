# Rust Core Architecture

`mars` is moving toward a shared Rust computational core with language bindings
for Python, R, Julia, Rust, C#, Go, and TypeScript.

The current Python implementation remains the compatibility baseline for the
public API. The existing `rust-runtime` crate is the first executable consumer
of the portable `ModelSpec` contract and should evolve into the shared core.

## Ownership Model

The Rust core owns behavior that must be identical across language bindings:

- `ModelSpec` parsing and validation
- basis-term representation
- basis-matrix evaluation
- prediction from coefficients and evaluated basis terms
- shared numerical error semantics
- eventually forward-pass candidate evaluation, pruning, and fitted-model export

The Python package owns behavior that is specific to the Python ecosystem:

- scikit-learn estimator classes
- Python import compatibility such as `import pymars as earth`
- NumPy/pandas input normalization
- plotting and explanation helpers
- Python CLI ergonomics

Each additional binding owns its host-language API shape, packaging, and
conversion into the Rust core boundary.

## Module Migration Order

The Rust core should absorb functionality in this order:

1. Stable runtime replay: validation, `design_matrix`, and `predict`.
2. Runtime error typing and fixture-backed conformance harness.
3. Basis-term evaluation internals currently mirrored in Python.
4. Rust-backed runtime bindings for Python, R, Julia, Rust, C#, Go, and TypeScript.
5. Fitted-model export/import normalization.
6. Forward-pass candidate scoring.
7. Pruning and GCV selection.
8. Full fitting orchestration.
9. Cross-language training APIs where each ecosystem can support them safely.

This order keeps the public Python API stable while moving shared deterministic
behavior into Rust behind existing tests.

## ModelSpec Boundary

`ModelSpec` remains the artifact contract across packages and languages.

Rust structs are implementation types for that contract; they are not allowed to
change artifact semantics independently of the documented `spec_version` rules.
Host-language wrappers should convert their native table/vector inputs into the
Rust core's row-major evaluation input and should return native outputs without
changing prediction semantics.

## Error Semantics

All bindings should expose the same error categories, even if host-language
exception types differ:

- malformed artifact
- unsupported artifact version
- missing required field
- unsupported basis term
- feature-count mismatch
- invalid categorical encoding
- numerical evaluation failure

Rust should define these categories first. Bindings should translate them into
idiomatic errors while preserving category names and actionable messages.

## Minimum Rust Crate API

The stable Rust core API should expose at least:

```rust
load_model_spec_str(raw: &str) -> Result<ModelSpec>
load_model_spec_path(path: impl AsRef<Path>) -> Result<ModelSpec>
validate_model_spec(spec: &ModelSpec) -> Result<()>
design_matrix(spec: &ModelSpec, rows: &[Vec<f64>]) -> Result<Vec<Vec<f64>>>
predict(spec: &ModelSpec, rows: &[Vec<f64>]) -> Result<Vec<f64>>
```

As fitting migrates, the core should add separate APIs for training inputs and
fitted artifact export rather than overloading the replay API.

## Binding Boundary

Bindings should be thin. They should:

- handle package installation and native data conversion
- call the Rust core for shared computation
- expose idiomatic names and types for the host ecosystem
- run the shared conformance fixtures in CI

Bindings should not reimplement basis semantics, model validation, or prediction
math independently unless they are explicitly experimental or pre-release.
Stable package publication is blocked until runtime bindings use the Rust core
or the remaining fallback is documented and tested.
