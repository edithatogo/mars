# Design

The current architecture is moving toward four explicit layers:

1. Python fit engine
2. Portable `ModelSpec`
3. Rust core
4. Language bindings

The short-term state is that Python remains the canonical training
implementation and Rust is already validating portable replay behavior against
the shared fixture corpus. The strategic direction is to move shared fitting and
runtime evaluation into a Rust core, then surface that core through Python, R,
Julia, Rust, C#, Go, and TypeScript bindings.

## Current target architecture

### 1. Current Python training layer

This is the existing Python `Earth` implementation:

- forward pass
- pruning pass
- sklearn-compatible wrappers
- plotting and explainability helpers

This remains the compatibility baseline for the Python package while the Rust
core is developed. Its public API should be preserved even as internals move.

### 2. Portable model layer

The portable contract is the JSON `ModelSpec`.

It should capture:

- hyperparameters required for replay
- feature schema
- fitted basis terms
- learned coefficients
- categorical preprocessing state
- minimal metrics and metadata

This layer is the most important dependency boundary for future bindings.

### 3. Rust core layer

The Rust core is the intended long-term implementation home for shared MARS
behavior:

- artifact validation
- basis-matrix evaluation
- prediction
- eventually forward pass, pruning, and fitting

The current Rust runtime is a reference replay prototype, not yet the complete
training core.

### 4. Binding layer

Bindings should be thin wrappers around the Rust core, shaped idiomatically for
each host ecosystem while sharing fixture-backed conformance tests.

Target bindings:

- Python
- R
- Julia
- Rust
- C#
- Go
- TypeScript

### Current Python runtime entry points

Current public runtime entry points:

- `pymars.load_model_spec(...)`
- `pymars.load_model(...)`
- `pymars.save_model(...)`
- `pymars.predict(...)`
- `pymars.design_matrix(...)`
- `pymars.inspect(...)`

## Direction for Broader Language Support

The most credible path is to turn the Rust replay prototype into a shared Rust
core and expose that core through language-specific bindings. Existing MARS
packages are useful historical/API references, not implementation dependencies
or runtime oracles.

Recommended sequence:

1. Freeze the `ModelSpec` schema and compatibility policy.
2. Keep Python as the compatibility baseline while the Rust core matures.
3. Keep the Rust reference runtime passing against the JSON fixture corpus.
4. Move replay hot paths and validation into the Rust core behind the Python API.
5. Move fitting internals, including forward pass and pruning, into Rust.
6. Add bindings in priority order: Rust crate, Python extension, R, Julia, Go,
   C#, and TypeScript/WASM or native package.
7. Require every binding to pass shared fixture and API conformance tests.

## Recommended hardening priorities

- preserve duplication-equivalent weighted fitting semantics as the runtime and API layers expand
- make unsupported surfaces explicit and tested
- keep Python fitted-model regression fixtures broad enough to catch accidental algorithm drift
- keep portable `ModelSpec` fixtures synchronized with every supported runtime
- define the Rust core crate boundaries before adding broad language bindings
- add binding-level conformance tests for Python, R, Julia, Rust, C#, Go, and TypeScript
- add versioned persistence compatibility tests
- mark experimental APIs clearly where behavior is still evolving
