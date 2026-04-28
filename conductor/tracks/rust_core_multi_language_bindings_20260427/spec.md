# Specification

## Objective

Move `mars` from a Python implementation with a Rust replay prototype toward a
shared Rust computational core surfaced through Python, R, Julia, Rust, C#, Go,
and TypeScript APIs.

## Scope

- Define the Rust core crate boundary for validation, basis evaluation,
  prediction, and fitting.
- Preserve the existing Python/scikit-learn public API while routing shared
  computation through the Rust core where practical.
- Define binding architecture and packaging expectations for Python, R, Julia,
  Rust, C#, Go, and TypeScript.
- Create shared conformance fixtures that every binding must pass.
- Decide the order in which runtime replay and fitting internals move from
  Python into Rust.

## Out of Scope

- Replacing the public Python estimator API.
- Depending on `py-earth` or R `earth` for implementation or validation.
- Shipping every production binding in the first phase.
- Abandoning the `ModelSpec` compatibility contract.

## Acceptance Criteria

- A Rust core architecture document defines crate boundaries, FFI/package
  boundaries, error semantics, and ownership of the `ModelSpec` contract.
- The existing Rust replay prototype has a migration plan toward the shared
  core.
- Python remains the compatibility baseline and has a clear integration plan for
  the Rust core.
- Binding plans exist for Python, R, Julia, Rust, C#, Go, and TypeScript.
- Shared fixture and conformance-test requirements are documented for all
  bindings.
