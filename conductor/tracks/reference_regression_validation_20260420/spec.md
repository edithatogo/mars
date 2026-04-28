# Specification

## Objective

Strengthen numerical confidence in `pymars` by locking down deterministic
Python outputs, validating the portable `ModelSpec` contract, and proving that
the contract can be consumed by the Rust runtime that will grow into the shared
core.

## Scope

- Maintain frozen in-repo regression fixtures for representative fitted Python models.
- Maintain portable-model fixtures that record expected `design_matrix` and
  `predict` outputs from Python-produced `ModelSpec` artifacts.
- Validate the Rust runtime against those portable fixtures.
- Add tests for invalid portable-model inputs, including missing fields, bad shapes, and incompatible versions.

## Out of Scope

- Performance benchmarking across implementations.
- Broad portability-contract design work.
- Public API documentation cleanups unrelated to validation.
- Live dependency on `py-earth` or R `earth` as required validation gates.

## Acceptance Criteria

- Deterministic Python reference fixtures exist for the intended comparison surface.
- Portable runtime fixtures are checked in and consumed by the Rust runtime.
- The validation suite covers missing fields, malformed shapes, and version incompatibilities.
- Numerical drift in the Python core or portable runtime path is caught by fixture-backed tests.

## Current Status

- Portable-model negative validation coverage is implemented.
- The frozen in-repo estimator regression corpus has been expanded with deterministic weighted, multi-feature, categorical, and missingness-sensitive cases.
- Runtime portability fixtures now cover continuous, interaction,
  categorical, combined, and missingness-sensitive artifacts.
- The Rust reference runtime consumes the checked-in `ModelSpec` fixtures and
  verifies `validate`, `design_matrix`, and `predict` behavior without Python,
  sklearn state, or pickle payloads.
- The next architectural step is to expand the Rust runtime into the shared core
  for Python, R, Julia, Rust, C#, Go, and TypeScript bindings.
- `py-earth` and R `earth` remain useful historical references, but they are not
  project dependencies and are not required validation gates.
