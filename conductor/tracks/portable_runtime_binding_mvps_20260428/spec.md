# Specification

## Objective

Implement thin portable runtime replay MVPs for Python, R, Julia, Rust, C#, Go,
and TypeScript before migrating training internals into Rust.

## Scope

- Keep the existing Python runtime as the compatibility baseline.
- Keep the existing Rust runtime passing shared fixtures.
- Add language package surfaces for R, Julia, C#, Go, and TypeScript.
- Implement `validate`, `design_matrix`, and `predict` replay semantics where
  local tooling is available.
- Wire implemented bindings into the shared conformance fixture corpus.
- Define publication metadata and release paths for each language package:
  PyPI, crates.io, npm, CRAN or r-universe, Julia General registry, NuGet, and
  Go modules via tagged releases.
- Add CI coverage for each binding family.

## Out of Scope

- Training/fitting in non-Python languages.
- Actually publishing packages before registry ownership, credentials, and
  release approvals are configured.
- Rust migration of forward pass or pruning.
- Dependence on `py-earth` or R `earth`.

## Acceptance Criteria

- Binding source trees exist for R, Julia, Rust, C#, Go, and TypeScript.
- Go and TypeScript bindings pass local fixture parity tests.
- Rust runtime still passes fixture parity tests.
- R, Julia, and C# package surfaces document their conformance commands and are
  ready for CI environments with their dependencies installed.
- Shared docs list binding status and validation coverage.
- CI runs binding validation for Python, Rust, Go, TypeScript, R, Julia, and
  C# where supported by hosted runners.
- Release workflow scaffolding exists for publishing to each relevant package
  manager, gated by manual dispatch or tags and registry credentials.
