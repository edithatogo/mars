# Specification

## Objective

After portable runtime bindings are validated, migrate shared MARS training
internals into Rust behind the established binding and conformance surfaces.

## Scope

- Use binding feedback to refine the Rust core boundary.
- Move basis evaluation, forward-pass candidate scoring, pruning, and fitting
  orchestration into Rust in staged increments.
- Preserve the Python/scikit-learn public API while routing internals through
  Rust where stable.
- Extend conformance from runtime replay to fitting only after replay bindings
  are passing.

## Out of Scope

- Starting Rust fitting migration before binding replay surfaces exist.
- Breaking Python estimator compatibility.
- Depending on external MARS implementations as validation gates.

## Acceptance Criteria

- Migration does not begin until binding replay MVPs pass their available
  conformance tests.
- Migration does not begin until CI and release scaffolding exists for the
  binding packages that will consume the Rust core.
- Each migrated training component has Python baseline fixtures and Rust parity
  tests.
- Python API compatibility remains stable during migration.
