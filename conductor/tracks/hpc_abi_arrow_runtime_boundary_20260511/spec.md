# Specification: H2 ABI and Arrow Runtime Boundary

## Overview

Implement the H2 contract from `docs/hpc_contracts.md`: a stable, narrow
runtime boundary for HPC consumers that need predictable FFI and batch data
interchange.

This track depends on H1 benchmark and replay semantics being stable enough to
expose through the boundary.

## Functional Requirements

- Define a versioned C ABI or equivalent FFI boundary for runtime replay.
- Define memory ownership, error code, and version negotiation rules.
- Evaluate and implement Arrow-compatible or Arrow-adjacent batch interchange.
- Add ABI/version tests and host-language conformance tests.
- Document the distinction between stable ABI and internal Rust APIs.

## Non-Claim Deferral Policy

- The H2 boundary is currently implemented only as a narrow RFC-style Rust FFI
  layer; release-facing claims must not state H2 completion until version
  negotiation, host-language boundary conformance, and non-Python interoperability
  evidence are complete.
- Until implementation is complete, all external text must include one of:
  - `not yet`
  - `deferred`
  - `requires explicit runtime-boundary review`
- Any H2 feature claim requires evidence links to:
  - symbol/version contract notes
  - boundary conformance tests
  - one maintained non-claim checkpoint entry

## Non-Functional Requirements

- Preserve current binding APIs.
- Keep training internals out of the stable boundary unless explicitly added by
  a later contract.
- Avoid breaking existing CLI bridge users.
- Do not choose ABI shapes that require accelerator or distributed execution.

## Acceptance Criteria

- H2 requirements in `docs/hpc_contracts.md` are implemented or explicitly
  deferred with rationale.
- ABI and batch interchange tests run in CI.
- Binding docs describe supported stable boundary calls and unsupported calls.
- H2 docs pass the claim-check gate.

## Out of Scope

- GPU, TPU, or distributed execution.
- Replacing language bindings with ABI-only packages.
