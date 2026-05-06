# Specification: ABI and Apache Arrow Interoperability Feasibility

## Overview

Evaluate an API-preserving ABI boundary and optional Apache Arrow
interoperability story for stable runtime primitives. The goal is to decide
what can safely be shared across language bindings without freezing unstable
training internals.

## Requirements

- Reconfirm the runtime ownership boundary before introducing ABI commitments.
- Define the narrowest stable ABI scope for model loading, validation, runtime
  evaluation, errors, and version negotiation.
- Evaluate Apache Arrow C Data or C Stream interoperability for table inputs.
- Document memory ownership, error handling, and versioning rules.
- Preserve current language APIs, including `import pymars as earth` and
  `earth.Earth(...)`.

## Dependencies

- Depends on Rust runtime ownership documentation and binding conformance
  expectations.
- Supports future HPC packaging and polyglot interoperability tracks.
- Can run in parallel with citation, supply-chain, packaging, and governance
  work.

## Acceptance Criteria

- ABI decision record clearly states what is in scope and out of scope.
- Arrow decision record identifies whether Arrow is optional, deferred, or worth
  a proof of concept.
- Any POC code is isolated and does not become a required runtime dependency.
- Existing public API tests continue to pass.

## Out of Scope

- Replacing current language APIs with raw ABI calls.
- Adding GPU, TPU, MPI, or distributed execution.
- Freezing training internals before parity evidence is complete.
