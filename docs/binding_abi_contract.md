# Binding ABI and API Contract

This contract defines the narrow shared boundary for the language bindings.
It keeps the public API stable while making the Rust core responsible for the
portable model semantics.

## Scope

The ABI boundary is intentionally narrow:

- portable model specs move as JSON strings or file paths
- runtime evaluation uses row-major numeric matrices
- training uses explicit request/response JSON payloads
- host wrappers remain responsible for local parsing, CLI wiring, and package
  ergonomics
- no general-purpose object ABI is introduced

## Ownership

Rust owns the shared computational semantics for supported portable models:

- model-spec validation
- portable basis evaluation
- prediction replay
- portable spec inspection for compatible specs
- supported training/export normalization paths

Host wrappers own compatibility glue:

- parsing host-native types into the portable boundary
- preserving `import pymars as earth`
- preserving `Earth(...)`, `fit`, `predict`, `score`, and export helpers
- fallback behavior for unsupported or transitional cases

## Memory Ownership

Memory ownership stays explicit and local to each boundary:

- Rust-allocated buffers are returned only through dedicated free functions
- opaque handles are freed by matching destructor functions
- caller-owned buffers stay caller-owned
- the ABI does not transfer borrowed references across language boundaries
- zero-copy exchange is not part of the current contract

## Error Handling

Errors are surfaced as structured status codes plus owned error text:

- null pointers are rejected explicitly
- invalid UTF-8 is reported separately from malformed artifacts
- unsupported artifacts and basis terms map to stable status codes
- numerical failures stay distinct from parse or validation failures
- host wrappers may translate these errors into idiomatic exceptions

## Arrow Decision

Apache Arrow is not part of the current ABI contract.

See [Binding ABI and Arrow Decision](binding_abi_arrow_decision.md) for the
decision record and feasibility notes.

## Stability Guarantees

The ABI contract must preserve:

- the current Python import surface
- the current `Earth` estimator API
- the current portable `ModelSpec` top-level shape
- the current JSON export/load contract
- existing compatibility wrappers for transitional fallback paths
