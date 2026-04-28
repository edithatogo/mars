# Specification: Rust-Backed Language Bindings

## Overview

The current binding MVPs prove that Python, Rust, Go, TypeScript, R, Julia, and
C# can consume the portable `ModelSpec` contract. They intentionally duplicate
runtime replay semantics in each host language. This track replaces that
duplication with thin bindings over the central Rust core.

The goal is one computational implementation, surfaced idiomatically in every
supported language.

## Dependency Notes

- Depends on the completed runtime portability contract, runtime prototypes,
  conformance harness, and MVP binding package layout.
- Must complete before stable publication of language packages that claim shared
  runtime semantics.
- Should complete before full Rust training APIs are exposed through every
  language binding, because the runtime FFI/error/memory boundary is the safer
  first proof point.

## Functional Requirements

- Define the stable Rust core interface used by foreign-language packages for
  model validation, design-matrix evaluation, prediction, and eventually
  training.
- Select and document binding mechanisms for each language before
  implementation.
- Provide Rust-backed packages for:
  - Python
  - R
  - Julia
  - Rust
  - C#
  - Go
  - TypeScript
- Keep host-language APIs idiomatic while preserving shared semantics and error
  categories.
- Reuse the shared conformance fixtures in every binding CI job.
- Remove or demote duplicated host-language replay logic once Rust-backed paths
  are passing.
- Define ABI/API memory ownership, allocator/free rules, null/NaN handling,
  error-code stability, and SemVer compatibility before implementation.
- Add install-from-built-artifact smoke tests for Rust-backed packages where
  practical.

## Non-Functional Requirements

- The Rust core remains the only source of basis evaluation and prediction
  semantics.
- Bindings must preserve memory safety, deterministic error handling, and
  versioned artifact compatibility.
- Packaging choices must be practical for each ecosystem and documented in
  `tech-stack.md`.
- TypeScript support may use WebAssembly or another documented Rust-backed
  mechanism if native Node bindings are not appropriate.
- FFI boundaries must include ownership, error, and null/NaN handling tests.

## Acceptance Criteria

- Every supported language has a Rust-backed runtime surface that passes shared
  conformance fixtures.
- CI builds and tests all Rust-backed binding packages.
- Host-language error translation preserves shared error categories.
- Documentation identifies install/build requirements and the relationship
  between the Rust core and host package APIs.
- Duplicated MVP replay logic is removed, isolated as a fallback, or explicitly
  marked temporary.
- No package is eligible for a stable publish while it still depends on
  duplicated replay logic unless it is explicitly labeled experimental or
  runtime-preview.

## Out of Scope

- Publishing packages to external registries.
- Changing `ModelSpec` artifact semantics without a schema-version update.
- Adding new languages beyond Python, R, Julia, Rust, C#, Go, and TypeScript.
