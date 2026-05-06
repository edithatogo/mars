# Specification: Rust Core Profiling and Observability Hardening

## Overview

Harden the Rust core runtime and CLI for performance visibility, benchmark
repeatability, and operational diagnostics so the project can spot regressions
before they leak into downstream bindings or release artifacts.

This track focuses on the Rust-native execution path that underpins portable
runtime replay and Rust-backed training. It does not change public model
semantics, package identity, or foreign-language binding APIs.

## Context

- The repository already contains a Rust runtime core and a release rehearsal
  path that builds and exercises the Rust binary.
- The project already documents a shared runtime contract and release
  governance, but the Rust performance and observability story is still spread
  across code, docs, and ad hoc checks.
- Profiling and observability need to be actionable for maintainers, stable
  across repeated runs, and cheap enough to keep in CI without creating noisy
  false positives.
- The work should support both local developer analysis and release-time
  regression detection.

## Functional Requirements

- Inventory the current Rust performance and observability surface:
  - runtime crate boundaries
  - runtime CLI entry points
  - existing benchmarks, if any
  - logging, tracing, and error-context hooks
  - build flags and release-only behavior
- Define a baseline for the current Rust core:
  - representative benchmark cases
  - current command timings
  - binary size and startup observations
  - memory or allocation hotspots where practical
- Specify a Rust-native benchmarking strategy:
  - benchmark targets for validation, design-matrix construction, prediction,
    and training-adjacent code paths
  - stable fixture inputs and output comparison rules
  - a repeatable command line for local and CI runs
- Specify an observability strategy:
  - structured logging or tracing spans for key runtime operations
  - correlation points for CLI invocation, validation, matrix construction,
    prediction, and training
  - clear error-context propagation for failures that matter to operators
- Define code-quality and CI gates for profiling and observability:
  - what is checked on every change
  - what remains benchmark-only or release-rehearsal-only
  - what thresholds are alerts versus hard failures
- Define documentation and release guidance:
  - how maintainers run the benchmarks
  - how to interpret performance deltas
  - how to record regressions and baseline updates
- Keep the plan parallelizable so six workers can operate without stepping on
  each other.

## Non-Functional Requirements

- Keep instrumentation low overhead in normal builds.
- Keep benchmark inputs deterministic and fixture-backed.
- Avoid introducing unstable, noisy, or environment-sensitive gates in the fast
  CI path.
- Prefer native Rust tooling and standard ecosystem crates for profiling and
  benchmark harnesses.
- Preserve compatibility with the existing portable runtime contract and
  release flow.
- Keep logging and profiling opt-in where possible in user-facing paths.

## Parallel Work Model

This track is designed for six parallel workstreams:

- Agent 1: inventory, baselines, and evidence capture.
- Agent 2: Rust-native benchmark harness design and implementation.
- Agent 3: observability instrumentation and error-context hardening.
- Agent 4: CI and code-quality gate design for profiling/regression checks.
- Agent 5: documentation, release guidance, and maintainer runbooks.
- Agent 6: final validation, cross-cutting integration, and evidence
  consolidation.

Each phase should have a review checkpoint at the end so the work can be
validated before moving on.

## Acceptance Criteria

- A baseline inventory exists for the current Rust runtime performance and
  observability surface.
- A benchmark strategy exists for the key Rust runtime paths, with stable input
  cases and repeatable invocation commands.
- The observability contract is defined for the Rust runtime and CLI, including
  the kinds of spans, logs, or error context that maintainers can rely on.
- CI and code-quality guidance exists for when to run fast checks versus
  benchmark or release-rehearsal checks.
- Documentation explains how to reproduce profiling runs and how to interpret
  regressions.
- The track is easy to execute across six workers with disjoint ownership.
- Every phase ends with a Conductor review checkpoint.
- Final validation confirms the plan is coherent with the current Rust runtime
  and does not require changes to public binding APIs.

## Out of Scope

- Changing Python, R, Julia, Go, C#, or TypeScript binding APIs.
- Changing portable `ModelSpec` semantics or serialized artifact structure.
- Adding new model families or algorithmic behavior unrelated to profiling and
  observability.
- Publishing release artifacts or changing registry workflows.
- Replacing the existing Rust runtime with a different execution engine.

## References

- [Rust core docs](../../../docs/rust_core.md)
- [Performance optimization plan](../../../docs/performance_optimization_plan.md)
- [CI quality notes](../../../docs/ci_quality.md)
- [Release inventory](../../../docs/release_inventory.md)
