# Specification: H1 CPU Parallel Runtime

## Overview

Implement the H1 contract from `docs/hpc_contracts.md`: deterministic,
resource-controlled CPU-parallel replay for batch prediction and design-matrix
construction.

## Functional Requirements

- Add Rust CPU-parallel batch prediction and design-matrix paths.
- Provide thread controls and deterministic single-thread fallback.
- Preserve existing ModelSpec semantics and binding APIs.
- Add benchmarks for representative small, medium, and large replay batches.
- Add parity tests across serial and parallel execution.
- Establish H1 benchmark acceptance thresholds before changing kernels.
- Expose H1 status only after claim-check validation passes.

## Non-Functional Requirements

- Parallel execution must be opt-in or resource-bounded by default.
- Numerical results must match serial CPU results within the documented
  tolerance.
- No accelerator, MPI, or distributed dependency may be introduced.
- H1 may run in parallel with H0 packaging cleanup, but H2/H3/H4 should treat
  H1 semantics and benchmarks as dependencies.

## Acceptance Criteria

- H1 requirements in `docs/hpc_contracts.md` are implemented.
- Rust and Python conformance tests cover serial and parallel replay.
- Benchmark artifacts and thresholds are documented.
- Existing binding conformance remains green.
- Single-thread mode avoids more than 5% benchmark median regression unless a
  documented tradeoff is approved.
- Parallel mode shows measurable large-batch speedup or records why replay is
  not parallelism-limited.

## Out of Scope

- GPU, TPU, MPI, or distributed execution.
- Stable C ABI work beyond what the current runtime already exposes.
