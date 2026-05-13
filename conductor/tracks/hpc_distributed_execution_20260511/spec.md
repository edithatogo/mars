# Specification: H4 Distributed Execution

## Overview

Implement the H4 contract from `docs/hpc_contracts.md`: explicit CPU cluster
parallelism and distributed or multi-worker replay semantics for partitioned
batch prediction.

This track depends on H1 partitioning and serial replay semantics. It depends
on H3 only if the distributed claim includes accelerator execution.

## Functional Requirements

- Define row-batch partitioning, output ordering, and aggregation semantics.
- Add a local distributed execution mode or adapter suitable for smoke tests.
- Add a local CPU cluster parallelism mode or adapter suitable for smoke tests.
- Add resource controls for worker count, chunk size, and memory limits.
- Add failure-mode documentation and deterministic retry behavior where
  practical.
- State whether the distributed path is replay-only or training-capable.

## Replay-Only Failure Semantics

- The current local distributed preview is fail-fast, not retrying:
  - invalid row shapes are rejected immediately
  - worker and chunk size hints are validated before execution
  - no hidden retry loop or cluster recovery behavior is implied
- Retry behavior remains a later contract if a scheduler-backed implementation
  is added.

## Non-Functional Requirements

- No hidden network activity during import or local prediction.
- Single-process CPU replay must remain available and deterministic.
- Distributed support must not be required for normal package installation.
- Distributed support must be opt-in and must not perform network activity on
  import or normal local prediction.

## Acceptance Criteria

- H4 requirements in `docs/hpc_contracts.md` are implemented for replay or
  explicitly scoped as replay-only.
- Distributed smoke tests validate partitioning and ordering.
- Docs include a CPU cluster-oriented execution recipe.
- H4 docs pass the claim-check gate.

## Out of Scope

- Implicit cluster provisioning or scheduler management.
- Accelerator-specific distributed kernels unless H3 has already landed and the
  H4 design explicitly depends on it.
