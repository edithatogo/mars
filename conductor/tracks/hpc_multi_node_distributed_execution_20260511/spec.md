# Specification: H4 Multi-Node Distributed Execution

## Overview

Implement the remaining H4 contract from `docs/hpc_contracts.md` for true
multi-node or scheduler-backed execution beyond the CPU cluster replay path.
The adjacent CPU-cluster replay API and shared cluster configuration layer are
already implemented; this track owns the scheduler-backed backend that remains
deferred.

This track depends on the H4 CPU-cluster replay slice and the existing H1
partitioning semantics. It does not depend on accelerator backends unless a
future design explicitly combines them.

## Functional Requirements

- Define scheduler-backed or node-backed partitioning semantics.
- Define retry, failure, and aggregation behavior for real multi-node replay.
- Define cluster-oriented smoke tests or recipe steps.
- Define any network and resource assumptions explicitly.
- Reuse the existing cluster configuration contract where possible rather than
  inventing a separate configuration surface.

## Non-Functional Requirements

- Multi-node execution must remain opt-in.
- Import-time behavior must remain local and deterministic.
- The cluster contract must remain separate from the CPU-cluster preview path.

## Acceptance Criteria

- The repo has a documented multi-node H4 contract separate from CPU-cluster replay.
- Cluster-oriented evidence or implementation exists for the real multi-node path.
- Claim-checks and docs remain honest about what is implemented.

## Out of Scope

- CPU cluster replay, which is now handled by the existing H4 replay slice.
- Accelerator-specific distributed kernels unless explicitly approved later.
