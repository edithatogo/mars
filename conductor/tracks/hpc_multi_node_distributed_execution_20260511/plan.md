# Implementation Plan

## Phase 0: Contract Definition

- [x] Task: Define the multi-node H4 contract [real scheduler backend previously deferred]
  - [x] Specify partitioning, aggregation, and retry rules
  - [x] Specify network and scheduler assumptions
  - [x] Specify what remains replay-only versus cluster-backed
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Contract Definition' (Protocol in workflow.md)

## Phase 1: Implementation

- [x] Task: Implement the multi-node execution path
  - [x] Add the real multi-node adapter or scheduler wiring
  - [x] Add failover and deterministic aggregation tests
  - [x] Add resource controls and smoke coverage
  - Evidence: `pymars.cluster.CommandMultiNodeBackend`,
    `pymars.cluster_worker`, and `tests/test_cluster_runtime.py`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Implementation' (Protocol in workflow.md)

## Phase 2: Docs and Validation

- [x] Task: Update docs and claim gates
  - [x] Document the multi-node H4 contract separately from CPU-cluster replay
  - [x] Update checkpoint notes and claim-check references
  - [x] Validate the final H4 wording and tests
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Docs and Validation' (Protocol in workflow.md)
