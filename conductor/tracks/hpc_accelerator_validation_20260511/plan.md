# Implementation Plan

## Phase 0: Validation Fixtures

- [x] Task: Add accelerator parity fixtures
  - [x] Create CPU-vs-accelerator fixture pairs
  - [x] Create fallback and no-device fixtures
  - [x] Add tests for deterministic replay behavior
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Validation Fixtures' (Protocol in workflow.md)

## Phase 1: Benchmarking and CI

- [x] Task: Add benchmark coverage
  - [x] Measure accelerator and CPU replay paths
  - [x] Compare supported backends against the fallback path
  - [x] Wire benchmark output into CI artifacts
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Benchmarking and CI' (Protocol in workflow.md)

## Phase 2: Documentation and Claim Hygiene

- [x] Task: Update docs and claim gates
  - [x] Document supported accelerator backends
  - [x] Update H3 claim-check and checkpoint notes
  - [x] Validate docs and registry alignment
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Documentation and Claim Hygiene' (Protocol in workflow.md)
