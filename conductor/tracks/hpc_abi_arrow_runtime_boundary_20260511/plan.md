# Implementation Plan

## Phase 0: Boundary Design

- [x] Task: Non-claim and dependency gate
  - [x] Add explicit H2 non-claim checkpoint entry to
    `docs/hpc_track_checkpoint_notes.md`.
  - [x] Confirm H1 stability evidence exists in:
    - `docs/hpc_cpu_parallel_runtime_benchmarks.md`
    - `docs/hpc_track_checkpoint_notes.md`
- [x] Task: Specify the stable runtime boundary
  - [x] Verify H1 semantics and benchmark thresholds are stable enough for H2
  - [x] Define supported calls, version negotiation, memory ownership, and error handling
  - [x] Decide Arrow-compatible versus Arrow-adjacent batch interchange
  - [x] Add design notes linked to `docs/hpc_contracts.md`
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Boundary Design' (Protocol in workflow.md)

## Phase 1: ABI and Batch Implementation

- [x] Task: Implement the H2 runtime boundary
  - [x] Keep Rust FFI ABI symbols additive and backward-compatible
  - [ ] Add version query/compatibility surface (`major.minor` contract surface)
    - [x] Add FFI entry points and version checks
    - [x] Add batch interchange adapter and validation
    - [x] Preserve CLI and existing binding behavior
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: ABI and Batch Implementation' (Protocol in workflow.md)

## Phase 2: Tests and Documentation

- [x] Task: Add ABI conformance coverage
  - [x] Add Rust-level ABI status/ownership tests for boundary-facing symbols
  - [x] Add at least one non-Python boundary conformance path (host binding smoke)
  - [x] Add ABI/version tests
  - [x] Add host-language boundary smoke tests
  - [x] Run the HPC claim-check gate
  - [x] Document H2 support and limitations
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Tests and Documentation' (Protocol in workflow.md)
