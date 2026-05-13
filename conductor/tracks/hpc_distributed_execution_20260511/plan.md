# Implementation Plan

## Phase 0: Distributed Semantics

- [x] Task: Define partitioning and output contracts
    - [x] Verify H1 partitioning and serial replay semantics are stable
    - [x] Decide whether this H4 slice depends on H3 accelerator behavior
    - [x] Define CPU cluster parallelism as the default distributed target
    - [x] Specify row partitioning and output ordering
    - [x] Specify failure behavior and retry semantics
    - [x] Specify replay-only versus training support
    - [x] Publish a first-cut contract for deterministic reassembly and chunk
      boundaries (for local distributed preview)
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Distributed Semantics' (Protocol in workflow.md)
- [x] [Checkpoint] Conductor - Non-Claim Gate 'Phase 0: H4 Not Yet Claimed'
  - [x] Add explicit H4 deferral or scope text to checkpoint notes
  - [x] Confirm current release-facing docs only say replay-only or deferred
  - [x] Confirm no "distributed" claims are made without a completed test trace

## Phase 1: Local Distributed Adapter (Replay-Only Preview)

- [x] Task: Implement distributed replay smoke path
    - [x] Keep CPU cluster parallelism preview explicit and opt-in
    - [x] Add worker/chunk resource controls
    - [x] Add deterministic aggregation and index-preserving join
    - [x] Add explicit opt-in entry points:
      - `pymars.runtime.predict_distributed`
      - `pymars.runtime.design_matrix_distributed`
      - `docs/hpc_parallel_execution_guide.md` (or equivalent evidence note)
    - [x] Preserve single-process fallback
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Local Distributed Adapter' (Protocol in workflow.md)

## Phase 2: Validation and Cluster Recipe

- [x] Task: Validate H4 behavior
    - [x] Add partitioning and ordering tests
    - [x] Add failure-mode tests where practical
    - [x] Add deterministic smoke proof for contiguous chunks and reordering safety
    - [x] Run the HPC claim-check gate
    - [x] Document CPU cluster-oriented execution guidance
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Validation and Cluster Recipe' (Protocol in workflow.md)
