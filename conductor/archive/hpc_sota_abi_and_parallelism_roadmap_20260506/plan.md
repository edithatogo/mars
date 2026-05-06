# Implementation Plan

## Phase 0: HPC Gap Inventory and ABI Decision

- [x] Task: Inventory the current HPC gaps
    - [x] Review CPU profiling and benchmarking coverage
    - [x] Review memory profiling and allocation visibility
    - [x] Review GPU, TPU, and distributed execution support
    - [x] Review current performance-portability tooling
- [x] Task: Decide the ABI posture
    - [x] Determine whether a stable ABI is justified
    - [x] If justified, define the narrowest API-preserving ABI scope
    - [x] If not justified, document why the current CLI/FFI strategy remains sufficient
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: HPC Gap Inventory and ABI Decision' (Protocol in workflow.md)

## Phase 1: Roadmap Design

- [x] Task: Define the SOTA HPC roadmap
    - [x] Split near-term profiling and benchmarking work from longer-term accelerator work
    - [x] Define the target state for HPC-ready runtime evaluation
    - [x] Define the target state for host-language bindings under an ABI-aware design
- [x] Task: Define the API-preserving migration strategy
    - [x] Identify what can move behind ABI boundaries without changing the public API
    - [x] Identify what should remain adapter-only
    - [x] Document the ownership and error-handling rules for any ABI boundary
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Roadmap Design' (Protocol in workflow.md)

## Phase 2: HPC Ecosystem Readiness

- [x] Task: Define HPSF readiness criteria
    - [x] Map the project to performance portability and HPC expectations
    - [x] Identify missing profiling, benchmarking, and tuning artifacts
    - [x] List repo/docs changes needed before an HPSF-style submission
- [x] Task: Define E4S readiness criteria
    - [x] Map the project to CPU/GPU portability and scientific workflow expectations
    - [x] Identify missing interoperability or build/release evidence
    - [x] List repo/docs changes needed before an E4S-style submission
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: HPC Ecosystem Readiness' (Protocol in workflow.md)

## Phase 3: Roadmap Validation and Handoff

- [x] Task: Publish the HPC roadmap page
    - [x] Add the roadmap to the docs nav
    - [x] Add the roadmap from the remaining-roadmap page
    - [x] Include the current-state and future-state diagrams
- [x] Task: Validate the final HPC artifacts
    - [x] Run docs and lint checks
    - [x] Confirm no public API changes were introduced
    - [x] Confirm the roadmap and ABI recommendation are internally consistent
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Roadmap Validation and Handoff' (Protocol in workflow.md)
