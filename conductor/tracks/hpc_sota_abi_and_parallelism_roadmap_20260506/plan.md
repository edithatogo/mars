# Implementation Plan

## Phase 0: HPC Gap Inventory and ABI Decision

- [ ] Task: Inventory the current HPC gaps
    - [ ] Review CPU profiling and benchmarking coverage
    - [ ] Review memory profiling and allocation visibility
    - [ ] Review GPU, TPU, and distributed execution support
    - [ ] Review current performance-portability tooling
- [ ] Task: Decide the ABI posture
    - [ ] Determine whether a stable ABI is justified
    - [ ] If justified, define the narrowest API-preserving ABI scope
    - [ ] If not justified, document why the current CLI/FFI strategy remains sufficient
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: HPC Gap Inventory and ABI Decision' (Protocol in workflow.md)

## Phase 1: Roadmap Design

- [ ] Task: Define the SOTA HPC roadmap
    - [ ] Split near-term profiling and benchmarking work from longer-term accelerator work
    - [ ] Define the target state for HPC-ready runtime evaluation
    - [ ] Define the target state for host-language bindings under an ABI-aware design
- [ ] Task: Define the API-preserving migration strategy
    - [ ] Identify what can move behind ABI boundaries without changing the public API
    - [ ] Identify what should remain adapter-only
    - [ ] Document the ownership and error-handling rules for any ABI boundary
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Roadmap Design' (Protocol in workflow.md)

## Phase 2: HPC Ecosystem Readiness

- [ ] Task: Define HPSF readiness criteria
    - [ ] Map the project to performance portability and HPC expectations
    - [ ] Identify missing profiling, benchmarking, and tuning artifacts
    - [ ] List repo/docs changes needed before an HPSF-style submission
- [ ] Task: Define E4S readiness criteria
    - [ ] Map the project to CPU/GPU portability and scientific workflow expectations
    - [ ] Identify missing interoperability or build/release evidence
    - [ ] List repo/docs changes needed before an E4S-style submission
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: HPC Ecosystem Readiness' (Protocol in workflow.md)

## Phase 3: Roadmap Validation and Handoff

- [ ] Task: Publish the HPC roadmap page
    - [ ] Add the roadmap to the docs nav
    - [ ] Add the roadmap from the remaining-roadmap page
    - [ ] Include the current-state and future-state diagrams
- [ ] Task: Validate the final HPC artifacts
    - [ ] Run docs and lint checks
    - [ ] Confirm no public API changes were introduced
    - [ ] Confirm the roadmap and ABI recommendation are internally consistent
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Roadmap Validation and Handoff' (Protocol in workflow.md)
