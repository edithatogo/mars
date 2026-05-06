# Implementation Plan

## Phase 0: Migration Inventory

- [ ] Task: Inventory the remaining Rust migration boundaries
    - [ ] Review adapter-only behavior
    - [ ] Review fallback policy
    - [ ] Review portable spec ownership
    - [ ] Review host-language bridge behavior
- [ ] Task: Decide the ABI posture
    - [ ] Determine whether a narrow ABI is justified
    - [ ] If justified, define the API-preserving ABI scope
    - [ ] If not justified, document why the current bridge model remains sufficient
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Migration Inventory' (Protocol in workflow.md)

## Phase 1: Compatibility Guidance

- [ ] Task: Define the migration compatibility rules
    - [ ] Identify what future Rust slices must preserve
    - [ ] Identify the tests required before a slice is accepted
    - [ ] Identify the docs required before a slice is accepted
- [ ] Task: Define the ABI handoff guidance
    - [ ] Explain ownership and lifecycle rules
    - [ ] Explain error handling rules
    - [ ] Explain fallback expectations for host-language bindings
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Compatibility Guidance' (Protocol in workflow.md)

## Phase 2: Handoff and Validation

- [ ] Task: Draft the migration handoff summary
    - [ ] Summarize the current migration boundary
    - [ ] Summarize the ABI decision
    - [ ] Summarize the next migration slices
- [ ] Task: Validate the migration artifacts
    - [ ] Run docs and lint checks
    - [ ] Confirm no API changes were introduced
    - [ ] Confirm the guidance is internally consistent
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Handoff and Validation' (Protocol in workflow.md)
