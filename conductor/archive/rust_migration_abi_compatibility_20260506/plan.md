# Implementation Plan

## Phase 0: Migration Inventory

- [x] Task: Inventory the remaining Rust migration boundaries
    - [x] Review adapter-only behavior
    - [x] Review fallback policy
    - [x] Review portable spec ownership
    - [x] Review host-language bridge behavior
- [x] Task: Decide the ABI posture
    - [x] Determine whether a narrow ABI is justified
    - [x] If justified, define the API-preserving ABI scope
    - [x] If not justified, document why the current bridge model remains sufficient
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Migration Inventory' (Protocol in workflow.md)

## Phase 1: Compatibility Guidance

- [x] Task: Define the migration compatibility rules
    - [x] Identify what future Rust slices must preserve
    - [x] Identify the tests required before a slice is accepted
    - [x] Identify the docs required before a slice is accepted
- [x] Task: Define the ABI handoff guidance
    - [x] Explain ownership and lifecycle rules
    - [x] Explain error handling rules
    - [x] Explain fallback expectations for host-language bindings
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Compatibility Guidance' (Protocol in workflow.md)

## Phase 2: Handoff and Validation

- [x] Task: Draft the migration handoff summary
    - [x] Summarize the current migration boundary
    - [x] Summarize the ABI decision
    - [x] Summarize the next migration slices
- [x] Task: Validate the migration artifacts
    - [x] Run docs and lint checks
    - [x] Confirm no API changes were introduced
    - [x] Confirm the guidance is internally consistent
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Handoff and Validation' (Protocol in workflow.md)
