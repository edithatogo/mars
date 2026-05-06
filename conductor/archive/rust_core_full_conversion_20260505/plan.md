# Implementation Plan

## Phase 0: Inventory and Boundary Definition

- [x] Task: Inventory the remaining Python-owned core behavior
    - [x] Agent 1: map runtime replay helpers and fallback logic
    - [x] Agent 2: map training orchestration and export helpers
    - [x] Agent 3: map Python-side validation, serialization, and adapter code
    - [x] Agent 4: map host-language bridge dependencies on Python behavior
    - [x] Agent 5: identify docs and release notes that describe current ownership
    - [x] Agent 6: consolidate the remaining-core inventory into a boundary map
- [x] Task: Define the Rust ownership boundary
    - [x] Agent 1: separate Rust-owned logic from adapter-only behavior
    - [x] Agent 2: identify transitional fallback points and removal conditions
    - [x] Agent 3: define API stability constraints for each migration slice
    - [x] Agent 4: define fixture and parity evidence needed for each boundary
    - [x] Agent 5: define documentation updates needed for the ownership shift
    - [x] Agent 6: normalize the boundary into a migration-ready checklist
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Inventory and Boundary Definition' (Protocol in workflow.md)

## Phase 1: Rust Ownership Migration Slices

The first landed slice in this phase is documented in
[evidence.md](./evidence.md): Rust-backed portable-spec loading, validation,
replay, inspect, export-normalization, and supported training paths for
compatible specs. The remaining phase-1 migration items have now landed, and
the remaining open work moves to Python-boundary retirement in phase 2.

- [x] Task: Migrate remaining replay and validation ownership into Rust
    - [x] Agent 1: move any remaining replay-heavy helpers behind Rust calls
    - [x] Agent 2: move inspect metadata helpers and model-spec loading to Rust where feasible
    - [x] Agent 3: preserve Python compatibility wrappers while Rust stabilizes
    - [x] Agent 4: add or extend parity tests for migrated replay paths
    - [x] Agent 5: document any intentionally retained adapter logic
    - [x] Agent 6: consolidate the migration slice evidence
- [x] Task: Migrate remaining training/export orchestration into Rust
    - [x] Agent 1: move orchestration glue that still lives in Python into Rust
    - [x] Agent 2: keep the Rust export format stable across the migration
    - [x] Agent 3: update host-language bridge calls to the Rust-owned path
    - [x] Agent 4: add tests that prove Python and Rust export the same artifact
    - [x] Agent 5: document any temporary flags or gates used during migration
    - [x] Agent 6: consolidate the training/export migration evidence
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Rust Ownership Migration Slices' (Protocol in workflow.md)

## Phase 2: Python Boundary Retirement

- [x] Task: Reduce Python to adapter-only behavior
    - [x] Agent 1: identify Python modules that can be simplified after Rust ownership lands
    - [x] Agent 2: remove duplicate execution paths now owned by Rust
    - [x] Agent 3: retain only compatibility glue and import surface in Python
    - [x] Agent 4: add tests that fail if Python regains core ownership accidentally
    - [x] Agent 5: document the remaining Python responsibilities clearly
    - [x] Agent 6: summarize the adapter-only contract
- [x] Task: Retire deliberate fallback paths
    - [x] Agent 1: remove fallbacks that are fully parity-proven
    - [x] Agent 2: keep any remaining fallback behind explicit, documented gates
    - [x] Agent 3: add skip/failure messages for unsupported transitional behavior
    - [x] Agent 4: ensure release docs reflect the narrower fallback surface
    - [x] Agent 5: ensure binding docs describe the Rust-first runtime path
    - [x] Agent 6: consolidate the fallback-retirement evidence
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Python Boundary Retirement' (Protocol in workflow.md)

## Phase 3: Documentation and Release Guidance

- [x] Task: Update runtime and binding docs for Rust-first ownership
    - [x] Agent 1: document Rust-owned replay and training responsibilities
    - [x] Agent 2: document what remains in Python and why
    - [x] Agent 3: document transitional flags, if any remain
    - [x] Agent 4: update release-facing guidance for maintainers
    - [x] Agent 5: update roadmap pointers so the new ownership is obvious
    - [x] Agent 6: consolidate the docs into a release-ready narrative
- [x] Task: Align CI and handoff language with the new ownership boundary
    - [x] Agent 1: confirm the fast CI gates reflect Rust ownership
    - [x] Agent 2: confirm release rehearsal covers the Rust-owned paths
    - [x] Agent 3: confirm the handoff docs do not promise Python ownership that has moved
    - [x] Agent 4: confirm the track registry and roadmap match the implementation state
    - [x] Agent 5: confirm any operator guidance is still accurate
    - [x] Agent 6: summarize the cross-document consistency checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Release Guidance' (Protocol in workflow.md)

## Phase 4: Final Validation and Handoff Readiness

- [x] Task: Run the final validation bundle
    - [x] Agent 1: validate the Rust-owned code paths against the existing fixtures
    - [x] Agent 2: validate the Python boundary no longer owns core behavior
    - [x] Agent 3: validate the binding surfaces still work unchanged
    - [x] Agent 4: validate the docs and release guidance are coherent
    - [x] Agent 5: validate the roadmap and track registry entries
    - [x] Agent 6: validate the end-to-end ownership story
- [x] Task: Complete the handoff summary
    - [x] Agent 1: summarize the migrated Rust ownership
    - [x] Agent 2: summarize the retired Python fallback logic
    - [x] Agent 3: summarize the binding compatibility status
    - [x] Agent 4: summarize any remaining transitional gates
    - [x] Agent 5: summarize documentation and release guidance updates
    - [x] Agent 6: collect remaining follow-ups, if any
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Final Validation and Handoff Readiness' (Protocol in workflow.md)
