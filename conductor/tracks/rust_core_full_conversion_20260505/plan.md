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
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Inventory and Boundary Definition' (Protocol in workflow.md)

## Phase 1: Rust Ownership Migration Slices

- [ ] Task: Migrate remaining replay and validation ownership into Rust
    - [ ] Agent 1: move any remaining replay-heavy helpers behind Rust calls
    - [ ] Agent 2: move validation and model-spec loading helpers to Rust where feasible
    - [ ] Agent 3: preserve Python compatibility wrappers while Rust stabilizes
    - [ ] Agent 4: add or extend parity tests for migrated replay paths
    - [ ] Agent 5: document any intentionally retained adapter logic
    - [ ] Agent 6: consolidate the migration slice evidence
- [ ] Task: Migrate remaining training/export orchestration into Rust
    - [ ] Agent 1: move orchestration glue that still lives in Python into Rust
    - [ ] Agent 2: keep the Rust export format stable across the migration
    - [ ] Agent 3: update host-language bridge calls to the Rust-owned path
    - [ ] Agent 4: add tests that prove Python and Rust export the same artifact
    - [ ] Agent 5: document any temporary flags or gates used during migration
    - [ ] Agent 6: consolidate the training/export migration evidence
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Rust Ownership Migration Slices' (Protocol in workflow.md)

## Phase 2: Python Boundary Retirement

- [ ] Task: Reduce Python to adapter-only behavior
    - [ ] Agent 1: identify Python modules that can be simplified after Rust ownership lands
    - [ ] Agent 2: remove duplicate execution paths now owned by Rust
    - [ ] Agent 3: retain only compatibility glue and import surface in Python
    - [ ] Agent 4: add tests that fail if Python regains core ownership accidentally
    - [ ] Agent 5: document the remaining Python responsibilities clearly
    - [ ] Agent 6: summarize the adapter-only contract
- [ ] Task: Retire deliberate fallback paths
    - [ ] Agent 1: remove fallbacks that are fully parity-proven
    - [ ] Agent 2: keep any remaining fallback behind explicit, documented gates
    - [ ] Agent 3: add skip/failure messages for unsupported transitional behavior
    - [ ] Agent 4: ensure release docs reflect the narrower fallback surface
    - [ ] Agent 5: ensure binding docs describe the Rust-first runtime path
    - [ ] Agent 6: consolidate the fallback-retirement evidence
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Python Boundary Retirement' (Protocol in workflow.md)

## Phase 3: Documentation and Release Guidance

- [ ] Task: Update runtime and binding docs for Rust-first ownership
    - [ ] Agent 1: document Rust-owned replay and training responsibilities
    - [ ] Agent 2: document what remains in Python and why
    - [ ] Agent 3: document transitional flags, if any remain
    - [ ] Agent 4: update release-facing guidance for maintainers
    - [ ] Agent 5: update roadmap pointers so the new ownership is obvious
    - [ ] Agent 6: consolidate the docs into a release-ready narrative
- [ ] Task: Align CI and handoff language with the new ownership boundary
    - [ ] Agent 1: confirm the fast CI gates reflect Rust ownership
    - [ ] Agent 2: confirm release rehearsal covers the Rust-owned paths
    - [ ] Agent 3: confirm the handoff docs do not promise Python ownership that has moved
    - [ ] Agent 4: confirm the track registry and roadmap match the implementation state
    - [ ] Agent 5: confirm any operator guidance is still accurate
    - [ ] Agent 6: summarize the cross-document consistency checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Release Guidance' (Protocol in workflow.md)

## Phase 4: Final Validation and Handoff Readiness

- [ ] Task: Run the final validation bundle
    - [ ] Agent 1: validate the Rust-owned code paths against the existing fixtures
    - [ ] Agent 2: validate the Python boundary no longer owns core behavior
    - [ ] Agent 3: validate the binding surfaces still work unchanged
    - [ ] Agent 4: validate the docs and release guidance are coherent
    - [ ] Agent 5: validate the roadmap and track registry entries
    - [ ] Agent 6: validate the end-to-end ownership story
- [ ] Task: Complete the handoff summary
    - [ ] Agent 1: summarize the migrated Rust ownership
    - [ ] Agent 2: summarize the retired Python fallback logic
    - [ ] Agent 3: summarize the binding compatibility status
    - [ ] Agent 4: summarize any remaining transitional gates
    - [ ] Agent 5: summarize documentation and release guidance updates
    - [ ] Agent 6: collect remaining follow-ups, if any
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Final Validation and Handoff Readiness' (Protocol in workflow.md)

