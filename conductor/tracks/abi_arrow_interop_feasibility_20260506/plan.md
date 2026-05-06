# Implementation Plan

## Phase 0: Boundary Review

- [ ] Task: Reconfirm runtime and binding ownership
    - [ ] Review Rust core boundary docs
    - [ ] Review binding ABI and API contract docs
    - [ ] Identify stable runtime primitives and unstable training internals
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Boundary Review' (Protocol in workflow.md)

## Phase 1: ABI Decision Record

- [ ] Task: Define API-preserving ABI feasibility
    - [ ] Specify version negotiation, ownership, error, and memory rules
    - [ ] Identify one non-Python binding path suitable for validation
    - [ ] Document dependencies and risks without adding required ABI consumers
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: ABI Decision Record' (Protocol in workflow.md)

## Phase 2: Apache Arrow Decision Record

- [ ] Task: Evaluate Arrow interoperability
    - [ ] Review Arrow C Data and C Stream fit for table inputs
    - [ ] Decide whether Arrow should be optional, deferred, or prototyped
    - [ ] Document dependency impact for each language binding
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Apache Arrow Decision Record' (Protocol in workflow.md)

## Phase 3: Validation and Handoff

- [ ] Task: Validate compatibility and docs
    - [ ] Run the public API test subset affected by binding docs
    - [ ] Run `uv run mkdocs build --strict`
    - [ ] Record follow-on POC tasks if justified
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and Handoff' (Protocol in workflow.md)
