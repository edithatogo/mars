# Implementation Plan

## Phase 0: Boundary Review

- [x] Task: Reconfirm runtime and binding ownership
    - [x] Review Rust core boundary docs
    - [x] Review binding ABI and API contract docs
    - [x] Identify stable runtime primitives and unstable training internals
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Boundary Review' (Protocol in workflow.md)

## Phase 1: ABI Decision Record

- [x] Task: Define API-preserving ABI feasibility
    - [x] Specify version negotiation, ownership, error, and memory rules
    - [x] Identify one non-Python binding path suitable for validation
    - [x] Document dependencies and risks without adding required ABI consumers
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: ABI Decision Record' (Protocol in workflow.md)

## Phase 2: Apache Arrow Decision Record

- [x] Task: Evaluate Arrow interoperability
    - [x] Review Arrow C Data and C Stream fit for table inputs
    - [x] Decide whether Arrow should be optional, deferred, or prototyped
    - [x] Document dependency impact for each language binding
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Apache Arrow Decision Record' (Protocol in workflow.md)

## Phase 3: Validation and Handoff

- [x] Task: Validate compatibility and docs
    - [x] Run the public API test subset affected by binding docs
    - [x] Run `uv run mkdocs build --strict`
    - [x] Record follow-on POC tasks if justified
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and Handoff' (Protocol in workflow.md)
