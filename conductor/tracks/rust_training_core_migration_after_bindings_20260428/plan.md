# Implementation Plan

## Phase 0: Post-Binding Core Boundary Review

- [x] Task: Review binding feedback before Rust training migration
    - [x] Confirm runtime replay bindings pass available conformance tests
    - [x] Capture data-shape, missingness, categorical, and error-model gaps
    - [x] Update Rust core API boundaries before training migration
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Post-Binding Core Boundary Review' (Protocol in workflow.md)

## Phase 1: Basis Evaluation Migration

- [x] Task: Move shared basis evaluation behind Rust core APIs
    - [x] Add Python baseline fixtures for basis evaluation
    - [x] Add Rust parity tests
    - [x] Preserve Python public API behavior
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Basis Evaluation Migration' (Protocol in workflow.md)

## Phase 2: Forward and Pruning Migration

- [x] Task: Move forward-pass and pruning internals into Rust
    - [x] Add deterministic candidate-scoring fixtures
    - [x] Add pruning/GCV parity fixtures
    - [x] Preserve sample-weight behavior
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Forward and Pruning Migration' (Protocol in workflow.md)
