# Implementation Plan

## Phase 0: Baseline and Architecture Lock

- [x] Task: Inventory current Python training orchestration and Rust primitives [789e77d]
    - [x] Map Python forward-pass, pruning, coefficient fitting, and export code to Rust equivalents
    - [x] Identify unsupported edge cases before implementation starts
    - [x] Document the selected Rust training API boundary
- [x] Task: Create failing Rust training-orchestration fixtures [e7579ac]
    - [x] Add Python-generated baseline fixtures for representative fits
    - [x] Add Rust tests that assert the full fit/export path matches those fixtures
    - [x] Include sample-weight and interaction coverage
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Baseline and Architecture Lock' (Protocol in workflow.md) [b00face]

## Phase 1: Rust Forward-Pass Orchestration

- [x] Task: Implement Rust forward-pass orchestration [1cc6430]
    - [x] Generate candidate hinge and interaction terms from normalized rows
    - [x] Apply max-terms, max-degree, minspan/endspan, and stopping rules
    - [x] Preserve deterministic candidate ordering and tie handling
    - [x] Fit coefficients for each accepted basis set
- [x] Task: Validate Rust forward-pass parity [2a170d5]
    - [x] Run Rust fixture tests for forward-pass structure and coefficients
    - [x] Add regression fixtures for deterministic tie cases
    - [x] Update migration docs with any bounded numerical differences
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Rust Forward-Pass Orchestration' (Protocol in workflow.md) [2fa741d]

## Phase 2: Rust Pruning and Final Model Export

- [x] Task: Implement Rust pruning orchestration [d9a05a8]
    - [x] Score pruning subsets with GCV and weighted RSS
    - [x] Refit coefficients after selected basis pruning
    - [x] Export final fitted state as `ModelSpec`
- [x] Task: Validate exported models through replay fixtures [1ba3648]
    - [x] Add Rust tests that run `predict` from exported specs
    - [x] Add Python tests that load Rust-exported specs through the shared runtime
    - [x] Verify conformance fixtures remain stable
- [~] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Rust Pruning and Final Model Export' (Protocol in workflow.md)

## Phase 3: Python Estimator Integration

- [ ] Task: Add controlled Python routing to the Rust training core
    - [ ] Add feature-gated or environment-gated Rust training invocation
    - [ ] Preserve current `Earth` constructor parameters and sklearn behavior
    - [ ] Keep Python fallback available for unsupported cases
- [ ] Task: Validate sklearn and artifact compatibility
    - [ ] Run estimator compatibility tests with Python fallback
    - [ ] Run targeted estimator tests with Rust routing enabled
    - [ ] Confirm fitted Rust-backed estimators export compatible `ModelSpec`
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Python Estimator Integration' (Protocol in workflow.md)

## Phase 4: Documentation and Migration Readiness

- [ ] Task: Document Rust training support and remaining gaps
    - [ ] Update Rust core and training migration docs
    - [ ] Update user-facing docs only where behavior is available
    - [ ] Record unsupported cases and fallback behavior
- [ ] Task: Run full validation
    - [ ] Run Python tests
    - [ ] Run Rust tests
    - [ ] Run binding conformance tests
    - [ ] Run docs build
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Documentation and Migration Readiness' (Protocol in workflow.md)
