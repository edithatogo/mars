# Implementation Plan

The checklist below records the helper-layer and documentation work that has
landed. Rust now owns a parent-linked numeric forward-pass and pruning
baseline in the training crate. Categorical handling, estimator routing, and
cross-language training surfaces remain open and are tracked separately in the
remaining roadmap.

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
    - [x] Generate numeric candidate terms from normalized rows
    - [x] Emit parent-linked linear and hinge terms from the baseline path
    - [x] Apply max-terms and GCV-driven stopping for the baseline path
    - [x] Preserve deterministic candidate ordering within the numeric baseline
    - [x] Fit coefficients for each accepted basis set
- [x] Task: Extend Rust forward-pass orchestration to full parity [1cc6430]
    - [x] Generate categorical and missingness-aware candidate terms from normalized rows
    - [x] Generate root interaction candidates from normalized rows
    - [x] Apply max-terms, max-degree, minspan/endspan, and full stopping rules
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
- [x] Task: Extend Rust pruning orchestration to full parity [d9a05a8]
    - [x] Score pruning subsets across the full production search space
    - [x] Refit coefficients after selected basis pruning
    - [x] Export final fitted state as `ModelSpec`
- [x] Task: Validate exported models through replay fixtures [1ba3648]
    - [x] Add Rust tests that run `predict` from exported specs
    - [x] Add Python tests that load Rust-exported specs through the shared runtime
    - [x] Verify conformance fixtures remain stable
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Rust Pruning and Final Model Export' (Protocol in workflow.md) [6fee520]

## Phase 3: Python Estimator Integration

- [x] Task: Add controlled Python routing to the Rust training core [58f1860]
    - [x] Add feature-gated or environment-gated Rust training invocation
    - [x] Preserve current `Earth` constructor parameters and sklearn behavior
    - [x] Keep Python fallback available for unsupported cases
- [x] Task: Validate sklearn and artifact compatibility [dc661ce]
    - [x] Run estimator compatibility tests with Python fallback
    - [x] Run targeted estimator tests with Rust routing enabled
    - [x] Confirm fitted Rust-backed estimators export compatible `ModelSpec`
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Python Estimator Integration' (Protocol in workflow.md) [dc661ce]

## Phase 4: Documentation and Migration Readiness

- [x] Task: Document Rust training support and remaining gaps
    - [x] Update Rust core and training migration docs
    - [x] Update user-facing docs only where behavior is available
    - [x] Record unsupported cases and fallback behavior
- [x] Task: Run full validation
    - [x] Run Python tests
    - [x] Run Rust tests
    - [x] Run binding conformance tests
    - [x] Run docs build
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Documentation and Migration Readiness' (Protocol in workflow.md)
