# Implementation Plan

## Phase 0: Training Binding API Design

- [ ] Task: Define shared training API contract
    - [ ] Specify input shape, sample weights, feature metadata, hyperparameters, and output artifacts
    - [ ] Specify error categories for unsupported options and numerical failures
    - [ ] Document runtime-only versus training-capable package expectations
- [ ] Task: Add failing training conformance fixtures
    - [ ] Create shared fixture cases for fit, predict, and export
    - [ ] Add expected failures for runtime-only package modes
    - [ ] Extend conformance runner to distinguish runtime and training suites
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Training Binding API Design' (Protocol in workflow.md)

## Phase 1: Python and Rust Training Surfaces

- [ ] Task: Expose Rust training through Python
    - [ ] Preserve sklearn-compatible `Earth.fit`
    - [ ] Add explicit runtime/core routing tests
    - [ ] Verify exported specs pass runtime conformance
- [ ] Task: Expose training through the Rust crate API
    - [ ] Add idiomatic Rust training entrypoints
    - [ ] Add Rust docs and examples
    - [ ] Verify cargo tests and package checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Python and Rust Training Surfaces' (Protocol in workflow.md)

## Phase 2: R, Julia, C#, and Go Training Surfaces

- [ ] Task: Add R and Julia training APIs
    - [ ] Preserve idiomatic package naming and data conversion
    - [ ] Translate Rust errors into host-language errors
    - [ ] Run shared training conformance fixtures
- [ ] Task: Add C# and Go training APIs
    - [ ] Preserve type-safe APIs and ownership rules
    - [ ] Translate Rust errors into host-language errors
    - [ ] Run shared training conformance fixtures
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: R, Julia, C#, and Go Training Surfaces' (Protocol in workflow.md)

## Phase 3: TypeScript Training Surface

- [ ] Task: Add TypeScript training support or runtime-only declaration
    - [ ] Implement training through the selected Rust-backed mechanism if feasible
    - [ ] Otherwise expose clear runtime-only package metadata and unsupported training errors
    - [ ] Add package docs and tests for the selected behavior
- [ ] Task: Validate TypeScript package behavior
    - [ ] Run runtime conformance
    - [ ] Run training conformance or expected unsupported-feature tests
    - [ ] Run npm package checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: TypeScript Training Surface' (Protocol in workflow.md)

## Phase 4: Cross-Language Documentation and CI

- [ ] Task: Update docs and examples
    - [ ] Add train/predict/export examples for each training-capable language
    - [ ] Mark runtime-only surfaces explicitly
    - [ ] Update package READMEs and central binding docs
- [ ] Task: Update CI for training conformance
    - [ ] Add training conformance jobs per supported package
    - [ ] Keep runtime conformance jobs separate
    - [ ] Run package dry-runs after training API changes
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Cross-Language Documentation and CI' (Protocol in workflow.md)
