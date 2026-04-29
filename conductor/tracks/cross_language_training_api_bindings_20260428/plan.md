# Implementation Plan

## Phase 0: Training Binding API Design

- [x] Task: Define shared training API contract
    - [x] Specify input shape, sample weights, feature metadata, hyperparameters, and output artifacts
    - [x] Specify error categories for unsupported options and numerical failures
    - [x] Document runtime-only versus training-capable package expectations
- [x] Task: Add failing training conformance fixtures
    - [x] Create shared fixture cases for fit, predict, and export
    - [x] Add expected failures for runtime-only package modes
    - [x] Extend conformance runner to distinguish runtime and training suites
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Training Binding API Design' (Protocol in workflow.md)

## Phase 1: Python and Rust Training Surfaces

- [x] Task: Expose Rust training through Python
    - [x] Preserve sklearn-compatible `Earth.fit`
    - [x] Add explicit runtime/core routing tests
    - [x] Verify exported specs pass runtime conformance
- [x] Task: Expose training through the Rust crate API
    - [x] Add idiomatic Rust training entrypoints
    - [x] Add Rust docs and examples
    - [x] Verify cargo tests and package checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Python and Rust Training Surfaces' (Protocol in workflow.md)

## Phase 2: R, Julia, C#, and Go Training Surfaces

- [x] Task: Add R and Julia training APIs
    - [x] Preserve idiomatic package naming and data conversion
    - [x] Translate Rust errors into host-language errors
    - [x] Run shared training conformance fixtures
    - [x] Expose R training through `fit_model` and the Rust CLI `fit` subcommand
    - [x] Expose Julia training through `fit_model` and the Rust CLI `fit` subcommand
- [x] Task: Add C# and Go training APIs
    - [x] Preserve type-safe APIs and ownership rules
    - [x] Translate Rust errors into host-language errors
    - [x] Run shared training conformance fixtures
    - [x] Expose C# training through `Runtime.FitModel` and the Rust CLI `fit` subcommand
    - [x] Expose Go training through `FitModel` and the Rust CLI `fit` subcommand
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: R, Julia, C#, and Go Training Surfaces' (Protocol in workflow.md)

## Phase 3: TypeScript Training Surface

- [x] Task: Add TypeScript training support or runtime-only declaration
    - [x] Implement training through the selected Rust-backed mechanism if feasible
    - [x] Otherwise expose clear runtime-only package metadata and unsupported training errors
    - [x] Add package docs and tests for the selected behavior
- [x] Task: Validate TypeScript package behavior
    - [x] Run runtime conformance
    - [x] Run training conformance or expected unsupported-feature tests
    - [x] Run npm package checks
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: TypeScript Training Surface' (Protocol in workflow.md)

## Phase 4: Cross-Language Documentation and CI

- [x] Task: Update docs and examples
    - [x] Add train/predict/export examples for each training-capable language
    - [x] Mark runtime-only surfaces explicitly
    - [x] Update package READMEs and central binding docs
- [x] Task: Update CI for training conformance
    - [x] Add training conformance jobs per supported package
    - [x] Keep runtime conformance jobs separate
    - [x] Run package dry-runs after training API changes
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Cross-Language Documentation and CI' (Protocol in workflow.md)
