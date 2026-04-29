# Implementation Plan

## Phase 0: Inventory and Contract

- [x] Task: Inventory the current operator-facing surface
    - [x] Review `summary`, `trace`, `feature_importances_`, plotting, PDP, and ICE helpers
    - [x] Identify what is already backed by the Rust-produced `ModelSpec`
    - [x] Record any gaps between Python-only helpers and portable runtime data
- [x] Task: Decide the public diagnostics contract
    - [x] Mark stable diagnostics APIs
    - [x] Mark experimental or internal helpers
    - [x] Decide the prediction-interval policy
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Inventory and Contract' (Protocol in workflow.md)

## Phase 1: Diagnostics Parity

- [x] Task: Preserve diagnostics for Rust-backed fits
    - [x] Ensure supported diagnostics remain available when the native backend is used
    - [x] Add tests for summary and feature-importance behavior on Rust-backed models
    - [x] Add tests for any intentional unsupported paths
- [x] Task: Tighten docs and user guidance
    - [x] Document stable vs experimental diagnostics
    - [x] Document logging and verbosity expectations
    - [x] Document how diagnostics relate to the portable runtime contract
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Diagnostics Parity' (Protocol in workflow.md)

## Phase 2: Uncertainty Reporting

- [x] Task: Define the uncertainty surface
    - [x] Decide whether prediction intervals are implemented
    - [x] If not implemented, add a tested unsupported-feature path
    - [x] Document model-family limitations and assumptions
- [x] Task: Validate uncertainty behavior
    - [x] Add regression tests for supported interval or reporting paths
    - [x] Add user-facing tests for deliberate rejection paths
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Uncertainty Reporting' (Protocol in workflow.md)
