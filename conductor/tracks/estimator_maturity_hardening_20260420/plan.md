# Implementation Plan

## Phase 0: Baseline and Gap Analysis

- [x] Task: Re-run estimator checks and capture the current failure surface
    - [x] Re-run `check_estimator` for `EarthRegressor`
    - [x] Re-run `check_estimator` for `EarthClassifier`
    - [x] Classify failures into implementation bugs versus intentional unsupported cases
    - [x] Record the baseline and the target reductions for this track
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Baseline and Gap Analysis' (Protocol in workflow.md)

## Phase 1: Contract Decisions

- [x] Task: Define explicit contract boundaries for sparse and multi-output inputs
    - [x] Decide whether sparse inputs are supported or rejected
    - [x] Decide whether multi-output regression is supported or rejected
    - [x] Decide whether multi-output classification is supported or rejected
    - [x] Specify required validation errors and user-facing documentation changes
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Contract Decisions' (Protocol in workflow.md)

## Phase 2: Estimator and Classifier Hardening

- [x] Task: Implement the chosen sklearn contract adjustments
    - [x] Fix estimator-check failures that should be compliant
    - [x] Add or tighten validation paths for unsupported boundaries
    - [x] Improve classifier probability and decision-function ergonomics
    - [x] Decide and implement the multiclass behavior that matches the documented contract
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Estimator and Classifier Hardening' (Protocol in workflow.md)

## Phase 3: Verification and Documentation

- [x] Task: Lock the contract down with tests and docs
    - [x] Add regression tests for every supported or explicitly rejected boundary
    - [x] Update estimator docs and API notes
    - [x] Re-run the estimator-check suite and confirm the final expected-failure list
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Verification and Documentation' (Protocol in workflow.md)
