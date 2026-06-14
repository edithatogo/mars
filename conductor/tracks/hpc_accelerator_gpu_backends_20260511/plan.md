# Implementation Plan

## Phase 0: GPU Backend Contract

- [x] Task: Define GPU backend behavior
  - [x] Specify supported device discovery
  - [x] Specify replay parity and tolerance rules
  - [x] Specify CPU fallback expectations
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: GPU Backend Contract' (Protocol in workflow.md)

## Phase 1: GPU Implementation

- [x] Task: Implement the first GPU-family backend adapter [optional module-backed H3 adapter; no vendor speedup claim]
  - [x] Add device wiring for the selected runtime
  - [x] Add supported-kernel replay paths through the shared optional array-module H3 replay surface
  - [x] Add no-device and unsupported-kernel fallbacks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: GPU Implementation' (Protocol in workflow.md)

## Phase 2: GPU Validation

- [x] Task: Validate GPU behavior [adapter-family validation; vendor-specific parity remains non-claimed]
  - [x] Add parity fixtures and benchmark evidence
  - [x] Update docs and package metadata
  - [x] Validate claim-check rules
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: GPU Validation' (Protocol in workflow.md)
