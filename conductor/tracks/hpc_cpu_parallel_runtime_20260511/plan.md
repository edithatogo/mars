# Implementation Plan

## Phase 0: Benchmark and Test Baseline

- [x] Task: Add CPU replay benchmark baselines
    - [x] Add representative Rust criterion cases for design matrix and prediction
    - [x] Add Python benchmark or smoke wrapper for the Rust path
    - [x] Define benchmark acceptance thresholds and regression policy
    - [x] Define serial/parallel numerical tolerance
    - [x] Record baseline commands and expected artifacts
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Benchmark and Test Baseline' (Protocol in workflow.md)
    - [x] Checkpoint evidence: `docs/hpc_track_checkpoint_notes.md`

## Phase 1: Parallel Runtime Implementation

- [x] Task: Implement CPU-parallel replay paths
    - [x] Add resource-controlled thread configuration
    - [x] Add deterministic single-thread fallback
    - [x] Preserve existing serial behavior and error semantics
    - [x] Measure memory use for large replay batches
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Parallel Runtime Implementation' (Protocol in workflow.md)
    - [x] Checkpoint evidence: `docs/hpc_track_checkpoint_notes.md`

## Phase 2: Binding and Conformance Coverage

- [x] Task: Extend conformance tests for H1
    - [x] Add serial-vs-parallel fixture parity tests
    - [x] Add Python and Rust smoke tests for thread controls
    - [x] Run the HPC claim-check gate before documenting H1 support
    - [x] Document H1 status in release/community docs
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Binding and Conformance Coverage' (Protocol in workflow.md)
    - [x] Checkpoint evidence: `docs/hpc_track_checkpoint_notes.md`
