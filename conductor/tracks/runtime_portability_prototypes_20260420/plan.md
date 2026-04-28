# Implementation Plan

## Phase 0: Fixture Corpus And Runtime Boundary

- [x] Task: Define the fixture corpus and prototype contract boundary
    - [x] Choose representative Python-exported `ModelSpec` fixtures for portability validation
    - [x] Record expected `design_matrix` outputs for each fixture
    - [x] Record expected `predict` outputs for each fixture
    - [x] Define the minimal file layout and invocation surface for the Rust prototype
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Fixture Corpus And Runtime Boundary' (Protocol in workflow.md)
    - [x] Validation: `cd /Users/doughnut/GitHub/pymars && .venv/bin/pytest tests/test_model_spec.py`

## Phase 1: Rust Reference Runtime

- [x] Task: Build the Rust reference runtime against the portability contract
    - [x] Load a portable model artifact in Rust
    - [x] Implement runtime-side `validate`
    - [x] Implement runtime-side `design_matrix`
    - [x] Implement runtime-side `predict`
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Rust Reference Runtime' (Protocol in workflow.md)
    - [x] Validation: `cd /Users/doughnut/GitHub/pymars/rust-runtime && cargo test`

## Phase 2: Cross-Runtime Validation And Gap Capture

- [x] Task: Validate Rust behavior against Python-produced fixtures
    - [x] Compare Rust `validate` results against fixture expectations
    - [x] Compare Rust `design_matrix` outputs against fixture expectations
    - [x] Compare Rust `predict` outputs against fixture expectations
    - [x] Record contract gaps and convert them into concrete follow-up tasks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Cross-Runtime Validation And Gap Capture' (Protocol in workflow.md)
    - [x] Validation: `cd /Users/doughnut/GitHub/pymars/rust-runtime && cargo test`
