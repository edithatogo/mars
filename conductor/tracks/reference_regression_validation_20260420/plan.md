# Implementation Plan

## Phase 0: Reference Surface Definition

- [x] Task: Define the comparison surface and fixtures
    - [x] Choose representative datasets and model settings for reference comparison
    - [x] Define comparison tolerances and deterministic fixture rules
    - [x] Define `pymars` Python fixture outputs as the canonical training reference
    - [x] Define portable `ModelSpec` fixtures as the cross-runtime comparison surface
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Reference Surface Definition' (Protocol in workflow.md)

## Phase 1: Internal and Cross-Runtime Reference Regression Tests

- [x] Task: Add internal fitted-model regression coverage
    - [x] Add frozen Python estimator fixtures for representative deterministic cases
    - [x] Cover weighted, multi-feature, categorical, missingness-sensitive, and interaction cases
    - [x] Lock basis strings, coefficients, predictions, GCV, RSS, and MSE outputs
- [x] Task: Add cross-runtime portable regression coverage
    - [x] Add Python-produced `ModelSpec` fixtures for representative runtime surfaces
    - [x] Add matching expected `design_matrix` and `predict` fixture outputs
    - [x] Validate the Rust reference runtime against the checked-in fixture corpus
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Internal and Cross-Runtime Reference Regression Tests' (Protocol in workflow.md)

## Phase 2: Portable Model Failure Validation

- [x] Task: Add negative validation coverage for portable-model inputs
    - [x] Add tests for missing required fields
    - [x] Add tests for bad array or coefficient shapes
    - [x] Add tests for incompatible or unknown model versions
    - [x] Ensure failures are deterministic and user-facing messages are actionable
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Portable Model Failure Validation' (Protocol in workflow.md)

## Progress Notes

- This track intentionally does not depend on live `py-earth` or R `earth`
  installs. Those projects are historical/API references, not validation
  authorities for the pure Python core or portable runtime contract.
- The current validation surface is the checked-in Python regression corpus plus
  the portable `ModelSpec` fixture corpus consumed by the Rust reference runtime.
