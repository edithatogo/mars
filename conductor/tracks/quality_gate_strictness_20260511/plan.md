# Implementation Plan

## Phase 0: Policy Audit

- [x] Task: Audit the quality gates
  - [x] Identify remaining advisory-only CI steps
  - [x] Identify remaining lint/type/docstring exceptions
  - [x] Define what must stay advisory
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Policy Audit' (Protocol in workflow.md)

## Phase 1: Gate Hardening

- [x] Task: Harden blocking quality gates
  - [x] Make strict checks fail fast in CI
  - [x] Align pre-commit, tox, and CI behavior
  - [x] Keep intentional advisory jobs explicit
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Gate Hardening' (Protocol in workflow.md)

## Phase 2: Verification

- [x] Task: Validate the strict policy
  - [x] Run the full check suite
  - [x] Update docs to reflect the strict policy
  - [x] Validate release and HPC claim checks
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Verification' (Protocol in workflow.md)
