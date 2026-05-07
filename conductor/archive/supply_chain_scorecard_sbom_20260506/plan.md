# Implementation Plan

## Phase 0: Supply-Chain Inventory

- [x] Task: Map current CI and release evidence
    - [x] Review GitHub Actions workflows and release scripts
    - [x] Identify existing security, dependency, and artifact checks
    - [x] Record where Scorecard, SBOM, and provenance can fit safely
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Supply-Chain Inventory' (Protocol in workflow.md)

## Phase 1: Scorecard and SBOM Evidence

- [x] Task: Add evidence generation
    - [x] Add or document OpenSSF Scorecard execution
    - [x] Add or document SBOM generation for Python and Rust artifacts
    - [x] Keep scheduled or manual jobs separate from fork-sensitive jobs
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Scorecard and SBOM Evidence' (Protocol in workflow.md)

## Phase 2: Provenance and Documentation

- [x] Task: Document release provenance expectations
    - [x] Update supply-chain documentation
    - [x] Link evidence to PyPA, HPSF, E4S, and foundation readiness pages
    - [x] Record external setup that maintainers must complete in GitHub
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Provenance and Documentation' (Protocol in workflow.md)

## Phase 3: Validation and Handoff

- [x] Task: Validate workflow and docs changes
    - [x] Run workflow syntax checks where available
    - [x] Run `uv run mkdocs build --strict`
    - [x] Confirm no secrets are introduced
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and Handoff' (Protocol in workflow.md)
