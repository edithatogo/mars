# Implementation Plan

## Phase 0: Supply-Chain Inventory

- [ ] Task: Map current CI and release evidence
    - [ ] Review GitHub Actions workflows and release scripts
    - [ ] Identify existing security, dependency, and artifact checks
    - [ ] Record where Scorecard, SBOM, and provenance can fit safely
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Supply-Chain Inventory' (Protocol in workflow.md)

## Phase 1: Scorecard and SBOM Evidence

- [ ] Task: Add evidence generation
    - [ ] Add or document OpenSSF Scorecard execution
    - [ ] Add or document SBOM generation for Python and Rust artifacts
    - [ ] Keep scheduled or manual jobs separate from fork-sensitive jobs
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Scorecard and SBOM Evidence' (Protocol in workflow.md)

## Phase 2: Provenance and Documentation

- [ ] Task: Document release provenance expectations
    - [ ] Update supply-chain documentation
    - [ ] Link evidence to PyPA, HPSF, E4S, and foundation readiness pages
    - [ ] Record external setup that maintainers must complete in GitHub
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Provenance and Documentation' (Protocol in workflow.md)

## Phase 3: Validation and Handoff

- [ ] Task: Validate workflow and docs changes
    - [ ] Run workflow syntax checks where available
    - [ ] Run `uv run mkdocs build --strict`
    - [ ] Confirm no secrets are introduced
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and Handoff' (Protocol in workflow.md)
