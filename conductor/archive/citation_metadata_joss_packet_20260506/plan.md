# Implementation Plan

## Phase 0: Metadata Inventory

- [x] Task: Confirm canonical metadata inputs
    - [x] Review release inventory, package names, license, author, and repository URLs
    - [x] Verify `mars-earth` distribution naming and `pymars` import namespace are preserved
    - [x] Identify missing DOI or archive identifiers as external gates
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Metadata Inventory' (Protocol in workflow.md)

## Phase 1: Citation Artifacts

- [x] Task: Add structured citation metadata
    - [x] Create `CITATION.cff`
    - [x] Create `codemeta.json`
    - [x] Validate structured files with local parsers or schema-aware checks where available
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Citation Artifacts' (Protocol in workflow.md)

## Phase 2: JOSS Packet

- [x] Task: Draft software-paper artifacts
    - [x] Create `paper.md`
    - [x] Create `paper.bib`
    - [x] Link the packet from community-readiness documentation
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: JOSS Packet' (Protocol in workflow.md)

## Phase 3: Validation and Handoff

- [x] Task: Validate docs and metadata
    - [x] Run `uv run mkdocs build --strict`
    - [x] Run JSON validation for `codemeta.json`
    - [x] Record external DOI and submission gates
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and Handoff' (Protocol in workflow.md)
