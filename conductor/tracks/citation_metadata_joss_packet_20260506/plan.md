# Implementation Plan

## Phase 0: Metadata Inventory

- [ ] Task: Confirm canonical metadata inputs
    - [ ] Review release inventory, package names, license, author, and repository URLs
    - [ ] Verify `mars-earth` distribution naming and `pymars` import namespace are preserved
    - [ ] Identify missing DOI or archive identifiers as external gates
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Metadata Inventory' (Protocol in workflow.md)

## Phase 1: Citation Artifacts

- [ ] Task: Add structured citation metadata
    - [ ] Create `CITATION.cff`
    - [ ] Create `codemeta.json`
    - [ ] Validate structured files with local parsers or schema-aware checks where available
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Citation Artifacts' (Protocol in workflow.md)

## Phase 2: JOSS Packet

- [ ] Task: Draft software-paper artifacts
    - [ ] Create `paper.md`
    - [ ] Create `paper.bib`
    - [ ] Link the packet from community-readiness documentation
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: JOSS Packet' (Protocol in workflow.md)

## Phase 3: Validation and Handoff

- [ ] Task: Validate docs and metadata
    - [ ] Run `uv run mkdocs build --strict`
    - [ ] Run JSON validation for `codemeta.json`
    - [ ] Record external DOI and submission gates
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Validation and Handoff' (Protocol in workflow.md)
