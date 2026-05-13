# Implementation Plan

## Phase 0: Versioning Inventory

- [x] Task: Inventory current version sources and release claims
    - [x] Record the authoritative version field for each package manifest
    - [x] Record which packages are published and which are still registry-only
    - [x] Record where logging/version parity is already documented
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Versioning Inventory' (Protocol in workflow.md)

## Phase 1: Canonical Metadata Source

- [x] Task: Define canonical release metadata
    - [x] Choose the metadata file format and location
    - [x] Record the package/version matrix and release-state fields
    - [x] Record intentional version skew policy
- [x] Task: Implement metadata validation
    - [x] Add a checker for manifest/package name alignment
    - [x] Add a checker for release-doc and metadata alignment
    - [x] Add a CI entrypoint for the checker
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Canonical Metadata Source' (Protocol in workflow.md)

## Phase 2: Documentation Sync

- [x] Task: Sync release-facing documentation
    - [x] Update release inventory to cite the canonical metadata source
    - [x] Update publication handoff and checklist docs to cite the same source
    - [x] Add a short operator note for intentional version skew
- [x] Task: Sync roadmap and registry pointers
    - [x] Update the remaining roadmap with the canonical versioning policy
    - [x] Refresh Conductor track pointers if needed
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Documentation Sync' (Protocol in workflow.md)

## Phase 3: Final Validation and Handoff

- [x] Task: Validate the canonical versioning flow
    - [x] Run the alignment checker
    - [x] Run strict docs build
    - [x] Run prose lint on the updated docs
- [x] Task: Archive the completed track
    - [x] Update track status and archive path
    - [x] Confirm the roadmap only lists truly open work
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Final Validation and Handoff' (Protocol in workflow.md)
