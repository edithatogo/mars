# Implementation Plan

## Phase 0: Docs Stack Inventory

- [x] Task: Inventory the current docs capabilities that must be preserved
    - [x] Document the current docs site behavior, navigation, and user-facing
      features
    - [x] Record which content patterns matter most for release docs, tracks,
      and API/reference pages
    - [x] Identify any current docs features that Starlight must match or
      improve
- [x] Task: Compare Starlight capabilities and candidate plugins
    - [x] Identify the Starlight core version(s) worth considering
    - [x] Inventory plugins or integrations needed for versioning, search,
      admonitions, code tabs, link validation, and related docs features
    - [x] Record any gaps or risks that would block adoption
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Docs Stack Inventory' (Protocol in workflow.md)

## Phase 1: Version and Plugin Policy

- [x] Task: Define the Starlight version pinning policy
    - [x] Decide the supported Starlight release line and upgrade cadence
    - [x] Record compatibility expectations for Node, astro, and any required
      toolchain packages
    - [x] Define how Starlight upgrades are reviewed and approved
- [x] Task: Lock the required plugin set
    - [x] Classify plugins as required, optional, or out of scope
    - [x] Record the rationale for each plugin and its maintenance burden
    - [x] Define how plugin updates will be tested before acceptance
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Version and Plugin Policy' (Protocol in workflow.md)

## Phase 2: Validation and Migration Path

- [x] Task: Define docs validation and CI expectations
    - [x] Document the build and smoke-test commands needed for the Starlight
      docs stack
    - [x] Define link, content, and release-doc validation expectations
    - [x] Record any provenance or artifact checks needed for docs releases
- [x] Task: Define the migration or coexistence path
    - [x] Describe whether Starlight replaces mkdocs or coexists with it
    - [x] Record the fallback and rollback path for the current docs site
    - [x] Capture the rollout criteria before any live migration
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Validation and Migration Path' (Protocol in workflow.md)

## Phase 3: Roadmap and Handoff

- [x] Task: Update roadmap and Conductor registry
    - [x] Add the Starlight docs stack work to the remaining roadmap
    - [x] Register any open decisions, risks, or owner/action/date rows
    - [x] Refresh the track metadata and registry links
- [x] Task: Capture handoff notes for future implementation
    - [x] Summarize the selected version/plugin policy
    - [x] Summarize any open migration blockers
    - [x] Record the next implementation phase if approval is granted
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Roadmap and Handoff' (Protocol in workflow.md)
