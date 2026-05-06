# Implementation Plan

## Phase 0: Docs Stack Inventory

- [ ] Task: Inventory the current docs capabilities that must be preserved
    - [ ] Document the current docs site behavior, navigation, and user-facing
      features
    - [ ] Record which content patterns matter most for release docs, tracks,
      and API/reference pages
    - [ ] Identify any current docs features that Starlight must match or
      improve
- [ ] Task: Compare Starlight capabilities and candidate plugins
    - [ ] Identify the Starlight core version(s) worth considering
    - [ ] Inventory plugins or integrations needed for versioning, search,
      admonitions, code tabs, link validation, and related docs features
    - [ ] Record any gaps or risks that would block adoption
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Docs Stack Inventory' (Protocol in workflow.md)

## Phase 1: Version and Plugin Policy

- [ ] Task: Define the Starlight version pinning policy
    - [ ] Decide the supported Starlight release line and upgrade cadence
    - [ ] Record compatibility expectations for Node, Astro, and any required
      toolchain packages
    - [ ] Define how Starlight upgrades are reviewed and approved
- [ ] Task: Lock the required plugin set
    - [ ] Classify plugins as required, optional, or out of scope
    - [ ] Record the rationale for each plugin and its maintenance burden
    - [ ] Define how plugin updates will be tested before acceptance
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Version and Plugin Policy' (Protocol in workflow.md)

## Phase 2: Validation and Migration Path

- [ ] Task: Define docs validation and CI expectations
    - [ ] Document the build and smoke-test commands needed for the Starlight
      docs stack
    - [ ] Define link, content, and release-doc validation expectations
    - [ ] Record any provenance or artifact checks needed for docs releases
- [ ] Task: Define the migration or coexistence path
    - [ ] Describe whether Starlight replaces MkDocs or coexists with it
    - [ ] Record the fallback and rollback path for the current docs site
    - [ ] Capture the rollout criteria before any live migration
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Validation and Migration Path' (Protocol in workflow.md)

## Phase 3: Roadmap and Handoff

- [ ] Task: Update roadmap and Conductor registry
    - [ ] Add the Starlight docs stack work to the remaining roadmap
    - [ ] Register any open decisions, risks, or owner/action/date rows
    - [ ] Refresh the track metadata and registry links
- [ ] Task: Capture handoff notes for future implementation
    - [ ] Summarize the selected version/plugin policy
    - [ ] Summarize any open migration blockers
    - [ ] Record the next implementation phase if approval is granted
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Roadmap and Handoff' (Protocol in workflow.md)
