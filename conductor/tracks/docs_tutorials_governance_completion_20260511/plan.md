# Implementation Plan

## Phase 0: Split Scope and Dependency Map

- [x] Task: Audit current documentation and tutorial coverage
    - [x] Inventory Python tutorials, examples, and API reference pages
    - [x] Inventory binding-specific docs and examples for R, Julia, Go, C#,
      and TypeScript
    - [x] Record which workflows are covered and which are only README-level
      examples
- [x] Task: Define the child-track boundaries
    - [x] docs/tutorial coverage child track
    - [x] docs-stack governance child track
    - [x] submission/registration synchronization child track
    - [x] Separate repo-side synchronization from user or maintainer decisions
- [x] Task: Audit remaining governance and registration state
    - [x] Record the current status and URLs for Spack, EasyBuild, and
      conda-forge PRs
    - [x] Record the HPSF TAC readiness inquiry URL and state
    - [x] Record the Julia General registration state and any follow-up notes
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Split Scope and Dependency Map' (Protocol in workflow.md)

## Phase 1: Documentation and Tutorial Coverage Track Coordination

- [x] Task: Expand tutorial and usage coverage
    - [x] Add or refine a Python tutorial that covers the supported public API
      more fully
    - [x] Add distinct example flows for fitting, prediction, model-spec
      export/validation, and interpretability
    - [x] Add or improve binding-specific walkthroughs where narrative tutorials
      are missing
- [x] Task: Keep the docs/tutorial child track dependency notes current
    - [x] Record tutorial pages that should stay in sync with demos and README snippets
    - [x] Record any example smoke-test coverage gaps
- [x] Task: Align examples with the shared conformance harness
    - [x] Ensure examples reference the shared fixture corpus or conformance
      harness where appropriate
    - [x] Keep code snippets reproducible and claim-safe
- [x] Task: Add or update documentation tests
    - [x] Add doc or example smoke tests where missing
    - [x] Verify docs pages and examples still build cleanly
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Documentation and Tutorial Coverage Track Coordination' (Protocol in workflow.md)

## Phase 2: Docs-Stack Governance Track Coordination

- [x] Task: Keep the docs stack record accurate
    - [x] Confirm mkdocs Material remains the live docs site
    - [x] Keep the Starlight governance page explicitly non-committal
    - [x] Update any migration notes if the governed evaluation changes
- [x] Task: Keep the docs-stack child track boundary explicit
    - [x] Preserve the distinction between live docs, governance record, and any future migration track
    - [x] Avoid implying that Starlight is selected unless a separate migration track says so
- [x] Task: Update docs navigation and cross-links
    - [x] Ensure tutorial pages, binding docs, and governance docs are linked
      from the main index
    - [x] Keep release and HPC docs linked from the docs map where relevant
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Docs-Stack Governance Track Coordination' (Protocol in workflow.md)

## Phase 3: Governance and Registration Synchronization Track Coordination

- [x] Task: Sync external submission and review status
    - [x] Update Spack, EasyBuild, and conda-forge tracker notes with current PR
      status and review feedback
    - [x] Keep the HPSF TAC inquiry and Julia registration notes current
    - [x] Capture any maintainer decision points that still need user input
- [x] Task: Keep the submission-sync child track separated from action tracks
    - [x] Treat PR opening, TAC submission, and Julia registration as separate external actions from documentation sync
    - [x] Surface any new feedback instead of hiding it inside the status notes
- [x] Task: Update release-facing docs
    - [x] Mirror current review/registration state in release inventory and
      publication handoff docs
    - [x] Keep the submission-readiness docs aligned with actual external state
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Governance and Registration Synchronization Track Coordination' (Protocol in workflow.md)

## Phase 4: Verification and Handoff

- [x] Task: Run documentation and claim checks
    - [x] Run the HPC claim-check gate
    - [x] Run docs build / link validation / relevant example smoke tests
- [x] Task: Prepare handoff summary
    - [x] Document what was completed
    - [x] List any remaining external blockers or decision points
- [x] Task: Confirm the child tracks are discoverable from the umbrella track
    - [x] Verify links and names in the registry
    - [x] Verify the child tracks do not duplicate the umbrella scope
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Verification and Handoff' (Protocol in workflow.md)
