# Implementation Plan

## Phase 0: Publication Readiness Audit

- [x] Task: Verify R package publication prerequisites
    - [x] Confirm package metadata, version, and release notes are ready
    - [x] Confirm build, check, and vignette outputs are available
    - [x] Confirm r-universe / CRAN submission path and maintainer access
- [x] Task: Prepare submission materials
    - [x] Gather the exact package URL, registry target, and required fields
    - [x] Assemble browser submission checklist and fallback notes
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Publication Readiness Audit' (Protocol in workflow.md)
    - [x] Checkpoint evidence: `docs/release_inventory.md` and `docs/publication_handoff.md`

## Phase 1: Browser-Based Submission

- [x] Task: Submit the R package through the registry workflow
    - [x] Open the submission browser flow
    - [x] Complete the r-universe submission steps
    - [x] If appropriate, prepare CRAN submission inputs for the maintainer
    - [x] Submit renamed `marsearth` tarball to CRAN
- [x] Task: Verify submission result
    - [x] Capture superseded `marsruntime` / `mars.earth` CRAN upload state
    - [x] Record the superseded upload timestamp and registry feedback
    - [x] Open maintainer confirmation link in the browser
    - [x] Check CRAN incoming/screening status after confirmation
    - [x] Maintain proof notes are in `docs/release_inventory.md` and
      `bindings/r/cran-comments.md`.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Browser-Based Submission' (Protocol in workflow.md)
    - [x] Checkpoint evidence: `docs/release_inventory.md` and `docs/publication_handoff.md`

## Phase 2: Post-Submission Verification and Documentation

- [x] Task: Verify registry visibility and installability
    - [x] Check CRAN incoming queue state and confirm the current submission artifact
    - [x] Confirm CRAN check page or screening status after queueing
    - [x] Confirm install or visibility instructions for maintainers
- [x] Task: Update release documentation
    - [x] Update release inventory and publication handoff state
    - [x] Update the release checklist with the submission outcome
    - [x] Update blocker notes if external review remains pending
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Post-Submission Verification and Documentation' (Protocol in workflow.md)
