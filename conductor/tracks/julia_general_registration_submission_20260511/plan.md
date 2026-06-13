# Implementation Plan

## Phase 0: Identity and Metadata Finalization

- [x] Task: Confirm package identity and ownership constraints
    - [x] Confirm the target release identity remains `MarsEarth` and not `MarsRuntime`.
    - [x] Confirm Julia package identity collision with `MarsRuntime` exists and must
      remain superseded.
    - [x] Confirm release docs and package metadata point to `bindings/julia`
      with the intended package name.
    - [x] Confirm this track stays review-only and does not alter Rust bindings.
- [x] Task: Run registration readiness validation
    - [x] Confirm Julia project metadata is complete in `bindings/julia/Project.toml`.
    - [x] Confirm release metadata and release inventory entries mention the pending
      registration state.
    - [x] Confirm release target path references do not point to `MarsRuntime` as the
      active target package.
    - [x] Record any local tooling blockers (for example, Registrator CLI or review
      account access).
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Identity and
      Metadata Finalization' (Protocol in workflow.md)

## Phase 1: Upstream Registration Draft

- [x] Task: Draft registration body
    - [x] Prepare a submission draft with concise package description and registry metadata.
    - [x] Include explicit note that this is a new package identity with `MarsRuntime`
      treated as legacy/superseded.
    - [x] Add the draft to [upstream_submission_draft.md](./upstream_submission_draft.md).
- [x] Task: Open or prepare Julia General registration PR [external Registrator access required]
    - [x] Prepare the Registrator.jl submission body and package metadata bundle.
    - [x] Confirm no compute-contract misclaims are included in registry metadata.
    - [x] Capture PR URL or blocker state in evidence notes.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Upstream Registration Draft' (Protocol in workflow.md)

## Phase 2: Verification and Handoff

- [x] Task: Confirm submission visibility and follow-up owner [blocked before external submission]
    - [x] Confirm registry status (or queue/blocked status) is recorded.
    - [x] Record response/owner/action/date and any follow-up review requirement.
    - [x] Update release and publication handoff docs with current status.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Verification and Handoff' (Protocol in workflow.md)
