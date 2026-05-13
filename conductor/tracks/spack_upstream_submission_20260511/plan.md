# Implementation Plan

## Phase 0: Recipe Finalization

- [x] Task: Replace feasibility placeholders
    - [x] Update source URL, version, checksum, and dependency names
    - [x] Align package name with published `mars-earth` registry identity where Spack policy allows
    - [x] Keep `pymars` only where referring to the Python import/module name
    - [x] Remove placeholder checksums from upstream-bound files
    - [x] Validate syntax locally (`python3 -m py_compile packaging/spack/package.py`)
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Recipe Finalization' (Protocol in workflow.md)
    - [x] Checkpoint evidence: `docs/hpc_track_checkpoint_notes.md`

## Phase 1: Local Spack Validation

- [x] Task: Run local Spack checks
    - [x] Run `spack spec` for the package
    - [x] Run install or dry-run validation where tooling is available
    - [x] Record any environment limitations
        - `spack` command is unavailable in this workspace.
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Local Spack Validation' (Protocol in workflow.md)

## Phase 2: Upstream Submission

- [x] Task: Prepare upstream PR
    - [x] Draft submission text with H0-only claims
    - [x] Run the HPC claim-check gate on submission docs
    - [x] Open or prepare the Spack PR
    - [x] Record URL, review status, and follow-up blockers
        - PR URL: https://github.com/spack/spack-packages/pull/4781
        - Review status: open
        - Follow-up blockers: await Spack maintainer review
    - Evidence: [upstream_submission_draft.md](upstream_submission_draft.md)
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Upstream Submission' (Protocol in workflow.md)
