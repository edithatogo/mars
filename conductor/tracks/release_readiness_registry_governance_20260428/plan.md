# Implementation Plan

## Phase 0: Registry and Ownership Inventory

- [ ] Task: Audit registry ownership and package names
    - [ ] Confirm PyPI project ownership and trusted publishing or token status
    - [ ] Confirm crates.io owner and crate name status
    - [ ] Confirm npm organization/package ownership and token status
    - [ ] Confirm NuGet package ID and API key status
    - [ ] Confirm Go module path and tag strategy
    - [ ] Confirm R r-universe/CRAN and Julia General submission requirements
- [ ] Task: Create release blocker table
    - [ ] Record owner, action, status, and date for each external blocker
    - [ ] Distinguish automation blockers from maintainer-approval blockers
    - [ ] Add the table to release documentation
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Registry and Ownership Inventory' (Protocol in workflow.md)

## Phase 1: Package Metadata and Artifact Contents

- [ ] Task: Verify package metadata
    - [ ] Check descriptions, licenses, authors, repository URLs, package READMEs, and keywords
    - [ ] Check package contents for generated files, missing docs, and accidental test/build artifacts
    - [ ] Align package versions or document intentionally independent versions
- [ ] Task: Add artifact inspection commands
    - [ ] Add Python build and metadata validation
    - [ ] Add `cargo package` validation
    - [ ] Add `npm pack --dry-run` validation
    - [ ] Add R build/check validation
    - [ ] Add Julia package validation
    - [ ] Add NuGet pack validation
    - [ ] Add Go module validation
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Package Metadata and Artifact Contents' (Protocol in workflow.md)

## Phase 2: Release Rehearsal Workflow

- [ ] Task: Implement non-publishing release rehearsal
    - [ ] Trigger all package dry-runs from one manual workflow or documented sequence
    - [ ] Upload release candidate artifacts for inspection
    - [ ] Run install-from-artifact smoke tests where feasible
- [ ] Task: Add release approval gates
    - [ ] Define protected release environments
    - [ ] Require explicit package selection and release approval
    - [ ] Ensure failed conformance blocks release rehearsal success
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Release Rehearsal Workflow' (Protocol in workflow.md)

## Phase 3: Release Documentation and Policy

- [ ] Task: Document release policy
    - [ ] Define pre-release versus stable release criteria
    - [ ] Define no-stable-publish rule for duplicated replay implementations
    - [ ] Define version-bump, changelog, release-notes, rollback, yanking, and deprecation policy
- [ ] Task: Document package-manager-specific steps
    - [ ] Document PyPI, crates.io, npm, r-universe/CRAN, Julia General, NuGet, and Go module release paths
    - [ ] Document required secrets and protected environments
    - [ ] Document registry blocker remediation steps
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Release Documentation and Policy' (Protocol in workflow.md)
