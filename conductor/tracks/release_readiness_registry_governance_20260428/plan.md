# Implementation Plan

Current external gate: Julia is still waiting out the mandatory General review
period after the retriggered registration. All other registry ownership and
credential checks in this lane are complete.

## Phase 0: Registry and Ownership Inventory [checkpoint: dd2bbec]

- [x] Task: Audit public package-name availability
    - [x] Confirm PyPI `mars-earth` exists and note the `pymars` namespace collision
    - [x] Confirm crates.io `mars-earth` is publicly registered as `0.1.0`
    - [x] Confirm npm `mars-earth` is publicly registered as `0.0.0`
    - [x] Confirm NuGet `mars-earth` is publicly registered as `0.0.0`
    - [x] Confirm Go module path is controlled by repository tags rather than a registry
    - [x] Confirm R `marsruntime` and Julia `MarsRuntime` remain release targets rather than published registry packages
- [~] Task: Confirm registry ownership and credential status [blocker: Julia waiting out mandatory General review period, retriggered with release notes, 2026-05-06]
    - [x] Confirm PyPI project ownership and trusted publishing or token status
    - [x] Confirm crates.io owner and crate name status [published on crates.io as `0.1.0`, 2026-05-04]
    - [x] Confirm npm organization/package ownership and publish-automation status [published on npm as `0.0.0`, 2026-05-04]
    - [x] Confirm NuGet trusted publishing policy and login path [published on NuGet as `0.0.0`, 2026-05-04]
    - [x] Confirm Go module path and tag strategy
    - [x] Confirm R r-universe/CRAN requirements; Julia registration submitted and waiting out mandatory General review period [blocker: Julia waiting out mandatory General review period, retriggered with release notes, 2026-05-06]
    - [x] Complete R package publication readiness track before treating R as release-ready
- [x] Task: Confirm repository-side release wiring
    - [x] Confirm PyPI publish configuration points at `mars-earth`
    - [x] Confirm crates.io, npm, and NuGet workflows reference release secrets or trusted publishing
    - [x] Confirm Go release path is tag-based rather than registry-based
    - [x] Confirm R and Julia release notes point to r-universe/CRAN and Registrator
- [x] Task: Create release blocker table
    - [x] Record owner, action, status, and date for each external blocker
    - [x] Distinguish automation blockers from maintainer-approval blockers
    - [x] Add the table to release documentation
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Registry and Ownership Inventory' (Protocol in workflow.md) [with blockers]

## Phase 1: Package Metadata and Artifact Contents [checkpoint: fa376e1]

- [x] Task: Verify package metadata
    - [x] Check descriptions, licenses, authors, repository URLs, package README files, and keywords
    - [x] Check package contents for generated files, missing docs, and accidental test/build artifacts
    - [x] Align package versions or document intentionally independent versions
    - [x] Define version-reporting and runtime/core version parity for each binding
    - [x] Define logging defaults and verbosity conventions for each binding
- [x] Task: Define cross-language operational parity
    - [x] Record package version, runtime/core version, and release-tag mappings
    - [x] Document logging defaults, verbosity toggles, and error-context behavior
    - [x] Verify package identities, repository URLs, and README links match the canonical project
- [x] Task: Add artifact inspection commands
    - [x] Add Python build and metadata validation
    - [x] Add `cargo package` validation
    - [x] Add `npm pack --dry-run` validation
    - [x] Add R build/check validation
    - [x] Add Julia package validation
    - [x] Add NuGet pack validation
    - [x] Add Go module validation
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Package Metadata and Artifact Contents' (Protocol in workflow.md)

## Phase 2: Release Rehearsal Workflow [checkpoint: 0957249]

- [x] Task: Implement non-publishing release rehearsal
    - [x] Trigger all package dry-runs from one manual workflow or documented sequence
    - [x] Upload release candidate artifacts for inspection
    - [x] Run install-from-artifact smoke tests where feasible
- [x] Task: Add release approval gates
    - [x] Define protected release environments
    - [x] Require explicit package selection and release approval
    - [x] Ensure failed conformance blocks release rehearsal success
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Release Rehearsal Workflow' (Protocol in workflow.md)

## Phase 3: Release Documentation and Policy [checkpoint: 5795166]

- [x] Task: Document release policy
    - [x] Define pre-release versus stable release criteria
    - [x] Define no-stable-publish rule for duplicated replay implementations
    - [x] Define version-bump, changelog, release-notes, rollback, yanking, and deprecation policy
    - [x] Include logging and versioning parity requirements for all bindings
- [x] Task: Document package-manager-specific steps
    - [x] Document PyPI, crates.io, npm, r-universe/CRAN, Julia General, NuGet, and Go module release paths
    - [x] Document required secrets and protected environments
    - [x] Document registry blocker remediation steps
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Release Documentation and Policy' (Protocol in workflow.md)
