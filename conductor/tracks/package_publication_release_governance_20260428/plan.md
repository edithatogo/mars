# Implementation Plan

## Phase 0: Release Readiness Handoff

- [x] Task: Verify release-readiness prerequisites [R confirmed in CRAN incoming/newbies pending screening; Julia MarsEarth needs new General registration]
    - [x] Confirm release-readiness track is complete [R confirmed in CRAN incoming/newbies pending screening; Julia MarsEarth needs new General registration]
    - [x] Confirm R package publication readiness track is complete before R publication
    - [x] Confirm CI quality gates are green for the selected release target
    - [x] Confirm Rust-backed runtime binding CI is green for stable runtime releases
    - [x] Confirm training API tracks are complete for packages that claim training support
- [x] Task: Lock final release candidate metadata [R CRAN screening and Julia MarsEarth registration pending]
    - [x] Confirm package versions, release notes, changelog entries, and package README links
    - [x] Confirm protected release environments and required maintainer approvals
    - [x] Confirm unresolved blocker rows are recorded for selected package targets [R CRAN screening and Julia MarsEarth registration remain external blockers]
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Release Readiness Handoff' (Protocol in workflow.md)

## Phase 1: Automated Registry Publication

- [x] Task: Publish approved automated registry packages
    - [x] Publish Python package to PyPI
    - [x] Publish Rust crate to crates.io
    - [x] Publish TypeScript package to npm
    - [x] Publish C# package to NuGet
    - [x] Publish Go module through a signed repository tag
- [x] Task: Verify automated registry packages
    - [x] Install each package from its registry
    - [x] Run smoke tests against the shared conformance fixture
    - [x] Record registry URLs, versions, and checksums where available
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Automated Registry Publication' (Protocol in workflow.md)

## Phase 2: Manual and Review-Based Registry Publication

- [x] Task: Publish review-based ecosystem packages [R published on CRAN as marsearth; Julia MarsEarth blocked pending Registrator access]
    - [x] Publish R package to r-universe or CRAN [CRAN published package is the canonical R registry release]
    - [x] Prepare CRAN submission when package maturity and maintainer approval are available [marsearth submitted and maintainer confirmation complete]
    - [x] Register Julia package in General via Registrator.jl [prepared and explicitly blocked pending maintainer Registrator access; MarsRuntime superseded]
- [x] Task: Verify review-based packages [R published; Julia registration blocked before registry install is possible]
    - [x] Install available R and Julia packages from their registries [R `marsearth` installed from CRAN; `MarsEarth` unavailable in Julia General]
    - [x] Run smoke tests against the shared conformance fixture [R CRAN `marsearth` load/predict/design-matrix smoke passed against `tests/fixtures/model_spec_v1.json`]
    - [x] Record any manual-review blockers with owner/action/date [Julia MarsEarth registration blocked pending maintainer access to Registrator.jl]
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Manual and Review-Based Registry Publication' (Protocol in workflow.md)

## Phase 3: Post-Publish Verification and Documentation

- [x] Task: Complete post-publish verification [published registries verified; Julia remains explicitly blocked]
    - [x] Verify registry package metadata and documentation links
    - [x] Verify install instructions in clean environments
    - [x] Verify release artifacts match expected versions
- [x] Task: Update release documentation [R confirmed pending screening; Julia MarsEarth registration pending]
    - [x] Update binding release status
    - [x] Update release notes and published package links
    - [x] Update unresolved blocker table for any unpublished package
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Post-Publish Verification and Documentation' (Protocol in workflow.md)
