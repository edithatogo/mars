# Implementation Plan

## Phase 0: Release Readiness Handoff

- [~] Task: Verify release-readiness prerequisites [blocker: Julia waiting out mandatory General review period, retriggered with release notes, 2026-05-06]
    - [~] Confirm release-readiness track is complete [blocker: Julia waiting out mandatory General review period, retriggered with release notes]
    - [x] Confirm R package publication readiness track is complete before R publication
    - [x] Confirm CI quality gates are green for the selected release target
    - [x] Confirm Rust-backed runtime binding CI is green for stable runtime releases
    - [x] Confirm training API tracks are complete for packages that claim training support
- [~] Task: Lock final release candidate metadata [blocker: Julia waiting out mandatory General review period, retriggered with release notes, 2026-05-06]
    - [x] Confirm package versions, release notes, changelog entries, and package README links
    - [x] Confirm protected release environments and required maintainer approvals
    - [~] Confirm no unresolved blocker exists for selected package targets [blocker: Julia registration submitted, waiting out mandatory period, retriggered with release notes]
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Release Readiness Handoff' (Protocol in workflow.md)

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
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Automated Registry Publication' (Protocol in workflow.md)

## Phase 2: Manual and Review-Based Registry Publication

- [~] Task: Publish review-based ecosystem packages [R ready for submission; Julia waiting period active]
    - [ ] Publish R package to r-universe [ready for external submission]
    - [ ] Prepare CRAN submission when package maturity and maintainer approval are available [ready for maintainer action]
    - [~] Register Julia package in General via Registrator.jl [submitted, waiting out mandatory General review period, retriggered with release notes]
- [~] Task: Verify review-based packages [Julia waiting period active]
    - [ ] Install available R and Julia packages from their registries
    - [ ] Run smoke tests against the shared conformance fixture
    - [ ] Record any manual-review blockers with owner/action/date [Julia waiting period active]
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Manual and Review-Based Registry Publication' (Protocol in workflow.md)

## Phase 3: Post-Publish Verification and Documentation

- [ ] Task: Complete post-publish verification [pending Julia release completion]
    - [ ] Verify registry package metadata and documentation links
    - [ ] Verify install instructions in clean environments
    - [ ] Verify release artifacts match expected versions
- [~] Task: Update release documentation [R ready for submission; Julia waiting period active]
    - [ ] Update binding release status
    - [ ] Update release notes and published package links
    - [ ] Update unresolved blocker table for any unpublished package
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Post-Publish Verification and Documentation' (Protocol in workflow.md)
