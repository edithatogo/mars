# Implementation Plan

## Phase 0: CI Inventory and Required Check Policy

- [x] Task: Inventory existing workflows and local validation commands
    - [x] Map each language package to its CI command
    - [x] Identify missing lint, format, type, security, package, and docs checks
    - [x] Document required versus advisory checks
- [x] Task: Define branch and release protection policy
    - [x] Document required PR checks
    - [x] Document protected release environments and manual approval gates
    - [x] Document least-privilege workflow permission expectations
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: CI Inventory and Required Check Policy' (Protocol in workflow.md)

## Phase 1: Language Quality Gates

- [~] Task: Add or harden Python quality gates
    - [x] Verify tests, docs build, formatting/linting, type checking, and package build checks
    - [x] Keep sklearn compatibility tests in CI
    - [x] Add coverage reporting or threshold policy where practical
- [~] Task: Add or harden Rust quality gates
    - [x] Run `cargo fmt --check`, `cargo clippy`, `cargo test`, and `cargo package`
    - [x] Add fixture conformance to Rust CI
    - [ ] Add dependency audit or deny policy where practical
- [x] Task: Add advisory mutation and profiling checks
    - [x] Add mutmut scheduled workflow
    - [x] Add Scalene and py-spy profiling workflow
- [x] Task: Add or harden binding language quality gates
    - [x] Add Go format/test/module checks
    - [x] Add TypeScript test/package checks
    - [x] Add R build/check entrypoints
    - [x] Add Julia test/package hygiene checks
    - [x] Add C# format/build/test/pack checks for .NET 11
- [~] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Language Quality Gates' (Protocol in workflow.md)

## Phase 2: Cross-Platform and Artifact Smoke Tests

- [ ] Task: Add cross-platform validation for native artifacts
    - [ ] Add Linux, macOS, and Windows matrix coverage where the binding mechanism requires native binaries
    - [ ] Cache Rust and language package artifacts with safe cache keys
    - [ ] Keep fast PR checks separate from heavier release checks
- [~] Task: Add install-from-built-artifact smoke tests
    - [ ] Install Python wheels or sdists before smoke tests
    - [x] Install NuGet, npm, R, Julia, Go, and Rust package artifacts where feasible
    - [x] Run a shared conformance smoke fixture against installed artifacts
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Cross-Platform and Artifact Smoke Tests' (Protocol in workflow.md)

## Phase 3: Security and Supply-Chain Automation

- [ ] Task: Add dependency automation
    - [ ] Configure dependency update automation for GitHub Actions and package ecosystems
    - [ ] Document review policy for automated dependency PRs
    - [ ] Keep lockfile updates reproducible
- [ ] Task: Add security checks
    - [ ] Configure vulnerability scanning for supported ecosystems
    - [ ] Add secret scanning and dependency review policy documentation
    - [ ] Add release provenance or attestation steps where practical
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Security and Supply-Chain Automation' (Protocol in workflow.md)

## Phase 4: Workflow Reliability and Documentation

- [ ] Task: Harden workflow reliability
    - [ ] Add concurrency controls to cancel stale CI runs
    - [ ] Add artifact retention settings and diagnostic uploads
    - [ ] Ensure workflow permissions are minimal by default
- [ ] Task: Document maintainer automation workflow
    - [ ] Update release and contributor docs with CI expectations
    - [ ] Add local command parity for every required check
    - [ ] Document common failure triage steps
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Workflow Reliability and Documentation' (Protocol in workflow.md)
