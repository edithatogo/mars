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

- [x] Task: Add or harden Python quality gates
    - [x] Verify tests, docs build, formatting/linting, type checking, and package build checks
    - [x] Keep sklearn compatibility tests in CI
    - [x] Add coverage reporting or threshold policy where practical
- [x] Task: Add or harden Rust quality gates
    - [x] Run `cargo fmt --check`, `cargo clippy`, `cargo test`, and `cargo package`
    - [x] Add fixture conformance to Rust CI
    - [x] Add dependency audit or deny policy where practical
- [x] Task: Add advisory mutation and profiling checks
    - [x] Add mutmut scheduled workflow
    - [x] Add Scalene and py-spy profiling workflow
- [x] Task: Add or harden binding language quality gates
    - [x] Add Go format/test/module checks
    - [x] Add TypeScript test/package checks
    - [x] Add R build/check entrypoints
    - [x] Add Julia test/package hygiene checks
    - [x] Add C# format/build/test/pack checks for .NET 11
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Language Quality Gates' (Protocol in workflow.md)

## Phase 2: Cross-Platform and Artifact Smoke Tests

- [x] Task: Add cross-platform validation for native artifacts
    - [x] Add Linux, macOS, and Windows matrix coverage where the binding mechanism requires native binaries
    - [x] Cache Rust and language package artifacts with safe cache keys
    - [x] Keep fast PR checks separate from heavier release checks
- [x] Task: Add install-from-built-artifact smoke tests
    - [x] Install Python wheels or sdists before smoke tests
    - [x] Install NuGet, npm, R, Julia, Go, and Rust package artifacts where feasible
    - [x] Run a shared conformance smoke fixture against installed artifacts
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Cross-Platform and Artifact Smoke Tests' (Protocol in workflow.md)

## Phase 3: Security and Supply-Chain Automation

- [x] Task: Add dependency automation
    - [x] Configure dependency update automation for GitHub Actions and package ecosystems
    - [x] Document review policy for automated dependency PRs
    - [x] Keep lockfile updates reproducible
- [x] Task: Add security checks
    - [x] Configure vulnerability scanning for supported ecosystems
    - [x] Add secret scanning and dependency review policy documentation
    - [x] Add release provenance or attestation steps where practical
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Security and Supply-Chain Automation' (Protocol in workflow.md)

## Phase 4: Workflow Reliability and Documentation

- [x] Task: Harden workflow reliability
    - [x] Add concurrency controls to cancel stale CI runs
    - [x] Add artifact retention settings and diagnostic uploads
    - [x] Ensure workflow permissions are minimal by default
- [x] Task: Document maintainer automation workflow
    - [x] Update release and contributor docs with CI expectations
    - [x] Add local command parity for every required check
    - [x] Document common failure triage steps
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Workflow Reliability and Documentation' (Protocol in workflow.md)
