# Implementation Plan

## Phase 0: Baseline, Isolation, and Control Architecture

- [x] Task: Establish the implementation baseline (7a342f2)
  - [x] Preserve and inventory existing staged and unstaged changes.
  - [x] Create a dedicated feature branch without losing prior scoped changes.
  - [x] Record the base commit, workflows, hosted settings, dependency PRs, required checks, durations, failures, coverage, security, and release evidence.
- [x] Task: Define the assurance control model (47598d8)
  - [x] Classify controls as required PR gates, scheduled checks, release gates, or preview canaries.
  - [x] Map each control to its command, workflow, trigger, evidence, failure policy, and remediation.
  - [x] Define solo-maintainer rules with automated checks and zero required reviewers.
  - [x] Define configured, enabled, executed, passing, deferred, and blocked states.
- [ ] Task: Write failing policy tests for the baseline gaps
  - [ ] Test action pinning, permissions, timeouts, concurrency, retention, merge-group coverage, and control declarations.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Baseline, Isolation, and Control Architecture' (Protocol in workflow.md)

## Phase 1: Canonical Local Harness and Context

- [ ] Task: Specify the one-command assurance harness
  - [ ] Define fast, full, security, release-rehearsal, and preview profiles.
  - [ ] Define deterministic environments, seeds, budgets, outputs, and platform behavior.
- [ ] Task: Write failing harness contract tests
  - [ ] Test discovery, non-interactive execution, exit codes, receipts, partial failures, and hosted parity.
- [ ] Task: Implement the canonical harness
  - [ ] Extend existing Make, tox, and uv orchestration instead of creating competing entry points.
  - [ ] Add locked quality, test, security, build, docs, and evidence commands.
  - [ ] Produce JUnit, coverage, SARIF, benchmark, and machine-readable receipts.
  - [ ] Add usage and troubleshooting documentation.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Canonical Local Harness and Context' (Protocol in workflow.md)

## Phase 2: Workflow Supply-Chain Hardening

- [ ] Task: Write failing workflow-security tests
  - [ ] Validate YAML, mutable references, interpolation, permissions, events, timeouts, concurrency, and retention.
- [ ] Task: Pin and govern workflow dependencies
  - [ ] Pin every action to a full commit SHA with version comments.
  - [ ] Configure Renovate to maintain SHA pins, workflow dependencies, and strict config validation.
- [ ] Task: Apply workflow runtime controls
  - [ ] Set least privilege, timeouts, concurrency cancellation, bounded retention, and trusted cache boundaries.
  - [ ] Add runner egress monitoring in audit mode.
- [ ] Task: Add workflow policy gates
  - [ ] Run `zizmor`, `actionlint`, YAML validation, custom policy tests, dependency inventory, EOL checks, and SARIF upload.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Workflow Supply-Chain Hardening' (Protocol in workflow.md)

## Phase 3: Fast Required Pull-Request Assurance

- [ ] Task: Write failing tests for the required-check contract
  - [ ] Verify check names, triggers, pull requests, merge groups, path behavior, and matrix failure propagation.
- [ ] Task: Modernize Python assurance
  - [ ] Use locked uv and pinned toolchains.
  - [ ] Run Ruff, `ty`, metadata, lock, unit, integration, property, estimator, CLI, and >=90% coverage checks.
  - [ ] Build and install wheels and sdists in clean environments.
- [ ] Task: Modernize Rust assurance
  - [ ] Run formatting, Clippy, nextest, docs, features, MSRV, stable, cargo-deny, RustSec, licences, and source checks.
  - [ ] Build release artifacts and exercise public interfaces.
- [ ] Task: Modernize binding and documentation assurance
  - [ ] Run targeted TypeScript, Go, C#, R, Julia, fixture, schema, generated-code, API-doc, Starlight, link, and package smoke checks.
- [ ] Task: Add cross-cutting security gates
  - [ ] Enable dependency review, ecosystem audits, CodeQL or SARIF analysis, and secret checks.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Fast Required Pull-Request Assurance' (Protocol in workflow.md)

## Phase 4: Scheduled Deep Assurance

- [ ] Task: Write failing tests for deep-check schedules and evidence
  - [ ] Verify nightly, weekly, release, and dispatch triggers, non-required status, and durable failures.
- [ ] Task: Add exhaustive compatibility matrices
  - [ ] Cover supported runtimes, operating systems, stable ARM64, clean installs, and all maintained bindings.
- [ ] Task: Add adversarial and semantic testing
  - [ ] Schedule mutation, fuzzing, Miri, sanitizers, nightly, SemVer, flaky-test, and deterministic-seed checks.
- [ ] Task: Add performance and reproducibility assurance
  - [ ] Enforce runtime and memory budgets, comparable benchmarks, repeated-build digests, and deterministic outputs.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Scheduled Deep Assurance' (Protocol in workflow.md)

## Phase 5: Release and Supply-Chain Integrity

- [ ] Task: Write failing release-policy tests
  - [ ] Detect persistent credentials, unattested artifacts, rebuild divergence, and missing source, digest, SBOM, or receipt evidence.
- [ ] Task: Harden build and publication identity
  - [ ] Use OIDC and trusted publishing, isolate release authority, protect pull-request boundaries, tags, and releases.
- [ ] Task: Produce and verify release evidence
  - [ ] Generate per-artifact SBOMs, provenance and SBOM attestations, verification, checksums, source and workflow identities.
- [ ] Task: Add release rehearsals and post-release tests
  - [ ] Build once, rehearse without registry mutation, install published packages, verify digests, and document rollback.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 5: Release and Supply-Chain Integrity' (Protocol in workflow.md)

## Phase 6: Hosted GitHub Security and Merge Controls

- [ ] Task: Capture hosted settings before mutation
  - [ ] Record repository, Actions, security, ruleset, environment, permission, dependency-bot, and rollback state.
- [ ] Task: Apply solo-maintainer-compatible rules
  - [ ] Require stable automated checks and linear history; prevent force pushes and deletion; require zero reviews.
- [ ] Task: Configure Actions and merge policy
  - [ ] Enforce SHA pins and safe events where available; configure merge queue or safe automerge; verify merge-group checks.
- [ ] Task: Enable hosted security controls
  - [ ] Enable and verify CodeQL, secret scanning, push protection, dependency features, private reporting, SARIF, and dashboards.
- [ ] Task: Complete the Renovate migration
  - [ ] Activate Renovate, verify validation and a real run, then retire Dependabot and reconcile its PRs.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 6: Hosted GitHub Security and Merge Controls' (Protocol in workflow.md)

## Phase 7: Preview Canaries and Emerging Capabilities

- [ ] Task: Define canary promotion and retirement policy
  - [ ] Set success runs, failure rates, maintenance budgets, promotion criteria, and expiry dates.
- [ ] Task: Add platform and toolchain canaries
  - [ ] Add Ubuntu 26.04, ARM64, free-threaded or pre-release Python, Rust nightly, and uv malware-check experiments where feasible.
- [ ] Task: Add guarded agentic experiments
  - [ ] Limit agents to triage, diagnosis, or docs; grant no secrets, merge authority, or required-check ownership; validate outputs deterministically.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 7: Preview Canaries and Emerging Capabilities' (Protocol in workflow.md)

## Phase 8: Evidence, Documentation, and Observability

- [ ] Task: Implement the assurance manifest
  - [ ] Map controls to implementation, triggers, evidence, ownership, status, schema validation, and drift rejection.
- [ ] Task: Implement CI receipts and observability
  - [ ] Record workflow identity, commit, environment, command, duration, outcome, reliability, flakiness, cost, and truth boundary.
- [ ] Task: Update operator and contributor documentation
  - [ ] Document topology, trust boundaries, parity commands, remediation, release verification, rollback, and self-review.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 8: Evidence, Documentation, and Observability' (Protocol in workflow.md)

## Phase 9: End-to-End Delivery and Closure

- [ ] Task: Run the complete local acceptance harness
  - [ ] Run quality, test, security, packaging, docs, policy, representative deep, and release-rehearsal checks.
  - [ ] Record deviations, external gates, and preview limitations.
- [ ] Task: Perform solo-maintainer formal self-review
  - [ ] Review the specification, diff, permissions, trust boundaries, rollback, evidence, and absence of second-human gates.
- [ ] Task: Deliver through a pull request
  - [ ] Commit scoped changes with notes and track references, push, open a PR, and monitor all required checks.
  - [ ] Diagnose and fix failures until every required check passes.
- [ ] Task: Merge and reconcile
  - [ ] Merge only when green, verify post-merge workflows, synchronize main, remove the merged branch, and confirm a clean checkout and synchronized submodules.
- [ ] Task: Synchronize Conductor and project documentation
  - [ ] Record final repository and hosted evidence, task and checkpoint commits, registry status, and explicit deferred or blocked items.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 9: End-to-End Delivery and Closure' (Protocol in workflow.md)
