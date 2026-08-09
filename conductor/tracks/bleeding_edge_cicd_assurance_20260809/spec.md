# Bleeding-Edge CI/CD and Solo-Maintainer Assurance

## Overview

Modernize the repository's CI/CD system into a comprehensive, auditable
assurance control plane for its Python, Rust, TypeScript, Go, C#, R, Julia,
documentation, and release surfaces.

The system must maximize automated quality and security without requiring a
second reviewer, team membership, CODEOWNERS approval, or another human input.

## Operating Model

- Stable capabilities are required merge or release gates.
- Preview capabilities run as visible, non-blocking canaries until promoted.
- Fast checks run on pull requests.
- Expensive assurance runs nightly, weekly, or during release rehearsals.
- Repository state and hosted GitHub settings are both in scope.
- A control is not complete until its execution evidence is recorded.

## Functional Requirements

### Workflow Security

- Pin every action to a full commit SHA with readable version comments.
- Configure Renovate to maintain action pins and dependency lockfiles.
- Validate workflows with `zizmor`, `actionlint`, and strict YAML checks.
- Apply least-privilege permissions at workflow and job level.
- Add explicit timeouts, concurrency control, cancellation, and retention limits.
- Add runner egress monitoring and progressively enforce network allowlists.
- Test trusted and untrusted pull-request boundaries.
- Detect unsafe interpolation, cache poisoning, and privilege escalation paths.
- Inventory workflow dependencies and detect unsupported runner/action versions.

### Required Pull-Request Gates

- Validate locked dependencies, generated files, configuration, and documentation.
- Run Python formatting, linting, typing, unit, integration, property-based,
  estimator, CLI, coverage, build, and clean-install checks.
- Run Rust formatting, Clippy, nextest, MSRV, licence, advisory, dependency,
  build, and public-interface checks.
- Run targeted cross-language binding conformance and documentation checks.
- Enforce repository and diff coverage of at least 90%.
- Run CodeQL, dependency review, secret detection, and ecosystem audits.
- Support GitHub merge queues through the `merge_group` event.

### Scheduled Deep Assurance

- Exercise supported operating systems, architectures, language runtimes, and
  bindings through explicit compatibility matrices.
- Schedule mutation testing, fuzzing, Miri, sanitizers, SemVer checks, flaky-test
  detection, and deterministic-seed validation.
- Enforce performance and memory regression budgets.
- Compare repeated builds and generated artifacts for reproducibility.
- Retain machine-readable results and bounded failure evidence.

### Supply Chain and Releases

- Use OIDC and trusted publishing instead of persistent publication tokens.
- Isolate release environments and permissions.
- Generate ecosystem-aware SBOMs for each distributable artifact.
- Produce and verify provenance and SBOM attestations.
- Verify registry packages match reviewed workflow artifacts.
- Protect tags and releases and use immutable releases where supported.
- Record release receipts, digests, source commits, workflow runs, and results.

### Hosted GitHub Controls

- Enable CodeQL for all supported repository languages.
- Enable secret scanning, push protection, dependency review, and private
  vulnerability reporting where available.
- Enforce full-SHA Actions policy where available.
- Add a ruleset requiring automated checks, linear history, and controlled
  force-push and deletion behavior, with zero required human approvals.
- Configure merge queue or automated merge only when required checks are green.
- Activate Renovate and retire Dependabot only after Renovate runs successfully.

### Harness and Context Engineering

- Provide one documented local command matching required hosted checks.
- Maintain a machine-readable assurance manifest mapping controls to evidence.
- Produce CI receipts and test reports in durable, inspectable formats.
- Document workflow topology, trust boundaries, schedules, ownership, failure
  handling, rollback, and solo-maintainer self-review.
- Detect documentation, fixture, schema, generated-code, and metadata drift.
- Track CI reliability, duration, cost, flakiness, and failure classification.
- Keep Conductor status synchronized with repository and hosted evidence.

### Preview Canaries

Non-blocking canaries may cover Ubuntu 26.04, ARM64 runners, free-threaded or
pre-release Python, Rust nightly, uv malware checking, and guarded agentic
automation for issue triage, CI diagnosis, or documentation drift.

Agentic workflows must receive no publication credentials, have no merge
authority, and must not replace deterministic required checks.

## Non-Functional Requirements

- Preserve public APIs and package behavior.
- Prefer deterministic, locked, non-interactive commands.
- Keep required pull-request feedback fast through path-aware and tiered checks.
- Keep expensive checks scheduled or release-bound unless their signal justifies
  promotion to the required gate.
- Preserve recovery paths for the sole maintainer without normalizing bypasses.
- Keep external registry, credential, preview, and hosted-service gates explicit.

## Acceptance Criteria

- Every required workflow passes strict syntax and security validation.
- Every action is pinned to a full commit SHA.
- Required jobs have explicit permissions, timeouts, concurrency, and retention.
- Local and hosted required-check inventories agree.
- Required checks run successfully on pull requests and merge groups.
- Hosted security controls are verified through authenticated readback.
- Renovate executes successfully before Dependabot automation is retired.
- Release artifacts have generated and successfully verified attestations.
- Stable controls pass and preview controls remain visibly non-blocking.
- No control requires a second reviewer or second human approval.
- Evidence distinguishes configured, enabled, executed, passing, deferred, and
  blocked controls.
- The completed work is delivered through a green pull request, merged, and the
  local checkout is synchronized and clean.

## Out of Scope

- Mandatory reviewer counts, CODEOWNERS approval, or team gates.
- Unbounded self-hosted runners.
- Preview features as required gates before reliability is demonstrated.
- Publishing or credential changes without verified registry authorization.
- Treating configuration files alone as proof that hosted controls are active.
