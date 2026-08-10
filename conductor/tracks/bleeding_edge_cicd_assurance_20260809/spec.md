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
- Run Python formatting, linting, typing, unit, integration, end-to-end, smoke,
  property-based, metamorphic, estimator, CLI, coverage, build, and
  clean-install checks.
- Run Rust formatting, Clippy, nextest, MSRV, licence, advisory, dependency,
  build, and public-interface checks.
- Run targeted cross-language binding conformance and documentation checks.
- Run consumer-driven contract tests for each supported binding against the
  versioned ModelSpec schema, Rust core capabilities, error semantics, and
  representative consumer fixtures.
- Enforce repository and diff coverage of at least 90%.
- Run CodeQL, dependency review, secret detection, and ecosystem audits.
- Support GitHub merge queues through the `merge_group` event.

### Scheduled Deep Assurance

- Exercise supported operating systems, architectures, language runtimes, and
  bindings through explicit compatibility matrices.
- Schedule mutation testing, fuzzing, Miri, sanitizers, SemVer checks, flaky-test
  detection, and deterministic-seed validation.
- Run Deterministic Simulation Testing (DST) with versioned scenarios, fixed
  seeds, deterministic clocks and fault schedules, replayable inputs, stable
  state digests, and cross-runtime equivalence assertions.
- Run broader metamorphic suites over transformations such as row permutation,
  feature scaling, serialization round trips, batching, and parallel execution.
- Measure mutation scores by maintained source surface and reject regressions
  against explicit, ratcheting thresholds.
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

Autonomous Agentic Testing may generate adversarial cases, explore state spaces,
triage failures, and propose tests inside an isolated, read-only or disposable
environment. Agents must receive no publication credentials, have no merge
authority, and must not replace deterministic required checks. Any agentic
finding must be minimized into a deterministic reproducer and pass ordinary
reviewable tests before it can affect a required gate.

### Test Architecture and Evidence

- Unit tests isolate public and internal behavior with no network dependency.
- Integration tests exercise Python, Rust, packaging, persistence, and binding
  boundaries with controlled fixtures.
- End-to-end tests build distributable artifacts, install them into clean
  environments, fit or load a model, and validate predictions across supported
  consumer paths.
- Smoke tests provide a fast import, CLI, native-extension, artifact, and
  binding sanity gate on every relevant change and release rehearsal.
- Property-based tests generate bounded structured inputs with recorded seeds
  and minimized counterexamples.
- Metamorphic tests assert domain relations when a single exact oracle is
  insufficient and retain the transformation and failing specimen.
- Consumer-driven contract tests version provider and consumer expectations,
  verify every supported binding, and fail on incompatible schema or behavior
  drift before publication.
- Every suite emits machine-readable results recording suite kind, seed or
  scenario, artifact and source identity, environment, duration, and outcome.

## Non-Functional Requirements

- Formally support a rolling three-version Python window consisting of the
  current stable release and its two immediate predecessors (3.12-3.14 for
  this track), with bounded package metadata and matching CI coverage.
- Use Node.js 24 for stable documentation and TypeScript gates, bound the
  package contract below Node 25, and exercise newer Node releases only as
  non-blocking preview canaries until compatibility is demonstrated.
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
- Unit, integration, end-to-end, smoke, property-based, metamorphic,
  consumer-driven contract, mutation, and deterministic simulation suites have
  passing evidence at their documented gate or schedule.
- Autonomous agentic tests are sandboxed and every promoted finding has a
  deterministic non-agentic reproducer.
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
