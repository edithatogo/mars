# Implementation Plan: GitHub Notification Hygiene and External-Human Signalling

## Phase 1: Evidence, contracts, and architecture

- [~] Task: Inventory standards and current automation
  - [ ] Inspect relevant `edithatogo` standards and templates read-only.
  - [ ] Inspect Mars automation, security policies, and scripting conventions.
  - [ ] Pin current GitHub notification contracts, permissions, pagination, and
    rate-limit behavior from primary documentation.
  - [ ] Confirm no third-party repository writes are required.
- [ ] Task: Define versioned notification and decision contracts
  - [ ] Define typed input, classification, decision, and audit schemas.
  - [ ] Define actor classes, reason classes, precedence, and stable rule IDs.
  - [ ] Define fail-open, privacy, approved-host, and retention invariants.
- [ ] Task: Record architecture before implementation
  - [ ] Document Python 3.12-3.14, GitHub CLI transport, and Windows Task
    Scheduler.
  - [ ] Document rejection of repository-hosted personal-token automation.
  - [ ] Record dependency ordering and explicit track state/lock conventions.
- [ ] Task: Automated Phase Verification & Checkpoint
  - [ ] Run documentation, schema, policy, link, and security checks.
  - [ ] Run `conductor-review`, automatically remediate findings, rerun gates,
    and record the checkpoint according to `conductor/workflow.md`.

## Phase 2: Test harness and deterministic fixtures

- [ ] Task: Write failing classification tests
  - [ ] Cover external users, self, bots, Apps, workflows, automation, and
    unknown actors.
  - [ ] Cover every supported reason and conflicting evidence.
  - [ ] Prove uncertain cases remain unread.
- [ ] Task: Write failing advanced harness tests
  - [ ] Add property-based tests for malformed and incomplete payloads.
  - [ ] Add metamorphic tests for irrelevant, reordered, or unknown fields.
  - [ ] Add contract tests using synthetic redacted API fixtures.
  - [ ] Add deterministic simulations for pagination, retries, rate limits,
    deletion, partial failure, overlap, and schema drift.
- [ ] Task: Establish mutation and security baselines
  - [ ] Target safety-critical classification and write-decision branches.
  - [ ] Test token disclosure, log injection, malicious URLs, command injection,
    and fixture privacy.
- [ ] Task: Automated Phase Verification & Checkpoint
  - [ ] Confirm Red-phase failures are attributable to missing implementation.
  - [ ] Run `conductor-review`, automatically remediate harness defects, rerun
    gates, and record the checkpoint.

## Phase 3: Classification and audit implementation

- [ ] Task: Implement pure classification logic
  - [ ] Implement typed models, actor/reason classifiers, precedence, and
    fail-open decisions.
  - [ ] Return explicit evidence and stable rule identifiers without network
    side effects.
- [ ] Task: Implement privacy-preserving audit and configuration
  - [ ] Persist sanitized metadata only and enforce 30-day retention.
  - [ ] Apply restrictive local permissions where supported.
  - [ ] Default to dry-run and require explicit enforcement.
  - [ ] Add emergency-disable and secure configuration validation.
- [ ] Task: Make classification gates green
  - [ ] Run unit, property, metamorphic, contract, simulation, mutation,
    coverage, typing, lint, and security gates.
- [ ] Task: Automated Phase Verification & Checkpoint
  - [ ] Run `conductor-review`, apply fixes automatically, rerun affected gates,
    and record the checkpoint.

## Phase 4: GitHub adapter and safe enforcement

- [ ] Task: Write failing adapter and processing tests
  - [ ] Cover authenticated pagination, subject resolution, host validation,
    actor extraction, rate limits, dry-run, and thread updates.
- [ ] Task: Implement credential-preserving GitHub CLI adapter
  - [ ] Invoke `gh api` without extracting or printing tokens.
  - [ ] Validate all responses and restrict subject resolution to approved
    targets.
  - [ ] Enforce bounded pagination, requests, retries, and backoff.
- [ ] Task: Implement idempotent processing engine
  - [ ] Enumerate owned-repository notifications.
  - [ ] Preserve external-human and unknown threads.
  - [ ] Mark only positively classified automation read.
  - [ ] Continue safely after per-thread failures and emit metadata receipts.
- [ ] Task: Run integration and end-to-end simulations
  - [ ] Exercise fake-API dry-run, enforcement, partial failure, recovery,
    concurrency, and third-party-write prohibitions.
- [ ] Task: Automated Phase Verification & Checkpoint
  - [ ] Run all adapter, integration, quality, privacy, and security gates.
  - [ ] Run `conductor-review`, apply fixes automatically, and record the
    checkpoint.

## Phase 5: Windows scheduling and operations

- [ ] Task: Write failing scheduler lifecycle tests
  - [ ] Cover five-minute scheduling, spaces in paths, credential-free task
    arguments, inspect, disable, enable, and uninstall.
- [ ] Task: Implement user-scoped Task Scheduler integration
  - [ ] Run non-interactively without elevation or a visible window.
  - [ ] Prevent overlapping instances and embedded secrets.
- [ ] Task: Implement lifecycle and health commands
  - [ ] Provide install, status, health, disable, emergency-stop, re-enable, and
    clean uninstall operations.
- [ ] Task: Document authorization and operation
  - [ ] Document notification-scope authorization, dry-run review,
    enforcement, privacy, offline behavior, revocation, and mobile-push limits.
- [ ] Task: Automated Phase Verification & Checkpoint
  - [ ] Run scheduler unit, integration, smoke, security, and documentation
    tests.
  - [ ] Run `conductor-review`, remediate automatically, and record the
    checkpoint.

## Phase 6: Live rollout and governance synchronization

- [ ] Task: Verify explicit GitHub authorization
  - [ ] Confirm account `edithatogo` and `notifications` scope without exposing
    the token.
  - [ ] Record absent authorization as an external blocker.
- [ ] Task: Conduct live dry-run
  - [ ] Enumerate live metadata without modifying threads.
  - [ ] Confirm external-human and ambiguous threads remain preserved.
  - [ ] Confirm proposed automated actions match policy.
- [ ] Task: Enable and smoke-test enforcement
  - [ ] Enable only after clean dry-run evidence.
  - [ ] Process eligible existing notifications and verify preserved threads.
  - [ ] Install, enable, and smoke-test the five-minute scheduled task.
- [ ] Task: Synchronize Conductor and GitHub governance
  - [ ] Maintain focused Mars GitHub issues for implementation,
    security/privacy, and operational rollout.
  - [ ] Cross-reference issues, track, pull request, and Mars project items.
  - [ ] Record dependencies on follow-on programme tracks.
  - [ ] Prove no third-party repository writes occurred.
- [ ] Task: Automated Phase Verification & Checkpoint
  - [ ] Run the full local quality, security, privacy, mutation, simulation, and
    operational suite.
  - [ ] Run `conductor-review`, remediate automatically, and record the
    checkpoint.

## Phase 7: Small PR delivery and closure

- [ ] Task: Perform final automated track review
  - [ ] Run `conductor-review` across the complete track.
  - [ ] Automatically fix all actionable in-scope findings and rerun gates.
  - [ ] Keep credential or infrastructure blockers explicit.
- [ ] Task: Deliver one focused pull request
  - [ ] Ensure the PR contains only this track.
  - [ ] Cross-reference issues and project items.
  - [ ] Require every hosted check to finish green; queued, unexpected skips,
    cancellation, timeout, or failure are not green.
- [ ] Task: Merge and verify
  - [ ] Merge only after green checks and verify post-merge workflows.
  - [ ] Verify local scheduled automation remains healthy.
  - [ ] Verify no third-party repository mutations occurred.
- [ ] Task: Archive and clean up
  - [ ] Reconcile specification, plan, repository, hosted, and operational
    evidence.
  - [ ] Archive all track artifacts and update registry/project references.
  - [ ] Remove only clean merged disposable branches and worktrees while
    preserving the dirty primary checkout.
- [ ] Task: Automated Phase Verification & Final Checkpoint
  - [ ] Attach final evidence as a git note and record the merged PR and archive
    state.
