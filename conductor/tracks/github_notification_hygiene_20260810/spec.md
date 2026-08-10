# Specification: GitHub Notification Hygiene and External-Human Signalling

## Overview

Create secure, local, account-wide notification automation for repositories
owned by `edithatogo`. Preserve actionable notifications initiated by external
humans while automatically clearing notifications produced by the maintainer,
bots, GitHub Apps, workflows, releases, dependency automation, and routine
machine-generated activity.

The automation supports a solo maintainer, requires no second reviewer, avoids
repository-held personal credentials, and fails open whenever classification
is uncertain.

## Functional requirements

1. Run locally under the Windows user account using the authenticated GitHub
   CLI credential.
2. Request only the GitHub `notifications` permission needed for notification
   inspection and updates.
3. Inspect notifications across repositories owned by `edithatogo`.
4. Classify initiating actors as external human, repository owner/self, bot,
   GitHub App, workflow, release automation, or unknown.
5. Keep external-human and unknown notifications unread.
6. Automatically mark positively identified self-generated, Renovate,
   Dependabot, GitHub App, bot, workflow, release, routine security-analysis,
   and other deterministic machine-generated notifications read.
7. Resolve notification subjects only as necessary to identify actor and
   reason.
8. Never persist notification titles, bodies, comments, private content, or
   credentials.
9. Maintain metadata-only audit records containing timestamp, repository,
   notification identifier, reason, actor classification, rule identifier,
   and action.
10. Retain audit metadata for 30 days.
11. Provide dry-run and enforcement modes, with dry-run as the default.
12. Run every five minutes while the Windows host and authenticated session
    are available.
13. Provide emergency-disable and clean uninstall mechanisms.
14. Prevent overlapping scheduled executions.
15. Use bounded retries and GitHub rate-limit awareness.
16. Leave notification state unchanged on API failure, ambiguous identity,
    incomplete evidence, or schema drift.
17. Provide deterministic fixtures for every classification path.
18. Produce a human-readable summary without notification content.
19. Cross-reference the Conductor track, focused Mars GitHub issues, and the
    Mars Conductor Roadmap project.
20. Never post, comment, open issues, or create pull requests in third-party
    repositories without explicit user approval for the specific action.

## Security and privacy requirements

- Never store a GitHub token in this repository.
- Use credential-manager-backed GitHub CLI authentication.
- Never print tokens or authorization headers.
- Treat notification contents as sensitive transient data.
- Store only minimal metadata required for audit and debugging.
- Restrict audit-file permissions to the local user where supported.
- Validate every GitHub API response before acting.
- Use an explicit external-human allow policy and fail open for unknown actors.
- Restrict subject resolution to approved GitHub API hosts.
- Include dependency, static-analysis, secret-scanning, and misuse-case tests.
- Document credential revocation and automation removal.

## Testing and harness requirements

- Unit tests for actor and reason classification.
- Property-based tests for payload combinations and malformed responses.
- Contract tests against recorded, synthetic, redacted GitHub API schemas.
- Metamorphic tests ensuring irrelevant fields cannot change classification.
- Deterministic simulation of pagination, rate limits, retries, races, schema
  drift, and partial failures.
- Integration tests using an isolated fake API.
- End-to-end dry-run and enforcement tests using synthetic fixtures.
- Mutation testing of safety-critical rules.
- Security tests for token disclosure, malicious URLs, log injection, command
  injection, and schema confusion.
- Smoke testing for the installed scheduled task.
- Idempotency and concurrency tests.
- Agentic exploratory testing whose findings require deterministic reproducers
  before affecting gates.

## Operational lifecycle

1. Obtain the required notification scope through explicit GitHub
   authorization.
2. Run fixture-backed validation.
3. Run against the live inbox in dry-run mode.
4. Compare proposed actions with expected classifications.
5. Enable enforcement only after clean dry-run evidence.
6. Schedule execution every five minutes.
7. Monitor metadata-only audit output and failures.
8. Provide one-command disable and uninstall paths.

## Conductor and delivery requirements

- Use current pinned upstream Conductor `main` for repository governance.
- Adopt experimental Conductor mechanics selectively: isolated worktrees,
  explicit state and locks, dependency-aware track ordering, bounded autonomous
  loops, and automated review/remediation.
- Do not pin production governance to divergent or unmerged upstream branches.
- Deliver one implementation phase per small pull request, splitting phases
  further when needed, and merge only after every required hosted check is
  green. No pull request may exceed one track.
- Preserve the dirty primary checkout and remove only clean, merged disposable
  worktrees and branches.
- Require no second-human approval or fictional reviewer.

## Acceptance criteria

- All external-human test cases remain unread.
- All unknown or ambiguous cases remain unread.
- Covered self, bot, App, and automation cases are marked read in enforcement
  mode.
- Dry-run mode never changes GitHub state.
- No notification content or credentials are persisted.
- API and identity failures cannot cause a notification to be marked read.
- Repeated execution is idempotent and overlapping execution is prevented.
- Audit retention is automatically bounded to 30 days.
- The scheduled task can be installed, inspected, disabled, and removed
  reproducibly.
- Local quality, security, contract, simulation, mutation, and end-to-end gates
  pass.
- The corresponding focused pull request and hosted checks are green before
  merge.
- Conductor artifacts, GitHub issues, pull requests, and project items
  cross-reference one another.

## Out of scope

- Suppressing notifications initiated by external humans.
- Modifying GitHub's global mobile push settings.
- Guaranteeing cancellation of a mobile push dispatched before filtering.
- Storing a personal notification token in GitHub Actions.
- Operating while the local Windows host is unavailable.
- Modifying third-party repositories.
- Rust core, bindings, framework adapters, documentation modernization, and
  `astro-polyglot` branch protection; these are coordinated follow-on tracks.

## Coordinated follow-on tracks

1. Contract-first architecture and machine-readable capability model.
2. Rust core modernization, parallelism, memory safety, SIMD, zero-copy
   boundaries, profiling, fuzzing, and experimental dependency canaries.
3. Python binding and strict py-earth/scikit-learn compatibility.
4. R binding using idiomatic R interfaces and native package advantages.
5. Mojo binding with equal API, testing, documentation, packaging, and
   performance status.
6. Julia binding using multiple dispatch, artifacts, package extensions, and
   native Julia testing.
7. Cross-language generated contracts and conformance harness.
8. Framework interoperability and adapter boundaries.
9. MLCommons Croissant support.
10. Governed plugin and extension architecture.
11. Product boundaries, non-goals, maturity levels, and compatibility promises.
12. Astro/Starlight modernization and `astro-polyglot` evaluation.
13. Repository standards/template alignment and continuous drift detection.
14. Solo-maintainer CI/CD, security, provenance, and maximal automation.
15. `astro-polyglot` solo-maintainer GitHub ruleset and branch protection.
