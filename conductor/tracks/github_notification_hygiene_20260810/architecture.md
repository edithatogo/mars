# Notification Hygiene Architecture

## Decision status

Accepted for Track `github_notification_hygiene_20260810` on 2026-08-10.

## Runtime boundary

- The utility supports **Python 3.12-3.14** independently of the wider Mars
  package runtime matrix.
- The **GitHub CLI** is the only credential-bearing transport. Runtime code
  invokes `gh api`; it never calls `gh auth token`, reads credential-manager
  storage, or persists an authorization value.
- **Windows Task Scheduler** provides user-scoped local scheduling. It does not
  require elevation, a service account, or a repository secret.
- **No repository-hosted personal token** is permitted. A hosted workflow would
  enlarge the credential blast radius and would not satisfy the local-only
  privacy boundary.

## Component boundaries

1. `contracts`: versioned notification, decision, and metadata-audit schemas.
2. `classifier`: pure deterministic actor/reason classification with no network
   or filesystem writes.
3. `github_adapter`: bounded `gh api` invocation, response validation, subject
   resolution, conditional polling, and individual thread updates.
4. `processor`: orchestration that preserves unknown/external-human threads and
   writes only positively classified automation in enforcement mode.
5. `audit`: sanitized metadata receipts and bounded 30-day retention.
6. `scheduler`: install, inspect, disable, enable, health, and uninstall of the
   user-scoped scheduled task.

Dependencies point inward: scheduler and adapter depend on the processor;
processor depends on classifier and contracts; pure contracts and classifier
never depend on GitHub or Windows infrastructure.

## Delivery and Conductor automation

- Delivery uses **one implementation phase per pull request** unless a phase
  must be split further to remain reviewable. A pull request is never larger
  than one track.
- Every pull request originates in an **isolated worktree** created from current
  `origin/main`.
- Track and task states are explicit in `plan.md`; only one task is marked
  in-progress in a worktree at a time.
- Experimental Conductor ideas are adopted as patterns, not as an unmerged
  submodule pin: dependency-aware sequencing, courtesy locks, persistent state,
  automated review/remediation, and **bounded autonomous** execution loops.
- Autonomous loops stop on authorization, credential, external-write,
  unresolved security, or non-green hosted gates.
- `conductor-review` automatically applies actionable fixes at phase and track
  checkpoints. No second human or fictional reviewer is required.

## Polling and concurrency

- A five-minute scheduler cadence is only a wake-up interval. GitHub's
  `X-Poll-Interval`, `Last-Modified`, rate-limit reset, and retry headers govern
  actual request timing.
- A per-user single-instance lock prevents overlapping processors.
- Retries are bounded and fail open. Exhausted, malformed, deleted, or
  contradictory threads remain unread.

## Security and privacy

- Subject resolution is restricted to `https://api.github.com/` and known
  endpoint shapes.
- Audit records contain no title, body, comment, notification content, token,
  or authorization header.
- Dry-run and classification code cannot call a write adapter.
- Individual thread updates are used; bulk notification-read endpoints are
  prohibited.
- Third-party repositories remain read-only without explicit, action-specific
  approval.

## Rejected alternatives

- **Ignore Mars:** rejects desired external-human notifications.
- **Native filters only:** cannot implement actor-aware automatic read policy.
- **Hosted GitHub Action:** requires a repository-held classic token and adds a
  privileged workflow attack surface.
- **Reason-only rules:** reasons are mutable and do not prove the latest actor.
- **Pin an experimental Conductor branch:** current experimental branches are
  divergent from the supported upstream `main`; patterns are safer to adopt
  behind local tests and contracts.
