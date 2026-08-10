# Notification Hygiene Evidence Baseline

## Scope and date

This baseline was refreshed on 2026-08-10. It records primary-source API
constraints, relevant `edithatogo` standards, and the current Mars automation
surface. External repositories were inspected read-only.

## GitHub notification contract

Primary sources:

- [REST notification endpoints](https://docs.github.com/en/rest/activity/notifications?apiVersion=2022-11-28)
- [REST repository watching endpoints](https://docs.github.com/en/rest/activity/watching?apiVersion=2022-11-28)
- [Inbox filters](https://docs.github.com/en/enterprise-cloud@latest/subscriptions-and-notifications/reference/inbox-filters)

Pinned observations:

1. Notifications are threads. A thread includes repository, subject, reason,
   unread state, timestamps, and API URLs, but it does not directly identify
   the actor responsible for the latest update.
2. The documented reason vocabulary includes `approval_requested`, `assign`,
   `author`, `ci_activity`, `comment`, `invitation`, `manual`,
   `member_feature_requested`, `mention`, `review_requested`,
   `security_advisory_credit`, `security_alert`, `state_change`, `subscribed`,
   and `team_mention`.
3. A thread reason is mutable. A later event can replace the earlier reason,
   so reason alone cannot prove that the current update was machine-generated.
4. Actor-aware classification therefore requires bounded resolution of the
   subject or latest-comment resource. If actor evidence is absent, stale,
   deleted, contradictory, or unsupported, the decision must be `unknown` and
   the thread must remain unread.
5. The API advertises `Last-Modified`, conditional polling, and
   `X-Poll-Interval`. The client must obey the server-provided interval, even
   when the local schedule runs more frequently.
6. Listing or changing notification threads requires classic-token
   notification access. Live GitHub CLI responses for the current account
   explicitly require the `notifications` scope; absence of that scope is an
   external authorization blocker, not a repository defect.
7. Marking an individual thread read is safer than bulk read operations because
   the policy must preserve external-human and ambiguous notifications.
8. Setting a repository to ignored blocks all notifications, including desired
   external-human activity, and therefore does not meet this track's policy.

## Repository-estate standards

Primary repository:

- [`edithatogo/repository-standards`](https://github.com/edithatogo/repository-standards)

Applicable requirements:

- Single-maintainer repositories use zero mandatory human approvals and do not
  depend on CODEOWNERS or fictional reviewers.
- Automated gates replace a second human for routine quality and security
  assurance.
- Repository state, live hosted controls, and verification receipts must agree
  before claiming conformance.
- Dependency automation is admitted only after stable checks.
- Destructive or externally consequential changes fail closed and remain
  review-required when evidence is insufficient.
- Managed policy and repository-owned context remain distinct.

Relevant template references inspected read-only:

- [`template-solo-python`](https://github.com/edithatogo/template-solo-python)
- [`template-solo-rust`](https://github.com/edithatogo/template-solo-rust)
- [`template-solo-docs`](https://github.com/edithatogo/template-solo-docs)
- [`template-solo-node`](https://github.com/edithatogo/template-solo-node)

The notification utility is Python-facing and local-only. It should follow the
Python template's testing, Renovate, Codecov, security, and solo-maintainer
posture without importing unrelated template files wholesale.

## Current Mars automation surface

Mars already provides:

- GitHub Actions matrices for Python, Rust, bindings, documentation, security,
  CodeQL, benchmarks, mutation testing, profiling, release rehearsal, and
  supply-chain evidence.
- Renovate configuration and a dependency dashboard.
- Issue and pull-request templates.
- A Conductor workflow with automated review and remediation.
- A dirty historical primary checkout that must not be used for implementation
  or cleanup.

The notification filter must not duplicate CI orchestration. It is a bounded
local account utility whose repository contribution is its contract,
implementation, tests, installer, and operational documentation.

## External-write boundary

Read-only standards research is permitted. No comment, issue, pull request,
discussion, reaction, release, or other mutation may be made in a third-party
repository without explicit approval for that specific external action.

## Consequences for implementation

- Classification is evidence-based and actor-aware, never reason-only.
- Unknown means preserve unread.
- Dry-run is the default and contains no state-changing API call.
- `gh api` remains the credential boundary; the implementation never extracts
  a token.
- Polling honors GitHub response headers and avoids fixed-rate request storms.
- Audit output is metadata-only and excludes notification titles, bodies,
  comment text, and credentials.
- Live enforcement cannot begin until the user explicitly authorizes the
  `notifications` scope and a clean dry-run is reviewed.
