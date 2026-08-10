# Notification Hygiene Contract 1.0.0

## Safety invariants

1. **Unknown means preserve unread.** Missing, stale, deleted, contradictory,
   unsupported, or malformed actor evidence cannot authorize a write.
2. **Reason alone is insufficient.** GitHub documents notification reasons as
   mutable per thread, and the thread payload does not identify the latest
   actor.
3. **Dry-run cannot write.** Dry-run code paths cannot invoke a thread update,
   repository mutation, comment, issue, pull request, or other write endpoint.
4. **No third-party repository writes.** External repositories may be inspected
   read-only; a specific external mutation requires explicit user approval.
5. **Positive evidence is required.** `mark_read` requires a supported actor
   identity and a deterministic self, bot, App, workflow, or release-automation
   rule.
6. **Content is transient.** Notification titles, bodies, comments, and
   credentials are excluded from audit records and fixtures.
7. **Server pacing wins.** Clients obey `X-Poll-Interval`, conditional request
   semantics, rate-limit resets, and bounded retry policy.
8. **Individual writes only.** The implementation marks an eligible thread
   read individually and never uses a bulk-read endpoint.

## Precedence

Decision precedence is deliberately conservative:

1. Invalid schema or unsafe URL: `unknown`, `preserve_unread`, `NH-001`.
2. Missing or contradictory actor evidence: `unknown`, `preserve_unread`,
   `NH-002`.
3. External human actor: `external_human`, `preserve_unread`, `NH-100`.
4. Authenticated repository owner: `self`, eligible for `mark_read`, `NH-200`.
5. GitHub actor type `Bot`: `bot`, eligible for `mark_read`, `NH-210`.
6. Verified GitHub App actor: `app`, eligible for `mark_read`, `NH-220`.
7. Verified workflow/check actor: `workflow`, eligible for `mark_read`,
   `NH-230`.
8. Verified release automation actor: `release_automation`, eligible for
   `mark_read`, `NH-240`.

Eligibility does not itself perform a write. Enforcement mode, authorization,
single-instance locking, and a successful final validation are separate gates.

## Contract evolution

- Additive optional fields require a minor schema version.
- New actor classes, actions, or changed safety semantics require a major
  schema version.
- Clarifications that do not change validation or decisions require a patch
  version.
- Readers reject unsupported major versions and preserve the thread unread.
