# CI/CD Assurance Baseline

Captured on 2026-08-10 at 18:08 AEST from local Git state and authenticated
GitHub API readback. This is decision evidence, not proof that a control is
remediated.

## Repository and Delivery State

- Repository: `edithatogo/mars` (public, active, not a fork)
- Default branch: `main`
- Delivery branch: `chore/cicd-assurance-baseline`
- Baseline parent commit: `cc49f6d6ba48f1fc1b50b7eb6a2ea7679ee93b86`
- Workflow files: 21
- Referenced GitHub Actions: 161
- Prerequisite changes already merged to `main`:
  - pinned Conductor plugin subrepo at `99ba10e1a11130fc159f681b7ba8803489239cbf`
  - Renovate config migration and monorepo-manager alignment
- The historical primary checkout was left untouched: its local `main` was one
  commit ahead and nine behind `origin/main`, with untracked `.agents/` and
  `docs/astro-site/.astro/`. This task runs in a clean disposable worktree.

## Hosted Control Readback

| Control | Observed state | Disposition |
| --- | --- | --- |
| Actions | Enabled; all actions allowed | Harden |
| Action SHA policy | Not required | Enforce after repository pins pass |
| Default workflow token | Write | Reduce to read |
| Actions PR approval permission | Enabled | Disable for least privilege |
| `main` protection | None | Add solo-maintainer ruleset |
| Repository rulesets | None | Add required automated checks |
| Dependabot alerts | Enabled (HTTP 204 readback) | Retain alerts |
| Dependabot security updates | Enabled and not paused | Retire after Renovate is healthy |
| Private vulnerability reporting | Enabled | Retain |
| Renovate | Dashboard active; no open update PRs | Validate update discovery |

## Current Admission Evidence

- PR #179 exposed 37 check contexts: 36 succeeded and one conditional check was
  skipped. Because `main` has no protection or ruleset, none is a required gate.
- The nine post-merge workflows for `cc49f6d` all completed successfully.
- Representative elapsed times were CI 5m37s, Code Quality 3m58s, Security
  3m30s, Bindings CI 2m51s, and Performance Benchmarks 1m53s.
- The CI matrix passed on Ubuntu, macOS, and Windows with Python 3.10-3.12.
  Codecov upload succeeded on Ubuntu/Python 3.12 and was intentionally skipped
  in the other matrix cells; no coverage percentage is asserted here.
- Recent failed hosted runs include Code Quality on superseded PR #175 and
  GitHub Advanced Security's dynamic "Code scanning AI findings" runs for PRs
  #175, #178, and #179. These runs are evidence to classify, not passing gates.
- The latest published release is `v1.0.4` (2026-04-17); release freshness and
  reproducibility have not yet been admitted by this track.

## Dependency Automation Evidence

Renovate's dependency dashboard (#167) is open and was updated after the config
migration. Of its recent security PRs, #165 and #166 merged and #164 closed
without merge. No Renovate or Dependabot update PR was open at capture time.
Dependabot alerts and security updates remain enabled, so completing the bot
migration still requires verified Renovate discovery before retiring the latter.

## Workflow Gaps

- Action references use mutable tags rather than full commit SHAs.
- Required action-SHA enforcement cannot be enabled safely until pins land.
- Jobs broadly lack explicit `timeout-minutes` declarations.
- Required workflows do not yet have a verified `merge_group` contract.
- Default write permissions amplify the impact of an unsafe workflow.
- `pull_request_target` automation requires explicit trust-boundary review.
- No branch or repository ruleset currently makes CI a merge gate.
- No canonical machine-readable control-to-evidence manifest exists.

## Evidence Boundary

- Local file presence proves configuration only.
- GitHub API readback proves a hosted setting at capture time.
- A completed workflow run proves execution, not durable future enforcement.
- Queued, running, canceled, skipped, or failed checks are not passing evidence.
- Preview controls remain non-blocking until their promotion criteria are met.
