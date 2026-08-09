# CI/CD Assurance Baseline

Captured on 2026-08-10 AEST from local Git state and authenticated GitHub API
readback. This is decision evidence, not proof that a control is remediated.

## Repository and Delivery State

- Repository: `edithatogo/mars` (public, active, not a fork)
- Default branch: `main`
- Implementation branch: `chore/bleeding-edge-cicd-assurance`
- Baseline track-status commit: `ae1cbeaaf7aa66f9cab864b8a7d8bb5ab594fe2f`
- Workflow files: 21
- Referenced GitHub Actions: 160
- Existing scoped working changes preserved:
  - pinned Conductor plugin submodule and `.gitmodules`
  - Renovate configuration corrections
  - release-summary migration from Dependabot to Renovate

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
| Renovate | Active with live PRs | Validate and complete migration |

## Dependency Automation Evidence

Renovate is active through PRs #164, #165, and #166. Dependabot remains active
through PRs #134 and #163. Recent Renovate-triggered workflows include queued,
running, successful, and failing states; activation must not be confused with a
healthy completed migration.

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
