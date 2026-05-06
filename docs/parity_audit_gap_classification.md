# Parity Audit Gap Classification

This note classifies the current repo gap matrix using the parity-audit rubric
and the matrices already recorded in `docs/`.

In this note, `nice-to-have` corresponds to the rubric's `optional` bucket,
and `upstream-only or intentionally out of scope` corresponds to the rubric's
`out of scope or intentional` bucket.

## Source Basis

- [Parity Audit Rubric](parity_audit_rubric.md)
- [mars / earth repo gap matrix](parity_audit_repo_gap_matrix.md)
- [Consolidated Parity Table](parity_audit_parity_table.md)
- [Parity Audit Recommendations](parity_audit_recommendations.md)

Rows already marked `compatible` in the repo gap matrix are treated as aligned
baseline, not gaps, and are not repeated below.

| Bucket | Matrix rows | Source-backed reading |
| --- | --- | --- |
| parity-critical | Example outputs and documented claims; Serialization and `multioutput` edge cases | The rubric treats canonical examples, serialization semantics, and docs claims that could mislead users as parity-critical (`docs/parity_audit_rubric.md:10-20`, `docs/parity_audit_rubric.md:77-78`). The repo gap matrix still says example-output parity is not fixture-locked and that pickle / multioutput remain expected failures (`docs/parity_audit_repo_gap_matrix.md:36,40`). The consolidated parity table already mirrors that split (`docs/parity_audit_parity_table.md:28-29`). |
| nice-to-have | Tie handling | The repo already has deterministic regression fixtures, but the current matrix does not document a user-facing tie policy (`docs/parity_audit_repo_gap_matrix.md:35,52-57`). That is audit-visible evidence hardening, not a confirmed behavioral break. The parity table keeps it as a partial contract gap rather than a defect (`docs/parity_audit_parity_table.md:26`). |
| upstream-only or intentionally out of scope | Warning behavior; Formula / interface ergonomics; Packaging / versioning / release behavior | These rows are already documented as deliberate scope choices: narrower warning policy, estimator-centric API surface, and Rust-first multi-registry packaging (`docs/parity_audit_repo_gap_matrix.md:33,39,41`, `docs/parity_audit_rubric.md:34-39`, `docs/parity_audit_parity_table.md:24,30`). |

## Track Implications

- Serialization and `multioutput` remain separate implementation-track
  candidates.
- Tie handling should stay in the evidence queue until the upstream contract is
  pinned down more explicitly.
- The upstream-only or intentionally out of scope boundaries should remain
  documented so they are not reclassified as missing parity.
- The current bucket list is limited to rows already present in the repo gap
  matrix; it does not exhaust the R `earth` parity-critical surface. R-side
  plots, prediction intervals / variance models, GLM-style extensions, and
  update workflows remain upstream requirements that are not yet represented
  as closed repo parity items.
