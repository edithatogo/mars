# Parity Audit Recommendations

This memo turns the `py-earth` feature matrix and the mars / earth repo gap
matrix into a Rust-first roadmap summary. It uses the bucket assignments from
[Parity Audit Gap Classification](parity_audit_gap_classification.md) so the
next work item can be assigned without reopening settled decisions.

## Source Basis

- [py-earth Feature Matrix](parity_audit_feature_matrix.md)
- [mars / earth repo gap matrix](parity_audit_repo_gap_matrix.md)
- [Parity Audit Rubric](parity_audit_rubric.md)
- [Parity Audit Gap Classification](parity_audit_gap_classification.md)

## Bucket Handoff

The classification note is the source of truth for what this memo promotes
into the roadmap:

- `parity-critical` rows become the next Rust-first implementation tracks
- `nice-to-have` rows stay in the evidence / contract-clarification queue
- `upstream-only or intentionally out of scope` rows stay documented as
  deliberate scope choices

## Executive Summary

The audit points to a small set of remaining parity-critical gaps that should
drive the next Rust-first implementation work:

- example outputs and documented claims
- estimator serialization and pickle round-trips
- multioutput support

This recommendation list is scoped to the current repo gap matrix. The R
`earth` matrix still contains additional parity-critical surfaces - plots,
prediction intervals / variance models, GLM-style extensions, and update
workflows - that remain upstream requirements and should be promoted into
separate tracks if the repo chooses to close them.

At the same time, several differences are intentional and should stay outside
the parity backlog:

- warning behavior stays narrower than upstream R `earth`
- formula/interface ergonomics remain estimator-centric by design
- packaging and release behavior stay Rust-first and multi-registry

The `nice-to-have` tie-handling row stays in the evidence queue until the
upstream contract is pinned down more explicitly.

## Parity-Critical Gaps To Close Next

| Area | Why it matters | Recommendation |
| --- | --- | --- |
| Example outputs and documented claims | The classification note marks canonical examples and claims as parity-critical, so they must remain fixture-backed and visible in the audit trail. | Lock down the canonical examples and documentation claims with fixtures before treating the public story as closed. |
| Serialization and pickle support | The repo gap matrix marks `check_estimators_pickle` as an expected failure, and the classification note keeps serialization in the parity-critical bucket. If fitted estimators cannot round-trip cleanly, the Rust-first core cannot yet be treated as a stable model artifact. | Prioritize a dedicated Rust-first state serialization track for `Earth` and its fitted model state. Add fixture-backed save/load coverage and keep the sklearn pickle check in the failure queue until the round-trip contract is proven. |
| multioutput regression | The repo gap matrix still treats multioutput regression as an expected failure, and the classification note keeps it parity-critical. That is a material compatibility gap for sklearn users and should not be folded into a generic cleanup bucket. | Create a separate multioutput implementation track with explicit success criteria for fit, predict, and estimator checks. Do not blur this into the single-output path until the supported contract is clear. |

## Recommended Rust-First Track Order

1. Close example outputs and documented claims first. This is parity-critical
   and keeps the public contract fixture-backed before deeper implementation
   work proceeds.
2. Close estimator artifact persistence next. The classification note keeps
   serialization in the parity-critical bucket, and fitted-model round-trips
   are a prerequisite for a stable Rust-first core artifact boundary.
3. Close multioutput after serialization. It is still parity-critical, but it
   is a separate behavioral contract from single-output fitting and should stay
   isolated.
4. Keep tie handling in the evidence queue. It is only `nice-to-have`, so it
   should inform audit coverage without becoming the next implementation track.
5. Leave warning behavior, formula/interface ergonomics, and
   packaging/versioning/release behavior documented as intentional boundaries.
6. Treat the R `earth` plot / interval / GLM / update-workflow surface as a
   separate future track family rather than folding it into the current repo
   gap list.

## Intentional Boundaries Versus Future Tracks

These `upstream-only or intentionally out of scope` differences should stay
explicit as boundaries, not parity defects:

- Formula/interface ergonomics: the repo remains estimator-centric, and a
  formula layer would be a future extension rather than a missing parity item.
- Packaging/versioning/release behavior: the repo is deliberately Rust-first
  and multi-registry, so it should not try to mimic the upstream single-package
  release model.
- Warning policy: the repo uses a narrower logging / exception approach rather
  than exposing the upstream warning knobs.

If any of these boundaries are revisited later, they should be opened as
separate implementation tracks with their own acceptance criteria.

## Roadmap Implication

The parity audit should feed the Rust-first roadmap in this order:

- promote the `parity-critical` rows into implementation tracks in the order
  above
- keep the `nice-to-have` rows in evidence / contract-clarification mode until
  the upstream contract is explicit
- keep `upstream-only or intentionally out of scope` rows visible in the docs
  so they are not reclassified as defects
- maintain fixture-backed evidence for the areas already marked compatible so
  regression work does not drift back into the audit backlog
