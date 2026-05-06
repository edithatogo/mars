# Implementation Plan

## Phase 0: Source Inventory and Audit Framing

- [ ] Task: Inventory the upstream mars/earth references
    - [x] Agent 1: audit the `py-earth` source and README
    - [x] Agent 2: audit the R `earth` package manual and reference docs
    - [x] Agent 3: list any other canonical mars/earth references that define expected behavior
    - [x] Agent 4: capture the upstream feature and option surface at a high level
    - [x] Agent 5: capture packaging and release practices that affect user expectations
    - [x] Agent 6: define the evidence collection format for the audit
    - [x] Companion note: `docs/parity_audit_evidence.md`
    - [x] Companion note: `docs/parity_audit_r_earth_matrix.md`
- [ ] Task: Define the parity audit rubric
    - [x] Agent 1: define what counts as parity-critical versus optional
    - [x] Agent 2: define how to compare defaults, errors, and warnings
    - [x] Agent 3: define how to compare examples and docs claims
    - [x] Agent 4: define how to compare packaging and versioning behavior
    - [x] Agent 5: define how to record intentional deviations
    - [x] Agent 6: consolidate the audit rubric for later phases
    - [x] Rubric recorded in `docs/parity_audit_rubric.md`
    - [x] Evidence template recorded in `docs/parity_audit_evidence_template.md`
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Source Inventory and Audit Framing' (Protocol in workflow.md)

## Phase 1: Behavioral and Feature Matrix

- [x] Task: Build the feature matrix
    - [x] Agent 1: map core model capabilities and basis-term support
    - [x] Agent 2: map training, pruning, and selection behavior
    - [x] Agent 3: map categorical and missingness support
    - [x] Agent 4: map diagnostics, summaries, plots, and uncertainty behavior
    - [x] Agent 5: map formula/interface ergonomics and defaults
    - [x] Agent 6: consolidate the feature matrix into a single parity table
- [ ] Task: Compare behavior against the current repo
    - [x] Agent 1: compare validation, error, and warning behavior
    - [x] Agent 2: compare deterministic outputs and tie handling
    - [x] Agent 3: compare sample-weight and edge-case behavior
    - [x] Agent 4: compare example outputs and documented claims
    - [x] Agent 5: compare packaging/versioning/release behavior
    - [x] Agent 6: consolidate the behavioral differences into a gap list
    - [x] Companion note: `docs/parity_audit_repo_gap_matrix.md`
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Behavioral and Feature Matrix' (Protocol in workflow.md)

## Phase 2: Gap Classification and Recommendation Synthesis

- [x] Task: Classify parity gaps
    - [x] Agent 1: mark parity-critical gaps
    - [x] Agent 2: mark nice-to-have gaps
    - [x] Agent 3: mark upstream-only or intentionally out-of-scope gaps
    - [x] Agent 4: verify each classification against source evidence
    - [x] Agent 5: identify any mismatches that need a separate implementation track
    - [x] Agent 6: consolidate the gap classes into a recommendation-ready list
    - [x] Companion note: `docs/parity_audit_gap_classification.md`
- [x] Task: Produce recommendations for the Rust-first roadmap
    - [x] Agent 1: recommend features that should be implemented next
    - [x] Agent 2: recommend behavioral boundaries that should stay explicit
    - [x] Agent 3: recommend docs or test coverage that should be added
    - [x] Agent 4: recommend any release or package guidance changes
    - [x] Agent 5: recommend any parity evidence that should be preserved as fixtures
    - [x] Agent 6: consolidate the recommendations into an implementation memo
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Gap Classification and Recommendation Synthesis' (Protocol in workflow.md)

## Phase 3: Documentation and Roadmap Sync

- [x] Task: Document the audit findings
    - [x] Agent 1: write the upstream source inventory summary
    - [x] Agent 2: write the feature parity summary
    - [x] Agent 3: write the behavioral divergence summary
    - [x] Agent 4: write the packaging and docs summary
    - [x] Agent 5: write the recommendation summary
    - [x] Agent 6: consolidate the audit narrative
    - [x] Companion note: `docs/parity_audit_findings.md`
- [x] Task: Sync the project roadmap and Conductor pointers
    - [x] Agent 1: update `docs/remaining_roadmap.md` if needed
    - [x] Agent 2: update any release-facing docs that should link to the audit
    - [x] Agent 3: ensure the Rust conversion track points to audit evidence where useful
    - [x] Agent 4: ensure the parity audit track is referenced from the roadmap
    - [x] Agent 5: ensure the track registry reflects the audit as a live track
    - [x] Agent 6: verify the doc/story is coherent across the project
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Documentation and Roadmap Sync' (Protocol in workflow.md)

## Phase 4: Final Validation and Handoff Readiness

- [x] Task: Validate the audit bundle
    - [x] Agent 1: validate the feature matrix against the upstream docs
    - [x] Agent 2: validate the behavior matrix against the current repo
    - [x] Agent 3: validate the gap classification against the source evidence
    - [x] Agent 4: validate the recommendations against the Rust-core roadmap
    - [x] Agent 5: validate the roadmap/doc pointers
    - [x] Agent 6: validate the end-to-end parity narrative
- [x] Task: Complete the handoff summary
    - [x] Agent 1: summarize the upstream source inventory
    - [x] Agent 2: summarize the behavior matrix
    - [x] Agent 3: summarize the parity-critical gaps
    - [x] Agent 4: summarize the recommended next implementation tracks
    - [x] Agent 5: summarize the documentation outcomes
    - [x] Agent 6: collect any open follow-ups
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: Final Validation and Handoff Readiness' (Protocol in workflow.md)
