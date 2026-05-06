# Specification: MARS / Earth Parity Audit

## Overview

Perform a comprehensive parity audit of the original mars/earth lineage and
document the feature, behavior, diagnostics, packaging, and test gaps between
the current repository and the authoritative upstream references.

This track is primarily investigative. Its job is to produce a high-confidence
parity matrix and an actionable gap list so the Rust-first core can keep
moving toward full feature equivalence without guessing at upstream behavior.

## Context

- The project has already moved many capabilities into Rust-backed runtime and
  training paths.
- The remaining risk is not only missing code, but also mismatched semantics,
  docs, defaults, error surfaces, and packaging behavior relative to the
  original mars/earth implementations.
- A complete audit should cover the original `py-earth` lineage, the R `earth`
  ecosystem, and any project-specific documentation or examples that define
  expected behavior.

## Functional Requirements

- Build a feature matrix for the original mars/earth implementations:
  - model families and supported objectives
  - basis term types and interaction rules
  - training, pruning, and selection behavior
  - categorical and missingness handling
  - diagnostics, summaries, plots, and uncertainty outputs
  - formula or interface ergonomics
  - packaging, versioning, and release behavior
- Compare behavior and defaults against the current repository:
  - supported and unsupported user-facing options
  - error and warning messages
  - deterministic behavior and tie handling
  - example outputs and documentation claims
- Capture parity evidence:
  - upstream docs or source references
  - fixture examples and reproductions where possible
  - notes on areas where behavior diverges intentionally
  - evidence records stored in `docs/parity_audit_evidence.md` with source
    family, canonical URL, retrieval date, claim summary, excerpt or line
    reference, local reproduction command, and parity classification
- Classify gaps:
  - parity-critical
  - nice-to-have
  - upstream-only or intentionally out of scope
- Produce an actionable recommendation set:
  - what should be implemented next
  - what should remain intentionally unsupported
  - what should be documented as a compatibility boundary

## Non-Functional Requirements

- Prefer source-backed audit findings over memory.
- Keep the audit traceable and reproducible.
- Avoid changing runtime behavior as part of the audit unless a direct gap fix
  is part of the accepted scope.
- Keep the audit parallelizable across six workers with disjoint ownership.

## Parallel Work Model

- Agent 1: py-earth source and documentation audit.
- Agent 2: R `earth` package source and documentation audit.
- Agent 3: behavior matrix and option/default comparison.
- Agent 4: diagnostics, plots, summaries, and uncertainty audit.
- Agent 5: packaging, versioning, and release workflow audit.
- Agent 6: gap consolidation, recommendation synthesis, and evidence map.

## Acceptance Criteria

- A comprehensive parity matrix exists for the original mars/earth references.
- The matrix covers functional, diagnostic, packaging, and release behavior.
- The audit distinguishes parity-critical gaps from lower-priority differences.
- The resulting recommendation set is actionable for future Rust-core work.
- The audit is structured so six workers can operate without overlapping files.
- Every phase ends with a Conductor review checkpoint.
- Final validation confirms the audit is coherent with the current repository
  and clearly identifies what remains intentionally out of parity.

## Out of Scope

- Implementing all identified parity gaps in this track.
- Changing the current public API surface.
- Reworking release registries or package ownership.
- Removing intentional deviations that are already documented as out of scope
  without a separate feature track.

## References

- py-earth project and README
- R `earth` package documentation
- R package documentation conventions and manual generation guidance
- The current repo docs and Conductor tracks
