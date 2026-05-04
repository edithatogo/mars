# Specification: R Package Publication Readiness

## Overview

Bring the R binding to a publish-ready package state for r-universe first and
CRAN later, while preserving the existing R runtime and training API. The track
must cover R package metadata, Rd help, generated manual/vignette readiness,
CRAN-safe tests, runtime-binary optional behavior, CI/release rehearsal, and
release handoff documentation.

This is a readiness and packaging track. It must not change mars model
semantics, the portable `ModelSpec` contract, or the public R function names
without a separate API-change track.

## Context

- The R package currently lives under `bindings/r`.
- The R package name is `marsruntime`; the public project brand is
  `mars-earth`.
- The package exposes runtime replay helpers and a Rust-backed `fit_model`
  helper when the Rust runtime binary is available.
- R remains the main unpublished language package after Python, Rust,
  TypeScript, and C# publication.
- Official R packaging guidance expects source packages to provide standard
  metadata, Rd help pages, package checks, and documentation artifacts that can
  be built by R tooling.
  Vignettes are built by R package tooling when present, and package manuals are
  generated from Rd files during checks/builds.
- r-universe can publish directly from GitHub-backed source repositories and
  exposes package manuals and binary/source artifacts on package pages.

## Functional Requirements

- Audit and align the R package metadata, namespace, README, help pages, tests,
  and release docs.
- Ensure every exported R function has complete Rd documentation with valid
  usage, documented arguments, return values, and examples where appropriate.
- Add or verify source documentation for a package manual and a lightweight
  vignette or equivalent article that can be built by R tooling.
- Add or verify a package-level help topic so the package has a clear
  `?marsruntime` entry point in addition to per-function help.
- Ensure `R CMD build` and `R CMD check --no-manual --as-cran` pass locally.
- Ensure CI uses `R CMD check` or `rcmdcheck`; build/install-only validation is
  insufficient for publication readiness.
- Ensure a documentation/manual validation path exists, including `R CMD Rd2pdf`
  or an equivalent R package manual build command where the local toolchain
  supports it.
- Keep tests CRAN-safe:
  - source-tree tests may use shared fixtures when available;
  - installed-package checks must not require repository-only fixtures;
  - Rust-backed training tests must run when `MARS_RUNTIME_BIN` is set;
  - tests must skip clearly when the Rust runtime binary is unavailable.
- Add CI or release rehearsal coverage for the R package build, package check,
  artifact inspection, and install-from-tarball smoke test.
- Decide whether the shared fixture corpus is source-tree-only validation or
  whether a minimized fixture subset should be included in the R package for
  self-contained package checks.
- Update release inventory, handoff, checklist, and package path docs so the R
  package state is unambiguous.
- Record any remaining external submission blocker with owner, action, and date.

## Non-Functional Requirements

- Do not add a dependency on `py-earth`, R `earth`, or another mars
  implementation.
- Do not commit registry credentials or maintainer-private submission data.
- Keep R dependencies minimal and declared in `DESCRIPTION`.
- Keep default logging/output quiet; skip messages must be explicit and
  actionable.
- Preserve compatibility with source-tree execution and installed-package
  execution.
- Prefer generated documentation sources over committed build artifacts unless
  the release path explicitly requires a committed artifact.
- The plan must be easy to parallelize across six agents with disjoint
  ownership areas.

## Parallel Work Model

The implementation should be decomposed so six agents can work without stepping
on each other:

- Agent 1: R package metadata, namespace, README, and package identity.
- Agent 2: Rd help pages, package manual generation, and vignette source.
- Agent 3: R tests, conformance fixtures, training smoke checks, and skip
  behavior.
- Agent 4: CI/release rehearsal, tarball inspection, and install smoke tests.
- Agent 5: release inventory, publication handoff, checklist, and package path
  documentation.
- Agent 6: integration, evidence collection, Conductor plan state, and final
  consistency validation.

## Acceptance Criteria

- `Rscript bindings/r/tests/conformance.R` passes from the repository root.
- `Rscript bindings/r/tests/training.R` passes from the repository root when the
  Rust runtime binary is available, and skips clearly when it is not.
- `R CMD build bindings/r` passes.
- `R CMD check --no-manual --as-cran` passes with no ERROR or WARNING; any NOTE
  is documented and acceptable.
- CI or release rehearsal runs an R package check, not only build/install smoke
  tests.
- The R manual build path is documented and validated locally when the required
  LaTeX/R toolchain is available.
- The package has complete Rd documentation for all exports.
- The package has a package-level help topic or vignette source that provides a
  durable user entry point.
- The R package has a source vignette or documented equivalent ready for
  r-universe/CRAN tooling.
- CI or release rehearsal covers the R package build/check/install smoke path.
- Release docs and Conductor publication tracks agree on the R package state.
- `conductor-review` has been invoked at the end of every phase, fixes have
  been applied automatically where high confidence, and the track has advanced
  to the next phase only after review passes or records an explicit blocker.

## Out of Scope

- Changing the public R API names.
- Changing Rust runtime or training semantics.
- Publishing to CRAN without maintainer approval.
- Adding GLM/classification, uncertainty, plotting, or diagnostics features.
- Replacing the current Rust CLI bridge with native R FFI.
- Committing generated site output, registry tokens, or private account data.

## References

- R Extensions manual: <https://stat.ethz.ch/R-manual/R-devel/doc/manual/R-exts.html>
- r-universe package documentation: <https://docs.r-universe.dev/browse/packages.html>
- r-universe setup guidance: <https://r-universe.dev/welcome>
