# Implementation Plan

## Phase 0: Inventory, Evidence, and Work Split

- [x] Task: Inventory the current R package surface
    - [x] Agent 1: audit `bindings/r/DESCRIPTION`, `NAMESPACE`, and `README.md`
    - [x] Agent 2: audit `bindings/r/man`, manual generation, and vignette/source article coverage
    - [x] Agent 3: audit `bindings/r/tests` and source-tree versus installed-package behavior
    - [x] Agent 4: audit R CI, release rehearsal, artifact upload, and install smoke coverage
    - [x] Agent 5: audit R rows in release inventory, handoff, checklist, and package-path docs
    - [x] Agent 6: consolidate the gap list and map each gap to a later phase
- [x] Task: Establish validation baseline
    - [x] Run `Rscript bindings/r/tests/conformance.R`
    - [x] Run `Rscript bindings/r/tests/training.R` with and without `MARS_RUNTIME_BIN`
    - [x] Run `R CMD build bindings/r`
    - [x] Run `R CMD check --no-manual --as-cran` on the built tarball
    - [x] Run the release alignment check after documenting current state
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Inventory, Evidence, and Work Split' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 1: R Package Metadata and API Documentation

- [x] Task: Align R package metadata and namespace
    - [x] Confirm package name, title, description, license, URLs, bug tracker, maintainer, and dependency declarations
    - [x] Confirm exports and imports in `NAMESPACE` match the callable API
    - [x] Confirm package README describes runtime replay, Rust-backed training, validation, and installation accurately
- [x] Task: Complete Rd documentation for every export
    - [x] Add or verify a package-level `marsruntime-package` help topic
    - [x] Ensure `load_model_spec` documents file and JSON-string inputs
    - [x] Ensure `validate_model_spec` documents invisible success and validation failure behavior
    - [x] Ensure `design_matrix` documents accepted row inputs and matrix return shape
    - [x] Ensure `predict_model` documents accepted row inputs and prediction output
    - [x] Ensure `fit_model` documents every training argument and runtime-binary requirement
    - [x] Add examples that are CRAN-safe and do not require unavailable external binaries
- [x] Task: Validate R documentation integrity
    - [x] Run Rd parse/check validation through `R CMD check`
    - [x] Verify no missing documentation, code/documentation mismatch, or wide usage warnings remain
    - [x] Confirm documentation names use `mars`/`mars-earth` branding consistently where appropriate
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: R Package Metadata and API Documentation' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 2: Manual, Vignette, and User Documentation Artifacts

- [x] Task: Add package-level documentation source that R tooling can build
    - [x] Decide whether the package should use a lightweight vignette, `inst/doc` article, or manual-only Rd path
    - [x] Add source documentation for installing, loading, validating, predicting, and Rust-backed fitting
    - [x] Ensure examples do not require private files, registry credentials, or unavailable binaries
    - [x] Add `VignetteBuilder` and `Suggests` only if a real vignette source is adopted
- [x] Task: Validate package manual generation
    - [x] Run `R CMD Rd2pdf bindings/r` when the local LaTeX/R toolchain supports it
    - [x] If PDF generation is unavailable locally, record the missing toolchain and keep the source path build-ready
    - [x] Ensure generated PDF/manual artifacts are ignored or handled according to release policy
- [x] Task: Sync project documentation
    - [x] Update `docs/package_release_paths.md` with the R manual/vignette policy
    - [x] Update `docs/publication_handoff.md` with the R documentation artifact status
    - [x] Update `docs/release_checklist.md` with manual/vignette verification steps
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Manual, Vignette, and User Documentation Artifacts' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 3: CRAN-Safe Tests and Runtime-Binary Behavior

- [x] Task: Harden source-tree and installed-package tests
    - [x] Keep source-tree fixture conformance active when repository fixtures exist
    - [x] Decide whether to vendor a minimized fixture subset into the R package or keep fixture conformance as a source-tree-only gate
    - [x] Ensure installed-package `R CMD check` does not require repository fixture files
    - [x] Ensure tests load the installed package under `R CMD check` and source files during local source-tree runs
- [x] Task: Harden Rust runtime optionality
    - [x] Ensure `fit_model` reports a clear error when the Rust runtime binary is unavailable
    - [x] Ensure training tests exercise Rust-backed fitting when `MARS_RUNTIME_BIN` is set
    - [x] Ensure training tests skip clearly when the binary is unavailable
    - [x] Ensure replay functions fall back to pure R runtime logic where that is the established package behavior
- [x] Task: Validate R package behavior
    - [x] Run `Rscript bindings/r/tests/conformance.R`
    - [x] Run `Rscript bindings/r/tests/training.R`
    - [x] Run `MARS_RUNTIME_BIN=<local binary> Rscript bindings/r/tests/training.R`
    - [x] Run `R CMD build bindings/r`
    - [x] Run `R CMD check --no-manual --as-cran` with `MARS_RUNTIME_BIN` set
    - [x] Run `R CMD check --no-manual --as-cran` without `MARS_RUNTIME_BIN` when feasible
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: CRAN-Safe Tests and Runtime-Binary Behavior' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 4: CI, Release Rehearsal, and Artifact Inspection

- [x] Task: Add or verify CI coverage for the R package
    - [x] Ensure GitHub Actions runs R package build/check in the binding or release-rehearsal workflow
    - [x] Replace build/install-only validation with `R CMD check` or `rcmdcheck`
    - [x] Consider Linux/macOS/Windows and current/release/devel R coverage where runtime availability makes sense
    - [x] Ensure the workflow installs R dependencies declared in `DESCRIPTION`
    - [x] Ensure the workflow builds the Rust runtime helper when training smoke tests are expected
    - [x] Ensure package check logs are uploaded as artifacts on failure
- [x] Task: Add artifact and install smoke validation
    - [x] Inspect the built R source tarball contents
    - [x] Install the built tarball into a clean R library
    - [x] Run a minimal validate/design/predict smoke test against the installed package
    - [x] Run the training smoke test only when the helper binary is explicitly provided
- [x] Task: Align release rehearsal docs
    - [x] Update `docs/ci_quality.md` and release docs with the R build/check/install commands
    - [x] Ensure `scripts/check_release_alignment.py` continues to pass after R documentation updates
    - [x] Record any platform-specific R check caveats in release guidance
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: CI, Release Rehearsal, and Artifact Inspection' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 5: r-universe and CRAN Submission Handoff

- [x] Task: Decide and document the R submission route
    - [x] Confirm whether r-universe is the first publication target
    - [x] Confirm whether CRAN is immediate, deferred until maturity, or explicitly out of the current release
    - [x] Reconcile maintainer-owner status versus submission-path status across release docs
    - [x] Record maintainer account, owner, action, and date without committing private credentials
- [x] Task: Prepare r-universe handoff
    - [x] Document the GitHub app/source repository setup needed for r-universe
    - [x] Document expected package page, manual, source artifact, binary artifact, and install command
    - [x] Add post-publish verification steps for r-universe
- [x] Task: Prepare CRAN handoff
    - [x] Document CRAN preflight checks, package check expectations, submission metadata, and manual review notes
    - [x] Record any CRAN-specific blockers such as package maturity, examples, maintainer approval, or runtime-binary policy
    - [x] Ensure CRAN handoff does not claim support for external binaries that CRAN cannot build or test
- [x] Task: Sync Conductor publication state
    - [x] Update release-readiness and publication tracks to consume this R readiness result
    - [x] Ensure only external submission/approval work remains after local readiness is complete
    - [x] Record evidence commands and results in the track plan or linked docs
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 5: r-universe and CRAN Submission Handoff' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 6: Final Track Review and Archive Readiness

- [x] Task: Run final validation bundle
    - [x] Run R package tests and package checks
    - [x] Run docs build and release alignment checks
    - [x] Run Vale on changed release/R documentation
    - [x] Run `git diff --check`
- [x] Task: Complete final evidence record
    - [x] Record package check status, accepted NOTE rationale, and generated artifact policy
    - [x] Record r-universe/CRAN remaining external actions, if any
    - [x] Confirm no core mars functionality or public API was changed
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 6: Final Track Review and Archive Readiness' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)
