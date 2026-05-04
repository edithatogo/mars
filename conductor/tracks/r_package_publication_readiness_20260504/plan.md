# Implementation Plan

## Phase 0: Inventory, Evidence, and Work Split

- [ ] Task: Inventory the current R package surface
    - [ ] Agent 1: audit `bindings/r/DESCRIPTION`, `NAMESPACE`, and `README.md`
    - [ ] Agent 2: audit `bindings/r/man`, manual generation, and vignette/source article coverage
    - [ ] Agent 3: audit `bindings/r/tests` and source-tree versus installed-package behavior
    - [ ] Agent 4: audit R CI, release rehearsal, artifact upload, and install smoke coverage
    - [ ] Agent 5: audit R rows in release inventory, handoff, checklist, and package-path docs
    - [ ] Agent 6: consolidate the gap list and map each gap to a later phase
- [ ] Task: Establish validation baseline
    - [ ] Run `Rscript bindings/r/tests/conformance.R`
    - [ ] Run `Rscript bindings/r/tests/training.R` with and without `MARS_RUNTIME_BIN`
    - [ ] Run `R CMD build bindings/r`
    - [ ] Run `R CMD check --no-manual --as-cran` on the built tarball
    - [ ] Run the release alignment check after documenting current state
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 0: Inventory, Evidence, and Work Split' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 1: R Package Metadata and API Documentation

- [ ] Task: Align R package metadata and namespace
    - [ ] Confirm package name, title, description, license, URLs, bug tracker, maintainer, and dependency declarations
    - [ ] Confirm exports and imports in `NAMESPACE` match the callable API
    - [ ] Confirm package README describes runtime replay, Rust-backed training, validation, and installation accurately
- [ ] Task: Complete Rd documentation for every export
    - [ ] Add or verify a package-level `marsruntime-package` help topic
    - [ ] Ensure `load_model_spec` documents file and JSON-string inputs
    - [ ] Ensure `validate_model_spec` documents invisible success and validation failure behavior
    - [ ] Ensure `design_matrix` documents accepted row inputs and matrix return shape
    - [ ] Ensure `predict_model` documents accepted row inputs and prediction output
    - [ ] Ensure `fit_model` documents every training argument and runtime-binary requirement
    - [ ] Add examples that are CRAN-safe and do not require unavailable external binaries
- [ ] Task: Validate R documentation integrity
    - [ ] Run Rd parse/check validation through `R CMD check`
    - [ ] Verify no missing documentation, code/documentation mismatch, or wide usage warnings remain
    - [ ] Confirm documentation names use `mars`/`mars-earth` branding consistently where appropriate
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: R Package Metadata and API Documentation' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 2: Manual, Vignette, and User Documentation Artifacts

- [ ] Task: Add package-level documentation source that R tooling can build
    - [ ] Decide whether the package should use a lightweight vignette, `inst/doc` article, or manual-only Rd path
    - [ ] Add source documentation for installing, loading, validating, predicting, and Rust-backed fitting
    - [ ] Ensure examples do not require private files, registry credentials, or unavailable binaries
    - [ ] Add `VignetteBuilder` and `Suggests` only if a real vignette source is adopted
- [ ] Task: Validate package manual generation
    - [ ] Run `R CMD Rd2pdf bindings/r` when the local LaTeX/R toolchain supports it
    - [ ] If PDF generation is unavailable locally, record the missing toolchain and keep the source path build-ready
    - [ ] Ensure generated PDF/manual artifacts are ignored or handled according to release policy
- [ ] Task: Sync project documentation
    - [ ] Update `docs/package_release_paths.md` with the R manual/vignette policy
    - [ ] Update `docs/publication_handoff.md` with the R documentation artifact status
    - [ ] Update `docs/release_checklist.md` with manual/vignette verification steps
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: Manual, Vignette, and User Documentation Artifacts' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 3: CRAN-Safe Tests and Runtime-Binary Behavior

- [ ] Task: Harden source-tree and installed-package tests
    - [ ] Keep source-tree fixture conformance active when repository fixtures exist
    - [ ] Decide whether to vendor a minimized fixture subset into the R package or keep fixture conformance as a source-tree-only gate
    - [ ] Ensure installed-package `R CMD check` does not require repository fixture files
    - [ ] Ensure tests load the installed package under `R CMD check` and source files during local source-tree runs
- [ ] Task: Harden Rust runtime optionality
    - [ ] Ensure `fit_model` reports a clear error when the Rust runtime binary is unavailable
    - [ ] Ensure training tests exercise Rust-backed fitting when `MARS_RUNTIME_BIN` is set
    - [ ] Ensure training tests skip clearly when the binary is unavailable
    - [ ] Ensure replay functions fall back to pure R runtime logic where that is the established package behavior
- [ ] Task: Validate R package behavior
    - [ ] Run `Rscript bindings/r/tests/conformance.R`
    - [ ] Run `Rscript bindings/r/tests/training.R`
    - [ ] Run `MARS_RUNTIME_BIN=<local binary> Rscript bindings/r/tests/training.R`
    - [ ] Run `R CMD build bindings/r`
    - [ ] Run `R CMD check --no-manual --as-cran` with `MARS_RUNTIME_BIN` set
    - [ ] Run `R CMD check --no-manual --as-cran` without `MARS_RUNTIME_BIN` when feasible
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: CRAN-Safe Tests and Runtime-Binary Behavior' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 4: CI, Release Rehearsal, and Artifact Inspection

- [ ] Task: Add or verify CI coverage for the R package
    - [ ] Ensure GitHub Actions runs R package build/check in the binding or release-rehearsal workflow
    - [ ] Replace build/install-only validation with `R CMD check` or `rcmdcheck`
    - [ ] Consider Linux/macOS/Windows and current/release/devel R coverage where runtime availability makes sense
    - [ ] Ensure the workflow installs R dependencies declared in `DESCRIPTION`
    - [ ] Ensure the workflow builds the Rust runtime helper when training smoke tests are expected
    - [ ] Ensure package check logs are uploaded as artifacts on failure
- [ ] Task: Add artifact and install smoke validation
    - [ ] Inspect the built R source tarball contents
    - [ ] Install the built tarball into a clean R library
    - [ ] Run a minimal validate/design/predict smoke test against the installed package
    - [ ] Run the training smoke test only when the helper binary is explicitly provided
- [ ] Task: Align release rehearsal docs
    - [ ] Update `docs/ci_quality.md` and release docs with the R build/check/install commands
    - [ ] Ensure `scripts/check_release_alignment.py` continues to pass after R documentation updates
    - [ ] Record any platform-specific R check caveats in release guidance
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: CI, Release Rehearsal, and Artifact Inspection' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 5: r-universe and CRAN Submission Handoff

- [ ] Task: Decide and document the R submission route
    - [ ] Confirm whether r-universe is the first publication target
    - [ ] Confirm whether CRAN is immediate, deferred until maturity, or explicitly out of the current release
    - [ ] Reconcile maintainer-owner status versus submission-path status across release docs
    - [ ] Record maintainer account, owner, action, and date without committing private credentials
- [ ] Task: Prepare r-universe handoff
    - [ ] Document the GitHub app/source repository setup needed for r-universe
    - [ ] Document expected package page, manual, source artifact, binary artifact, and install command
    - [ ] Add post-publish verification steps for r-universe
- [ ] Task: Prepare CRAN handoff
    - [ ] Document CRAN preflight checks, package check expectations, submission metadata, and manual review notes
    - [ ] Record any CRAN-specific blockers such as package maturity, examples, maintainer approval, or runtime-binary policy
    - [ ] Ensure CRAN handoff does not claim support for external binaries that CRAN cannot build or test
- [ ] Task: Sync Conductor publication state
    - [ ] Update release-readiness and publication tracks to consume this R readiness result
    - [ ] Ensure only external submission/approval work remains after local readiness is complete
    - [ ] Record evidence commands and results in the track plan or linked docs
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 5: r-universe and CRAN Submission Handoff' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)

## Phase 6: Final Track Review and Archive Readiness

- [ ] Task: Run final validation bundle
    - [ ] Run R package tests and package checks
    - [ ] Run docs build and release alignment checks
    - [ ] Run Vale on changed release/R documentation
    - [ ] Run `git diff --check`
- [ ] Task: Complete final evidence record
    - [ ] Record package check status, accepted NOTE rationale, and generated artifact policy
    - [ ] Record r-universe/CRAN remaining external actions, if any
    - [ ] Confirm no core mars functionality or public API was changed
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 6: Final Track Review and Archive Readiness' (Protocol in workflow.md; invoke conductor-review, apply high-confidence fixes, rerun validation, then progress)
