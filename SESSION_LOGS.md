# pymars Development Session Logs

This file records significant decisions, tool outputs, and progress during development sessions for the `pymars` library.

## Session 1: Initial Project Setup & Planning

**Date:** 2023-10-27 (Placeholder Date)

**Developer/Agent:** Jules (AI Agent)

**Goal:** Set up the basic project structure and create initial planning/documentation files.

**Actions Taken:**

1.  **Plan Creation:**
    *   A detailed multi-step plan was formulated to address the user's request for creating a pure Python adaptation of `py-earth` with scikit-learn compatibility.
    *   The plan emphasizes phased development: core model, scikit-learn layer, advanced features, testing, and documentation.
    *   Tool used: `set_plan`

2.  **Directory Structure:**
    *   Created `pymars/` directory for the main library code.
    *   Created `pymars/__init__.py`.
    *   Created `tests/` directory for test code.
    *   Created `tests/__init__.py`.
    *   Tool used: `create_file_with_block` for each file.

3.  **Documentation & Planning Files Created:**
    *   `ROADMAP.md`: Outlined the multi-phase development plan, including key objectives and tasks for each phase.
        *   Tool used: `create_file_with_block`
    *   `AGENTS.md`: Provided guidelines for AI agents working on the project, focusing on scikit-learn compatibility, `py-earth` structure, pure Python implementation, coding standards, and testing.
        *   Tool used: `create_file_with_block`
    *   `GEMINI.md`: Added specific notes for Gemini models, covering code generation strategies, tool usage, and testing considerations.
        *   Tool used: `create_file_with_block`
    *   `TODO.md`: Created a detailed checklist of tasks, broken down by phase and module, to track progress.
        *   Tool used: `create_file_with_block`
    *   `SESSION_LOGS.md`: This file was created to log these initial actions.
        *   Tool used: `create_file_with_block`

**Key Decisions:**

*   The project will be named `pymars`.
*   The import convention `import pymars as earth` will be a target, similar to `py-earth`.
*   Emphasis on scikit-learn compatibility from the outset.
*   Phased approach to manage complexity.

**Next Steps (as per plan):**

*   Populate `AGENTS.md` (already done by creating it with content).
*   Populate `ROADMAP.md` (already done).
*   Populate `TODO.md` (already done).
*   Begin core `Earth` model implementation.

**Tool Outputs (Summary):**
*   All `create_file_with_block` calls were successful.
*   `set_plan` call was successful.

---
*(This marks the end of the initial setup phase as per step 1 of the plan)*

---

## 2026-04-18 00:55:04 AEST

**Summary:**

Recovered previously uncommitted local work, reconciled it onto current `origin/main`,
validated the result, published `mars-earth` `1.0.4` to PyPI, and advanced the conda
release work. PyPI is now live at `1.0.4`. The conda-forge automation token in GitHub
Actions is still invalid for push operations, but the staged-recipes PR was created
manually. The Anaconda workflow is still blocked because no Anaconda token is exposed
to the workflow.

**Key Actions:**

1. Recovered local work that had not all previously been committed/pushed.
   * Captured the local working tree and reconciled it cleanly onto latest `main`.
2. Bumped the package version from `1.0.3` to `1.0.4`.
   * Avoided colliding with the already-published PyPI release.
3. Validated the reconciled branch before publishing.
   * `ruff check .` passed.
   * `ty check pymars/` passed.
   * `pytest` passed with `175 passed, 3 skipped`.
4. Published release `v1.0.4`.
   * Pushed `main` and tag `v1.0.4`.
   * Confirmed PyPI artifacts for `mars-earth==1.0.4` are live.
5. Diagnosed the two conda publishing tracks.
   * `anaconda-publish.yml`: workflow receives no `ANACONDA_TOKEN` or `ANACONDA_API_TOKEN`.
   * `conda-publish.yml`: `CONDA_FORGE_PAT` is present, but fails on `git push` to `edithatogo/staged-recipes`.
6. Improved CI diagnostics for conda publishing.
   * Added explicit credential-resolution and early validation steps to both workflows.
7. Completed the conda-forge submission manually.
   * Pushed branch `mars-earth-1.0.4` to `edithatogo/staged-recipes`.
   * Opened conda-forge staged-recipes PR: `https://github.com/conda-forge/staged-recipes/pull/33010`.

**Current State:**

* `main` is clean and at commit `f5ea47f`.
* Release tag `v1.0.4` points at commit `37d2699`.
* PyPI release is complete.
* conda-forge submission is open and awaiting review.
* Anaconda.org publication is still blocked on missing workflow secret exposure.

**Remaining Blockers / Follow-up:**

1. Add or expose an Anaconda token to the workflow under `ANACONDA_TOKEN` or `ANACONDA_API_TOKEN`.
2. Replace or fix `CONDA_FORGE_PAT` if workflow-based conda-forge automation is still desired.
3. Monitor conda-forge staged-recipes PR `#33010` until merged.

---

## 2026-04-18 Packaging Follow-up

**Summary:**

Packaging and release automation were driven to a stable end state.

**Resolved:**

1. `ANACONDA_API_TOKEN` is now exposed to the GitHub Actions workflow.
2. `Publish to Anaconda.org` was fixed and now succeeds.
   * Root cause 1: the token was missing from the repo's Actions secret scope during earlier runs.
   * Root cause 2: the workflow incorrectly defaulted the upload user to the GitHub repository owner (`edithatogo`), which is not the Anaconda.org account for the token.
   * Fix: the workflow now uploads as the explicit `ANACONDA_USER` only when that variable is set; otherwise it lets the token owner account be inferred by the Anaconda CLI.
3. `Publish to conda-forge` was hardened.
   * The workflow now checks for an already-open staged-recipes PR before attempting push/PR creation.
   * The workflow now performs a real push-capability validation instead of relying on a public `git ls-remote` access check.

**External State:**

* PyPI: `mars-earth==1.0.4` is live.
* Anaconda.org: publish workflow completed successfully (`24596198520`).
* conda-forge: staged-recipes PR is open and healthy: `conda-forge/staged-recipes#33010`.

**Current Limitation:**

* `CONDA_FORGE_PAT` still does not appear to be a valid GitHub HTTPS push token for first-time automated staged-recipes submissions.
* This does not block the current release because the staged-recipes PR already exists and passes checks.
* Future brand-new versions will still require either:
  * a corrected `CONDA_FORGE_PAT`, or
  * a repeat of the manual staged-recipes PR process.

---

## 2026-04-18 Profiling Follow-up

**Summary:**

Validated the repository again after the release work and completed the last open
performance-profiling follow-up item without merging any risky optimisation.

**Validation:**

* `uv run ruff check pymars tests` passed.
* `uv run ty check pymars/` passed.
* `uv run pytest -q` passed with `175 passed, 3 skipped`.

**Profiling Findings:**

* `scripts/profile_pymars.py` still shows the forward pass as the dominant cost centre.
* The main remaining hotspot is candidate evaluation in `pymars._forward.ForwardPasser`:
  * repeated `_build_basis_matrix(...)` calls during candidate scoring
  * repeated `_calculate_rss_and_coeffs(...)` least-squares solves via `numpy.linalg.lstsq`
* Prediction and pruning remain comparatively cheap.

**Decision:**

* A memoization-based optimisation for basis evaluation was explored and validated,
  but not kept.
* It changed the internal call profile without delivering a clear end-to-end runtime
  improvement, so it was reverted to avoid unnecessary complexity and regression risk.

---

## 2026-04-18 Release Verification Follow-up

**Summary:**

Verified the post-release state end-to-end against the current `main` checkout and
recorded the exact package publication status across PyPI, Anaconda.org, and
conda-forge.

**Repository State:**

* Current local `main` is clean and synced with `origin/main`.
* Current HEAD: `19f4b92` (`19f4b92be81869eae0b96fadb6031b72e120030e`).
* Repository validation on the current checkout passed again:
  * `uv run ruff check pymars tests`
  * `uv run ty check pymars/`
  * `uv run pytest -q`
  * Result: `175 passed, 3 skipped`

**Packaging State:**

* PyPI:
  * Confirmed `mars-earth==1.0.4` is live.
  * Published artifacts present:
    * `mars_earth-1.0.4-py3-none-any.whl`
    * `mars_earth-1.0.4.tar.gz`
* Anaconda.org:
  * Confirmed successful workflow run `24596198520`.
  * Verified the workflow built `mars-earth-1.0.4-py_0.conda` and uploaded it successfully.
  * Verified upload destination:
    * `https://anaconda.org/doughnut/mars-earth`
  * Important clarification:
    * earlier checks against the `edithatogo` namespace were misleading
      because the token resolves to the `doughnut` Anaconda account
* conda-forge:
  * Staged-recipes PR remains open:
    * `https://github.com/conda-forge/staged-recipes/pull/33010`
  * The PR is healthy and passing checks.
  * It is still waiting on conda-forge maintainer review/merge.

**Additional Release Automation Fixes Verified:**

* GitHub Actions workflows on the current branch are green after correcting the
  Codecov action input in `.github/workflows/ci.yml` from `file:` to `files:`.
* The repository-side release automation is no longer the active blocker for the
  current release line.

**Current Remaining External Blocker:**

* The only remaining incomplete distribution path is final conda-forge publication,
  which depends on maintainers merging staged-recipes PR `#33010`.

---

## 2026-04-19 Runtime and Roadmap Hardening

**Summary:**

Introduced a slightly more explicit portable runtime surface and aligned the repo's
planning documents with the actual state of the codebase and the broader
multi-language direction.

**Changes:**

1. Added runtime file helpers:
   * `pymars.save_model(...)`
   * `pymars.load_model(...)`
2. Added `ModelSpec` validation:
   * validates supported schema version
   * validates presence of required top-level fields
3. Added tests for:
   * runtime save/load round-trip
   * invalid model-spec version rejection
4. Updated roadmap and TODO:
   * corrected the prior overstatement that `sample_weight` was already implemented
   * added explicit phases for runtime portability, maturity hardening, and broader language bindings
5. Updated docs:
   * expanded design guidance around training vs portable-model vs runtime layers
   * documented the new runtime API surface
   * aligned MkDocs metadata more closely with the current project naming and package URLs

---

## 2026-04-19 Weighted Fitting Support

**Summary:**

Implemented end-to-end `sample_weight` support in the core `Earth` fit path and
the sklearn-compatible wrappers, then updated tests and planning docs to reflect
the new state precisely.

**Changes:**

1. Added weighted least-squares support in:
   * `ForwardPasser._calculate_rss_and_coeffs(...)`
   * `PruningPasser._calculate_rss_and_coeffs(...)`
2. Threaded `sample_weight` through:
   * `Earth.fit(...)`
   * `ForwardPasser.run(...)`
   * `PruningPasser.run(...)`
   * `EarthRegressor.fit(...)`
   * `EarthClassifier.fit(...)`
3. Updated intercept-only and fallback model handling so weighted means, RSS,
   and MSE are computed correctly.
4. Added tests covering:
   * weighted least-squares coefficient equivalence
   * `EarthRegressor.fit(..., sample_weight=...)`
   * weighted-fit behavior changing the learned regressor
   * `EarthClassifier.fit(..., sample_weight=...)`
5. Updated roadmap/TODO/docs:
   * marked baseline `sample_weight` support as implemented
   * explicitly called out the remaining duplication-equivalence gap in
     estimator-check semantics

**Validation:**

* `uv run ruff check pymars tests`
* `uv run pytest -q tests/test_forward.py tests/test_sklearn_compat.py`
* `uv run pytest -q`

---

## 2026-04-19 Weighted Model-Selection Equivalence

**Summary:**

Completed the remaining `sample_weight` hardening work so weighted fitting now
behaves duplication-equivalently under sklearn's estimator checks, including the
`weight == 0` case behaving like row removal.

**Changes:**

1. Updated forward-pass heuristics to use effective weighted sample counts for:
   * GCV complexity calculations
   * default `max_terms` limits
   * parent-activity counts used by `minspan` logic
2. Filtered zero-weight rows out of knot and categorical candidate generation so
   candidate search matches repeated/removed-row semantics.
3. Updated pruning-pass GCV bookkeeping to use weighted valid-row counts
   consistently.
4. Removed the temporary sklearn expected-failure exemptions for
   `check_sample_weight_equivalence_on_dense_data`.
5. Refreshed tests and roadmap/docs to reflect that the duplication-equivalence
   gap is now closed.

**Validation:**

* `uv run ruff check pymars tests`
* `uv run pytest -q tests/test_sklearn_compat.py -k 'check_estimator or sample_weight'`
* `uv run pytest -q`

---

## 2026-04-19 Portable Model Contract Fixtures

**Summary:**

Added a checked-in `ModelSpec` fixture, tightened loader validation, and
documented the schema contract so persistence compatibility is now enforced by
artifact-backed tests instead of only round-tripping live models.

**Changes:**

1. Added structural validation for portable model specs:
   * top-level type checks
   * required container-type checks
   * coefficient/basis-term cardinality checks
   * basis-term `kind` validation
2. Added checked-in fixture:
   * `tests/fixtures/model_spec_v1.json`
3. Added compatibility tests covering:
   * loading a historical checked-in `1.0` fixture
   * stable probe predictions from that fixture
   * loading specs from both path and raw JSON text
   * explicit rejection of malformed contract violations
4. Added dedicated documentation:
   * `docs/model_spec.md`
   * navigation links from the docs home and usage pages
5. Updated roadmap/TODO to mark schema-contract documentation and persistence
   fixture coverage complete.

**Validation:**

* `uv run ruff check pymars tests`
* `uv run pytest -q tests/test_model_spec.py`
* `uv run pytest -q`

---

## 2026-04-20 Deterministic Numerical Regression Fixtures

**Summary:**

Added checked-in regression fixtures for representative `Earth` fits so the
current implementation has stable numerical lockfiles for forward and prediction
behavior without depending on external MARS packages as validation oracles.

**Changes:**

1. Added fixture-backed coverage for two representative models:
   * a simple 1D piecewise-linear regression case
   * a mixed 3D interaction case
2. Locked down the following outputs in tests:
   * basis-function strings
   * fitted coefficients
   * probe predictions
   * `gcv_`, `rss_`, and `mse_`
3. Added checked-in fixture data:
   * `tests/fixtures/reference_regression_cases.json`
4. Added the corresponding regression test:
   * `tests/test_reference_regression.py`
5. Validated the repo after the addition:
   * `uv run ruff check pymars tests`
   * `uv run pytest -q tests/test_reference_regression.py`
   * `uv run pytest -q`

**Validation Direction:**

* `py-earth` and R `earth` remain historical/API references, not required
  implementation or validation dependencies.
* The reference path is the pure Python trainer plus checked-in estimator
  fixtures and portable `ModelSpec` fixtures.
* Cross-runtime confidence comes from replaying those portable fixtures in
  non-Python runtimes, currently the Rust reference runtime.

---

## 2026-04-27 Reference Validation Track Realignment

**Summary:**

Realigned the reference regression validation track with the project goal:
`mars` is a pure Python training implementation moving toward a portable
runtime contract, not a wrapper around or test harness for `py-earth` or R
`earth`.

**Changes:**

1. Marked the reference validation track complete because the required
   validation surface now exists:
   * frozen Python fitted-model regression cases
   * portable `ModelSpec` fixtures
   * Rust runtime parity tests for `validate`, `design_matrix`, and `predict`
2. Replaced the stale external-package blocker with the current validation
   authority:
   * Python remains the compatibility baseline while Rust becomes the core
   * `ModelSpec` is the cross-runtime contract
   * Rust is the current foreign-language reference consumer
3. Updated user-facing design/model-spec docs to state that `py-earth` and R
   `earth` are historical/API references, not required validation gates.

---

## 2026-04-27 Rust Core Direction

**Summary:**

Updated the product and planning documents to make the intended architecture
explicit: `mars` should move toward a shared Rust computational core surfaced
through Python, R, Julia, Rust, C#, Go, and TypeScript APIs.

**Changes:**

1. Updated the product definition and user-facing docs:
   * current Python implementation remains the compatibility baseline
   * Rust replay prototype is the seed of the shared core
   * language bindings are target products, not incidental future consumers
2. Added a new Conductor track:
   * `Rust core and multi-language bindings`
3. Defined planned phases for:
   * Rust core ownership and crate boundaries
   * runtime consolidation
   * Python binding strategy
   * R, Julia, Rust, C#, Go, and TypeScript package surfaces
   * shared conformance testing across bindings
