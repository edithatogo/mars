# Implementation Plan: Strict Ruff, Strict Typing, >90% Coverage

## Phase 0: Baseline Measurement
[checkpoint: completed]

- [x] Task: Measure current ruff strictness level [7ad4a42]
    - [x] Count current ruff violations (auto-fixable vs manual)
    - [x] Document which rule categories are enabled/disabled
    - [x] Record violations by file for prioritization
    - [x] Capture baseline snapshot in `baseline.md`
- [x] Task: Measure current ty strictness level [ceee702]
    - [x] Run ty in current config, count errors
    - [x] Run ty --strict, count errors
    - [x] Identify files with most type errors
- [x] Task: Measure current test coverage [ceee702]
    - [x] Run pytest --cov and record per-module coverage
    - [x] Identify modules below 90%
    - [x] Identify uncovered lines/functions

## Phase 1: Ruff Strict Mode
[checkpoint: completed]

- [x] Task: Enable strict ruff rules incrementally
    - [x] Enable `ANN` (type annotation) rules — handled by ty, keep ignored
    - [x] Enable `ARG` (unused arguments) — fix legitimate violations
    - [x] Enable `PTH` (pathlib) — convert os.path to pathlib where practical
    - [x] Enable `PERF` (performance) — fix anti-patterns
    - [x] Enable `RET` (return statement analysis) — fix implicit returns
    - [x] Enable `SIM102`, `SIM108`, `SIM101` — simplify conditional logic
    - [x] Remove broad per-file ignores, fix violations
- [x] Task: Add strict ruff preview rules
    - [x] Add `PL` (pylint) rules incrementally
    - [x] Add `FURB` (refurb) rules incrementally
    - [x] Add `TRY` (tryceratops) rules incrementally
    - [x] Add `DOC` (docstring) rules if applicable
- [x] Task: Set ruff `target-version` to `py39` (current minimum)
- [x] Task: Run `ruff check . --fix` and verify no regressions
- [x] Task: Automated Phase Review & Progression (Phase 1)

## Phase 2: Ty Strict Mode (replacing mypy)
[checkpoint: completed]

- [x] Task: Configure ty for strict mode
    - [x] Add ty to pyproject.toml dev dependencies
    - [x] Create `ty.toml` or configure in pyproject.toml
    - [x] Remove mypy.ini (no longer needed)
    - [x] Remove mypy from CI workflows, add ty
- [x] Task: Fix ty errors incrementally (65 diagnostics)
    - [x] Fix `invalid-assignment` — use `Optional[T]` or `T | None` where None is valid
    - [x] Fix `invalid-parameter-default` — make default-None parameters `Optional[T]`
    - [x] Fix `unsupported-operator` — replace `X | None` with `Optional[X]` or `"X | None"` for py39
    - [x] Fix `invalid-type-form` — replace `any` with `Any` from typing
    - [x] Fix `not-subscriptable` / `unresolved-attribute` — add None guards before subscripting
    - [x] Fix `invalid-argument-type` — add None guards or widen parameter types
    - [x] Fix `invalid-return-type` — widen return types where None can be returned
    - [x] Run `ty check pymars/` until zero diagnostics
- [x] Task: Add type annotations to untyped modules
    - [x] Add return types to all public functions
    - [x] Add parameter types to all public functions
    - [x] Add type annotations to class attributes
    - [x] Use `typing` module imports (Optional, Union, List, Dict, etc.)
    - [x] Use `collections.abc` imports (Iterator, Mapping, etc.)
- [x] Task: Automated Phase Review & Progression (Phase 2)

## Phase 3: Test Coverage >90%
[checkpoint: completed]

- [x] Task: Raise coverage for core algorithm modules
    - [x] `pymars/earth.py` — forward pass, pruning, fit, predict
    - [x] `pymars/_basis.py` — all basis function types
    - [x] `pymars/_forward.py` — forward pass logic
    - [x] `pymars/_pruning.py` — GCV pruning logic
    - [x] `pymars/_categorical.py` — categorical imputation
    - [x] `pymars/_missing.py` — missing value handling
- [x] Task: Raise coverage for sklearn compatibility modules
    - [x] `pymars/_sklearn_compat.py` — EarthRegressor, EarthClassifier
    - [x] `pymars/glm.py` — GLMEarth
    - [x] `pymars/cv.py` — EarthCV
- [x] Task: Raise coverage for utilities
    - [x] `pymars/plot.py` — plotting functions
    - [x] `pymars/cli.py` — CLI commands
    - [x] `pymars/demos/` — demo scripts
- [x] Task: Add property-based tests with hypothesis
    - [x] Test MARS invariants (e.g., prediction shape matches input)
    - [x] Test parameter validation (invalid inputs raise expected errors)
    - [x] Test numerical stability across input ranges
- [x] Task: Raise coverage threshold to >90%
    - [x] Update `fail_under` in pyproject.toml to 90
    - [x] Verify all modules meet threshold
    - [x] Automated Phase Review & Progression (Phase 3)

## Phase 4: Final Verification & Quality Gates
[checkpoint: completed]

- [x] Task: Run full quality gate suite
    - [x] `ruff check .` — zero violations
    - [x] `ruff format --check .` — zero formatting issues
    - [x] `ty pymars` — zero errors in strict mode
    - [x] `pytest --cov=pymars --cov-report=term-missing` — >90% coverage
    - [x] All tests pass
- [x] Task: Update CI workflows
    - [x] Ensure code-quality workflow runs ruff strict + ty strict
    - [x] Ensure CI workflow fails on coverage below 90%
    - [x] Add ty strict check to code-quality workflow
- [x] Task: Final documentation update
    - [x] Update DEVELOPMENT.md with strict mode requirements
    - [x] Update CONTRIBUTING.md with type annotation guidelines
    - [x] Update README badges for coverage
- [x] Task: Automated Phase Review & Progression (Phase 4)

## Track Completion Protocol (Automated)

- [x] Task: Track Completion & Auto-Archival
    - [x] Push all remaining changes to remote
    - [x] Monitor GitHub Actions runs until ALL pass
    - [x] Verify all 4 phase checkpoints are complete
    - [x] Run final comprehensive quality gate check
    - [x] Update track metadata status to 'completed'
    - [x] Update tracks.md to mark track as complete [x]
    - [x] Commit all final changes with descriptive message
    - [x] Archive track (update metadata.json with completed_at timestamp)
    - [x] Announce track completion

- [x] Task: Auto-Progress to Next Track
    - [x] Check if next track exists in tracks.md
    - [x] If next track exists, set up next track
    - [x] If no next track exists, all tracks complete
