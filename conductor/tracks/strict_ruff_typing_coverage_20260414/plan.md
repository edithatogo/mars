# Implementation Plan: Strict Ruff, Strict Typing, >90% Coverage

## Phase 0: Baseline Measurement
[checkpoint: pending]

- [x] Task: Measure current ruff strictness level
    - [x] Count current ruff violations (auto-fixable vs manual)
    - [x] Document which rule categories are enabled/disabled
    - [x] Record violations by file for prioritization
    - [x] Capture baseline snapshot in `baseline.md`
- [ ] Task: Measure current mypy strictness level
    - [ ] Run mypy in current config, count errors
    - [ ] Run mypy --strict, count errors
    - [ ] Identify files with most type errors
- [ ] Task: Measure current test coverage
    - [ ] Run pytest --cov and record per-module coverage
    - [ ] Identify modules below 90%
    - [ ] Identify uncovered lines/functions

## Phase 1: Ruff Strict Mode
[checkpoint: pending]

- [ ] Task: Enable strict ruff rules incrementally
    - [ ] Enable `ANN` (type annotation) rules — handled by mypy, keep ignored
    - [ ] Enable `ARG` (unused arguments) — fix legitimate violations
    - [ ] Enable `PTH` (pathlib) — convert os.path to pathlib where practical
    - [ ] Enable `PERF` (performance) — fix anti-patterns
    - [ ] Enable `RET` (return statement analysis) — fix implicit returns
    - [ ] Enable `SIM102`, `SIM108`, `SIM101` — simplify conditional logic
    - [ ] Remove broad per-file ignores, fix violations
- [ ] Task: Add strict ruff preview rules
    - [ ] Add `PL` (pylint) rules incrementally
    - [ ] Add `FURB` (refurb) rules incrementally
    - [ ] Add `TRY` (tryceratops) rules incrementally
    - [ ] Add `DOC` (docstring) rules if applicable
- [ ] Task: Set ruff `target-version` to `py39` (current minimum)
- [ ] Task: Run `ruff check . --fix` and verify no regressions
- [ ] Task: Automated Phase Review & Progression (Phase 1)

## Phase 2: Ty Strict Mode (replacing mypy)
[checkpoint: pending]

- [ ] Task: Configure ty for strict mode
    - [ ] Add ty to pyproject.toml dev dependencies
    - [ ] Create `ty.toml` or configure in pyproject.toml
    - [ ] Remove mypy.ini (no longer needed)
    - [ ] Remove mypy from CI workflows, add ty
- [ ] Task: Fix ty errors incrementally (65 diagnostics)
    - [ ] Fix `invalid-assignment` — use `Optional[T]` or `T | None` where None is valid
    - [ ] Fix `invalid-parameter-default` — make default-None parameters `Optional[T]`
    - [ ] Fix `unsupported-operator` — replace `X | None` with `Optional[X]` or `"X | None"` for py39
    - [ ] Fix `invalid-type-form` — replace `any` with `Any` from typing
    - [ ] Fix `not-subscriptable` / `unresolved-attribute` — add None guards before subscripting
    - [ ] Fix `invalid-argument-type` — add None guards or widen parameter types
    - [ ] Fix `invalid-return-type` — widen return types where None can be returned
    - [ ] Run `ty check pymars/` until zero diagnostics
- [ ] Task: Add type annotations to untyped modules
    - [ ] Add return types to all public functions
    - [ ] Add parameter types to all public functions
    - [ ] Add type annotations to class attributes
    - [ ] Use `typing` module imports (Optional, Union, List, Dict, etc.)
    - [ ] Use `collections.abc` imports (Iterator, Mapping, etc.)
- [ ] Task: Automated Phase Review & Progression (Phase 2)

## Phase 3: Test Coverage >90%
[checkpoint: pending]

- [ ] Task: Raise coverage for core algorithm modules
    - [ ] `pymars/earth.py` — forward pass, pruning, fit, predict
    - [ ] `pymars/_basis.py` — all basis function types
    - [ ] `pymars/_forward.py` — forward pass logic
    - [ ] `pymars/_pruning.py` — GCV pruning logic
    - [ ] `pymars/_categorical.py` — categorical imputation
    - [ ] `pymars/_missing.py` — missing value handling
- [ ] Task: Raise coverage for sklearn compatibility modules
    - [ ] `pymars/_sklearn_compat.py` — EarthRegressor, EarthClassifier
    - [ ] `pymars/glm.py` — GLMEarth
    - [ ] `pymars/cv.py` — EarthCV
- [ ] Task: Raise coverage for utilities
    - [ ] `pymars/plot.py` — plotting functions
    - [ ] `pymars/cli.py` — CLI commands
    - [ ] `pymars/demos/` — demo scripts
- [ ] Task: Add property-based tests with hypothesis
    - [ ] Test MARS invariants (e.g., prediction shape matches input)
    - [ ] Test parameter validation (invalid inputs raise expected errors)
    - [ ] Test numerical stability across input ranges
- [ ] Task: Raise coverage threshold to >90%
    - [ ] Update `fail_under` in pyproject.toml to 90
    - [ ] Verify all modules meet threshold
    - [ ] Automated Phase Review & Progression (Phase 3)

## Phase 4: Final Verification & Quality Gates
[checkpoint: pending]

- [ ] Task: Run full quality gate suite
    - [ ] `ruff check .` — zero violations
    - [ ] `ruff format --check .` — zero formatting issues
    - [ ] `mypy pymars` — zero errors in strict mode
    - [ ] `pytest --cov=pymars --cov-report=term-missing` — >90% coverage
    - [ ] All tests pass
- [ ] Task: Update CI workflows
    - [ ] Ensure code-quality workflow runs ruff strict + mypy strict
    - [ ] Ensure CI workflow fails on coverage below 90%
    - [ ] Add mypy strict check to code-quality workflow
- [ ] Task: Final documentation update
    - [ ] Update DEVELOPMENT.md with strict mode requirements
    - [ ] Update CONTRIBUTING.md with type annotation guidelines
    - [ ] Update README badges for coverage
- [ ] Task: Automated Phase Review & Progression (Phase 4)

## Track Completion Protocol (Automated)

- [ ] Task: Track Completion & Auto-Archival
    - [ ] Push all remaining changes to remote
    - [ ] Monitor GitHub Actions runs until ALL pass
    - [ ] Verify all 4 phase checkpoints are complete
    - [ ] Run final comprehensive quality gate check
    - [ ] Update track metadata status to 'completed'
    - [ ] Update tracks.md to mark track as complete [x]
    - [ ] Commit all final changes with descriptive message
    - [ ] Archive track (update metadata.json with completed_at timestamp)
    - [ ] Announce track completion

- [ ] Task: Auto-Progress to Next Track
    - [ ] Check if next track exists in tracks.md
    - [ ] If next track exists, set up next track
    - [ ] If no next track exists, all tracks complete
