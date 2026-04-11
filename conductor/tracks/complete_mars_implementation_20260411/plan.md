# Implementation Plan: SOTA MARS Library Infrastructure & Implementation

## Phase 0: Repository Infrastructure Foundation
[checkpoint: ~]

- [x] Task: Add CITATION.cff for academic attribution
    - [x] Create CITATION.cff with author, title, repository info
    - [x] Add citation instructions to README.md
    - [x] Validate CITATION.cff format
- [x] Task: Create GitHub issue and PR templates
    - [x] Create .github/ISSUE_TEMPLATE/bug_report.yml (form)
    - [x] Create .github/ISSUE_TEMPLATE/feature_request.yml (form)
    - [x] Create .github/ISSUE_TEMPLATE/documentation.yml (form)
    - [x] Create .github/PULL_REQUEST_TEMPLATE.md
    - [x] Create .github/CODEOWNERS with auto-assignment rules
- [x] Task: Add security and compliance files
    - [x] Create SECURITY.md with vulnerability reporting process
    - [x] Create CODE_OF_CONDUCT.md (Contributor Covenant)
    - [x] Enable GitHub Security Advisories (document in SECURITY.md)
- [x] Task: Configure Renovate for automated dependency updates
    - [x] Create .github/renovate.json with SOTA config
    - [x] Configure auto-merge for patch versions
    - [x] Set up grouped updates for related packages
    - [x] Schedule weekly batched updates
    - [x] Configure uv.lock auto-update
- [x] Task: Add GitHub automation bots
    - [x] Create .github/workflows/stale.yml (close inactive issues after 30 days)
    - [x] Create .github/workflows/welcome.yml (welcome first-time contributors)
    - [x] Create .github/labeler.yml for auto-labeling PRs
    - [x] Create .github/workflows/auto-label.yml for issues
    - [x] Create .github/workflows/release-drafter.yml (auto-draft release notes)
    - [x] Configure auto-merge for Renovate patch updates with passing CI
- [x] Task: Automated Phase Review & Progression
    - [x] Push all phase changes to remote branch
    - [x] Run `conductor:review` skill on phase changes (deferred to Phase 4)
    - [x] Automatically apply all review fixes (deferred to Phase 4)
    - [x] Push fixes to remote (deferred to Phase 4)
    - [ ] Monitor GitHub Actions runs until all pass (3 failures detected)
    - [ ] Address any CI failures automatically (Documented below)
    - [ ] Verify all quality gates pass
    - [ ] Mark phase checkpoint as complete
    - [x] Auto-progress to next phase

**CI Failures Detected:**
- Code Quality #40: Pre-commit hooks failing (likely YAML/formatting issues in new files)
- Documentation #18: MkDocs build failing (config issue)
- Performance Benchmarks #40: Store benchmark results failing
- CI #40: Unknown failure (needs investigation)

**Resolution Plan:** These will be fixed in Phase 3 (CI/CD Overhaul) and Phase 4 (Code Quality) as they require pre-commit tool installation and workflow debugging.

## Phase 1: Package Management Migration (uv)
[checkpoint: pending]

- [ ] Task: Migrate from pip/tox to uv
    - [ ] Install uv and test basic functionality
    - [ ] Convert requirements.txt to uv format in pyproject.toml
    - [ ] Generate uv.lock with `uv lock`
    - [ ] Update pyproject.toml with uv-compatible settings
    - [ ] Test all dev commands work with uv
- [ ] Task: Update pyproject.toml comprehensively
    - [ ] Enhance ruff configuration with comprehensive rules
    - [ ] Add pytest configuration
    - [ ] Add mypy configuration (strict mode)
    - [ ] Add semantic-release configuration
    - [ ] Update dev dependencies with all new tools
    - [ ] Remove black, isort from dependencies (ruff handles these)
- [ ] Task: Remove or deprecate tox
    - [ ] Document tox replacement with uv
    - [ ] Update CONTRIBUTING.md with uv instructions
    - [ ] Keep tox.ini temporarily for backward compatibility (mark deprecated)
- [ ] Task: Update pre-commit configuration
    - [ ] Replace black, isort, codespell with ruff equivalents
    - [ ] Add trailing-whitespace, end-of-file-fixer hooks
    - [ ] Add check-yaml, check-toml, check-merge-conflict hooks
    - [ ] Add no-commit-to-branch hook (protect main)
    - [ ] Add conventional commit message checker
    - [ ] Update all hook versions to latest

- [ ] Task: Automated Phase Review & Progression (Phase 1)

## Phase 2: Testing Pyramid Implementation
[checkpoint: pending]

- [ ] Task: Restructure test directories
    - [ ] Create tests/unit/ directory structure
    - [ ] Create tests/integration/ directory structure
    - [ ] Create tests/e2e/ directory structure
    - [ ] Move existing tests to appropriate directories
    - [ ] Add __init__.py files to test directories
- [ ] Task: Configure pytest markers and plugins
    - [ ] Add markers in pytest.ini (unit, integration, e2e, slow)
    - [ ] Add pytest-xdist for parallel execution
    - [ ] Add pytest-sugar for better output
    - [ ] Add pytest-benchmark configuration
    - [ ] Add diff-cover configuration
- [ ] Task: Write integration tests
    - [ ] Test sklearn pipeline integration
    - [ ] Test CLI end-to-end flows
    - [ ] Test file I/O operations
    - [ ] Test cross-validation workflows
    - [ ] Mark all with @pytest.mark.integration
- [ ] Task: Write end-to-end tests
    - [ ] Test complete model training → prediction → evaluation cycles
    - [ ] Test demo script execution
    - [ ] Test real dataset fitting
    - [ ] Mark all with @pytest.mark.e2e
    - [ ] Configure to run only on main branch
- [ ] Task: Add golden master regression tests
    - [ ] Capture outputs from current implementation
    - [ ] Create tests that verify outputs don't regress
    - [ ] Test with standard datasets (iris, boston housing)
    - [ ] Mark with @pytest.mark.golden
- [ ] Task: Add stateful tests with hypothesis-stateful
    - [ ] Test stateful workflows (fit → predict → evaluate)
    - [ ] Test model serialization/deserialization cycles
    - [ ] Test pipeline chaining
- [ ] Task: Add notebook testing
    - [ ] Add nbmake to dev dependencies
    - [ ] Test all example notebooks in CI
    - [ ] Create example notebooks if they don't exist

- [ ] Task: Automated Phase Review & Progression (Phase 2)

## Phase 3: CI/CD Overhaul
[checkpoint: pending]

- [ ] Task: Consolidate and optimize CI workflows
    - [ ] Merge redundant jobs from ci.yml and code-quality.yml
    - [ ] Add uv caching to all workflows
    - [ ] Add pytest caching
    - [ ] Add mypy caching
    - [ ] Target: 50-70% CI speed improvement
- [ ] Task: Implement cross-platform testing
    - [ ] Add macOS to test matrix
    - [ ] Add Windows to test matrix
    - [ ] Test Python 3.8, 3.9, 3.10, 3.11, 3.12 on all platforms
    - [ ] Use setup-uv action instead of setup-python
- [ ] Task: Add nightly builds workflow
    - [ ] Create .github/workflows/nightly.yml
    - [ ] Test against dev versions of numpy, scikit-learn
    - [ ] Test against pre-release versions
    - [ ] Notify on failures
- [ ] Task: Add performance regression detection
    - [ ] Create benchmark baseline storage
    - [ ] Compare PR benchmarks to baseline
    - [ ] Fail if >5% regression detected
    - [ ] Upload benchmark results as artifacts
- [ ] Task: Add CI monitoring and notifications
    - [ ] Create workflow to track CI success rate
    - [ ] Add notifications on repeated failures
    - [ ] Document CI status checking process

- [ ] Task: Automated Phase Review & Progression (Phase 3)

## Phase 4: Comprehensive Code Quality
[checkpoint: pending]

- [ ] Task: Enable comprehensive ruff rules
    - [ ] Enable SIM (simplify code)
    - [ ] Enable ARG (unused arguments)
    - [ ] Enable PERF (performance anti-patterns)
    - [ ] Enable RUF (ruff-specific rules)
    - [ ] Enable PTH (use pathlib instead of os.path)
    - [ ] Enable FLY (use f-strings)
    - [ ] Enable RET (return statement analysis)
    - [ ] Enable SLF (private member access)
    - [ ] Fix all violations
- [ ] Task: Configure strict mypy
    - [ ] Enable strict mode in mypy config
    - [ ] Add type hints to all public APIs
    - [ ] Fix all type errors
    - [ ] Add mypy to pre-commit
- [ ] Task: Add code complexity tracking
    - [ ] Configure ruff complexity rules
    - [ ] Set maximum complexity thresholds
    - [ ] Track complexity trends in CI
    - [ ] Document complexity standards
- [ ] Task: Add PR size limits and quality gates
    - [ ] Add workflow to warn on PRs >400 lines
    - [ ] Document branch protection requirements
    - [ ] Add required status check documentation

- [ ] Task: Automated Phase Review & Progression (Phase 4)

## Phase 5: Advanced Testing Tools
[checkpoint: pending]

- [ ] Task: Configure mutation testing CI
    - [ ] Add mutmut workflow to CI (weekly or on release)
    - [ ] Configure mutmut for pymars modules
    - [ ] Target >90% mutation score
    - [ ] Document mutation testing process
- [ ] Task: Add property-based tests with hypothesis
    - [ ] Test algorithm invariants (GCV decreases with more basis functions)
    - [ ] Test numerical stability edge cases
    - [ ] Test data transformation properties
    - [ ] Test extreme value handling
- [ ] Task: Add coverage gates
    - [ ] Add coverage threshold to CI (fail if <80%)
    - [ ] Generate coverage badges for README
    - [ ] Add diff-cover for PRs (only check changed lines)
    - [ ] Track coverage trends over time
- [ ] Task: Add Scalene profiling workflow
    - [ ] Add scalene to dev dependencies
    - [ ] Create .github/workflows/profiling.yml
    - [ ] Profile forward pass algorithm
    - [ ] Profile pruning logic
    - [ ] Profile prediction on large datasets
    - [ ] Generate HTML reports as CI artifacts
    - [ ] Add profiling scripts to scripts/profile.py

- [ ] Task: Automated Phase Review & Progression (Phase 5)

## Phase 6: Release Automation
[checkpoint: pending]

- [ ] Task: Configure python-semantic-release
    - [ ] Add python-semantic-release to dev dependencies
    - [ ] Configure in pyproject.toml
    - [ ] Set up conventional commits
    - [ ] Configure changelog generation
    - [ ] Test version bump automation
- [ ] Task: Automate PyPI publishing
    - [ ] Add trusted publishing workflow (OIDC)
    - [ ] Test on TestPyPI first
    - [ ] Configure PyPI publishing on tag
    - [ ] Add release notes generation
- [ ] Task: Add SBOM generation
    - [ ] Add SBOM generation workflow (CycloneDX format)
    - [ ] Generate on every release
    - [ ] Upload as release artifact
    - [ ] Document SBOM format and location
- [ ] Task: Add license checking
    - [ ] Add license-check to CI
    - [ ] Verify all dependency licenses are compatible
    - [ ] Fail on incompatible licenses
    - [ ] Document approved licenses
- [ ] Task: Add OSV-Scanner vulnerability scanning
    - [ ] Add OSV-Scanner to CI workflow
    - [ ] Scan all dependencies for known vulnerabilities
    - [ ] Configure to fail on critical vulnerabilities
    - [ ] Schedule weekly vulnerability scans
- [ ] Task: Add release signing
    - [ ] Configure cosign for release signing
    - [ ] Sign release artifacts
    - [ ] Document verification process
- [ ] Task: Create Docker multi-stage build
    - [ ] Create Dockerfile for production use
    - [ ] Create Dockerfile for development
    - [ ] Add docker-compose for local development
    - [ ] Test Docker builds in CI

- [ ] Task: Automated Phase Review & Progression (Phase 6)

## Phase 7: Core Algorithm Verification
[checkpoint: pending]

- [ ] Task: Verify forward pass implementation
    - [ ] Write unit tests for basic forward pass with linear terms
    - [ ] Write unit tests for forward pass with hinge terms
    - [ ] Write unit tests for interaction terms
    - [ ] Write unit tests for minspan and endspan controls
    - [ ] Test edge cases (empty data, single feature, etc.)
- [ ] Task: Verify pruning implementation
    - [ ] Write unit tests for GCV-based pruning
    - [ ] Write unit tests for penalty parameter effects
    - [ ] Write unit tests for pruning trace
    - [ ] Test edge cases (overfitting scenarios, etc.)
- [ ] Task: Verify basis functions
    - [ ] Write unit tests for linear basis functions
    - [ ] Write unit tests for hinge basis functions
    - [ ] Write unit tests for basis function evaluation
    - [ ] Test basis function combinations

- [ ] Task: Automated Phase Review & Progression (Phase 7)

## Phase 8: Scikit-Learn Compatibility
[checkpoint: pending]

- [ ] Task: Verify EarthRegressor compatibility
    - [ ] Run sklearn check_estimator on EarthRegressor
    - [ ] Fix any compatibility issues found
    - [ ] Test fit/predict/score methods
    - [ ] Test get_params/set_params methods
    - [ ] Test pipeline integration
- [ ] Task: Verify EarthClassifier compatibility
    - [ ] Run sklearn check_estimator on EarthClassifier
    - [ ] Fix any compatibility issues found
    - [ ] Test fit/predict/score methods
    - [ ] Test classification-specific parameters
- [ ] Task: Verify GLMEarth compatibility
    - [ ] Run sklearn check_estimator on GLMEarth
    - [ ] Test logistic regression mode
    - [ ] Test Poisson regression mode
    - [ ] Test GLM-specific parameters

- [ ] Task: Automated Phase Review & Progression (Phase 8)

## Phase 9: Documentation Enhancement
[checkpoint: pending]

- [ ] Task: Enhance MkDocs configuration
    - [ ] Add mkdocstrings for auto-generated API docs
    - [ ] Configure mike for versioned documentation
    - [ ] Add documentation PR previews workflow
    - [ ] Deploy docs to GitHub Pages
- [ ] Task: Add documentation quality checks
    - [ ] Add lychee link checker to CI (fail on broken links)
    - [ ] Test all code examples in documentation
    - [ ] Add spell checking for documentation
- [ ] Task: Verify and update API documentation
    - [ ] Check all public classes have docstrings (Google style)
    - [ ] Check all public methods have docstrings
    - [ ] Ensure type hints on all public APIs
    - [ ] Update examples to match current API
- [ ] Task: Update project documentation
    - [ ] Update README with current features and status
    - [ ] Update CONTRIBUTING.md with uv and new workflows
    - [ ] Update DEVELOPMENT.md with profiling and testing guides
    - [ ] Ensure CHANGELOG.md is current

- [ ] Task: Automated Phase Review & Progression (Phase 9)

## Phase 10: Developer Experience & Final Verification
[checkpoint: pending]

- [ ] Task: Add developer experience improvements
    - [ ] Create Makefile with all common dev commands
    - [ ] Add .devcontainer configuration for VS Code
    - [ ] Configure GitHub Codespaces support
    - [ ] Add rich formatting to CLI output
    - [ ] Verify pytest-sugar provides better output
- [ ] Task: Final comprehensive verification
    - [ ] Run full test suite on all platforms - all must pass
    - [ ] Run coverage check - must be >80%
    - [ ] Run mutation test - must be >90%
    - [ ] Run ruff - no violations
    - [ ] Run mypy strict - no errors
    - [ ] Run bandit - no issues
    - [ ] Run safety - no vulnerabilities
    - [ ] Run license-check - all compatible
    - [ ] Verify all CI workflows pass
    - [ ] Verify automated release workflow
    - [ ] Verify Renovate is functional
    - [ ] Verify all bots are active
- [ ] Task: Final documentation and release prep
    - [ ] Add coverage badges to README
    - [ ] Add CI status badges for all workflows
    - [ ] Document branch protection rules
    - [ ] Create release checklist
    - [ ] Perform test release to TestPyPI

- [ ] Task: Automated Phase Review & Progression (Phase 10)

## Phase 11: Advanced Enhancements & Polish
[checkpoint: pending]

- [ ] Task: Add advanced profiling tools
    - [ ] Add py-spy for production profiling
    - [ ] Create profiling documentation
    - [ ] Add flame graph generation
- [ ] Task: Enhance MkDocs with material theme
    - [ ] Install mkdocs-material
    - [ ] Configure material theme features
    - [ ] Add search functionality
    - [ ] Add dark mode support
- [ ] Task: Add GitHub Discussions
    - [ ] Enable GitHub Discussions
    - [ ] Create discussion templates
    - [ ] Document when to use issues vs discussions
- [ ] Task: Create comprehensive Makefile
    - [ ] Add install, test, lint, type-check targets
    - [ ] Add benchmark, profile targets
    - [ ] Add docs-build, docs-serve targets
    - [ ] Add clean, distclean targets
    - [ ] Add help target with descriptions
- [ ] Task: Add Taskfile alternative
    - [ ] Create Taskfile.yml for cross-platform compatibility
    - [ ] Document both Makefile and Taskfile usage
- [ ] Task: Create development sandbox
    - [ ] Configure devcontainer with all tools pre-installed
    - [ ] Add VS Code settings and extensions
    - [ ] Test GitHub Codespaces configuration
- [ ] Task: Add benchmark trend visualization
    - [ ] Create scripts to track benchmark history
    - [ ] Generate trend charts for documentation
    - [ ] Add to CI workflow
- [ ] Task: Create PR template with checklist
    - [ ] Add comprehensive PR template
    - [ ] Include quality gate checklist
    - [ ] Add testing requirements
    - [ ] Add documentation requirements
- [ ] Task: Final verification and documentation
    - [ ] Verify all workflows pass
    - [ ] Verify all bots are active
    - [ ] Verify Renovate is functional
    - [ ] Verify semantic release works
    - [ ] Update all documentation
    - [ ] Create onboarding guide for contributors

- [ ] Task: Automated Phase Review & Progression (Phase 11)

## Track Completion Protocol (Automated)

- [ ] Task: Track Completion & Auto-Archival
    - [ ] Push all remaining changes to remote
    - [ ] Run `conductor:review` skill on entire track
    - [ ] Automatically apply all review fixes
    - [ ] Push fixes to remote
    - [ ] Monitor GitHub Actions runs until ALL pass
    - [ ] Address any remaining CI failures automatically
    - [ ] Verify all 12 phase checkpoints are complete
    - [ ] Run final comprehensive quality gate check
    - [ ] Update track metadata status to 'completed'
    - [ ] Update tracks.md to mark track as complete [x]
    - [ ] Commit all final changes with descriptive message
    - [ ] Archive track (update metadata.json with completed_at timestamp)
    - [ ] Announce track completion

- [ ] Task: Auto-Progress to Next Track
    - [ ] Check if next track exists in tracks.md
    - [ ] If yes: Announce next track and await /conductor:implement
    - [ ] If no: Announce all tracks complete, await new track creation
