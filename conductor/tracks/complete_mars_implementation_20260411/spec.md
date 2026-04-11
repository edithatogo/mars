# Track Specification: SOTA MARS Library Infrastructure & Implementation

## Overview
Transform the mars library into a state-of-the-art Python package with maximum automation, comprehensive testing, modern tooling, and minimal maintenance overhead. This track covers both infrastructure modernization and core algorithm verification.

## Objectives
1. Modernize package management with `uv` and automated dependency updates via Renovate
2. Implement full testing pyramid (unit, integration, e2e) with mutation and property-based testing
3. Consolidate and optimize CI/CD with cross-platform testing, caching, and nightly builds
4. Automate releases with semantic versioning, changelog generation, and PyPI publishing
5. Enhance code quality with comprehensive ruff rules, strict type checking, and complexity tracking
6. Add performance profiling with Scalene and automated regression detection
7. Improve documentation with auto-generated APIs, PR previews, and versioned docs
8. Automate community management with bots for stale issues, welcoming, and labeling
9. Ensure security compliance with SBOM generation, license checking, and signed releases
10. Verify core MARS algorithm functionality and scikit-learn compatibility

## Scope
### In Scope
- All existing source code in `pymars/`
- Test suite restructuring (`tests/unit/`, `tests/integration/`, `tests/e2e/`)
- CI/CD workflows (consolidation and enhancement)
- GitHub automation (bots, templates, renovate)
- Documentation infrastructure (MkDocs enhancements)
- Package management migration (pip/tox → uv)
- Release automation (python-semantic-release)
- Performance profiling infrastructure
- Security and compliance tooling
- Developer experience improvements (Makefile, devcontainer)

### Out of Scope
- Algorithm feature additions beyond py-earth parity
- C/Cython performance optimizations
- Major API changes
- External service integrations beyond GitHub

## Technical Requirements
- **Package Manager:** uv with lockfile
- **Dependency Updates:** Renovate with auto-merge for safe updates
- **Testing:** pytest with xdist, hypothesis (property-based), mutmut (mutation), hypothesis-stateful (stateful testing)
- **Linting:** ruff for everything possible, mypy for type checking
- **CI/CD:** GitHub Actions with caching, matrix testing, nightly builds
- **Releases:** python-semantic-release with PyPI auto-publish
- **Profiling:** Scalene for performance analysis, py-spy for production profiling
- **Documentation:** MkDocs with mkdocstrings, mike for versioning, mkdocs-material for theme
- **Security:** SBOM generation, license checking, signed releases, OSV-Scanner for vulnerability scanning
- **Automation:** stale, welcome, auto-label bots, release drafter
- **Monitoring:** CI trend tracking, benchmark trend visualization, CodeCov trends
- **Containerization:** Docker multi-stage builds for reproducible builds, devcontainer for instant dev setup

## Acceptance Criteria
- [ ] All CI workflows pass on Ubuntu, macOS, Windows for Python 3.8-3.12
- [ ] Code coverage >80% with diff-cover for PRs
- [ ] Mutation score >90%
- [ ] Zero ruff violations (comprehensive rule set)
- [ ] mypy strict mode passes
- [ ] Nightly builds test against dev dependency versions
- [ ] Automated releases work end-to-end (tag → PyPI)
- [ ] Renovate auto-merges safe updates
- [ ] Stale/welcome bots active and functional
- [ ] Documentation deploys with PR previews
- [ ] Scalene profiling workflow operational
- [ ] SBOM generated on every release (CycloneDX format)
- [ ] Release artifacts signed with cosign
- [ ] OSV-Scanner vulnerability scans pass
- [ ] Docker builds work for production and development
- [ ] CITATION.cff present and valid
- [ ] GitHub Discussions enabled with templates
- [ ] Makefile/Taskfile provide single-entry dev interface
- [ ] Devcontainer configured for instant environment setup
- [ ] All existing MARS algorithm tests pass
- [ ] sklearn check_estimator passes for all estimators
- [ ] CLI commands work end-to-end
- [ ] Demo scripts execute without errors
- [ ] Benchmark trend visualization available in docs
- [ ] All phase reviews completed via `conductor:review` skill
- [ ] All CI failures automatically resolved
- [ ] Track archived with completed_at timestamp
- [ ] All changes pushed to remote with passing GitHub Actions

## Dependencies
- **Package Management:** uv
- **Testing:** pytest, pytest-cov, pytest-xdist, pytest-benchmark, hypothesis, hypothesis-stateful, mutmut, nbmake, diff-cover
- **Linting:** ruff (comprehensive), mypy
- **CI/CD:** GitHub Actions, setup-uv
- **Release:** python-semantic-release, cosign
- **Profiling:** scalene, py-spy
- **Documentation:** mkdocs, mkdocs-material, mkdocstrings, mike, lychee
- **Automation:** renovate, stale action, welcome action, release-drafter
- **Security:** bandit, safety, license-check, osv-scanner, sbom-tool (CycloneDX)
- **Containerization:** Docker, devcontainer
- **Task Runners:** make, task (Taskfile)

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| uv migration breaks existing workflows | High | Keep tox temporarily, parallel run |
| Renovate auto-merge introduces breaking changes | Medium | Only auto-merge patches, require CI pass |
| Semantic release misconfiguration | High | Test on test-pypi first |
| Cross-platform test failures | Medium | Start with ubuntu, add others incrementally |
| Performance regression detection false positives | Low | Tune thresholds, use statistical analysis |
| Bot spam/over-automation | Low | Conservative initial settings, monitor |
| Docker builds add maintenance burden | Low | Optional, well-documented, tested in CI |
| Too many tools increase complexity | Medium | Phase rollout, comprehensive docs |
| OSV-Scanner false positives | Low | Review manually, configure exceptions |

## Success Metrics
- 100% CI pass rate across all platforms/versions
- >80% code coverage, >90% mutation score
- Zero security vulnerabilities
- <5 minute CI execution time (with caching and xdist)
- Automated releases with zero manual intervention
- <1 hour/week maintainer time for dependency updates
- Clear contribution guidelines with automated triage
- Comprehensive documentation with examples
- Benchmark trends show stable or improving performance
- All releases signed and include SBOM

## References
- uv: https://docs.astral.sh/uv/
- Renovate: https://docs.renovatebot.com/
- python-semantic-release: https://python-semantic-release.readthedocs.io/
- Scalene: https://github.com/plasma-umass/scalene
- ruff: https://docs.astral.sh/ruff/
- hypothesis: https://hypothesis.readthedocs.io/
- mutmut: https://mutmut.readthedocs.io/
- OSV-Scanner: https://osv-scanner.dev/
- CycloneDX SBOM: https://cyclonedx.org/
- cosign: https://docs.sigstore.dev/cosign/overview/
- mkdocs-material: https://squidfunk.github.io/mkdocs-material/
- mike: https://github.com/jimporter/mike
- pytest-xdist: https://pytest-xdist.readthedocs.io/
- diff-cover: https://diff-cover.readthedocs.io/
- nbmake: https://github.com/treebeardtech/nbmake
- py-spy: https://github.com/benfred/py-spy
- Taskfile: https://taskfile.dev/
- devcontainer: https://containers.dev/
