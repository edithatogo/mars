# Technology Stack: mars

## Core Language
- **Python 3.9+** - Current public package and scikit-learn compatibility layer
- **Rust 2021** - Shared computational core direction and current portable runtime prototype

## Dependencies
- **numpy** - Numerical computing and array operations
- **scikit-learn** - Estimator base classes, input validation, model selection utilities
- **matplotlib** - Plotting utilities for diagnostics and visualization

## Rust Runtime/Core
- **cargo** - Rust build and test orchestration
- **serde / serde_json** - `ModelSpec` artifact parsing
- **anyhow** - Test harness error context
- **criterion** - Rust core benchmarking and regression measurement without
  expanding the public API
- **tracing** - Structured internal spans and events for Rust-core diagnostics
- **cargo deny** - Dependency and license hygiene checks for the Rust core
- **cargo nextest** - Faster Rust test execution for core and integration suites

## Rust Binding Strategy
- **PyO3 / maturin** - Python extension mechanism for Rust-backed estimator and runtime integration
- **Rust CLI bridge** - Temporary runtime bridge for R, Julia, C#, and Go while native interop is prepared
- **WASM / WebAssembly** - TypeScript runtime surface backed by `wasm-bindgen` where a portable browser/Node package is preferable

## Optional Dependencies
- **pandas** - Required for CLI functionality and full scikit-learn estimator checks

## Testing
- **pytest** - Primary test framework
- **pytest-cov** - Code coverage reporting
- **hypothesis** - Property-based testing for edge cases
- **pytest-benchmark** - Performance benchmarking
- **coverage/profiling artifacts** - Keep benchmark and profiling outputs
  reproducible and inspectable when tracking regressions
- **flamegraphs / memory profiling** - Optional local tooling for targeted Rust
  performance investigation

## Code Quality
- **ruff** - Fast Python linter (replaces flake8, pyflakes, isort)
- **ruff format** - Code formatting
- **isort** - Import sorting (handled by ruff)
- **ty** - Static type checking

## Security & Mutation Testing
- **bandit** - Security linting
- **safety** - Dependency vulnerability scanning
- **mutmut** - Mutation testing

## CI/CD
- **GitHub Actions** - Continuous integration and deployment
- **tox** - Multi-environment test orchestration
- **pre-commit** - Git hook management

## Workspace Automation
- **notionctl** - Local Notion CLI for page, database, block, comment, user,
  team, search, schema, and raw API operations once authenticated
- **linear-cli** - Local Linear CLI for issues, projects, documents,
  initiatives, roadmaps, milestones, labels, comments, search, and raw GraphQL
  operations once authenticated

## Documentation
- **Starlight (Astro)** — Static site generation for documentation (ACTIVE; replaced mkdocs Material 2026-05-13)
- **@astrojs/sitemap** — Sitemap generation for SEO
- **starlight-polyglot** — Plugin generating API docs from Python source code (entryPoints: `pymars`)
- **starlight-links-validator** — Internal link validation in CI
- **starlight-versions** — Versioned docs support
- **starlight-llms-txt** — LLM-friendly documentation export
