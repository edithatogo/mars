# Technology Stack: mars

## Core Language
- **Python 3.9+** - Current public package and scikit-learn compatibility layer
- **Rust 2021** - Shared computational core direction and current portable runtime prototype

## Dependencies
- **numpy** - Numerical computing and array operations
- **scikit-learn** - Estimator base classes, input validation, model selection utilities
- **matplotlib** - Plotting utilities for diagnostics and visualization

## Rust Runtime/Core
- **Cargo** - Rust build and test orchestration
- **serde / serde_json** - `ModelSpec` artifact parsing
- **anyhow** - Test harness error context

## Rust Binding Strategy
- **PyO3 / maturin** - Python extension mechanism for Rust-backed estimator and runtime integration
- **extendr** - R binding bridge for Rust-backed package surfaces
- **C ABI + cbindgen / cgo / PInvoke / Julia `ccall`** - Shared foreign-function boundary for Go, C#, and Julia wrappers
- **wasm-bindgen / WebAssembly** - TypeScript runtime surface where a portable browser/Node package is preferable

## Optional Dependencies
- **pandas** - Required for CLI functionality and full scikit-learn estimator checks

## Testing
- **pytest** - Primary test framework
- **pytest-cov** - Code coverage reporting
- **hypothesis** - Property-based testing for edge cases
- **pytest-benchmark** - Performance benchmarking

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

## Documentation
- **MkDocs** - Static site generation for documentation
