# Technology Stack: mars

## Core Language
- **Python 3.8+** - Primary implementation language

## Dependencies
- **numpy** - Numerical computing and array operations
- **scikit-learn** - Estimator base classes, input validation, model selection utilities
- **matplotlib** - Plotting utilities for diagnostics and visualization

## Optional Dependencies
- **pandas** - Required for CLI functionality and full scikit-learn estimator checks

## Testing
- **pytest** - Primary test framework
- **pytest-cov** - Code coverage reporting
- **hypothesis** - Property-based testing for edge cases
- **pytest-benchmark** - Performance benchmarking

## Code Quality
- **ruff** - Fast Python linter (replaces flake8, pyflakes, isort)
- **black** - Code formatting
- **isort** - Import sorting (handled by ruff)
- **mypy** - Static type checking

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
