# Development Guidelines for pymars

## Overview
This document outlines the development process, coding standards, and best practices for contributing to pymars.

## Prerequisites
- Python 3.8 or higher
- Git
- A code editor with Python support

## Getting Started

### Setting up the development environment

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/pymars.git
cd pymars
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e ".[dev]"
```

4. Install and set up pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### Branch Strategy
- Use feature branches for new features
- Name branches descriptively: `feature/meaningful-name`, `bugfix/issue-description`
- Keep branches up-to-date with main through rebase

### Commit Messages
We follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding or updating tests
- `refactor:` for code restructuring
- `perf:` for performance improvements
- `chore:` for maintenance tasks

### Code Standards

1. **Formatting**: Use Ruff for formatting (configured in pyproject.toml)
```bash
ruff format pymars tests
```

2. **Linting**: Use Ruff for linting
```bash
ruff check pymars tests
```

3. **Type checking**: Use MyPy
```bash
mypy pymars/
```

4. **Testing**: All code must have appropriate tests
```bash
python -m pytest tests/
```

## Testing

### Test Structure
- Place tests in the `tests/` directory
- Follow the same directory structure as the source code
- Name test files with `test_` prefix
- Use descriptive test function names

### Test Coverage
- Aim for >90% test coverage for each module
- Run coverage with: `python -m pytest tests/ --cov=pymars`
- Check coverage by file: `python scripts/analyze_coverage.py`

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=pymars

# Run specific test file
python -m pytest tests/test_earth.py

# Run with verbose output
python -m pytest tests/ -v
```

## Code Quality

### Type Hints
- Use type hints for all public functions
- Use `typing` module for complex types
- Use type comments only when necessary

### Documentation
- All public functions and classes need docstrings
- Follow Google or NumPy docstring style
- Include examples where helpful

### Performance
- Profile code using `python scripts/profile_pymars.py`
- Optimize critical path code
- Avoid unnecessary computations

## CI/CD Pipeline

Our CI/CD pipeline includes:
- Automated testing on multiple Python versions
- Code formatting and linting
- Type checking
- Security scanning
- Test coverage checking
- Performance benchmarks

## Pull Request Process

1. Ensure all tests pass locally
2. Update documentation as needed
3. Add tests for new functionality
4. Follow the PR template
5. Request review from maintainers

## Release Process

Releases are automated through GitHub Actions:
- Create a git tag with semantic versioning (e.g., `v1.0.0`)
- Push the tag to trigger the release workflow
- Package is automatically published to PyPI

## Security

- Report security issues through GitHub's security advisory feature
- Use `bandit` for security scanning: `bandit -r pymars`
- Use `safety` for dependency scanning: `safety check`

## Performance Monitoring

- Run benchmarks with: `python -m pytest tests/ -k "benchmark"`
- Monitor performance regressions through CI
- Profile code using `scripts/profile_pymars.py`

## Documentation

- API documentation is generated from docstrings
- User guides go in the `docs/` directory
- Build docs locally: `sphinx-build -b html docs docs/_build`

## Troubleshooting

### Common Issues
- **Import errors**: Ensure the package is installed in development mode
- **Test failures**: Check that all dependencies are installed
- **Formatting issues**: Run `ruff format` to auto-format code

### Useful Commands
```bash
# Run the full CI pipeline locally
bash scripts/cicd_pipeline.sh

# Format and fix all code issues
tox -e format

# Run type checking
tox -e type-check

# Run all tests across Python versions
tox
```

## Contact

For questions about development, create an issue in the repository.