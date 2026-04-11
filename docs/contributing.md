# Contributing

Thank you for your interest in contributing to pymars! We welcome contributions from the community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pymars.git
   cd pymars
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests if applicable
4. Run the tests to ensure your changes work:
   ```bash
   pytest
   ```
5. Make sure your code passes linting:
   ```bash
   ruff check .
   ```
6. Commit your changes with a descriptive commit message following the [conventional commits](https://www.conventionalcommits.org/) specification
7. Push your changes to your fork
8. Create a pull request to the main repository

## Code Style

We use `ruff` for linting and formatting. Please make sure your code follows the project's style:

```bash
ruff check .  # Check for issues
ruff format .  # Format code
```

## Testing

All contributions should include appropriate tests. We use pytest for testing:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=pymars

# Run specific test file
pytest tests/test_earth.py
```

## Documentation

When adding new features, please update the documentation accordingly. This includes both docstrings in the code and user documentation.

## Questions?

If you have questions, feel free to open an issue or contact the maintainers.