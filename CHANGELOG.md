# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-beta.1] - 2025-02-02
### Added
- Initial release of pure Python MARS implementation
- Full scikit-learn compatibility with EarthRegressor and EarthClassifier
- Generalized Linear Models support with GLMEarth
- Cross-validation helper with EarthCV
- Feature importance calculations (nb_subsets, gcv, rss)
- Missing value and categorical feature support
- Plotting utilities for diagnostics
- Comprehensive test suite with >90% coverage
- State-of-the-art CI/CD pipeline with automated testing, linting, type checking, and security scanning
- Property-based testing with Hypothesis
- Performance benchmarking with pytest-benchmark
- Advanced interpretability tools (partial dependence, ICE plots, model explanations)
- Enhanced command-line interface for model fitting, prediction, and scoring
- Comprehensive documentation and development guidelines
- Automated release workflow to PyPI
- Code coverage integration with Codecov

[Unreleased]: https://github.com/pymars/pymars/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/pymars/pymars/releases/tag/v1.0.0