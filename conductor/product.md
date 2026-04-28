# Initial Concept

mars (formerly pymars) is a MARS implementation project moving from a pure
Python training implementation toward a shared Rust computational core surfaced
through Python, R, Julia, Rust, C#, Go, and TypeScript APIs. The project is
inspired by Jerome H. Friedman's MARS work and prior open-source APIs such as
`py-earth` and R `earth`, but it does not depend on those packages for
implementation or validation.

# Product Guide: mars (MARS Algorithm Library)

## Vision
Provide a well-documented, production-ready MARS implementation with a
scikit-learn-compatible Python API today and a shared Rust core that can power
first-class bindings for Python, R, Julia, Rust, C#, Go, and TypeScript.

## Target Users
- **Data Scientists & ML Practitioners:** Needing an interpretable, non-parametric regression/classification algorithm that integrates seamlessly with scikit-learn pipelines and other language ecosystems.
- **Researchers & Academics:** Requiring a transparent MARS implementation with reproducible fixtures and portable model artifacts.
- **Software Engineers:** Building production ML systems who value easy installation, clear APIs, comprehensive test coverage, and reusable runtime behavior across services and languages.

## Core Goals
1. **Algorithm Completeness:** Implement the full MARS algorithm including forward pass, GCV-based pruning, interaction terms, and refined knot placement controls (`minspan`, `endspan`).
2. **Scikit-learn Compatibility:** Full compliance with scikit-learn estimator interface (fit, predict, score, get_params, set_params) and compatibility with pipelines, model selection, and cross-validation utilities.
3. **Interpretability:** Provide feature importance calculations, partial dependence plots, Individual Conditional Expectation (ICE) plots, and model explanation tools.
4. **Generalized Linear Models:** Support logistic and Poisson regression via the `GLMEarth` subclass.
5. **CLI Support:** Command-line interface for model fitting, prediction, and evaluation.
6. **Portable Runtime Contract:** Export fitted models as a versioned `ModelSpec` that can be validated and replayed outside Python without pickle, Python object reconstruction, or dependency on external MARS packages.
7. **Rust Core Migration:** Move shared fitting and runtime evaluation logic into a Rust core while preserving the public Python API.
8. **Multi-Language Bindings:** Surface the Rust core through Python, R, Julia, Rust, C#, Go, and TypeScript packages with shared fixtures and conformance tests.

## Non-Functional Requirements
- **Current Python Package:** Preserve the current scikit-learn-compatible Python API and avoid C/Cython extensions in Python-specific code.
- **Rust Core:** Use Rust for the shared computational core and expose stable FFI/package boundaries for supported languages.
- **Performance:** Optimize correctness first, then move shared hot paths into Rust where this improves portability or runtime performance.
- **Test Coverage:** Maintain high test coverage using pytest, hypothesis (property-based testing), and mutation testing (mutmut).
- **Code Quality:** Enforce linting (ruff), formatting (ruff format), type checking (ty), and security checks (bandit, safety).
- **Documentation:** Comprehensive API docs, user guides, and academic paper.

## Success Metrics
- Passes scikit-learn's `check_estimator` suite.
- Achieves >90% code coverage.
- Published on PyPI with stable releases.
- Rust core validates against the shared fixture corpus.
- Python, R, Julia, Rust, C#, Go, and TypeScript bindings pass shared conformance tests.
- Active community contributions and issue resolution.
