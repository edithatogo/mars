# Initial Concept

mars (formerly pymars) is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), inspired by the popular `py-earth` library by Jason Friedman and an R package `earth` by Stephen Milborrow. The goal of **mars** is to provide an easy-to-install, scikit-learn compatible version of the MARS algorithm without C/Cython dependencies.

# Product Guide: mars (MARS Algorithm Library)

## Vision
Provide a pure Python, scikit-learn-compatible implementation of the Multivariate Adaptive Regression Splines (MARS) algorithm that is easy to install, well-documented, and production-ready for regression, classification, and generalized linear modeling tasks.

## Target Users
- **Data Scientists & ML Practitioners:** Needing an interpretable, non-parametric regression/classification algorithm that integrates seamlessly with scikit-learn pipelines.
- **Researchers & Academics:** Requiring a transparent, pure-Python implementation for experimentation and teaching without C/Cython compilation barriers.
- **Software Engineers:** Building production ML systems who value easy installation, clear APIs, and comprehensive test coverage.

## Core Goals
1. **Algorithm Completeness:** Implement the full MARS algorithm including forward pass, GCV-based pruning, interaction terms, and refined knot placement controls (`minspan`, `endspan`).
2. **Scikit-learn Compatibility:** Full compliance with scikit-learn estimator interface (fit, predict, score, get_params, set_params) and compatibility with pipelines, model selection, and cross-validation utilities.
3. **Interpretability:** Provide feature importance calculations, partial dependence plots, Individual Conditional Expectation (ICE) plots, and model explanation tools.
4. **Generalized Linear Models:** Support logistic and Poisson regression via the `GLMEarth` subclass.
5. **CLI Support:** Command-line interface for model fitting, prediction, and evaluation.

## Non-Functional Requirements
- **Pure Python:** No C, Cython, or compiled extensions. Cross-platform compatibility is paramount.
- **Performance:** Optimize Python code for correctness and clarity; accept that pure Python may be slower than compiled alternatives.
- **Test Coverage:** Maintain high test coverage using pytest, hypothesis (property-based testing), and mutation testing (mutmut).
- **Code Quality:** Enforce linting (ruff), formatting (ruff format), type checking (ty), and security checks (bandit, safety).
- **Documentation:** Comprehensive API docs, user guides, and academic paper.

## Success Metrics
- Passes scikit-learn's `check_estimator` suite.
- Achieves >90% code coverage.
- Published on PyPI with stable releases.
- Active community contributions and issue resolution.
