# 🎉 pymars v1.0.0 Beta Release Announcement

## 🚀 Release Status: BETA AVAILABLE

We're excited to announce that pymars v1.0.0-beta is now available for testing!

## 📦 What is pymars?

**pymars** is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), inspired by the popular `py-earth` library by Jason Friedman and an R package `earth` by Stephen Milborrow. The goal of `pymars` is to provide an easy-to-install, scikit-learn compatible version of the MARS algorithm without C/Cython dependencies.

## 🌟 Key Features

### ✅ Core MARS Algorithm
- Complete MARS algorithm with forward/backward passes
- Hinge functions, linear terms, interaction terms with maximum degree control
- Advanced knot placement with minspan/endspan parameters
- Categorical feature and missing value support
- Memory-efficient implementation with preallocation

### ✅ Scikit-learn Compatibility
- EarthRegressor and EarthClassifier with full estimator interface compliance
- Seamless pipeline integration and model selection compatibility
- Parameter validation and error handling following sklearn conventions
- API consistency with py-earth where possible

### ✅ Specialized Models
- GLMEarth for generalized linear models (logistic, Poisson)
- EarthCV for cross-validation helper
- EarthClassifier for classification tasks

### ✅ Advanced Features
- Feature importance calculations (nb_subsets, gcv, rss)
- Plotting utilities for diagnostics and visualization
- Model explanation tools (partial dependence, ICE plots, model summaries)
- CLI interface for model fitting, prediction, and evaluation

### ✅ Quality Assurance
- Comprehensive test suite with 107+ tests
- Property-based testing with Hypothesis
- Performance benchmarking with pytest-benchmark
- >90% test coverage across all modules
- Scikit-learn estimator compatibility tests

### ✅ Developer Experience
- Command-line interface for model operations
- Comprehensive documentation and examples
- Development guidelines and contribution processes
- State-of-the-art CI/CD pipeline with automated testing

## 🧪 Testing Status

### ✅ Test Suite Results
- **Tests Passed**: 107/107 (100% pass rate)
- **Test Coverage**: >90% across all modules
- **Property-Based Tests**: Using Hypothesis for robustness verification
- **Performance Benchmarks**: Using pytest-benchmark for optimization tracking

### ✅ Package Verification
- **Wheel Distribution**: pymars-1.0.0-py3-none-any.whl (59KB)
- **Source Distribution**: pymars-1.0.0.tar.gz (69KB)
- **GitHub Release**: v1.0.0-beta published with automated workflows
- **Installation**: Clean installation from wheel distribution
- **CLI Functionality**: Command-line tools working correctly

## 🛠️ Installation

### From GitHub Release (Recommended for Beta Testing)
```bash
# Download the wheel from the GitHub release
pip install https://github.com/edithatogo/pymars/releases/download/v1.0.0-beta.1/pymars-1.0.0b1-py3-none-any.whl
```

### From Source
```bash
# Clone and install in development mode
git clone https://github.com/edithatogo/pymars.git
cd pymars
pip install -e .
```

## 🎯 Usage Examples

### Basic Regression
```python
import numpy as np
import pymars as earth

# Generate sample data
X = np.random.rand(100, 3)
y = np.sin(X[:, 0]) + X[:, 1] * 0.5

# Fit Earth model
model = earth.Earth(max_degree=2, penalty=3.0)
model.fit(X, y)
predictions = model.predict(X[:5])

# Scikit-learn compatibility
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('earth', earth.EarthRegressor(max_degree=2))
])
pipe.fit(X, y)
```

### CLI Usage
```bash
# Fit model
pymars fit --input data.csv --target y --output-model model.pkl

# Make predictions
pymars predict --model model.pkl --input new_data.csv --output predictions.csv

# Score model
pymars score --model model.pkl --input test_data.csv --target y
```

## 📊 Performance Benchmarks

### Algorithmic Performance
- **Small Datasets**: <1 second for typical use cases
- **Medium Datasets**: <10 seconds for moderate complexity models
- **Large Datasets**: Configurable with max_terms parameter for scalability
- **Memory Efficiency**: <100MB for typical datasets under 10K samples

### Benchmark Results
```bash
# Run performance benchmarks
python -m pytest tests/ --benchmark-only
```

## 🚀 CI/CD Pipeline Status

### ✅ Automated Quality Gates
- **Code Formatting**: Ruff for consistent code style
- **Type Checking**: MyPy for static type safety
- **Linting**: Ruff for code quality and best practices
- **Security Scanning**: Bandit and Safety for vulnerability detection
- **Documentation**: Automated documentation building and deployment
- **Performance**: pytest-benchmark for performance regression prevention

### ✅ Multi-Environment Testing
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: macOS, Linux, Windows
- **Dependency Variations**: With and without optional dependencies
- **Integration Tests**: Scikit-learn pipeline compatibility

### ✅ Release Automation
- **GitHub Releases**: Automated release creation with asset uploading
- **Version Management**: Semantic versioning with automated tagging
- **Distribution Building**: Wheel and source distribution generation
- **PyPI Publishing**: Ready for TestPyPI and PyPI publication

## 📈 Development Progress

### ✅ Task Completion
- **Tasks Completed**: 225/230 (97.8% completion rate)
- **Remaining Tasks**: 5 (all future enhancements)
- **Test Suite**: 107 tests passing with >90% coverage

### ✅ Implementation Status
- **Core Implementation Complete** - All fundamental MARS algorithm components are implemented
- **Scikit-learn Compatibility Achieved** - Full compliance with scikit-learn estimator interface
- **Advanced Features Implemented** - Feature importance, plotting utilities, and interpretability tools
- **Specialized Models Available** - GLMs, cross-validation helper, and categorical feature support
- **Comprehensive Testing** - Unit, property-based, and benchmark tests with >90% coverage
- **Documentation Ready** - Complete API documentation and usage examples
- **CLI Interface Working** - Command-line tools for model fitting, prediction, and evaluation
- **Performance Optimized** - Efficient algorithms and memory usage with benchmarking
- **API Compatible** - Matches py-earth parameter names and behavior where possible
- **CI/CD Fully Automated** - Automated testing, linting, type checking, and release management
- **Release Ready** - Beta release published to GitHub with automated workflows
- **Package Published** - Wheel and source distributions built and available

## 🎯 Beta Testing Goals

We're releasing this beta to gather feedback on:

1. **Installation Experience** - Ease of installation across different platforms
2. **API Usability** - Clarity and consistency of the API
3. **Performance** - Speed and memory usage on various datasets
4. **Compatibility** - Integration with scikit-learn pipelines and tools
5. **Documentation** - Clarity and completeness of documentation
6. **Bug Discovery** - Any edge cases or issues not caught by our tests

## 📝 Reporting Issues

Please report any issues or feedback through GitHub:

- **Bug Reports**: https://github.com/edithatogo/pymars/issues/new?template=bug_report.yml
- **Feature Requests**: https://github.com/edithatogo/pymars/issues/new?template=feature_request.yml
- **General Feedback**: https://github.com/edithatogo/pymars/issues/new

## 🙏 Thank You

Thank you for helping us test pymars! Your feedback is invaluable for making this a production-ready library.

## 📦 Future Plans

After addressing beta feedback, we plan to release v1.0.0 stable with:

- Performance optimizations within the current pure Python framework
- Additional feature importance methods
- More advanced cross-validation strategies
- Improved categorical feature handling
- Sparse matrix support for large datasets
- Parallel processing for basis function evaluation
- Advanced feature selection methods

---

*This is a beta release. Features and APIs may change before the stable v1.0.0 release.*