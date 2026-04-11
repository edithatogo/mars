# Release Process for pymars v1.0.1

## Overview
This document outlines the steps needed to release version 1.0.1 of pymars to TestPyPI, PyPI, and GitHub.

## Changes in v1.0.1
- Fixed sklearn deprecation warnings by updating `force_all_finite` parameter to `ensure_all_finite` with backward compatibility
- Improved documentation with comprehensive tutorial, API reference, and usage examples
- Updated license from MIT to Apache 2.0
- Fixed GitHub Pages workflow to properly deploy documentation to docs folder on main branch

## Prerequisites
- Python 3.8+
- `build` package: `pip install build`
- `twine` package: `pip install twine`
- Valid PyPI and TestPyPI account credentials
- Valid GitHub CLI installed with appropriate permissions

## Steps

### 1. Prepare Environment
```bash
# Create a clean environment
python -m venv release_env
source release_env/bin/activate  # On Windows: release_env\Scripts\activate
pip install --upgrade pip build twine
```

### 2. Verify Changes
```bash
# Run tests to ensure everything is working
python -m pytest tests/ --tb=short

# Test package installation
pip install -e .

# Test imports
python -c "import pymars; print(f'Version: {pymars.__version__}')"
```

### 3. Build Distribution Packages
```bash
# First, clean any previous builds
rm -rf dist/ build/ *.egg-info/ || true

# Build both source distribution and wheel
python -m build

# Verify the distributions
twine check dist/*
```

### 4. Upload to TestPyPI (Optional but Recommended)
```bash
# Upload to TestPyPI for verification
twine upload --repository testpypi dist/*

# Test installation from TestPyPI (in a separate environment)
# pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pymars==1.0.1
```

### 5. Upload to PyPI
```bash
# Upload to PyPI
twine upload dist/*
```

### 6. Create GitHub Release
```bash
# Using GitHub CLI (requires gh to be installed and configured)
gh release create v1.0.1 dist/* --title "v1.0.1" --notes "Fixed sklearn deprecation warnings and improved documentation"

# Or create via web interface
# Go to: https://github.com/edithatogo/mars/releases/new
# Tag: v1.0.1
# Title: v1.0.1
# Description: Fixed sklearn deprecation warnings and improved documentation
```

### 7. Post-Release Verification
```bash
# Install from PyPI and verify
pip install pymars==1.0.1
python -c "import pymars; print(f'Version: {pymars.__version__}')"
```

### 4. Upload to TestPyPI (Optional but Recommended)
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pymars==1.0.1
```

### 5. Upload to PyPI
```bash
# Upload to PyPI
twine upload dist/*
```

### 6. Create GitHub Release
```bash
# Using GitHub CLI
gh release create v1.0.1 dist/* --title "v1.0.1" --notes "Fixed sklearn deprecation warnings and improved documentation"
```

### 7. Post-Release Verification
```bash
# Install from PyPI and verify
pip install pymars==1.0.1
python -c "import pymars; print(f'Version: {pymars.__version__}')"
```

## Additional Notes
You may see deprecation warnings during the build process related to the license format in pyproject.toml, but these do not affect the functionality of the package:
- SetuptoolsDeprecationWarning: `project.license` as a TOML table is deprecated

## Verification Steps
1. Check PyPI page to confirm version 1.0.1 is available
2. Verify documentation site is updated
3. Test installation in a fresh environment
4. Run a simple example to ensure functionality