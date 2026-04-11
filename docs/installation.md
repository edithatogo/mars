# Installation

You can install pymars using pip:

```bash
pip install pymars
```

## Prerequisites

- Python 3.8 or higher
- numpy
- scikit-learn
- matplotlib

## Optional Dependencies

For full functionality, you may want to install:

```bash
pip install "pymars[pandas]"
```

This will install pandas which is needed for full scikit-learn estimator checks.

## Development Installation

To install pymars for development:

```bash
git clone https://github.com/pymars/pymars.git
cd pymars
pip install -e ".[dev]"
```

This will install pymars in editable mode along with all development dependencies.