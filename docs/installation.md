# Installation

## Installing mars

mars can be installed directly from PyPI:

```bash
pip install mars
```

## Installing from Source

To install from the source code:

```bash
git clone https://github.com/edithatogo/mars.git
cd mars
pip install -e .
```

## Dependencies

mars requires Python >= 3.8 and has the following dependencies:

- numpy
- scikit-learn 
- matplotlib

Optional dependencies for extended functionality:

- pandas (for DataFrame support)
- jax (for experimental JAX backend)