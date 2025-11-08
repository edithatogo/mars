# Installation

## Installing pymars

pymars can be installed directly from PyPI:

```bash
pip install pymars
```

## Installing from Source

To install from the source code:

```bash
git clone https://github.com/edithatogo/pymars.git
cd pymars
pip install -e .
```

## Dependencies

pymars requires Python >= 3.8 and has the following dependencies:

- numpy
- scikit-learn 
- matplotlib

Optional dependencies for extended functionality:

- pandas (for DataFrame support)
- jax (for experimental JAX backend)