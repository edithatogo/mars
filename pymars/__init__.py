# Pure Python Earth (Multivariate Adaptive Regression Splines)
# Inspired by py-earth: https://github.com/scikit-learn-contrib/py-earth

__version__ = "0.0.1"

# Alias for py-earth compatibility (user can do this in their own code if they prefer)
# import pymars as earth

# Core Earth model
from .earth import Earth

# Scikit-learn compatible estimators
from ._sklearn_compat import EarthRegressor, EarthClassifier
from .glm import GLMEarth
from .cv import EarthCV
from .plot import plot_basis_functions, plot_residuals
from ._categorical import CategoricalImputer

# TODO: Add other classes/functions to expose at the top level if desired.
# e.g., from ._basis import BasisFunction (if users need to interact with it directly)

__all__ = [
    'Earth',
    'EarthRegressor',
    'EarthClassifier',
    'CategoricalImputer'
    ,'GLMEarth'
    ,'EarthCV'
    ,'plot_basis_functions'
    ,'plot_residuals'
]
