# Pure Python Earth (Multivariate Adaptive Regression Splines)
# Inspired by py-earth: https://github.com/scikit-learn-contrib/py-earth

__version__ = "1.0.0"

# Alias for py-earth compatibility (user can do this in their own code if they prefer)
# import pymars as earth

# Core Earth model
from ._categorical import CategoricalImputer

# Scikit-learn compatible estimators
from ._sklearn_compat import EarthClassifier, EarthRegressor
from .cv import EarthCV
from .earth import Earth
from .explain import (
    get_model_explanation,
    plot_individual_conditional_expectation,
    plot_partial_dependence,
)
from .glm import GLMEarth
from .plot import plot_basis_functions, plot_residuals

# TODO: Add other classes/functions to expose at the top level if desired.
# e.g., from ._basis import BasisFunction (if users need to interact with it directly)

__all__ = [
    'Earth',
    'EarthRegressor',
    'EarthClassifier',
    'CategoricalImputer',
    'GLMEarth',
    'EarthCV',
    'plot_basis_functions',
    'plot_residuals',
    'plot_partial_dependence',
    'plot_individual_conditional_expectation',
    'get_model_explanation'
]
