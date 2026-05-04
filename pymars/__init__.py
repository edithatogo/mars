from __future__ import annotations

__version__ = "1.0.4"

from ._categorical import CategoricalImputer
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
from .runtime import (
    design_matrix,
    inspect,
    load_model,
    load_model_spec,
    predict,
    save_model,
    validate,
)

__all__ = [
    "CategoricalImputer",
    "Earth",
    "EarthCV",
    "EarthClassifier",
    "EarthRegressor",
    "GLMEarth",
    "design_matrix",
    "get_model_explanation",
    "inspect",
    "load_model",
    "load_model_spec",
    "plot_basis_functions",
    "plot_individual_conditional_expectation",
    "plot_partial_dependence",
    "plot_residuals",
    "predict",
    "save_model",
    "validate",
]
