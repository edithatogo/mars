# Pure Python Earth (Multivariate Adaptive Regression Splines)
# Inspired by py-earth: https://github.com/scikit-learn-contrib/py-earth

__version__ = "1.0.0"

# Enable fault handler for debugging during development
import faulthandler
import sys
import os

# Only enable faulthandler in development/debugging environments
# Check for environment variables that indicate a debugging scenario
if os.getenv('PYMARS_DEBUG', '').lower() in ('1', 'true', 'yes') or \
   os.getenv('PYTHONFAULTHANDLER', '').lower() in ('1', 'true', 'yes') or \
   '--debug' in sys.argv or '-d' in sys.argv:
    faulthandler.enable()
    # Register dump to be triggered by SIGUSR1 (where supported)
    try:
        import signal
        faulthandler.register(signal.SIGUSR1, chain=True)
    except (AttributeError, ValueError):
        # SIGUSR1 not available on all platforms (e.g., Windows)
        pass
else:
    # In production, only enable for SIGSEGV, SIGFPE, SIGABRT, and SIGBUS signals
    # This only dumps on actual crashes, not on demand
    try:
        faulthandler.enable()
    except Exception:
        # Don't crash the module import if fault handler can't be enabled
        pass

# Alias for py-earth compatibility (user can do this in their own code if they prefer)
# import pymars as earth

# Core Earth model
from ._categorical import CategoricalImputer

# Scikit-learn compatible estimators
from ._sklearn_compat import EarthClassifier, EarthRegressor
from .cv import EarthCV
from .earth import Earth
from .cache import (
    CachedEarth,
    enable_basis_function_caching,
    disable_basis_function_caching,
    get_basis_function_cache_info
)
from .parallel import (
    ParallelEarth,
    ParallelBasisEvaluator,
    ParallelBasisMatrixBuilder
)
from .sparse import (
    SparseEarth,
    SparseBasisFunction,
    convert_to_sparse
)
from .advanced_cv import (
    AdvancedEarthCV,
    BootstrapEarthCV
)
from .advanced_glm import (
    AdvancedGLMEarth,
    GammaRegressor,
    TweedieRegressor,
    InverseGaussianRegressor
)
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
    'CachedEarth',
    'ParallelEarth',
    'SparseEarth',
    'enable_basis_function_caching',
    'disable_basis_function_caching',
    'get_basis_function_cache_info',
    'plot_basis_functions',
    'plot_residuals',
    'plot_partial_dependence',
    'plot_individual_conditional_expectation',
    'get_model_explanation'
]
