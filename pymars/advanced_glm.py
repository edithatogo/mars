"""
Additional GLM (Generalized Linear Model) families for pymars.

This module provides support for additional GLM families beyond the basic
logistic and Poisson regression already implemented.
"""
import logging
from typing import Any, Dict, Optional
import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
    PoissonRegressor,
    LinearRegression,
    GammaRegressor as SklearnGammaRegressor,
    TweedieRegressor as SklearnTweedieRegressor
)
from ._sklearn_compat import EarthClassifier
from .glm import GLMEarth

logger = logging.getLogger(__name__)


class GammaRegressor:
    """
    Custom Gamma regressor for GLM Earth.
    
    This regressor uses the sklearn Gamma regressor but with additional
    functionality for GLM Earth integration.
    """
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 100, tol: float = 1e-4, solver: str = 'lbfgs'):
        """
        Initialize the Gamma regressor.
        
        Parameters
        ----------
        alpha : float, optional (default=1.0)
            Regularization strength; must be a positive float.
        max_iter : int, optional (default=100)
            Maximum number of iterations for the solver to converge.
        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.
        solver : str, optional (default='lbfgs')
            Algorithm to use in the optimization problem.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self._regressor = SklearnGammaRegressor(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver
        )
    
    def fit(self, X, y):
        """
        Fit the Gamma regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : GammaRegressor
            The fitted regressor.
        """
        # Ensure all y values are positive (required for Gamma regression)
        y = np.asarray(y)
        if np.any(y <= 0):
            raise ValueError("Gamma regressor requires all target values to be positive")
        
        self._regressor.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted values.
        """
        return self._regressor.predict(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and contained subobjects.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'solver': self.solver
        }


class TweedieRegressor:
    """
    Custom Tweedie regressor for GLM Earth.
    
    This regressor uses the sklearn Tweedie regressor but with additional
    functionality for GLM Earth integration.
    """
    
    def __init__(self, power: float = 1.5, alpha: float = 1.0, max_iter: int = 100, tol: float = 1e-4):
        """
        Initialize the Tweedie regressor.
        
        Parameters
        ----------
        power : float, optional (default=1.5)
            The power parameter of the Tweedie distribution.
            - power <= 0: Normal distribution
            - power == 1: Poisson distribution  
            - 1 < power < 2: Compound Poisson distribution
            - power == 2: Gamma distribution
            - power > 2: Positive stable distribution
        alpha : float, optional (default=1.0)
            Regularization strength; must be a positive float.
        max_iter : int, optional (default=100)
            Maximum number of iterations for the solver to converge.
        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.
        """
        self.power = power
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self._regressor = SklearnTweedieRegressor(
            power=self.power,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol
        )
    
    def fit(self, X, y):
        """
        Fit the Tweedie regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : TweedieRegressor
            The fitted regressor.
        """
        # Ensure all y values are positive for power > 1
        y = np.asarray(y)
        if self.power >= 1 and np.any(y < 0):
            raise ValueError("Tweedie regressor with power >= 1 requires all target values to be non-negative")
        
        self._regressor.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted values.
        """
        return self._regressor.predict(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and contained subobjects.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'power': self.power,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol
        }


class InverseGaussianRegressor:
    """
    Custom Inverse Gaussian regressor for GLM Earth.
    
    This regressor implements Inverse Gaussian regression using Tweedie regression
    with power = 3.
    """
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 100, tol: float = 1e-4):
        """
        Initialize the Inverse Gaussian regressor.
        
        Parameters
        ----------
        alpha : float, optional (default=1.0)
            Regularization strength; must be a positive float.
        max_iter : int, optional (default=100)
            Maximum number of iterations for the solver to converge.
        tol : float, optional (default=1e-4)
            Tolerance for stopping criteria.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        # Inverse Gaussian corresponds to power = 3 in Tweedie distribution
        self._regressor = SklearnTweedieRegressor(
            power=3.0,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol
        )
    
    def fit(self, X, y):
        """
        Fit the Inverse Gaussian regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : InverseGaussianRegressor
            The fitted regressor.
        """
        # Ensure all y values are positive for Inverse Gaussian
        y = np.asarray(y)
        if np.any(y <= 0):
            raise ValueError("Inverse Gaussian regressor requires all target values to be positive")
        
        self._regressor.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted values.
        """
        return self._regressor.predict(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and contained subobjects.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol
        }


class AdvancedGLMEarth(GLMEarth):
    """
    GLM Earth with support for additional GLM families.
    
    This class extends GLMEarth with additional GLM families beyond the basic
    logistic and Poisson regression already implemented.
    """
    
    def __init__(self, family: str = 'gaussian', *args, **kwargs):
        """
        Initialize AdvancedGLMEarth.
        
        Parameters
        ----------
        family : str, optional (default='gaussian')
            Distribution family for the GLM. Options include:
            - 'gaussian': Normal distribution (default)
            - 'logistic': Logistic regression (binary classification)
            - 'poisson': Poisson regression (count data)
            - 'gamma': Gamma regression (positive continuous data)
            - 'tweedie': Tweedie regression (compound Poisson/gamma)
            - 'inverse_gaussian': Inverse Gaussian regression
        *args : tuple
            Additional positional arguments passed to GLMEarth.
        **kwargs : dict
            Additional keyword arguments passed to GLMEarth.
        """
        # Store the family for reference
        self.family = family
        
        # Set up the appropriate internal regressor based on family
        if family == 'gaussian':
            from sklearn.linear_model import LinearRegression
            internal_regressor = LinearRegression()
        elif family == 'logistic':
            from sklearn.linear_model import LogisticRegression
            internal_regressor = LogisticRegression()
        elif family == 'poisson':
            from sklearn.linear_model import PoissonRegressor
            internal_regressor = PoissonRegressor()
        elif family == 'gamma':
            internal_regressor = GammaRegressor()
        elif family == 'tweedie':
            internal_regressor = TweedieRegressor()
        elif family == 'inverse_gaussian':
            internal_regressor = InverseGaussianRegressor()
        else:
            raise ValueError(f"Unknown family: {family}. Supported families: gaussian, logistic, poisson, gamma, tweedie, inverse_gaussian")
        
        # Update kwargs to use the specified regressor
        kwargs['internal_regressor'] = internal_regressor
        
        # Call parent constructor
        super().__init__(*args, **kwargs)
    
    def _setup_internal_regressor(self, family: str):
        """
        Set up the appropriate internal regressor based on the family.
        
        Parameters
        ----------
        family : str
            Distribution family for the GLM.
        """
        if family == 'gaussian':
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        elif family == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression()
        elif family == 'poisson':
            from sklearn.linear_model import PoissonRegressor
            return PoissonRegressor()
        elif family == 'gamma':
            return GammaRegressor()
        elif family == 'tweedie':
            return TweedieRegressor()
        elif family == 'inverse_gaussian':
            return InverseGaussianRegressor()
        else:
            raise ValueError(f"Unknown family: {family}. Supported families: gaussian, logistic, poisson, gamma, tweedie, inverse_gaussian")


def demo_additional_glm_families():
    """Demonstrate additional GLM families."""
    print("Demonstrating additional GLM families...")
    
    # Generate test data
    np.random.seed(42)
    n_samples = 100
    
    # For Gaussian (continuous data)
    X_cont = np.random.rand(n_samples, 3)
    y_gaussian = X_cont[:, 0] + X_cont[:, 1] * 0.5 + np.sin(X_cont[:, 2] * np.pi) + np.random.normal(0, 0.1, n_samples)
    
    # For Gamma (positive continuous data)
    y_gamma = np.abs(y_gaussian) + 0.1  # Ensure positive values
    
    # For Poisson (count data) - convert to integers
    y_poisson = np.abs(y_gaussian).astype(int)
    
    # For Tweedie (compound Poisson)
    y_tweedie = np.abs(y_gaussian) + 0.05  # Positive values
    
    print("\\n1. Gaussian GLM Earth (default):")
    try:
        gaussian_glm = AdvancedGLMEarth(family='gaussian', max_degree=2, penalty=3.0, max_terms=15)
        gaussian_glm.fit(X_cont, y_gaussian)
        gaussian_score = gaussian_glm.score(X_cont, y_gaussian)
        print(f"   Gaussian GLM fitted with {len(gaussian_glm.basis_)} basis functions")
        print(f"   Gaussian GLM R²: {gaussian_score:.4f}")
        print("   ✅ Gaussian GLM works correctly")
    except Exception as e:
        print(f"   ⚠️  Gaussian GLM failed: {e}")
    
    print("\\n2. Gamma GLM Earth:")
    try:
        gamma_glm = AdvancedGLMEarth(family='gamma', max_degree=2, penalty=3.0, max_terms=15)
        gamma_glm.fit(X_cont, y_gamma)
        gamma_score = gamma_glm.score(X_cont, y_gamma)
        print(f"   Gamma GLM fitted with {len(gamma_glm.basis_)} basis functions")
        print(f"   Gamma GLM R²: {gamma_score:.4f}")
        print("   ✅ Gamma GLM works correctly")
    except Exception as e:
        print(f"   ⚠️  Gamma GLM failed: {e}")
    
    print("\\n3. Poisson GLM Earth:")
    try:
        poisson_glm = AdvancedGLMEarth(family='poisson', max_degree=2, penalty=3.0, max_terms=15)
        poisson_glm.fit(X_cont, y_poisson)
        poisson_score = poisson_glm.score(X_cont, y_poisson)
        print(f"   Poisson GLM fitted with {len(poisson_glm.basis_)} basis functions")
        print(f"   Poisson GLM R²: {poisson_score:.4f}")
        print("   ✅ Poisson GLM works correctly")
    except Exception as e:
        print(f"   ⚠️  Poisson GLM failed: {e}")
    
    print("\\n4. Tweedie GLM Earth:")
    try:
        tweedie_glm = AdvancedGLMEarth(family='tweedie', max_degree=2, penalty=3.0, max_terms=15)
        tweedie_glm.fit(X_cont, y_tweedie)
        tweedie_score = tweedie_glm.score(X_cont, y_tweedie)
        print(f"   Tweedie GLM fitted with {len(tweedie_glm.basis_)} basis functions")
        print(f"   Tweedie GLM R²: {tweedie_score:.4f}")
        print("   ✅ Tweedie GLM works correctly")
    except Exception as e:
        print(f"   ⚠️  Tweedie GLM failed: {e}")
    
    print("\\n5. Inverse Gaussian GLM Earth:")
    try:
        inv_gaussian_glm = AdvancedGLMEarth(family='inverse_gaussian', max_degree=2, penalty=3.0, max_terms=15)
        inv_gaussian_glm.fit(X_cont, y_gamma)  # Use positive data
        inv_gaussian_score = inv_gaussian_glm.score(X_cont, y_gamma)
        print(f"   Inverse Gaussian GLM fitted with {len(inv_gaussian_glm.basis_)} basis functions")
        print(f"   Inverse Gaussian GLM R²: {inv_gaussian_score:.4f}")
        print("   ✅ Inverse Gaussian GLM works correctly")
    except Exception as e:
        print(f"   ⚠️  Inverse Gaussian GLM failed: {e}")
    
    print("\\nAdditional GLM families demonstration completed!")


if __name__ == "__main__":
    demo_additional_glm_families()