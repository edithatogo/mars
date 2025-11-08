"""
Caching mechanisms for pymars to optimize repeated computations.

This module provides caching functionality to improve performance
when the same computations are repeated multiple times.
"""
import hashlib
import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Import Earth to avoid NameError
from .earth import Earth

logger = logging.getLogger(__name__)


class BasisFunctionCache:
    """
    Cache for basis function computations to avoid repeated evaluations.
    
    This cache stores the results of basis function transformations to improve
    performance when the same input data is used multiple times.
    """
    
    def __init__(self, maxsize: int = 128):
        """
        Initialize the basis function cache.
        
        Parameters
        ----------
        maxsize : int, optional (default=128)
            Maximum number of items to store in the cache.
        """
        self._cache: Dict[str, np.ndarray] = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def _hash_input(self, X: np.ndarray, missing_mask: np.ndarray) -> str:
        """
        Create a hash of the input data for cache key generation.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data array.
        missing_mask : numpy.ndarray
            Boolean mask indicating missing values.
            
        Returns
        -------
        str
            Hash string of the input data.
        """
        # Create a combined hash of X and missing_mask
        x_hash = hashlib.sha256(X.tobytes()).hexdigest()
        mask_hash = hashlib.sha256(missing_mask.tobytes()).hexdigest()
        return f"{x_hash}_{mask_hash}"
    
    def get(self, X: np.ndarray, missing_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Retrieve cached result for the given input.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data array.
        missing_mask : numpy.ndarray
            Boolean mask indicating missing values.
            
        Returns
        -------
        numpy.ndarray or None
            Cached result if found, None otherwise.
        """
        key = self._hash_input(X, missing_mask)
        if key in self._cache:
            self.hits += 1
            logger.debug(f"Cache hit for key {key[:8]}...")
            return self._cache[key]
        else:
            self.misses += 1
            logger.debug(f"Cache miss for key {key[:8]}...")
            return None
    
    def put(self, X: np.ndarray, missing_mask: np.ndarray, result: np.ndarray) -> None:
        """
        Store result in cache for the given input.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data array.
        missing_mask : numpy.ndarray
            Boolean mask indicating missing values.
        result : numpy.ndarray
            Computed result to cache.
        """
        # Check if cache is full
        if len(self._cache) >= self.maxsize:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        key = self._hash_input(X, missing_mask)
        self._cache[key] = result.copy()  # Store a copy to avoid modification
        logger.debug(f"Stored result in cache for key {key[:8]}...")
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("Cache cleared")
    
    def info(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Dictionary containing cache statistics.
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self._cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate
        }


# Global cache instance for basis function computations
_basis_function_cache = BasisFunctionCache(maxsize=128)


def enable_basis_function_caching(maxsize: int = 128) -> None:
    """
    Enable basis function caching with the specified maximum size.
    
    Parameters
    ----------
    maxsize : int, optional (default=128)
        Maximum number of items to store in the cache.
    """
    global _basis_function_cache
    _basis_function_cache = BasisFunctionCache(maxsize=maxsize)
    logger.info(f"Basis function caching enabled with maxsize={maxsize}")


def disable_basis_function_caching() -> None:
    """Disable basis function caching."""
    global _basis_function_cache
    _basis_function_cache.clear()
    _basis_function_cache = None
    logger.info("Basis function caching disabled")


def get_basis_function_cache_info() -> Dict[str, Any]:
    """
    Get information about the current basis function cache.
    
    Returns
    -------
    dict
        Dictionary containing cache statistics.
    """
    if _basis_function_cache is None:
        return {'enabled': False}
    return {'enabled': True, **_basis_function_cache.info()}


@lru_cache(maxsize=128)
def _cached_transform_wrapper(basis_func_hash: str, X_bytes: bytes, 
                              missing_mask_bytes: bytes) -> np.ndarray:
    """
    Wrapper for cached basis function transformations.
    
    This function is decorated with lru_cache to provide automatic caching
    of basis function transformations.
    
    Parameters
    ----------
    basis_func_hash : str
        Hash of the basis function for identification.
    X_bytes : bytes
        Serialized input data.
    missing_mask_bytes : bytes
        Serialized missing mask.
        
    Returns
    -------
    numpy.ndarray
        Transformed data.
    """
    # This is a placeholder - in practice, we would deserialize the inputs
    # and apply the basis function transformation
    pass


class CachedEarth(Earth):
    """
    Earth model with caching mechanisms for repeated computations.
    
    This class extends the standard Earth model with caching functionality
    to improve performance when the same computations are repeated.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize CachedEarth model.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments passed to Earth constructor.
        **kwargs : dict
            Keyword arguments passed to Earth constructor.
        """
        super().__init__(*args, **kwargs)
        self._cache_enabled = True
        self._transformation_cache = BasisFunctionCache(maxsize=128)
    
    def fit(self, X, y):
        """
        Fit the Earth model with caching.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : CachedEarth
            The fitted model.
        """
        # Clear cache before fitting to ensure fresh start
        self._transformation_cache.clear()
        
        # Call parent fit method
        super().fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Predict target values for X with caching.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted target values.
        """
        # Use cache if enabled and available
        if self._cache_enabled and self.basis_ is not None:
            # Build basis matrix with caching
            missing_mask = np.zeros_like(X, dtype=bool)
            B_matrix = self._build_basis_matrix_cached(X, self.basis_, missing_mask)
            return B_matrix @ self.coef_
        else:
            # Fall back to standard prediction
            return super().predict(X)
    
    def _build_basis_matrix_cached(self, X: np.ndarray, basis_functions: list, 
                                   missing_mask: np.ndarray) -> np.ndarray:
        """
        Build basis matrix with caching.
        
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input data.
        basis_functions : list
            List of basis functions.
        missing_mask : numpy.ndarray of shape (n_samples, n_features)
            Boolean mask indicating missing values.
            
        Returns
        -------
        numpy.ndarray of shape (n_samples, n_basis_functions)
            Basis matrix.
        """
        # Check cache first
        cached_result = self._transformation_cache.get(X, missing_mask)
        if cached_result is not None:
            return cached_result
        
        # Compute result
        B_matrix = self._build_basis_matrix(X, basis_functions, missing_mask)
        
        # Store in cache
        self._transformation_cache.put(X, missing_mask, B_matrix)
        
        return B_matrix
    
    def enable_caching(self) -> None:
        """Enable caching for this model."""
        self._cache_enabled = True
        logger.info("Caching enabled for Earth model")
    
    def disable_caching(self) -> None:
        """Disable caching for this model."""
        self._cache_enabled = False
        self._transformation_cache.clear()
        logger.info("Caching disabled for Earth model")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information for this model.
        
        Returns
        -------
        dict
            Dictionary containing cache statistics.
        """
        return self._transformation_cache.info()


def demo_caching():
    """Demonstrate caching functionality."""
    print("Demonstrating caching functionality...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 50)
    
    # Create cached Earth model
    model = CachedEarth(max_degree=2, penalty=3.0, max_terms=15)
    model.enable_caching()
    
    # Fit model
    model.fit(X, y)
    print(f"Model fitted with {len(model.basis_)} basis functions")
    
    # Make predictions (first time - cache miss)
    preds1 = model.predict(X[:10])
    print(f"First prediction: {preds1[0]:.4f}")
    
    # Make predictions again (second time - cache hit)
    preds2 = model.predict(X[:10])
    print(f"Second prediction: {preds2[0]:.4f}")
    
    # Check that predictions are identical
    assert np.allclose(preds1, preds2), "Cached predictions should be identical"
    print("✅ Cached predictions are identical")
    
    # Get cache info
    cache_info = model.get_cache_info()
    print(f"Cache info: {cache_info}")
    
    # Disable caching
    model.disable_caching()
    preds3 = model.predict(X[:10])
    print(f"Prediction without caching: {preds3[0]:.4f}")
    
    # Check that predictions are still identical
    assert np.allclose(preds1, preds3), "Predictions should be identical with/without caching"
    print("✅ Predictions are identical with/without caching")
    
    print("Caching demonstration completed successfully!")