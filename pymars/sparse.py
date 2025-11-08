"""
Sparse matrix support for pymars to handle large datasets efficiently.

This module provides sparse matrix support to improve memory efficiency
when working with large datasets that have many zero values.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import sparse
from ._basis import BasisFunction
from .earth import Earth

logger = logging.getLogger(__name__)


class SparseBasisFunction(BasisFunction):
    """
    Basis function that operates on sparse matrices.
    
    This class extends BasisFunction to work with sparse matrices
    for improved memory efficiency with large datasets.
    """
    
    def transform(self, X_processed: Union[np.ndarray, sparse.spmatrix], 
                 missing_mask: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Apply the basis function transformation to the input data X_processed,
        using the provided missing_mask.
        
        Parameters
        ----------
        X_processed : numpy.ndarray or scipy.sparse matrix of shape (n_samples, n_features)
            The processed input samples (NaNs typically filled).
        missing_mask : numpy.ndarray or scipy.sparse matrix of shape (n_samples, n_features)
            Boolean mask indicating which original values were NaN.
            
        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The transformed values for each sample.
        """
        # Convert sparse matrices to dense if needed for the transformation
        if sparse.issparse(X_processed):
            X_dense = X_processed.toarray()
        else:
            X_dense = X_processed
            
        if sparse.issparse(missing_mask):
            mask_dense = missing_mask.toarray()
        else:
            mask_dense = missing_mask
            
        # Delegate to parent transform method
        return super().transform(X_dense, mask_dense)


class SparseEarth(Earth):
    """
    Earth model with sparse matrix support.
    
    This class extends the standard Earth model with sparse matrix
    capabilities to handle large datasets more efficiently.
    """
    
    def __init__(self, *args, sparse_support: bool = False, **kwargs):
        """
        Initialize SparseEarth model.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments passed to Earth constructor.
        sparse_support : bool, optional (default=False)
            Whether to enable sparse matrix support.
        **kwargs : dict
            Keyword arguments passed to Earth constructor.
        """
        super().__init__(*args, **kwargs)
        self.sparse_support = sparse_support
    
    def fit(self, X, y):
        """
        Fit the Earth model to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values. Multi-output y is not currently supported.
            
        Returns
        -------
        self : SparseEarth
            The fitted model.
        """
        # Convert to sparse matrix if enabled
        if self.sparse_support:
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
            if not sparse.issparse(y):
                y = sparse.csr_matrix(y.reshape(-1, 1))
        
        # Call parent fit method
        super().fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Predict target values for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted target values.
        """
        # Convert to sparse matrix if enabled
        if self.sparse_support:
            if not sparse.issparse(X):
                X = sparse.csr_matrix(X)
        
        # Call parent predict method
        return super().predict(X)
    
    def _scrub_input_data(self, X, y):
        """
        Helper to validate and preprocess X and y, with sparse matrix support.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns
        -------
        tuple
            Tuple containing (X_processed, missing_mask, y_processed).
        """
        # Handle sparse matrices
        if sparse.issparse(X):
            X_array = X.toarray()
        else:
            X_array = np.asarray(X, dtype=float)
        
        if sparse.issparse(y):
            y_array = y.toarray().ravel()
        else:
            y_array = np.asarray(y, dtype=float).ravel()
        
        # Call parent scrubbing method
        X_processed, missing_mask, y_processed = super()._scrub_input_data(X_array, y_array)
        
        # Convert back to sparse if needed
        if self.sparse_support:
            X_processed = sparse.csr_matrix(X_processed)
            missing_mask = sparse.csr_matrix(missing_mask, dtype=bool)
        
        return X_processed, missing_mask, y_processed
    
    def _build_basis_matrix(self, X_processed: Union[np.ndarray, sparse.spmatrix], 
                           basis_functions: list, 
                           missing_mask: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Constructs the basis matrix B from X_processed and a list of basis functions,
        using the provided missing_mask, with sparse matrix support.
        
        Parameters
        ----------
        X_processed : numpy.ndarray or scipy.sparse matrix of shape (n_samples, n_features)
            The processed input samples.
        basis_functions : list of BasisFunction
            The list of basis functions to evaluate.
        missing_mask : numpy.ndarray or scipy.sparse matrix of shape (n_samples, n_features)
            Boolean mask indicating which original values were NaN.
            
        Returns
        -------
        numpy.ndarray of shape (n_samples, n_basis_functions)
            The basis matrix B.
        """
        if not basis_functions:
            if sparse.issparse(X_processed):
                return np.empty((X_processed.shape[0], 0))
            else:
                return super()._build_basis_matrix(X_processed, basis_functions, missing_mask)
        
        # For sparse matrices, convert to dense for basis function evaluation
        # (basis functions typically operate on dense arrays)
        if sparse.issparse(X_processed):
            X_dense = X_processed.toarray()
            mask_dense = missing_mask.toarray() if sparse.issparse(missing_mask) else missing_mask
            return super()._build_basis_matrix(X_dense, basis_functions, mask_dense)
        else:
            return super()._build_basis_matrix(X_processed, basis_functions, missing_mask)


def convert_to_sparse(X: np.ndarray, sparsity_threshold: float = 0.1) -> Union[np.ndarray, sparse.spmatrix]:
    """
    Convert a dense array to sparse if it's sufficiently sparse.
    
    Parameters
    ----------
    X : numpy.ndarray
        Dense array to potentially convert to sparse.
    sparsity_threshold : float, optional (default=0.1)
        Threshold for sparsity (fraction of zeros). If array has more zeros
        than this fraction, it will be converted to sparse format.
        
    Returns
    -------
    numpy.ndarray or scipy.sparse matrix
        Dense or sparse representation of the input array.
    """
    # Calculate sparsity
    if X.size == 0:
        return X
    
    sparsity = np.sum(X == 0) / X.size
    
    # Convert to sparse if sufficiently sparse
    if sparsity > sparsity_threshold:
        logger.info(f"Converting array to sparse format (sparsity: {sparsity:.2%})")
        return sparse.csr_matrix(X)
    else:
        return X


def demo_sparse_support():
    """Demonstrate sparse matrix support functionality."""
    print("Demonstrating sparse matrix support...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 50)
    
    # Create sparse versions
    X_sparse = sparse.csr_matrix(X)
    y_sparse = sparse.csr_matrix(y.reshape(-1, 1))
    
    print(f"Dense X shape: {X.shape}")
    print(f"Sparse X shape: {X_sparse.shape}")
    print(f"Dense y shape: {y.shape}")
    print(f"Sparse y shape: {y_sparse.shape}")
    
    # Test SparseEarth model
    try:
        model = SparseEarth(max_degree=2, penalty=3.0, max_terms=15, sparse_support=True)
        model.fit(X_sparse, y)
        print(f"✅ SparseEarth model fitted successfully with {len(model.basis_)} basis functions")
        print(f"   Model GCV: {model.gcv_:.6f}")
        print(f"   Model R²: {model.score(X_sparse, y):.4f}")
        
        # Test predictions
        preds = model.predict(X_sparse[:10])
        print(f"   Predictions shape: {preds.shape}")
        print(f"   Sample predictions: {preds[:3]}")
        
    except Exception as e:
        print(f"❌ SparseEarth model failed: {e}")
    
    # Test conversion utility
    X_dense_part = X.copy()
    X_dense_part[X_dense_part < 0.1] = 0  # Make it sparse
    X_auto_sparse = convert_to_sparse(X_dense_part, sparsity_threshold=0.05)
    print(f"\\n✅ Auto-sparse conversion:")
    print(f"   Original sparsity: {np.sum(X_dense_part == 0) / X_dense_part.size:.2%}")
    print(f"   Converted to sparse: {sparse.issparse(X_auto_sparse)}")
    
    print("\\nSparse matrix support demonstration completed!")


# Automatically determine if sparse matrix support should be used
def auto_sparse_support(X: Union[np.ndarray, sparse.spmatrix], 
                      threshold: float = 0.1) -> bool:
    """
    Automatically determine if sparse matrix support should be used.
    
    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix
        Input data array.
    threshold : float, optional (default=0.1)
        Sparsity threshold for automatic sparse conversion.
        
    Returns
    -------
    bool
        Whether sparse matrix support should be used.
    """
    if sparse.issparse(X):
        return True
    
    # Calculate sparsity of dense array
    sparsity = np.sum(X == 0) / X.size if X.size > 0 else 0.0
    return sparsity > threshold


# Utility functions for sparse matrix operations
def sparse_matrix_info(X: Union[np.ndarray, sparse.spmatrix]) -> Dict[str, Any]:
    """
    Get information about a matrix (dense or sparse).
    
    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix
        Input matrix.
        
    Returns
    -------
    dict
        Dictionary containing matrix information.
    """
    if sparse.issparse(X):
        return {
            'type': 'sparse',
            'format': X.format,
            'shape': X.shape,
            'nnz': X.nnz,  # Number of non-zero elements
            'sparsity': 1.0 - (X.nnz / (X.shape[0] * X.shape[1])) if X.shape[0] * X.shape[1] > 0 else 0.0,
            'density': X.nnz / (X.shape[0] * X.shape[1]) if X.shape[0] * X.shape[1] > 0 else 0.0,
            'dtype': X.dtype
        }
    else:
        nnz = np.sum(X != 0)
        return {
            'type': 'dense',
            'shape': X.shape,
            'nnz': int(nnz),
            'sparsity': 1.0 - (nnz / (X.shape[0] * X.shape[1])) if X.shape[0] * X.shape[1] > 0 else 0.0,
            'density': nnz / (X.shape[0] * X.shape[1]) if X.shape[0] * X.shape[1] > 0 else 0.0,
            'dtype': X.dtype
        }