"""
Parallel processing utilities for pymars to speed up computations.

This module provides parallel processing capabilities to improve performance
for basis function evaluation and other computationally intensive operations.
"""
import logging
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ._basis import BasisFunction

# Import Earth to avoid NameError
from .earth import Earth

logger = logging.getLogger(__name__)


def _evaluate_basis_function_parallel(args: Tuple[BasisFunction, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Helper function to evaluate a single basis function in parallel.
    
    This function is designed to be used with concurrent.futures to evaluate
    basis functions in parallel processes or threads.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (basis_function, X_processed, missing_mask)
        
    Returns
    -------
    numpy.ndarray
        Evaluated basis function values
    """
    basis_function, X_processed, missing_mask = args
    return basis_function.transform(X_processed, missing_mask)


class ParallelBasisEvaluator:
    """
    Parallel evaluator for basis functions.
    
    This class provides parallel processing capabilities for evaluating
    basis functions, which can significantly speed up computation for
    large numbers of basis functions.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize the parallel basis evaluator.
        
        Parameters
        ----------
        max_workers : int, optional (default=None)
            Maximum number of worker threads/processes. If None, uses
            the default from concurrent.futures.
        use_processes : bool, optional (default=False)
            Whether to use processes (True) or threads (False) for parallelization.
            Processes are better for CPU-bound tasks, threads for I/O-bound tasks.
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor = None
    
    def __enter__(self):
        """Context manager entry."""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def evaluate_basis_functions(self, basis_functions: List[BasisFunction], 
                                X_processed: np.ndarray, missing_mask: np.ndarray) -> List[np.ndarray]:
        """
        Evaluate basis functions in parallel.
        
        Parameters
        ----------
        basis_functions : list of BasisFunction
            List of basis functions to evaluate
        X_processed : numpy.ndarray
            Processed input data
        missing_mask : numpy.ndarray
            Boolean mask indicating missing values
            
        Returns
        -------
        list of numpy.ndarray
            List of evaluated basis function values
        """
        if not basis_functions:
            return []
        
        if self.executor is None:
            raise RuntimeError("ParallelBasisEvaluator must be used as a context manager")
        
        # Prepare arguments for parallel evaluation
        args_list = [(bf, X_processed, missing_mask) for bf in basis_functions]
        
        # Submit all evaluations
        futures = [self.executor.submit(_evaluate_basis_function_parallel, args) for args in args_list]
        
        # Collect results
        results = [future.result() for future in futures]
        
        return results


def _build_basis_matrix_column(args: Tuple[int, BasisFunction, np.ndarray, np.ndarray]) -> Tuple[int, np.ndarray]:
    """
    Helper function to build a single column of the basis matrix in parallel.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (column_index, basis_function, X_processed, missing_mask)
        
    Returns
    -------
    tuple
        Tuple containing (column_index, column_values)
    """
    col_idx, basis_function, X_processed, missing_mask = args
    column_values = basis_function.transform(X_processed, missing_mask)
    return col_idx, column_values


class ParallelBasisMatrixBuilder:
    """
    Parallel builder for basis matrices.
    
    This class provides parallel processing capabilities for building
    basis matrices, which can significantly speed up computation for
    large numbers of basis functions.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize the parallel basis matrix builder.
        
        Parameters
        ----------
        max_workers : int, optional (default=None)
            Maximum number of worker threads/processes. If None, uses
            the default from concurrent.futures.
        use_processes : bool, optional (default=False)
            Whether to use processes (True) or threads (False) for parallelization.
            Processes are better for CPU-bound tasks, threads for I/O-bound tasks.
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor = None
    
    def __enter__(self):
        """Context manager entry."""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def build_basis_matrix(self, X_processed: np.ndarray, basis_functions: List[BasisFunction], 
                          missing_mask: np.ndarray) -> np.ndarray:
        """
        Build basis matrix in parallel.
        
        Parameters
        ----------
        X_processed : numpy.ndarray
            Processed input data
        basis_functions : list of BasisFunction
            List of basis functions to evaluate
        missing_mask : numpy.ndarray
            Boolean mask indicating missing values
            
        Returns
        -------
        numpy.ndarray
            Basis matrix of shape (n_samples, n_basis_functions)
        """
        if not basis_functions:
            return np.empty((X_processed.shape[0], 0))
        
        if self.executor is None:
            raise RuntimeError("ParallelBasisMatrixBuilder must be used as a context manager")
        
        n_samples = X_processed.shape[0]
        n_basis_functions = len(basis_functions)
        
        # Preallocate basis matrix
        B_matrix = np.empty((n_samples, n_basis_functions), dtype=float)
        
        # Prepare arguments for parallel column building
        args_list = [(i, bf, X_processed, missing_mask) for i, bf in enumerate(basis_functions)]
        
        # Submit all column building tasks
        futures = [self.executor.submit(_build_basis_matrix_column, args) for args in args_list]
        
        # Collect results and fill matrix columns
        for future in futures:
            col_idx, column_values = future.result()
            B_matrix[:, col_idx] = column_values
        
        return B_matrix


class ParallelEarth(Earth):
    """
    Earth model with parallel processing capabilities.
    
    This class extends the standard Earth model with parallel processing
    functionality to improve performance for basis function evaluation.
    """
    
    def __init__(self, *args, max_workers: Optional[int] = None, 
                 use_processes: bool = False, **kwargs):
        """
        Initialize ParallelEarth model.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments passed to Earth constructor.
        max_workers : int, optional (default=None)
            Maximum number of worker threads/processes. If None, uses
            the default from concurrent.futures.
        use_processes : bool, optional (default=False)
            Whether to use processes (True) or threads (False) for parallelization.
            Processes are better for CPU-bound tasks, threads for I/O-bound tasks.
        **kwargs : dict
            Keyword arguments passed to Earth constructor.
        """
        super().__init__(*args, **kwargs)
        self.max_workers = max_workers
        self.use_processes = use_processes
    
    def _build_basis_matrix(self, X_processed: np.ndarray, basis_functions: list, 
                           missing_mask: np.ndarray) -> np.ndarray:
        """
        Constructs the basis matrix B from X_processed and a list of basis functions,
        using parallel processing if enabled.
        
        Parameters
        ----------
        X_processed : numpy.ndarray of shape (n_samples, n_features)
            The processed input samples.
        basis_functions : list of BasisFunction
            The list of basis functions to evaluate.
        missing_mask : numpy.ndarray of shape (n_samples, n_features)
            Boolean mask indicating which original values were NaN.
            
        Returns
        -------
        numpy.ndarray of shape (n_samples, n_basis_functions)
            The basis matrix B.
        """
        if not basis_functions:
            return np.empty((X_processed.shape[0], 0))
        
        # Use parallel processing for large numbers of basis functions
        if len(basis_functions) > 10:  # Threshold for parallelization
            try:
                with ParallelBasisMatrixBuilder(
                    max_workers=self.max_workers, 
                    use_processes=self.use_processes
                ) as builder:
                    return builder.build_basis_matrix(X_processed, basis_functions, missing_mask)
            except Exception as e:
                logger.warning(f"Parallel basis matrix building failed: {e}. Falling back to sequential.")
                # Fall back to sequential processing
                pass
        
        # Sequential processing (fallback or small number of basis functions)
        n_samples = X_processed.shape[0]
        B_matrix = np.empty((n_samples, len(basis_functions)), dtype=float)
        for idx, bf in enumerate(basis_functions):
            B_matrix[:, idx] = bf.transform(X_processed, missing_mask)
        return B_matrix


def demo_parallel_processing():
    """Demonstrate parallel processing functionality."""
    print("Demonstrating parallel processing functionality...")
    
    # Generate test data
    np.random.seed(42)
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] * 0.5 + np.sin(X[:, 2] * np.pi) + np.random.normal(0, 0.1, 50)
    missing_mask = np.zeros_like(X, dtype=bool)
    
    # Create test basis functions
    from ._basis import ConstantBasisFunction, LinearBasisFunction, HingeBasisFunction
    
    bf_const = ConstantBasisFunction()
    bf_linear = LinearBasisFunction(variable_idx=0, variable_name="x0")
    bf_hinge1 = HingeBasisFunction(variable_idx=1, knot_val=0.3, is_right_hinge=True, variable_name="x1_k0.3R")
    bf_hinge2 = HingeBasisFunction(variable_idx=1, knot_val=0.7, is_right_hinge=False, variable_name="x1_k0.7L")
    bf_hinge3 = HingeBasisFunction(variable_idx=2, knot_val=0.5, is_right_hinge=True, variable_name="x2_k0.5R")
    
    basis_functions = [bf_const, bf_linear, bf_hinge1, bf_hinge2, bf_hinge3]
    
    print(f"Testing with {len(basis_functions)} basis functions...")
    
    # Test sequential evaluation
    print("\\n1. Sequential evaluation:")
    B_seq = np.empty((X.shape[0], len(basis_functions)), dtype=float)
    for idx, bf in enumerate(basis_functions):
        B_seq[:, idx] = bf.transform(X, missing_mask)
    print(f"   Sequential result shape: {B_seq.shape}")
    print(f"   Sequential sample values: {B_seq[:3, :3]}")
    
    # Test parallel evaluation with threads
    print("\\n2. Parallel evaluation (threads):")
    try:
        with ParallelBasisMatrixBuilder(max_workers=4, use_processes=False) as builder:
            B_parallel_threads = builder.build_basis_matrix(X, basis_functions, missing_mask)
        print(f"   Thread-parallel result shape: {B_parallel_threads.shape}")
        print(f"   Thread-parallel sample values: {B_parallel_threads[:3, :3]}")
        assert np.allclose(B_seq, B_parallel_threads), "Thread-parallel results should match sequential"
        print("   ✅ Thread-parallel results match sequential")
    except Exception as e:
        print(f"   ⚠️  Thread-parallel evaluation failed: {e}")
    
    # Test parallel evaluation with processes
    print("\\n3. Parallel evaluation (processes):")
    try:
        with ParallelBasisMatrixBuilder(max_workers=4, use_processes=True) as builder:
            B_parallel_processes = builder.build_basis_matrix(X, basis_functions, missing_mask)
        print(f"   Process-parallel result shape: {B_parallel_processes.shape}")
        print(f"   Process-parallel sample values: {B_parallel_processes[:3, :3]}")
        assert np.allclose(B_seq, B_parallel_processes), "Process-parallel results should match sequential"
        print("   ✅ Process-parallel results match sequential")
    except Exception as e:
        print(f"   ⚠️  Process-parallel evaluation failed: {e}")
    
    # Test ParallelEarth model
    print("\\n4. ParallelEarth model:")
    try:
        model = ParallelEarth(max_degree=2, penalty=3.0, max_terms=15, 
                             max_workers=4, use_processes=False)
        model.fit(X, y)
        preds = model.predict(X[:5])
        print(f"   ParallelEarth fitted with {len(model.basis_)} basis functions")
        print(f"   ParallelEarth predictions: {preds}")
        print(f"   ParallelEarth R²: {model.score(X, y):.4f}")
        print("   ✅ ParallelEarth model works correctly")
    except Exception as e:
        print(f"   ⚠️  ParallelEarth model failed: {e}")
    
    print("\\nParallel processing demonstration completed!")