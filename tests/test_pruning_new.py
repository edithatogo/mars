"""
Tests for pymars._pruning module - pruning pass functionality
"""
import numpy as np
import pytest
from pymars._pruning import PruningPasser
from pymars.earth import Earth
from pymars._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
from pymars._util import calculate_gcv, gcv_penalty_cost_effective_parameters


class TestPruningPasser:
    """Test PruningPasser class."""
    
    def test_initialization(self):
        """Test initialization of PruningPasser."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        assert pruner.model is model
        assert pruner.best_gcv_so_far == np.inf
        assert pruner.best_basis_functions_so_far == []
        assert pruner.best_coeffs_so_far is None
        assert pruner.X_train is None
        assert pruner.y_train is None
        assert pruner.n_samples == 0
        assert pruner.missing_mask is None
        assert pruner.X_fit_original is None
    
    def test_calculate_rss_and_coeffs_basic(self):
        """Test _calculate_rss_and_coeffs with basic data."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create simple test data
        B_matrix = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        y_data = np.array([3.0, 4.0, 5.0])
        
        rss, coeffs, num_valid_rows = pruner._calculate_rss_and_coeffs(B_matrix, y_data)
        
        assert isinstance(rss, (float, np.floating))
        assert coeffs is not None
        assert isinstance(coeffs, np.ndarray)
        assert num_valid_rows == 3
    
    def test_calculate_rss_and_coeffs_empty_matrix(self):
        """Test _calculate_rss_and_coeffs with empty matrix."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        B_matrix = np.empty((3, 0))  # 3 samples, 0 basis functions (intercept only)
        y_data = np.array([3.0, 4.0, 5.0])
        
        rss, coeffs, num_valid_rows = pruner._calculate_rss_and_coeffs(B_matrix, y_data)
        
        assert isinstance(rss, (float, np.floating))
        assert num_valid_rows == 3
    
    def test_calculate_rss_and_coeffs_with_nan(self):
        """Test _calculate_rss_and_coeffs with NaN values."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        B_matrix = np.array([[1.0, 2.0], [np.nan, 3.0], [1.0, 4.0]])
        y_data = np.array([3.0, 4.0, 5.0])
        
        rss, coeffs, num_valid_rows = pruner._calculate_rss_and_coeffs(B_matrix, y_data)
        
        assert num_valid_rows == 2  # Only 2 rows are valid (no NaNs)
    
    def test_calculate_rss_and_coeffs_insufficient_data(self):
        """Test _calculate_rss_and_coeffs with insufficient data."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        B_matrix = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])  # 2 samples, 3 columns (underdetermined)
        y_data = np.array([3.0, 4.0])
        
        rss, coeffs, num_valid_rows = pruner._calculate_rss_and_coeffs(B_matrix, y_data)
        
        assert rss == np.inf or num_valid_rows == 0
    
    def test_build_basis_matrix(self):
        """Test _build_basis_matrix function."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        missing_mask = np.zeros_like(X_data, dtype=bool)
        
        # Create some basis functions
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        hinge_bf = HingeBasisFunction(variable_idx=1, knot_val=3.0, is_right_hinge=True, variable_name="x1_k3.0R")
        
        bfs = [constant_bf, linear_bf, hinge_bf]
        
        B_matrix = pruner._build_basis_matrix(X_data, bfs, missing_mask)
        
        assert B_matrix.shape == (3, 3)  # 3 samples, 3 basis functions
        assert B_matrix[0, 0] == 1.0  # Constant basis function
        assert B_matrix[0, 1] == X_data[0, 0]  # Linear basis function for first variable
    
    def test_build_basis_matrix_empty(self):
        """Test _build_basis_matrix with empty list."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        X_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        missing_mask = np.zeros_like(X_data, dtype=bool)
        
        B_matrix = pruner._build_basis_matrix(X_data, [], missing_mask)
        
        assert B_matrix.shape == (2, 0)  # 2 samples, 0 basis functions
    
    def test_compute_gcv_for_subset(self):
        """Test _compute_gcv_for_subset function."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_fit = np.array([3.0, 7.0, 11.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Create some basis functions
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        bfs = [constant_bf, linear_bf]
        
        gcv, rss, coeffs = pruner._compute_gcv_for_subset(
            X_fit_processed, y_fit, missing_mask, X_fit_original, bfs
        )
        
        assert isinstance(gcv, (int, float, np.floating))
        assert isinstance(rss, (int, float, np.floating))
        assert coeffs is not None
    
    def test_compute_gcv_for_empty_subset(self):
        """Test _compute_gcv_for_subset with empty subset.""" 
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        gcv, rss, coeffs = pruner._compute_gcv_for_subset(
            X_fit_processed, y_fit, missing_mask, X_fit_original, []
        )
        
        assert isinstance(gcv, (int, float, np.floating))
        assert isinstance(rss, (int, float, np.floating))
    
    def test_run_with_empty_initial_bfs(self):
        """Test run method with empty initial basis functions."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, [], np.array([]))
        
        assert basis_functions == []
        assert len(coeffs) == 0
        assert gcv == np.inf

    def test_run_with_single_basis_function(self):
        """Test run method with single basis function."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Create a single basis function
        constant_bf = ConstantBasisFunction()
        initial_bfs = [constant_bf]
        initial_coeffs = np.array([np.mean(y_fit)])  # Initial coefficients
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert isinstance(basis_functions, list)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))
    
    def test_run_with_multiple_basis_functions(self):
        """Test run method with multiple basis functions."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_fit = np.array([3.0, 7.0, 11.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Create multiple basis functions
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        initial_bfs = [constant_bf, linear_bf]
        initial_coeffs = np.array([1.0, 1.0])  # Initial coefficients
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert isinstance(basis_functions, list)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))
        assert len(basis_functions) <= len(initial_bfs)


class TestPruningUtilFunctions:
    """Test utility functions used in pruning."""
    
    def test_gcv_penalty_cost_effective_parameters(self):
        """Test gcv_penalty_cost_effective_parameters function."""
        num_terms = 5
        num_hinge_terms = 3
        penalty = 3.0
        num_samples = 20
        
        effective_params = gcv_penalty_cost_effective_parameters(num_terms, num_hinge_terms, penalty, num_samples)
        
        assert isinstance(effective_params, (int, float, np.floating))
        assert effective_params >= num_terms  # Effective params should be >= number of terms
    
    def test_calculate_gcv(self):
        """Test calculate_gcv function."""
        rss = 10.0
        num_samples = 20
        num_effective_params = 5.0
        
        gcv = calculate_gcv(rss, num_samples, num_effective_params)
        
        assert isinstance(gcv, (int, float, np.floating))
        assert gcv > 0


if __name__ == "__main__":
    test = TestPruningPasser()
    test.test_initialization()
    test.test_calculate_rss_and_coeffs_basic()
    test.test_calculate_rss_and_coeffs_empty_matrix()
    test.test_calculate_rss_and_coeffs_with_nan()
    test.test_calculate_rss_and_coeffs_insufficient_data()
    test.test_build_basis_matrix()
    test.test_build_basis_matrix_empty()
    test.test_compute_gcv_for_subset()
    test.test_compute_gcv_for_empty_subset()
    test.test_run_with_empty_initial_bfs()
    test.test_run_with_single_basis_function()
    test.test_run_with_multiple_basis_functions()
    
    util_test = TestPruningUtilFunctions()
    util_test.test_gcv_penalty_cost_effective_parameters()
    util_test.test_calculate_gcv()
    
    print("All _pruning.py tests passed!")