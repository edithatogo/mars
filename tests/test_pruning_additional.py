"""
Additional tests for pymars._pruning module to achieve >90% coverage
"""
import numpy as np
import pytest
from pymars._pruning import PruningPasser
from pymars.earth import Earth
from pymars._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction


class TestPruningPasserAdditional:
    """Additional test for PruningPasser class to cover missed lines."""
    
    def test_calculate_rss_and_coeffs_error_conditions(self):
        """Test _calculate_rss_and_coeffs with various error conditions."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Test with underdetermined system (more columns than rows)
        B_matrix = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 5.0, 6.0, 7.0]])  # 2 samples, 4 columns
        y_data = np.array([3.0, 4.0])
        
        rss, coeffs, num_valid_rows = pruner._calculate_rss_and_coeffs(B_matrix, y_data)
        
        # Should return error condition
        assert rss == np.inf or num_valid_rows == 0
    
    def test_calculate_rss_and_coeffs_no_valid_rows(self):
        """Test _calculate_rss_and_coeffs when there are no valid rows."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create matrix where all rows have NaN
        B_matrix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        y_data = np.array([3.0, 4.0])
        
        rss, coeffs, num_valid_rows = pruner._calculate_rss_and_coeffs(B_matrix, y_data)
        
        assert num_valid_rows == 0
        assert rss == np.inf or coeffs is None
    
    def test_calculate_rss_and_coeffs_empty_columns(self):
        """Test _calculate_rss_and_coeffs when B_matrix has no columns."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Matrix with 0 columns
        B_matrix = np.empty((3, 0))  # 3 samples, 0 basis functions
        y_data = np.array([3.0, 4.0, 5.0])
        
        rss, coeffs, num_valid_rows = pruner._calculate_rss_and_coeffs(B_matrix, y_data)
        
        assert num_valid_rows == 3
        assert coeffs is not None  # Should return intercept coefficient
    
    def test_compute_gcv_for_subset_with_bad_data(self):
        """Test _compute_gcv_for_subset with bad data that should result in errors."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_fit_processed = np.array([[np.nan, 2.0], [3.0, 4.0]])  # First row has NaN
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        missing_mask[0, 0] = True  # Mark first element as missing
        X_fit_original = X_fit_processed.copy()
        
        # Create a single basis function
        constant_bf = ConstantBasisFunction()
        bfs = [constant_bf]
        
        gcv, rss, coeffs = pruner._compute_gcv_for_subset(
            X_fit_processed, y_fit, missing_mask, X_fit_original, bfs
        )
        
        # This should compute successfully despite one NaN
        assert isinstance(gcv, (int, float, np.floating))
    
    def test_compute_gcv_for_subset_empty(self):
        """Test _compute_gcv_for_subset with empty basis functions list."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Empty basis functions list
        bfs = []
        
        gcv, rss, coeffs = pruner._compute_gcv_for_subset(
            X_fit_processed, y_fit, missing_mask, X_fit_original, bfs
        )
        
        assert isinstance(gcv, (int, float, np.floating))
        assert isinstance(rss, (int, float, np.floating))
    
    def test_build_basis_matrix_1d_data(self):
        """Test _build_basis_matrix with 1D data."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create 1D data
        X_data = np.array([1.0, 2.0, 3.0])
        X_data_2d = X_data.reshape(-1, 1)  # Make it 2D as expected by basis functions
        missing_mask = np.zeros_like(X_data_2d, dtype=bool)
        
        # Create basis functions that work with 1D data
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        bfs = [constant_bf, linear_bf]
        
        # Need to use 2D data for basis functions
        B_matrix = pruner._build_basis_matrix(X_data_2d, bfs, missing_mask)
        
        assert B_matrix.shape == (3, 2)  # 3 samples, 2 basis functions
        assert B_matrix[0, 0] == 1.0  # Constant basis function
    
    def test_run_pruning_with_singular_matrix(self):
        """Test run when the matrix becomes singular during pruning."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create data that might cause singularity
        X_fit_processed = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])  # Two identical rows
        y_fit = np.array([3.0, 3.0, 7.0])  # Matches the identical inputs
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Very simple basis functions
        constant_bf = ConstantBasisFunction()
        initial_bfs = [constant_bf]
        initial_coeffs = np.array([np.mean(y_fit)])
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert isinstance(basis_functions, list)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))
    
    def test_pruning_when_no_improvement_possible(self):
        """Test pruning when no improvements are possible."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create minimal test case
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Single basis function - nothing to prune
        constant_bf = ConstantBasisFunction()
        initial_bfs = [constant_bf]
        initial_coeffs = np.array([np.mean(y_fit)])
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert len(basis_functions) <= len(initial_bfs)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))
    
    def test_pruning_with_all_constant_basis_functions(self):
        """Test pruning when all basis functions are constant (edge case)."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Use only constant basis functions (degenerate case)
        constant_bf1 = ConstantBasisFunction()
        constant_bf2 = ConstantBasisFunction()  # Duplicate basis function
        initial_bfs = [constant_bf1, constant_bf2]
        initial_coeffs = np.array([3.0, 1.0])
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert isinstance(basis_functions, list)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))
    
    def test_pruning_with_underdetermined_system(self):
        """Test pruning with underdetermined system (more basis functions than samples)."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Only 2 samples but we'll try to create more basis functions
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])  # Only 2 samples
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Create basis functions that would lead to underdetermined system
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        linear_bf2 = LinearBasisFunction(variable_idx=1, variable_name="x1")
        # Add more basis functions than samples
        initial_bfs = [constant_bf, linear_bf, linear_bf2]  # 3 basis functions, 2 samples
        initial_coeffs = np.array([1.0, 1.0, 1.0])
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert isinstance(basis_functions, list)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))
    
    def test_compute_gcv_for_subset_with_error_case(self):
        """Test _compute_gcv_for_subset when it encounters an error in coefficient calculation."""
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        pruner = PruningPasser(model)
        
        # Create test data that might cause numerical instability
        X_fit_processed = np.array([[1e10, 2e10], [3e10, 4e10]])  # Very large numbers
        y_fit = np.array([3e10, 7e10])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        # Create basis functions
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        initial_bfs = [constant_bf, linear_bf]
        
        gcv, rss, coeffs = pruner._compute_gcv_for_subset(
            X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs
        )
        
        # Should handle numerical instability gracefully
        assert isinstance(gcv, (int, float, np.floating))
        assert isinstance(rss, (int, float, np.floating))
        assert coeffs is None or isinstance(coeffs, np.ndarray)
    
    def test_pruning_with_zero_penalty(self):
        """Test pruning with zero penalty."""
        model = Earth(max_degree=2, penalty=0.0, max_terms=10)  # Zero penalty
        pruner = PruningPasser(model)
        
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        initial_bfs = [constant_bf, linear_bf]
        initial_coeffs = np.array([3.0, 1.0])
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert isinstance(basis_functions, list)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))
    
    def test_pruning_with_maximum_penalty(self):
        """Test pruning with very high penalty."""
        model = Earth(max_degree=2, penalty=1000.0, max_terms=10)  # Very high penalty
        pruner = PruningPasser(model)
        
        X_fit_processed = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_fit = np.array([3.0, 7.0])
        missing_mask = np.zeros_like(X_fit_processed, dtype=bool)
        X_fit_original = X_fit_processed.copy()
        
        constant_bf = ConstantBasisFunction()
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        initial_bfs = [constant_bf, linear_bf]
        initial_coeffs = np.array([3.0, 1.0])
        
        basis_functions, coeffs, gcv = pruner.run(X_fit_processed, y_fit, missing_mask, X_fit_original, initial_bfs, initial_coeffs)
        
        assert isinstance(basis_functions, list)
        assert isinstance(coeffs, np.ndarray)
        assert isinstance(gcv, (int, float, np.floating))


if __name__ == "__main__":
    test = TestPruningPasserAdditional()
    test.test_calculate_rss_and_coeffs_error_conditions()
    test.test_calculate_rss_and_coeffs_no_valid_rows()
    test.test_calculate_rss_and_coeffs_empty_columns()
    test.test_compute_gcv_for_subset_with_bad_data()
    test.test_compute_gcv_for_subset_empty()
    test.test_build_basis_matrix_1d_data()
    test.test_run_pruning_with_singular_matrix()
    test.test_pruning_when_no_improvement_possible()
    test.test_pruning_with_all_constant_basis_functions()
    test.test_pruning_with_underdetermined_system()
    test.test_compute_gcv_for_subset_with_error_case()
    test.test_pruning_with_zero_penalty()
    test.test_pruning_with_maximum_penalty()
    
    print("All additional _pruning.py tests passed!")