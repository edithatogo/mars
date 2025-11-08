"""
Tests for pymars._record module - EarthRecord class
"""
import numpy as np
import pytest
from pymars._record import EarthRecord
from pymars.earth import Earth
from pymars._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction


class TestEarthRecord:
    """Test EarthRecord class functionality."""
    
    def test_initialization(self):
        """Test initialization of EarthRecord with basic data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        assert record.n_samples == 2
        assert record.n_features == 2
        assert record.model_params is not None
        assert isinstance(record.fwd_basis_, list)
        assert isinstance(record.fwd_coeffs_, list)
        assert isinstance(record.fwd_rss_, list)
        assert isinstance(record.pruning_trace_basis_functions_, list)
        assert isinstance(record.pruning_trace_coeffs_, list)
        assert isinstance(record.pruning_trace_gcv_, list)
        assert isinstance(record.pruning_trace_rss_, list)
        assert record.final_basis_ is None
        assert record.final_coeffs_ is None
        assert record.final_gcv_ is None
        assert record.final_rss_ is None
        assert record.final_mse_ is None
    
    def test_log_forward_pass_step(self):
        """Test logging forward pass steps."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        # Create some basis functions and coefficients
        bfs = [ConstantBasisFunction()]
        coeffs = np.array([1.0])
        rss = 2.0
        
        record.log_forward_pass_step(bfs, coeffs, rss)
        
        assert len(record.fwd_basis_) == 1
        assert len(record.fwd_coeffs_) == 1
        assert len(record.fwd_rss_) == 1
        assert record.fwd_rss_[0] == rss
        assert len(record.fwd_basis_[0]) == 1
        assert isinstance(record.fwd_coeffs_[0], np.ndarray)
        
        # Add another step
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        bfs2 = [ConstantBasisFunction(), linear_bf]
        coeffs2 = np.array([1.0, 2.0])
        rss2 = 1.5
        
        record.log_forward_pass_step(bfs2, coeffs2, rss2)
        
        assert len(record.fwd_basis_) == 2
        assert len(record.fwd_coeffs_) == 2
        assert len(record.fwd_rss_) == 2
        assert record.fwd_rss_[1] == rss2
        assert len(record.fwd_basis_[1]) == 2
    
    def test_log_pruning_step(self):
        """Test logging pruning steps."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        # Create some basis functions and coefficients
        bfs = [ConstantBasisFunction()]
        coeffs = np.array([1.0])
        gcv = 2.5
        rss = 1.0
        
        record.log_pruning_step(bfs, coeffs, gcv, rss)
        
        assert len(record.pruning_trace_basis_functions_) == 1
        assert len(record.pruning_trace_coeffs_) == 1
        assert len(record.pruning_trace_gcv_) == 1
        assert len(record.pruning_trace_rss_) == 1
        assert record.pruning_trace_gcv_[0] == gcv
        assert record.pruning_trace_rss_[0] == rss
        assert len(record.pruning_trace_basis_functions_[0]) == 1
        
        # Add another step
        linear_bf = LinearBasisFunction(variable_idx=0, variable_name="x0")
        bfs2 = [ConstantBasisFunction(), linear_bf]
        coeffs2 = np.array([1.0, 2.0])
        gcv2 = 1.5
        rss2 = 0.8
        
        record.log_pruning_step(bfs2, coeffs2, gcv2, rss2)
        
        assert len(record.pruning_trace_basis_functions_) == 2
        assert len(record.pruning_trace_coeffs_) == 2
        assert len(record.pruning_trace_gcv_) == 2
        assert len(record.pruning_trace_rss_) == 2
        assert record.pruning_trace_gcv_[1] == gcv2
        assert record.pruning_trace_rss_[1] == rss2
        assert len(record.pruning_trace_basis_functions_[1]) == 2
    
    def test_set_final_model(self):
        """Test setting final model details."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        # Create basis functions and coefficients
        bfs = [ConstantBasisFunction(), LinearBasisFunction(variable_idx=0, variable_name="x0")]
        coeffs = np.array([1.0, 2.0])
        gcv = 2.5
        rss = 1.0
        mse = 0.5
        
        record.set_final_model(bfs, coeffs, gcv, rss, mse)
        
        assert record.final_basis_ is not None
        assert record.final_coeffs_ is not None
        assert record.final_gcv_ == gcv
        assert record.final_rss_ == rss
        assert record.final_mse_ == mse
        assert len(record.final_basis_) == 2
        assert len(record.final_coeffs_) == 2
    
    def test_str_method_empty_record(self):
        """Test string representation of an empty record."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        str_repr = str(record)
        assert "Earth Model Fit Record" in str_repr
        assert "Initial Samples: 2" in str_repr
        assert "Features: 2" in str_repr
        
        # Should not contain forward pass or pruning details as they're empty
        assert "Forward Pass completed" not in str_repr
        assert "Pruning Pass Trace" not in str_repr
        assert "Final Selected Model" not in str_repr
    
    def test_str_method_with_forward_pass(self):
        """Test string representation with forward pass data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        # Add forward pass data
        bfs = [ConstantBasisFunction()]
        coeffs = np.array([1.0])
        rss = 2.0
        record.log_forward_pass_step(bfs, coeffs, rss)
        
        str_repr = str(record)
        assert "Earth Model Fit Record" in str_repr
        assert "Forward Pass completed" in str_repr
        assert "Final RSS from forward pass:" in str_repr
        assert f"{rss:.4f}" in str_repr
    
    def test_str_method_with_pruning_trace(self):
        """Test string representation with pruning trace data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        # Add pruning trace data
        bfs = [ConstantBasisFunction()]
        coeffs = np.array([1.0])
        gcv = 2.5
        rss = 1.0
        record.log_pruning_step(bfs, coeffs, gcv, rss)
        
        str_repr = str(record)
        assert "Earth Model Fit Record" in str_repr
        assert "Pruning Pass Trace (models considered): 1" in str_repr
    
    def test_str_method_with_final_model(self):
        """Test string representation with final model data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        # Set final model data
        bfs = [ConstantBasisFunction(), LinearBasisFunction(variable_idx=0, variable_name="x0")]
        coeffs = np.array([1.0, 2.0])
        gcv = 2.5
        rss = 1.0
        mse = 0.5
        record.set_final_model(bfs, coeffs, gcv, rss, mse)
        
        str_repr = str(record)
        assert "Earth Model Fit Record" in str_repr
        assert "Final Selected Model (after pruning):" in str_repr
        assert f"  Number of terms: {len(bfs)}" in str_repr
        assert f"  GCV: {gcv:.4f}" in str_repr
        assert f"  RSS: {rss:.4f}" in str_repr
        assert f"  MSE: {mse:.4f}" in str_repr
    
    def test_str_method_with_all_data(self):
        """Test string representation with all types of data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([3.0, 7.0, 11.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        # Add forward pass data
        bfs1 = [ConstantBasisFunction()]
        coeffs1 = np.array([1.0])
        rss1 = 2.0
        record.log_forward_pass_step(bfs1, coeffs1, rss1)
        
        # Add pruning trace data
        bfs2 = [ConstantBasisFunction()]
        coeffs2 = np.array([1.5])
        gcv = 2.5
        rss2 = 1.0
        record.log_pruning_step(bfs2, coeffs2, gcv, rss2)
        
        # Set final model data
        bfs_final = [ConstantBasisFunction(), LinearBasisFunction(variable_idx=0, variable_name="x0")]
        coeffs_final = np.array([1.0, 2.0])
        gcv_final = 2.0
        rss_final = 0.8
        mse_final = 0.4
        record.set_final_model(bfs_final, coeffs_final, gcv_final, rss_final, mse_final)
        
        str_repr = str(record)
        assert "Earth Model Fit Record" in str_repr
        assert "Initial Samples: 3" in str_repr
        assert "Features: 2" in str_repr
        assert "Forward Pass completed" in str_repr
        assert "Pruning Pass Trace" in str_repr
        assert "Final Selected Model (after pruning):" in str_repr
        
    def test_record_with_different_dimensions(self):
        """Test EarthRecord with different dimensional data."""
        X = np.array([[1.0], [3.0], [5.0]])  # Single feature
        y = np.array([2.0, 6.0, 10.0])
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        assert record.n_samples == 3
        assert record.n_features == 1
        
        # Add data to all fields
        constant_bf = ConstantBasisFunction()
        record.log_forward_pass_step([constant_bf], np.array([np.mean(y)]), np.var(y))
        record.log_pruning_step([constant_bf], np.array([np.mean(y)]), 1.0, 1.0)
        record.set_final_model([constant_bf], np.array([np.mean(y)]), 1.0, 1.0, 0.5)
        
        # Verify data was stored
        assert len(record.fwd_basis_) == 1
        assert len(record.pruning_trace_basis_functions_) == 1
        assert record.final_basis_ is not None
    
    def test_record_with_multidimensional_data(self):
        """Test EarthRecord with high-dimensional data."""
        X = np.random.rand(10, 5)  # 10 samples, 5 features
        y = np.sum(X, axis=1)  # Dependent variable
        model = Earth(max_degree=2, penalty=3.0, max_terms=10)
        
        record = EarthRecord(X, y, model)
        
        assert record.n_samples == 10
        assert record.n_features == 5
        
        # Create multiple basis functions
        bfs = [ConstantBasisFunction()]
        for i in range(2):  # Add 2 linear functions
            bfs.append(LinearBasisFunction(variable_idx=i, variable_name=f"x{i}"))
        
        coeffs = np.array([1.0, 0.5, 0.3])
        rss = 2.0
        gcv = 1.5
        mse = 0.4
        
        # Test logging
        record.log_forward_pass_step(bfs, coeffs, rss)
        record.log_pruning_step(bfs, coeffs, gcv, rss)
        record.set_final_model(bfs, coeffs, gcv, rss, mse)
        
        # Verify everything was stored properly
        assert len(record.fwd_basis_[0]) == 3
        assert len(record.pruning_trace_basis_functions_[0]) == 3
        assert len(record.final_basis_) == 3
        assert len(record.final_coeffs_) == 3
        assert record.final_gcv_ == gcv
        assert record.final_rss_ == rss
        assert record.final_mse_ == mse
        

if __name__ == "__main__":
    test = TestEarthRecord()
    test.test_initialization()
    test.test_log_forward_pass_step()
    test.test_log_pruning_step()
    test.test_set_final_model()
    test.test_str_method_empty_record()
    test.test_str_method_with_forward_pass()
    test.test_str_method_with_pruning_trace()
    test.test_str_method_with_final_model()
    test.test_str_method_with_all_data()
    test.test_record_with_different_dimensions()
    test.test_record_with_multidimensional_data()
    
    print("All _record.py tests passed!")