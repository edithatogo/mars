"""
Test to specifically trigger the scenario where forward pass returns no basis functions
"""
import numpy as np
from unittest.mock import Mock, patch
from pymars import Earth


def test_forward_pass_returns_no_basis_functions():
    """Test the specific edge case where forward pass returns no basis functions."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([3.0, 7.0])
    
    model = Earth(max_degree=1, penalty=0.0, max_terms=1)
    
    # Mock the ForwardPasser to return empty basis functions
    with patch('pymars._forward.ForwardPasser') as mock_forward_passer_class:
        # Create a mock instance
        mock_forward_passer_instance = Mock()
        mock_forward_passer_instance.run.return_value = ([], np.array([]))  # No basis functions
        
        # Have the class return our mock instance
        mock_forward_passer_class.return_value = mock_forward_passer_instance
        
        # Fit the model - this should trigger the edge case code path
        model.fit(X, y)
        
        # Should still be fitted but with just intercept
        assert model.fitted_
        assert model.basis_ is not None  # Should have constant basis function as fallback


if __name__ == "__main__":
    test_forward_pass_returns_no_basis_functions()
    print("Successfully tested forward pass with no basis functions edge case!")