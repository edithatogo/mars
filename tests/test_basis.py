# -*- coding: utf-8 -*-

"""
Unit tests for basis functions in pymars._basis
"""

import pytest
import numpy as np
# from pymars._basis import (
#     BasisFunction,
#     ConstantBasisFunction,
#     HingeBasisFunction,
#     LinearBasisFunction,
#     # InteractionBasisFunction # If implemented
# )

def test_basis_module_importable():
    """Test that the _basis module can be imported."""
    try:
        from pymars import _basis
        assert _basis is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pymars._basis: {e}")

def test_constant_basis_function():
    """Test ConstantBasisFunction."""
    from pymars._basis import ConstantBasisFunction
    bf = ConstantBasisFunction()
    X_sample = np.array([[1,2],[3,4]])
    transformed_X = bf.transform(X_sample)
    assert np.all(transformed_X == 1.0)
    assert str(bf) == "Intercept"
    assert bf.degree() == 0 # Or 1, depending on convention used in Earth class

def test_hinge_basis_function_right():
    """Test HingeBasisFunction (right hinge)."""
    from pymars._basis import HingeBasisFunction
    # h(x0 - 2.0)
    bf = HingeBasisFunction(variable_idx=0, knot_val=2.0, is_right_hinge=True, variable_name="X0")
    X_sample = np.array([[1,5],[3,5],[0,5]]) # X0 values: 1, 3, 0
    expected = np.array([
        max(0, 1 - 2.0), # 0
        max(0, 3 - 2.0), # 1
        max(0, 0 - 2.0)  # 0
    ])
    transformed_X = bf.transform(X_sample)
    assert np.allclose(transformed_X, expected)
    assert str(bf) == "max(0, X0 - 2.00)"
    assert bf.degree() == 1

def test_hinge_basis_function_left():
    """Test HingeBasisFunction (left hinge)."""
    from pymars._basis import HingeBasisFunction
    # h(2.0 - x0)
    bf = HingeBasisFunction(variable_idx=0, knot_val=2.0, is_right_hinge=False, variable_name="X0")
    X_sample = np.array([[1,5],[3,5],[0,5]]) # X0 values: 1, 3, 0
    expected = np.array([
        max(0, 2.0 - 1), # 1
        max(0, 2.0 - 3), # 0
        max(0, 2.0 - 0)  # 2
    ])
    transformed_X = bf.transform(X_sample)
    assert np.allclose(transformed_X, expected)
    assert str(bf) == "max(0, 2.00 - X0)"

def test_linear_basis_function():
    """Test LinearBasisFunction."""
    from pymars._basis import LinearBasisFunction
    bf = LinearBasisFunction(variable_idx=1, variable_name="X1") # Operates on X1 (second column)
    X_sample = np.array([[10, 1],[20, 2],[30, 3]])
    expected = np.array([1, 2, 3])
    transformed_X = bf.transform(X_sample)
    assert np.allclose(transformed_X, expected)
    assert str(bf) == "X1"
    assert bf.degree() == 1

# Placeholder for InteractionBasisFunction tests
# def test_interaction_basis_function():
#     """Test InteractionBasisFunction."""
#     from pymars._basis import InteractionBasisFunction, HingeBasisFunction, LinearBasisFunction
#     X_sample = np.array([[1,2,3],[4,5,6]])
#
#     h_bf = HingeBasisFunction(0, 2.0) # h(x0-2) -> [max(0,1-2), max(0,4-2)] -> [0, 2]
#     l_bf = LinearBasisFunction(1)     # x1      -> [2, 5]
#
#     inter_bf = InteractionBasisFunction(h_bf, l_bf)
#     expected = np.array([0 * 2, 2 * 5]) # [0, 10]
#     transformed_X = inter_bf.transform(X_sample)
#
#     assert np.allclose(transformed_X, expected)
#     assert str(inter_bf) == "(max(0, x0 - 2.00)) * (x1)"
#     assert inter_bf.degree() == h_bf.degree() + l_bf.degree()


if __name__ == '__main__':
    # pytest.main([__file__])
    print("Run tests using 'pytest tests/test_basis.py'")
