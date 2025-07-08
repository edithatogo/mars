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

    # Test __init__
    assert bf.get_name() == "Intercept"
    assert bf.is_linear_term is False
    assert bf.is_hinge_term is False
    assert bf.variable_idx is None
    assert bf.knot_val is None

    # Test transform method
    X_2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
    transformed_2d = bf.transform(X_2d)
    assert isinstance(transformed_2d, np.ndarray), "Transform should return a numpy array"
    assert transformed_2d.shape == (X_2d.shape[0],), "Shape of transformed output is incorrect for 2D input"
    assert np.all(transformed_2d == 1.0), "All values should be 1.0 for 2D input"

    X_1d = np.array([10, 11, 12, 13])
    transformed_1d = bf.transform(X_1d)
    assert transformed_1d.shape == (X_1d.shape[0],), "Shape of transformed output is incorrect for 1D input"
    assert np.all(transformed_1d == 1.0), "All values should be 1.0 for 1D input"

    X_empty = np.array([[]]) # Should handle 0-sample case gracefully if shape[0] is used
    if X_empty.shape[1] == 0: #This case is tricky, if X_empty = np.empty((0,N)) then shape[0] is 0.
         X_empty_0_samples = np.empty((0,3))
         transformed_empty = bf.transform(X_empty_0_samples)
         assert transformed_empty.shape == (0,), "Should return empty array for 0 samples"

    # Test transform with invalid input types
    with pytest.raises(TypeError, match="Input X must be a numpy array"):
        bf.transform([[1,2],[3,4]]) # list of lists

    with pytest.raises(ValueError, match="Input X must be 1D or 2D"):
        bf.transform(np.array([[[1],[2]],[[3],[4]]])) # 3D array

    # Test __str__ method
    assert str(bf) == "Intercept"

    # Test degree method
    assert bf.degree() == 0, "Degree of ConstantBasisFunction should be 0"

def test_hinge_basis_function_right():
    """Test HingeBasisFunction (right hinge, degree 1)."""
    from pymars._basis import HingeBasisFunction
    var_idx, knot = 0, 2.5
    var_name = "FeatA"

    bf = HingeBasisFunction(variable_idx=var_idx, knot_val=knot, is_right_hinge=True, variable_name=var_name)

    # Test __init__
    assert bf.variable_idx == var_idx
    assert bf.knot_val == knot
    assert bf.is_right_hinge is True
    assert bf.is_hinge_term is True
    assert bf.is_linear_term is False
    assert bf.parent1 is None
    expected_name = f"max(0, {var_name} - {knot:.2f})"
    assert str(bf) == expected_name
    assert bf.get_name() == expected_name
    assert bf.degree() == 1

    # Test transform
    X_2d = np.array([[1.0, 10.0], [2.5, 20.0], [3.0, 30.0], [0.0, 40.0]]) # Var0 values: 1.0, 2.5, 3.0, 0.0
    expected_transform_2d = np.array([
        max(0, 1.0 - knot), # 0
        max(0, 2.5 - knot), # 0
        max(0, 3.0 - knot), # 0.5
        max(0, 0.0 - knot)  # 0
    ])
    transformed_2d = bf.transform(X_2d)
    assert isinstance(transformed_2d, np.ndarray)
    assert transformed_2d.shape == (X_2d.shape[0],)
    assert np.allclose(transformed_2d, expected_transform_2d)

    X_1d = np.array([1.0, 2.5, 3.0, 0.0])
    # Recreate bf for 1D case if variable_idx needs to be 0, or ensure HingeBasisFunction handles it.
    # Current HingeBasisFunction requires variable_idx=0 for 1D X.
    bf_1d = HingeBasisFunction(variable_idx=0, knot_val=knot, is_right_hinge=True, variable_name=var_name)
    transformed_1d = bf_1d.transform(X_1d)
    assert transformed_1d.shape == (X_1d.shape[0],)
    assert np.allclose(transformed_1d, expected_transform_2d) # Same logic applies

    # Test error handling for transform
    with pytest.raises(TypeError, match="Input X must be a numpy array"):
        bf.transform([[1,2],[3,4]])
    with pytest.raises(ValueError, match="Input X must be 1D or 2D"):
        bf.transform(np.zeros((2,2,2)))
    with pytest.raises(IndexError): # variable_idx out of bounds
        bf.transform(np.zeros((5,0))) # X with 0 features
    bf_idx1 = HingeBasisFunction(variable_idx=1, knot_val=knot, is_right_hinge=True, variable_name="FeatB")
    with pytest.raises(IndexError):
         bf_idx1.transform(np.zeros((5,1))) # X with 1 feature, bf wants index 1

    # Test with 1D X and variable_idx != 0
    bf_for_1d_fail = HingeBasisFunction(variable_idx=1, knot_val=knot, is_right_hinge=True, variable_name=var_name)
    with pytest.raises(ValueError, match="For 1D X input, variable_idx must be 0"):
        bf_for_1d_fail.transform(X_1d)


def test_hinge_basis_function_left():
    """Test HingeBasisFunction (left hinge, degree 1)."""
    from pymars._basis import HingeBasisFunction
    var_idx, knot = 1, -0.5 # Using second variable
    var_name = "FeatB"

    bf = HingeBasisFunction(variable_idx=var_idx, knot_val=knot, is_right_hinge=False, variable_name=var_name)

    # Test __init__
    assert bf.variable_idx == var_idx
    assert bf.knot_val == knot
    assert bf.is_right_hinge is False
    assert bf.is_hinge_term is True
    expected_name = f"max(0, {knot:.2f} - {var_name})"
    assert str(bf) == expected_name
    assert bf.degree() == 1

    # Test transform
    X_2d = np.array([[10.0, -1.0], [20.0, -0.5], [30.0, 0.0], [40.0, 1.0]]) # Var1 values: -1.0, -0.5, 0.0, 1.0
    expected_transform_2d = np.array([
        max(0, knot - (-1.0)), # max(0, -0.5 + 1.0) = 0.5
        max(0, knot - (-0.5)), # max(0, -0.5 + 0.5) = 0
        max(0, knot - 0.0),    # max(0, -0.5 - 0.0) = 0
        max(0, knot - 1.0)     # max(0, -0.5 - 1.0) = 0
    ])
    transformed_2d = bf.transform(X_2d)
    assert np.allclose(transformed_2d, expected_transform_2d)


def test_hinge_basis_function_interaction():
    """Test HingeBasisFunction as an interaction term."""
    from pymars._basis import HingeBasisFunction, ConstantBasisFunction

    # Parent is a constant function (not typical but tests mechanism)
    # This would mean the "hinge" is just scaled by 1.
    parent = ConstantBasisFunction()
    var_idx, knot = 0, 1.0
    var_name = "X0"

    # Interaction: Intercept * max(0, X0 - 1.0)
    bf_inter = HingeBasisFunction(variable_idx=var_idx, knot_val=knot,
                                  is_right_hinge=True, variable_name=var_name, parent_bf=parent)

    assert bf_inter.parent1 is parent
    assert bf_inter.degree() == parent.degree() + 1 # 0 + 1 = 1
    expected_name = f"({str(parent)}) * max(0, {var_name} - {knot:.2f})"
    assert str(bf_inter) == expected_name

    X_sample = np.array([[0.5, 10], [1.0, 20], [1.5, 30]]) # X0 values: 0.5, 1.0, 1.5

    # Expected: parent.transform(X) * hinge_on_X0
    # parent.transform(X) is [1,1,1]
    # hinge_on_X0 is [max(0,0.5-1), max(0,1.0-1), max(0,1.5-1)] = [0, 0, 0.5]
    # result: [1*0, 1*0, 1*0.5] = [0, 0, 0.5]
    expected_transform = np.array([0, 0, 0.5])
    transformed_inter = bf_inter.transform(X_sample)
    assert np.allclose(transformed_inter, expected_transform)

    # Test interaction with another hinge
    parent_hinge = HingeBasisFunction(variable_idx=1, knot_val=20.0, is_right_hinge=False, variable_name="X1") # max(0, 20-X1)
    # bf_inter2: max(0, 20-X1) * max(0, X0 - 1.0)
    bf_inter2 = HingeBasisFunction(variable_idx=var_idx, knot_val=knot,
                                   is_right_hinge=True, variable_name=var_name, parent_bf=parent_hinge)

    assert bf_inter2.degree() == parent_hinge.degree() + 1 # 1 + 1 = 2
    expected_name2 = f"({str(parent_hinge)}) * max(0, {var_name} - {knot:.2f})"
    assert str(bf_inter2) == expected_name2

    # X_sample = np.array([[0.5, 10], [1.0, 20], [1.5, 30]])
    # parent_hinge transform: [max(0,20-10), max(0,20-20), max(0,20-30)] = [10, 0, 0]
    # current_hinge transform (on X0): [0, 0, 0.5] (from above)
    # bf_inter2 transform: [10*0, 0*0, 0*0.5] = [0,0,0]
    expected_transform2 = np.array([0,0,0])
    transformed_inter2 = bf_inter2.transform(X_sample)
    assert np.allclose(transformed_inter2, expected_transform2)


def test_linear_basis_function():
    """Test LinearBasisFunction (degree 1)."""
    from pymars._basis import LinearBasisFunction
    var_idx = 1
    var_name = "VarOne"

    bf = LinearBasisFunction(variable_idx=var_idx, variable_name=var_name)

    # Test __init__
    assert bf.variable_idx == var_idx
    assert bf.variable_name == var_name
    assert bf.is_linear_term is True
    assert bf.is_hinge_term is False
    assert bf.parent1 is None
    assert str(bf) == var_name
    assert bf.get_name() == var_name
    assert bf.degree() == 1

    # Test transform
    X_2d = np.array([[100, 10, 1000], [200, 20, 2000], [300, 30, 3000]]) # Var1 values: 10,20,30
    expected_transform_2d = np.array([10, 20, 30])
    transformed_2d = bf.transform(X_2d)
    assert isinstance(transformed_2d, np.ndarray)
    assert transformed_2d.shape == (X_2d.shape[0],)
    assert np.allclose(transformed_2d, expected_transform_2d)

    X_1d = np.array([-5.5, 0.0, 5.5])
    bf_1d = LinearBasisFunction(variable_idx=0, variable_name="VarZero") # Must use var_idx=0 for 1D X
    transformed_1d = bf_1d.transform(X_1d)
    assert transformed_1d.shape == (X_1d.shape[0],)
    assert np.allclose(transformed_1d, X_1d)

    # Test error handling for transform
    with pytest.raises(TypeError, match="Input X must be a numpy array"):
        bf.transform([[1,2],[3,4]])
    with pytest.raises(ValueError, match="Input X must be 1D or 2D"):
        bf.transform(np.zeros((2,2,2)))
    with pytest.raises(IndexError): # variable_idx out of bounds
        bf.transform(np.zeros((5,1))) # bf is for var_idx=1

    bf_idx0 = LinearBasisFunction(variable_idx=0, variable_name="VarZero")
    with pytest.raises(IndexError):
        bf_idx0.transform(np.zeros((5,0))) # X with 0 features

    bf_for_1d_fail = LinearBasisFunction(variable_idx=1, variable_name="VarOne")
    with pytest.raises(ValueError, match="For 1D X input, variable_idx must be 0"):
        bf_for_1d_fail.transform(X_1d)


def test_linear_basis_function_interaction():
    """Test LinearBasisFunction as an interaction term."""
    from pymars._basis import LinearBasisFunction, HingeBasisFunction

    parent_hinge = HingeBasisFunction(variable_idx=0, knot_val=5.0, is_right_hinge=True, variable_name="H0") # max(0, H0-5)

    inter_var_idx = 1
    inter_var_name = "L1"

    # Interaction: max(0, H0-5) * L1
    bf_inter = LinearBasisFunction(variable_idx=inter_var_idx, variable_name=inter_var_name, parent_bf=parent_hinge)

    assert bf_inter.parent1 is parent_hinge
    assert bf_inter.degree() == parent_hinge.degree() + 1 # 1 + 1 = 2
    expected_name = f"({str(parent_hinge)}) * {inter_var_name}"
    assert str(bf_inter) == expected_name

    X_sample = np.array([[4.0, 10.0], [5.0, 20.0], [6.0, 30.0]])
    # H0 values: 4,5,6. L1 values: 10,20,30

    # parent_hinge.transform(X_sample): [max(0,4-5), max(0,5-5), max(0,6-5)] = [0, 0, 1]
    # linear_term (L1): [10, 20, 30]
    # Expected result: [0*10, 0*20, 1*30] = [0, 0, 30]
    expected_transform = np.array([0, 0, 30.0])
    transformed_inter = bf_inter.transform(X_sample)
    assert np.allclose(transformed_inter, expected_transform)


# Placeholder for InteractionBasisFunction tests (a more generic one)
# def test_generic_interaction_basis_function():
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
