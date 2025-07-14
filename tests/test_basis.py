# -*- coding: utf-8 -*-

"""
Unit tests for the basis functions in pymars._basis
"""
import pytest
import numpy as np
from pymars._basis import BasisFunction, ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction

# --- Test ConstantBasisFunction ---
def test_constant_basis_function():
    """Test ConstantBasisFunction."""
    bf = ConstantBasisFunction()

    assert bf.get_name() == "Intercept"
    assert bf.is_linear_term is False
    assert bf.is_hinge_term is False
    assert bf.variable_idx is None
    assert bf.knot_val is None
    assert bf.degree() == 0
    assert bf.is_constant() is True
    assert bf.get_involved_variables() == frozenset()

    X_2d = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=float)
    dummy_missing_mask_2d = np.zeros_like(X_2d, dtype=bool)
    transformed_2d = bf.transform(X_2d, dummy_missing_mask_2d)
    assert np.array_equal(transformed_2d, np.ones(X_2d.shape[0]))

    X_1d = np.array([1,2,3,4,5], dtype=float)
    dummy_missing_mask_1d = np.zeros_like(X_1d, dtype=bool) # For 1D X, mask should be (N,) or (N,1)
    if X_1d.ndim == 1 and dummy_missing_mask_1d.ndim == 2 and dummy_missing_mask_1d.shape[1]==1:
        dummy_missing_mask_1d = dummy_missing_mask_1d.ravel()
    elif X_1d.ndim == 1 and dummy_missing_mask_1d.ndim == 1:
        pass # it's fine
    else: # if mask is 2d and X is 1d, adjust mask for consistency if needed by basis
        dummy_missing_mask_1d = np.zeros(X_1d.shape[0], dtype=bool)


    transformed_1d = bf.transform(X_1d, dummy_missing_mask_1d)
    assert np.array_equal(transformed_1d, np.ones(X_1d.shape[0]))

    with pytest.raises(TypeError):
        bf.transform("not an array", np.array([False])) # type: ignore
    with pytest.raises(ValueError):
        bf.transform(np.array([[[1]]]), np.array([[[False]]])) # 3D array

@pytest.mark.parametrize("is_right", [True, False])
@pytest.mark.parametrize("parent_type", [None, "Constant", "Hinge", "Linear"])
def test_hinge_basis_function(is_right, parent_type):
    base_var_idx, base_knot = 0, 2.5
    base_var_name = "X0"

    parent_bf = None
    expected_parent_str = ""
    parent_degree = 0
    parent_involved_vars = frozenset()

    if parent_type == "Constant":
        parent_bf = ConstantBasisFunction()
        expected_parent_str = f"({str(parent_bf)}) * "
    elif parent_type == "Hinge":
        parent_bf = HingeBasisFunction(variable_idx=1, knot_val=1.5, is_right_hinge=True, variable_name="X1")
        expected_parent_str = f"({str(parent_bf)}) * "
        parent_degree = 1
        parent_involved_vars = {1}
    elif parent_type == "Linear":
        parent_bf = LinearBasisFunction(variable_idx=1, variable_name="X1")
        expected_parent_str = f"({str(parent_bf)}) * "
        parent_degree = 1
        parent_involved_vars = {1}

    bf = HingeBasisFunction(variable_idx=base_var_idx, knot_val=base_knot,
                            is_right_hinge=is_right, variable_name=base_var_name,
                            parent_bf=parent_bf)

    assert bf.variable_idx == base_var_idx
    assert bf.knot_val == base_knot
    assert bf.is_right_hinge == is_right
    assert bf.is_hinge_term is True
    assert bf.is_linear_term is False
    assert bf.parent1 == parent_bf

    expected_hinge_str = f"max(0, {base_var_name} - {base_knot:.2f})" if is_right else f"max(0, {base_knot:.2f} - {base_var_name})"
    expected_name = expected_parent_str + expected_hinge_str
    assert str(bf) == expected_name
    assert bf.degree() == parent_degree + 1
    assert bf.is_constant() is False
    assert bf.get_involved_variables() == parent_involved_vars.union({base_var_idx})

    X_data = np.array([[1.0, 0.0], [2.5, 1.5], [3.0, 3.5], [0.0, -1.0]], dtype=float)
    dummy_missing_mask = np.zeros_like(X_data, dtype=bool)

    parent_transform_values = np.ones(X_data.shape[0])
    if parent_bf and not isinstance(parent_bf, ConstantBasisFunction):
        parent_transform_values = parent_bf.transform(X_data, dummy_missing_mask)

    x_col_for_hinge = X_data[:, base_var_idx]
    if is_right:
        hinge_values = np.maximum(0, x_col_for_hinge - base_knot)
    else:
        hinge_values = np.maximum(0, base_knot - x_col_for_hinge)

    expected_transform = parent_transform_values * hinge_values
    transformed_values = bf.transform(X_data, dummy_missing_mask)
    assert np.allclose(transformed_values, expected_transform, equal_nan=True)

    X_data_nan = X_data.copy()
    X_data_nan[0, base_var_idx] = np.nan
    missing_mask_nan = np.isnan(X_data_nan)

    X_processed_nan = X_data_nan.copy()
    X_processed_nan[missing_mask_nan] = 0.0

    transformed_nan = bf.transform(X_processed_nan, missing_mask_nan)
    assert np.isnan(transformed_nan[0])
    if parent_type is None:
         assert not np.isnan(transformed_nan[1])

@pytest.mark.parametrize("parent_type", [None, "Constant", "Hinge", "Linear"])
def test_linear_basis_function(parent_type):
    base_var_idx = 0
    base_var_name = "X0"

    parent_bf = None
    expected_parent_str = ""
    parent_degree = 0
    parent_involved_vars = frozenset()

    if parent_type == "Constant":
        parent_bf = ConstantBasisFunction()
        expected_parent_str = f"({str(parent_bf)}) * "
    elif parent_type == "Hinge":
        parent_bf = HingeBasisFunction(variable_idx=1, knot_val=1.5, is_right_hinge=True, variable_name="X1")
        expected_parent_str = f"({str(parent_bf)}) * "
        parent_degree = 1
        parent_involved_vars = {1}
    elif parent_type == "Linear":
        parent_bf = LinearBasisFunction(variable_idx=1, variable_name="X1")
        expected_parent_str = f"({str(parent_bf)}) * "
        parent_degree = 1
        parent_involved_vars = {1}

    bf = LinearBasisFunction(variable_idx=base_var_idx, variable_name=base_var_name, parent_bf=parent_bf)

    assert bf.variable_idx == base_var_idx
    assert bf.variable_name == base_var_name
    assert bf.is_linear_term is True
    assert bf.is_hinge_term is False
    assert bf.parent1 == parent_bf
    expected_name = expected_parent_str + base_var_name
    assert str(bf) == expected_name
    assert bf.degree() == parent_degree + 1
    assert bf.is_constant() is False
    assert bf.get_involved_variables() == parent_involved_vars.union({base_var_idx})

    X_data = np.array([[1.0, 10.0], [2.5, 20.0], [3.0, 30.0]], dtype=float)
    dummy_missing_mask = np.zeros_like(X_data, dtype=bool)

    parent_transform_values = np.ones(X_data.shape[0])
    if parent_bf and not isinstance(parent_bf, ConstantBasisFunction):
        parent_transform_values = parent_bf.transform(X_data, dummy_missing_mask)

    linear_values = X_data[:, base_var_idx]
    expected_transform = parent_transform_values * linear_values

    transformed_values = bf.transform(X_data, dummy_missing_mask)
    assert np.allclose(transformed_values, expected_transform, equal_nan=True)

    X_data_nan = X_data.copy()
    X_data_nan[0, base_var_idx] = np.nan
    missing_mask_nan = np.isnan(X_data_nan)

    X_processed_nan = X_data_nan.copy()
    X_processed_nan[missing_mask_nan] = 0.0

    transformed_nan = bf.transform(X_processed_nan, missing_mask_nan)
    assert np.isnan(transformed_nan[0])
    if parent_type is None:
        assert not np.isnan(transformed_nan[1])

# --- Test MissingnessBasisFunction ---
def test_missingness_basis_function():
    """Test MissingnessBasisFunction."""
    from pymars._basis import MissingnessBasisFunction # Import locally for this test

    var_idx, var_name = 0, "FeatureA"
    bf = MissingnessBasisFunction(variable_idx=var_idx, variable_name=var_name)

    assert bf.variable_idx == var_idx
    assert bf.variable_name == var_name
    assert str(bf) == f"is_missing({var_name})"
    assert bf.degree() == 1
    assert bf.is_constant() is False
    assert bf.get_involved_variables() == {var_idx}
    assert bf.is_linear_term is False
    assert bf.is_hinge_term is False
    assert bf.knot_val is None
    assert bf.parent1 is None

    # Test transform method
    # X_processed is not directly used by transform, but its shape might be checked by other parts if we used a mock X
    # missing_mask is key

    # Scenario 1: No missing values for the monitored variable
    X_dummy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 3 samples, 2 features
    missing_mask_none = np.array([
        [False, False],
        [False, False],
        [False, False]
    ], dtype=bool)
    transformed_none = bf.transform(X_dummy, missing_mask_none)
    assert np.array_equal(transformed_none, np.array([0, 0, 0]))

    # Scenario 2: Some missing values for the monitored variable (var_idx=0)
    missing_mask_some = np.array([
        [True,  False], # FeatureA missing for sample 0
        [False, True],  # FeatureB missing for sample 1 (should not affect bf for FeatureA)
        [True,  False]  # FeatureA missing for sample 2
    ], dtype=bool)
    transformed_some = bf.transform(X_dummy, missing_mask_some)
    assert np.array_equal(transformed_some, np.array([1, 0, 1]))

    # Scenario 3: All missing values for the monitored variable
    missing_mask_all = np.array([
        [True, False],
        [True, False],
        [True, False]
    ], dtype=bool)
    transformed_all = bf.transform(X_dummy, missing_mask_all)
    assert np.array_equal(transformed_all, np.array([1, 1, 1]))

    # Scenario 4: Different variable index
    bf_var1 = MissingnessBasisFunction(variable_idx=1, variable_name="FeatureB")
    transformed_var1 = bf_var1.transform(X_dummy, missing_mask_some) # Using mask where var_idx=1 has a missing value
    assert np.array_equal(transformed_var1, np.array([0, 1, 0]))


    # Test input validation for transform
    with pytest.raises(TypeError, match="Input missing_mask must be a numpy array."):
        bf.transform(X_dummy, "not a mask") # type: ignore
    with pytest.raises(ValueError, match="Input missing_mask must be 2D"):
        bf.transform(X_dummy, np.array([False, True])) # 1D mask for 2D X
    with pytest.raises(IndexError, match="variable_idx 0 is out of bounds for missing_mask with 0 features"):
        bf.transform(X_dummy, np.empty((X_dummy.shape[0],0), dtype=bool) ) # Mask with 0 features

    bf_high_idx = MissingnessBasisFunction(variable_idx=5, variable_name="FeatureF")
    with pytest.raises(IndexError, match="variable_idx 5 is out of bounds for missing_mask with 2 features"):
        bf_high_idx.transform(X_dummy, missing_mask_none)


if __name__ == '__main__':
    pytest.main()
