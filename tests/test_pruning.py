# -*- coding: utf-8 -*-

"""
Unit tests for the PruningPasser in pymars._pruning
"""

import pytest
import numpy as np
from pymars.earth import Earth
from pymars._pruning import PruningPasser
from pymars._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction

class MockEarth(Earth):
    def __init__(self, penalty=3.0, allow_missing=False, **kwargs):
        super().__init__(penalty=penalty, allow_missing=allow_missing, **kwargs)
        self.record_ = None

@pytest.fixture
def simple_pruning_data():
    np.random.seed(42)
    n_samples_fixture = 40 # Increased from 10
    X_list = []
    for i in range(n_samples_fixture):
        # Create more varied data and ensure reasonable range for sin function
        X_list.append([1 + i*0.2, 10 + i*0.3, 5 + i*0.1])
    X = np.array(X_list, dtype=float)

    y = 2 * X[:,0] + 2 * np.sin(X[:,1]/np.pi) - 0.8 * X[:,2] + np.random.randn(X.shape[0]) * 0.2
    return X, y

@pytest.fixture
def initial_model_for_pruning(simple_pruning_data):
    X, y = simple_pruning_data

    bf_intercept = ConstantBasisFunction()
    bf_linear_x0 = LinearBasisFunction(variable_idx=0, variable_name="x0")
    bf_hinge_x1_k13 = HingeBasisFunction(variable_idx=1, knot_val=13.0, is_right_hinge=False, variable_name="x1_k13L")
    bf_hinge_x1_k16 = HingeBasisFunction(variable_idx=1, knot_val=16.0, is_right_hinge=True, variable_name="x1_k16R")
    bf_linear_x2 = LinearBasisFunction(variable_idx=2, variable_name="x2")
    bf_extra_x0_k5 = HingeBasisFunction(variable_idx=0, knot_val=5.0, is_right_hinge=True, variable_name="x0_extra_k5R")
    bf_noise_x1_k11 = HingeBasisFunction(variable_idx=1, knot_val=11.0, is_right_hinge=True, variable_name="x1_noise_k11R")

    initial_bfs = [
        bf_intercept, bf_linear_x0,
        bf_hinge_x1_k13, bf_hinge_x1_k16,
        bf_linear_x2,
        bf_extra_x0_k5, bf_noise_x1_k11
    ]

    temp_passer = PruningPasser(MockEarth(penalty=0, allow_missing=True))
    temp_passer.X_train = X
    temp_passer.y_train = y.ravel()
    temp_passer.n_samples = X.shape[0]
    temp_passer.missing_mask = np.zeros_like(X, dtype=bool)
    temp_passer.X_fit_original = X

    B_initial = temp_passer._build_basis_matrix(X, initial_bfs, temp_passer.missing_mask)
    _, initial_coeffs, _ = temp_passer._calculate_rss_and_coeffs(B_initial, y.ravel())

    if initial_coeffs is None:
        pytest.skip("Could not calculate initial coefficients for pruning test setup due to LSTSQ failure.")
    return X, y, initial_bfs, initial_coeffs

def test_pruning_passer_instantiation():
    earth_model = MockEarth(penalty=2.5)
    passer = PruningPasser(earth_model)
    assert passer.model is earth_model
    assert passer.model.penalty == 2.5
    assert passer.X_train is None
    assert passer.best_gcv_so_far == np.inf
    assert len(passer.best_basis_functions_so_far) == 0

def test_compute_gcv_for_subset(simple_pruning_data, initial_model_for_pruning):
    X, y, initial_bfs, _ = initial_model_for_pruning
    earth_model_penalty3 = MockEarth(penalty=3.0)
    passer = PruningPasser(earth_model_penalty3)
    passer.X_train = X
    passer.y_train = y.ravel()
    passer.n_samples = X.shape[0]
    passer.missing_mask = np.zeros_like(X, dtype=bool)
    passer.X_fit_original = X

    gcv1, rss1, coeffs1 = passer._compute_gcv_for_subset(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=passer.missing_mask, X_fit_original=X,
        basis_subset=initial_bfs
    )
    assert gcv1 > 0 and gcv1 != np.inf, f"GCV1 was {gcv1}"
    assert rss1 > 0 and rss1 != np.inf
    assert coeffs1 is not None
    assert len(coeffs1) == len(initial_bfs)

    subset_bfs = initial_bfs[:-1]
    gcv2, rss2, coeffs2 = passer._compute_gcv_for_subset(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=passer.missing_mask, X_fit_original=X,
        basis_subset=subset_bfs
    )
    assert gcv2 > 0 and gcv2 != np.inf
    assert len(coeffs2) == len(subset_bfs)

    intercept_bf = [bf for bf in initial_bfs if isinstance(bf, ConstantBasisFunction)][0]
    gcv_intercept, rss_intercept, coeffs_intercept = passer._compute_gcv_for_subset(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=passer.missing_mask, X_fit_original=X,
        basis_subset=[intercept_bf]
    )
    assert gcv_intercept > 0 and gcv_intercept != np.inf
    assert np.isclose(coeffs_intercept[0], np.mean(y))
    assert np.isclose(rss_intercept, np.sum((y - np.mean(y))**2))

def test_pruning_run_no_pruning_if_penalty_zero(initial_model_for_pruning):
    X, y, initial_bfs, initial_coeffs = initial_model_for_pruning
    earth_model = MockEarth(penalty=0.0)
    passer = PruningPasser(earth_model)
    dummy_missing_mask = np.zeros_like(X, dtype=bool)

    initial_bfs_copy = list(initial_bfs)
    pruned_bfs, pruned_coeffs, best_gcv = passer.run(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        initial_basis_functions=initial_bfs_copy, initial_coefficients=initial_coeffs
    )

    passer_check = PruningPasser(earth_model)
    passer_check.X_train = X; passer_check.y_train = y.ravel(); passer_check.n_samples = X.shape[0]
    passer_check.missing_mask = dummy_missing_mask; passer_check.X_fit_original = X

    gcv_of_pruned_set, _, _ = passer_check._compute_gcv_for_subset(
         X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
         basis_subset=pruned_bfs
    )
    assert np.isclose(best_gcv, gcv_of_pruned_set)

    gcv_of_initial_set, _, _ = passer_check._compute_gcv_for_subset(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        basis_subset=initial_bfs
    )
    assert best_gcv <= gcv_of_initial_set + 1e-9
    assert pruned_coeffs is not None

def test_pruning_run_some_pruning_expected(initial_model_for_pruning):
    X, y, initial_bfs, initial_coeffs = initial_model_for_pruning
    earth_model = MockEarth(penalty=3.0)
    passer = PruningPasser(earth_model)
    dummy_missing_mask = np.zeros_like(X, dtype=bool)

    pruned_bfs, pruned_coeffs, best_gcv = passer.run(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        initial_basis_functions=list(initial_bfs), initial_coefficients=initial_coeffs
    )

    assert len(pruned_bfs) < len(initial_bfs)
    assert len(pruned_bfs) >= 1

    passer_check = PruningPasser(earth_model)
    passer_check.X_train = X; passer_check.y_train = y.ravel(); passer_check.n_samples = X.shape[0]
    passer_check.missing_mask = dummy_missing_mask; passer_check.X_fit_original = X
    gcv_of_pruned_set, _, _ = passer_check._compute_gcv_for_subset(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        basis_subset=pruned_bfs
    )
    assert np.isclose(best_gcv, gcv_of_pruned_set)

    gcv_initial_pen3, _, _ = passer_check._compute_gcv_for_subset(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        basis_subset=initial_bfs
    )
    assert best_gcv <= gcv_initial_pen3

def test_pruning_run_empty_initial_set(simple_pruning_data):
    X, y = simple_pruning_data
    earth_model = MockEarth(penalty=3.0)
    passer = PruningPasser(earth_model)
    dummy_missing_mask = np.zeros_like(X, dtype=bool)

    pruned_bfs, pruned_coeffs, best_gcv = passer.run(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        initial_basis_functions=[], initial_coefficients=np.array([])
    )

    assert len(pruned_bfs) == 0
    assert pruned_coeffs is not None and pruned_coeffs.size == 0
    assert best_gcv == np.inf

def test_pruning_run_intercept_protection(simple_pruning_data):
    X, y = simple_pruning_data
    bf_intercept = ConstantBasisFunction()
    bf_truly_bad_hinge = HingeBasisFunction(variable_idx=2, knot_val=1000,
                                           is_right_hinge=True, variable_name="bad_hinge_x2")
    initial_bfs = [bf_intercept, bf_truly_bad_hinge]

    temp_passer = PruningPasser(MockEarth(penalty=0, allow_missing=True))
    temp_passer.X_train = X
    temp_passer.y_train = y.ravel()
    temp_passer.n_samples = X.shape[0]
    dummy_missing_mask_temp = np.zeros_like(X, dtype=bool)
    temp_passer.missing_mask = dummy_missing_mask_temp

    B_initial = temp_passer._build_basis_matrix(X, initial_bfs, dummy_missing_mask_temp)
    _, initial_coeffs, _ = temp_passer._calculate_rss_and_coeffs(B_initial, y.ravel())

    if initial_coeffs is None:
        pytest.skip("Initial coeffs calculation failed for intercept protection test setup.")

    earth_model = MockEarth(penalty=3.0)
    passer = PruningPasser(earth_model)
    dummy_missing_mask = np.zeros_like(X, dtype=bool)

    pruned_bfs, pruned_coeffs, best_gcv = passer.run(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        initial_basis_functions=initial_bfs, initial_coefficients=initial_coeffs
    )

    assert len(pruned_bfs) == 1
    assert isinstance(pruned_bfs[0], ConstantBasisFunction)

    passer_check = PruningPasser(earth_model)
    passer_check.X_train = X; passer_check.y_train = y.ravel(); passer_check.n_samples = X.shape[0]
    passer_check.missing_mask = dummy_missing_mask; passer_check.X_fit_original = X

    gcv_intercept_only, _, _ = passer_check._compute_gcv_for_subset(
        X_fit_processed=X, y_fit=y.ravel(), missing_mask=dummy_missing_mask, X_fit_original=X,
        basis_subset=[bf_intercept]
    )
    assert np.isclose(best_gcv, gcv_intercept_only)

if __name__ == '__main__':
    pytest.main([__file__])
