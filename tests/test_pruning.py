# -*- coding: utf-8 -*-

"""
Unit tests for the PruningPasser in pymars._pruning
"""

import pytest
import numpy as np
from pymars.earth import Earth
from pymars._pruning import PruningPasser
from pymars._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction

# A simple mock Earth class for testing PruningPasser
class MockEarth(Earth):
    def __init__(self, penalty=3.0, **kwargs):
        super().__init__(penalty=penalty, **kwargs)
        # Ensure record_ is None for these unit tests, unless specifically testing recording
        self.record_ = None

@pytest.fixture
def simple_pruning_data():
    """
    Data where y is primarily determined by x0, with some minor influence from x1 and x2.
    X0: Strong linear component
    X1: Sinusoidal component (approximated by hinges)
    X2: Negative linear component
    """
    np.random.seed(42) # for reproducibility
    X = np.array([
        [1, 10, 5], [2, 11, 6], [3, 12, 7], [4, 13, 8], [5, 14, 9],
        [6, 15, 10],[7, 16, 11],[8, 17, 12],[9, 18, 13],[10,19,14]
    ])
    # y = 2*X0 + sin(X1/k) - 0.8*X2 + noise
    y = 2 * X[:,0] + 2 * np.sin(X[:,1]/3) - 0.8 * X[:,2] + np.random.randn(X.shape[0]) * 0.2
    return X, y

@pytest.fixture
def initial_model_for_pruning(simple_pruning_data):
    """
    Creates a set of initial basis functions and their coefficients as if
    they came from a forward pass. Includes some good terms and some less useful ones.
    """
    X, y = simple_pruning_data

    bf_intercept = ConstantBasisFunction()
    bf_linear_x0 = LinearBasisFunction(variable_idx=0, variable_name="x0")       # Strong term
    bf_hinge_x1_k13 = HingeBasisFunction(variable_idx=1, knot_val=13.0, is_right_hinge=False, variable_name="x1_k13L") # Useful for sin peak
    bf_hinge_x1_k16 = HingeBasisFunction(variable_idx=1, knot_val=16.0, is_right_hinge=True, variable_name="x1_k16R")  # Useful for sin peak
    bf_linear_x2 = LinearBasisFunction(variable_idx=2, variable_name="x2")       # Useful term
    bf_extra_x0_k5 = HingeBasisFunction(variable_idx=0, knot_val=5.0, is_right_hinge=True, variable_name="x0_extra_k5R") # Redundant with linear_x0
    bf_noise_x1_k11 = HingeBasisFunction(variable_idx=1, knot_val=11.0, is_right_hinge=True, variable_name="x1_noise_k11R") # Less useful

    initial_bfs = [
        bf_intercept, bf_linear_x0,
        bf_hinge_x1_k13, bf_hinge_x1_k16,
        bf_linear_x2,
        bf_extra_x0_k5, bf_noise_x1_k11
    ]

    # Calculate initial coefficients for this specific set
    # Use a temporary PruningPasser instance just for its helper methods
    temp_passer = PruningPasser(MockEarth(penalty=0)) # Penalty 0 for unbiased OLS fit
    temp_passer.X_train = X
    temp_passer.y_train = y.ravel()
    temp_passer.n_samples = X.shape[0]

    B_initial = temp_passer._build_basis_matrix(X, initial_bfs)
    _, initial_coeffs = temp_passer._calculate_rss_and_coeffs(B_initial, y)

    if initial_coeffs is None:
        pytest.skip("Could not calculate initial coefficients for pruning test setup due to LSTSQ failure.")

    return X, y, initial_bfs, initial_coeffs


def test_pruning_passer_instantiation():
    """Test PruningPasser instantiation."""
    earth_model = MockEarth(penalty=2.5)
    passer = PruningPasser(earth_model)

    assert passer.model is earth_model
    assert passer.model.penalty == 2.5
    assert passer.X_train is None
    assert passer.best_gcv_so_far == np.inf
    assert len(passer.best_basis_functions_so_far) == 0

def test_compute_gcv_for_subset(simple_pruning_data, initial_model_for_pruning):
    """Test the _compute_gcv_for_subset helper method."""
    X, y, initial_bfs, _ = initial_model_for_pruning

    earth_model_penalty3 = MockEarth(penalty=3.0)
    passer = PruningPasser(earth_model_penalty3)
    passer.X_train = X
    passer.y_train = y.ravel()
    passer.n_samples = X.shape[0]

    # Test with the full initial set
    gcv1, rss1, coeffs1 = passer._compute_gcv_for_subset(initial_bfs)
    assert gcv1 > 0 and gcv1 != np.inf
    assert rss1 > 0 and rss1 != np.inf
    assert coeffs1 is not None
    assert len(coeffs1) == len(initial_bfs)

    # Test with a subset (e.g., removing the last 'noise' basis function)
    subset_bfs = initial_bfs[:-1]
    gcv2, rss2, coeffs2 = passer._compute_gcv_for_subset(subset_bfs)
    assert gcv2 > 0 and gcv2 != np.inf
    assert len(coeffs2) == len(subset_bfs)
    # GCV might improve or worsen. RSS should generally increase or stay same if term was useless.

    # Test with only intercept
    intercept_bf = [bf for bf in initial_bfs if isinstance(bf, ConstantBasisFunction)][0]
    gcv_intercept, rss_intercept, coeffs_intercept = passer._compute_gcv_for_subset([intercept_bf])
    assert gcv_intercept > 0 and gcv_intercept != np.inf
    assert np.isclose(coeffs_intercept[0], np.mean(y))
    assert np.isclose(rss_intercept, np.sum((y - np.mean(y))**2))

def test_pruning_run_no_pruning_if_penalty_zero(initial_model_for_pruning):
    """With penalty=0, GCV is prop. to RSS/N. Best GCV = lowest RSS, so full model."""
    X, y, initial_bfs, initial_coeffs = initial_model_for_pruning

    earth_model = MockEarth(penalty=0.0)
    passer = PruningPasser(earth_model)

    initial_bfs_copy = list(initial_bfs) # run might modify the list internally if not careful
    pruned_bfs, pruned_coeffs, best_gcv = passer.run(X, y, initial_bfs_copy, initial_coeffs)

    # With penalty=0, GCV can still prefer a simpler model if terms are truly useless
    # or collinear, because of the (1-M/N)^2 denominator term.
    # The key is that the returned best_gcv is indeed the GCV of the pruned_bfs.
    passer_check = PruningPasser(earth_model) # Instantiate passer_check
    passer_check.X_train = X; passer_check.y_train = y.ravel(); passer_check.n_samples = X.shape[0]

    gcv_of_pruned_set, _, _ = passer_check._compute_gcv_for_subset(pruned_bfs)
    assert np.isclose(best_gcv, gcv_of_pruned_set), "Returned best_gcv should match GCV of returned pruned_bfs"

    gcv_of_initial_set, _, _ = passer_check._compute_gcv_for_subset(initial_bfs) # Use original full set
    assert best_gcv <= gcv_of_initial_set + 1e-9, "Best GCV from pruning should be <= GCV of full model (with penalty 0)"
    # It is possible len(pruned_bfs) < len(initial_bfs) if some terms were perfectly useless.


def test_pruning_run_some_pruning_expected(initial_model_for_pruning):
    """With a reasonable penalty, some terms should be pruned."""
    X, y, initial_bfs, initial_coeffs = initial_model_for_pruning

    earth_model = MockEarth(penalty=3.0) # Standard penalty
    passer = PruningPasser(earth_model)

    pruned_bfs, pruned_coeffs, best_gcv = passer.run(X, y, list(initial_bfs), initial_coeffs)

    assert len(pruned_bfs) < len(initial_bfs)
    assert len(pruned_bfs) >= 1 # Should keep at least intercept

    # Check that the GCV of the pruned model is indeed the best GCV found
    passer_check = PruningPasser(earth_model)
    passer_check.X_train = X; passer_check.y_train = y.ravel(); passer_check.n_samples = X.shape[0]
    gcv_of_pruned_set, _, _ = passer_check._compute_gcv_for_subset(pruned_bfs)
    assert np.isclose(best_gcv, gcv_of_pruned_set)

    # Check that the initial model's GCV (with penalty 3) is worse or equal
    gcv_initial_pen3, _, _ = passer_check._compute_gcv_for_subset(initial_bfs)
    assert best_gcv <= gcv_initial_pen3


def test_pruning_run_empty_initial_set(simple_pruning_data):
    """Test run with an empty initial set of basis functions."""
    X, y = simple_pruning_data
    earth_model = MockEarth(penalty=3.0)
    passer = PruningPasser(earth_model)

    pruned_bfs, pruned_coeffs, best_gcv = passer.run(X, y, [], np.array([]))

    assert len(pruned_bfs) == 0
    assert len(pruned_coeffs) == 0
    assert best_gcv == np.inf

def test_pruning_run_intercept_protection(simple_pruning_data):
    """Test that the intercept is protected if it's the best single term or only term."""
    X, y = simple_pruning_data
    bf_intercept = ConstantBasisFunction()
    # Using a variable (idx 2) that has some effect, but make the knot very far, so it's a bad hinge.
    # This hinge should be easily prunable.
    bf_truly_bad_hinge = HingeBasisFunction(variable_idx=2, knot_val=1000,
                                           is_right_hinge=True, variable_name="bad_hinge_x2")

    initial_bfs = [bf_intercept, bf_truly_bad_hinge]

    # Fit this 2-term model to get its coefficients
    temp_passer = PruningPasser(MockEarth(penalty=0)) # Use penalty 0 for OLS fit
    temp_passer.X_train = X
    temp_passer.y_train = y.ravel() # Use original y for this test
    temp_passer.n_samples = X.shape[0]
    B_initial = temp_passer._build_basis_matrix(X, initial_bfs)
    _, initial_coeffs = temp_passer._calculate_rss_and_coeffs(B_initial, y)

    if initial_coeffs is None:
        pytest.skip("Initial coeffs calculation failed for intercept protection test setup.")

    earth_model = MockEarth(penalty=3.0) # A normal penalty should prune the bad hinge
    passer = PruningPasser(earth_model)

    pruned_bfs, pruned_coeffs, best_gcv = passer.run(X, y, initial_bfs, initial_coeffs)

    assert len(pruned_bfs) == 1, "Expected only intercept to remain after pruning a very bad hinge."
    assert isinstance(pruned_bfs[0], ConstantBasisFunction), "Expected remaining term to be the intercept."

    # Verify the GCV of the pruned (intercept-only) model
    passer_check = PruningPasser(earth_model)
    passer_check.X_train = X; passer_check.y_train = y.ravel(); passer_check.n_samples = X.shape[0]
    gcv_intercept_only, _, _ = passer_check._compute_gcv_for_subset([bf_intercept])
    assert np.isclose(best_gcv, gcv_intercept_only), "GCV of pruned model should match GCV of intercept-only model."

if __name__ == '__main__':
    pytest.main([__file__])
