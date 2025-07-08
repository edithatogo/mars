# -*- coding: utf-8 -*-

"""
Unit tests for the ForwardPasser in pymars._forward
"""

import pytest
# import numpy as np
# from pymars._forward import ForwardPasser
# from pymars.earth import Earth # Or a mock Earth

def test_forward_module_importable():
    """Test that the _forward module can be imported."""
    try:
        from pymars import _forward
        assert _forward is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pymars._forward: {e}")

import numpy as np
from pymars.earth import Earth # Using the actual Earth class for instantiation
from pymars._forward import ForwardPasser
from pymars._basis import ConstantBasisFunction, HingeBasisFunction

# A simple mock Earth class for testing ForwardPasser in isolation where needed
class MockEarth(Earth):
    def __init__(self, max_degree=1, max_terms=10, minspan_alpha=0.0, endspan_alpha=0.0, penalty=3.0):
        super().__init__(max_degree=max_degree, penalty=penalty, max_terms=max_terms,
                         minspan_alpha=minspan_alpha, endspan_alpha=endspan_alpha)
        self.record_ = None # Mock: no recording for these unit tests unless specifically tested

@pytest.fixture
def simple_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 5.5, 8.5, 10.0]) # Approx 2*X, with some non-linearity
    return X, y

@pytest.fixture
def multi_feature_data():
    X = np.array([[1,10], [2,20], [3,15], [4,25], [5,12]])
    y = X[:,0] * 2 + X[:,1] * 0.5 + np.random.randn(5) * 0.1
    return X,y


def test_forward_passer_instantiation_and_initial_state(simple_data):
    """Test ForwardPasser instantiation and initial state variables."""
    X, y = simple_data
    earth_model = MockEarth(max_degree=1, max_terms=5)
    passer = ForwardPasser(earth_model)

    assert passer.model is earth_model
    assert passer.X_train is None # Not set until run()
    assert passer.y_train is None
    assert passer.current_basis_functions == [] # Empty list before run()
    assert passer.current_rss == np.inf

def test_initial_model_setup_in_run(simple_data):
    """Test the setup of the initial (intercept-only) model in run()."""
    X, y = simple_data
    earth_model = MockEarth(max_degree=1, max_terms=3)
    passer = ForwardPasser(earth_model)

    # Call run, but we are only testing the initial setup part for now.
    # The run method will try to loop, but with no way to generate further candidates yet
    # or if max_terms is small, it might stop after intercept.
    # For this test, we can mock _find_best_candidate_pair to stop iteration.

    def mock_find_best_candidate_pair():
        passer._best_candidate_pair = None # Simulate no improvement
        passer._min_candidate_rss = passer.current_rss

    original_find_best = passer._find_best_candidate_pair
    passer._find_best_candidate_pair = mock_find_best_candidate_pair

    bfs, coeffs = passer.run(X, y)

    passer._find_best_candidate_pair = original_find_best # Restore

    assert len(passer.current_basis_functions) == 1
    assert isinstance(passer.current_basis_functions[0], ConstantBasisFunction)
    assert passer.current_B_matrix is not None
    assert passer.current_B_matrix.shape == (X.shape[0], 1)
    assert np.all(passer.current_B_matrix == 1.0)
    assert passer.current_coefficients is not None
    assert len(passer.current_coefficients) == 1
    assert np.isclose(passer.current_coefficients[0], np.mean(y))
    expected_rss_intercept_only = np.sum((y - np.mean(y))**2)
    assert np.isclose(passer.current_rss, expected_rss_intercept_only)

    # Check returned values from run (as it stops early due to mock)
    assert len(bfs) == 1
    assert isinstance(bfs[0], ConstantBasisFunction)
    assert np.isclose(coeffs[0], np.mean(y))


def test_build_basis_matrix(simple_data):
    X, _ = simple_data
    earth_model = MockEarth()
    passer = ForwardPasser(earth_model)
    passer.X_train = X # Manually set for this test

    bf_const = ConstantBasisFunction()
    bf_hinge = HingeBasisFunction(variable_idx=0, knot_val=3.0, variable_name="x0")

    # Test with empty list
    B_empty = passer._build_basis_matrix(X, [])
    assert B_empty.shape == (X.shape[0], 0)

    # Test with one basis function
    B_one = passer._build_basis_matrix(X, [bf_const])
    assert B_one.shape == (X.shape[0], 1)
    assert np.all(B_one == 1.0)

    # Test with multiple basis functions
    B_two = passer._build_basis_matrix(X, [bf_const, bf_hinge])
    assert B_two.shape == (X.shape[0], 2)
    expected_col1 = np.ones(X.shape[0])
    expected_col2 = np.maximum(0, X[:,0] - 3.0)
    assert np.allclose(B_two[:,0], expected_col1)
    assert np.allclose(B_two[:,1], expected_col2)

def test_calculate_rss_and_coeffs(simple_data):
    X, y = simple_data
    earth_model = MockEarth()
    passer = ForwardPasser(earth_model)
    passer.X_train = X # Needed for _build_basis_matrix if used indirectly

    # Intercept only
    bf_const = ConstantBasisFunction()
    B_const = passer._build_basis_matrix(X, [bf_const])
    rss_const, coeffs_const = passer._calculate_rss_and_coeffs(B_const, y)
    assert np.isclose(coeffs_const[0], np.mean(y))
    assert np.isclose(rss_const, np.sum((y - np.mean(y))**2))

    # Intercept + one hinge
    bf_hinge = HingeBasisFunction(variable_idx=0, knot_val=2.5, variable_name="x0_k2.5")
    B_two = passer._build_basis_matrix(X, [bf_const, bf_hinge])
    rss_two, coeffs_two = passer._calculate_rss_and_coeffs(B_two, y)
    assert coeffs_two is not None
    assert len(coeffs_two) == 2
    # Manual calculation for this specific case:
    # y = c0*1 + c1*max(0, x-2.5)
    # X = [1,2,3,4,5]', y = [2,4,5.5,8.5,10]'
    # Hinge values for X: [0,0,0.5,1.5,2.5]'
    # B_two = [[1,0],[1,0],[1,0.5],[1,1.5],[1,2.5]]
    # Solving manually or with np.linalg.lstsq(B_two, y)
    manual_coeffs, _, _, _ = np.linalg.lstsq(B_two, y, rcond=None)
    assert np.allclose(coeffs_two, manual_coeffs)
    manual_y_pred = B_two @ manual_coeffs
    manual_rss = np.sum((y - manual_y_pred)**2)
    assert np.isclose(rss_two, manual_rss)
    assert rss_two < rss_const # Model should improve

    # Test with singular matrix (e.g. two identical basis functions)
    B_singular = passer._build_basis_matrix(X, [bf_const, bf_const]) # Two intercept columns
    rss_singular, coeffs_singular = passer._calculate_rss_and_coeffs(B_singular, y)
    # lstsq handles rank deficiency; it will give a solution. RSS should be same as single intercept.
    assert np.isclose(rss_singular, rss_const)
    # Coeffs might be distributed, e.g. [mean/2, mean/2] or one might be zero.
    # Check if their sum effect is like the single intercept:
    y_pred_singular = B_singular @ coeffs_singular
    assert np.allclose(y_pred_singular, np.mean(y))


def test_get_allowable_knot_values(simple_data):
    X, _ = simple_data # X = [[1],[2],[3],[4],[5]]
    earth_model = MockEarth(endspan_alpha=0.0) # Default endspan behavior
    passer = ForwardPasser(earth_model)
    passer.X_train = X # Initialize X_train for the passer

    parent_intercept = ConstantBasisFunction()

    # Test with parent_bf = intercept (additive term)
    # Current simplified logic: unique_X_vals[:-1] if len > 2 and endspan_alpha > 0
    # If endspan_alpha = 0, it should return all unique values
    # Let's refine the test based on current _get_allowable_knot_values

    # Case 1: endspan_alpha = 0.0 (should return all unique values if >2, else empty)
    knots1 = passer._get_allowable_knot_values(X[:,0], parent_intercept, 0)
    assert np.array_equal(knots1, np.array([1.,2.,3.,4.,5.]))

    # Case 2: endspan_alpha > 0 (e.g. 0.1, doesn't matter much for current simplified logic)
    earth_model_endspan = MockEarth(endspan_alpha=0.1)
    passer_endspan = ForwardPasser(earth_model_endspan)
    passer_endspan.X_train = X
    knots2 = passer_endspan._get_allowable_knot_values(X[:,0], parent_intercept, 0)
    assert np.array_equal(knots2, np.array([1.,2.,3.,4.])) # Excludes max (5.0)

    # Test with too few unique values
    X_few_unique = np.array([[1.0],[1.0],[2.0]])
    passer.X_train = X_few_unique
    knots_few = passer._get_allowable_knot_values(X_few_unique[:,0], parent_intercept, 0)
    # Current logic: if parent_bf.degree == 0 and endspan_alpha > 0, and len(unique) > 2
    # If endspan_alpha=0, it returns unique_X_vals if len > 2.
    # If len(unique_X_vals) is 2 (e.g. [1,2]), current logic returns empty. This needs review.
    # The `if len(unique_X_vals) > 2:` check at the start of `_get_allowable_knot_values`
    # means for [1,2] it returns []. For [1,2,3] it returns [1,2,3] (if endspan_alpha=0) or [1,2] (if endspan_alpha > 0 & parent=intercept)
    assert np.array_equal(knots_few, np.array([]))

    # Test with an interaction parent (degree > 0)
    parent_hinge = HingeBasisFunction(0, 2.0) # Dummy parent
    passer.X_train = X # Reset to original simple_data X
    knots_inter = passer._get_allowable_knot_values(X[:,0], parent_hinge, 0)
    # Should return all unique values as parent is not intercept (for current simplified logic)
    assert np.array_equal(knots_inter, np.array([1.,2.,3.,4.,5.]))


def test_generate_candidates_simple(simple_data):
    X, y = simple_data # X = [[1],[2],[3],[4],[5]]
    earth_model = MockEarth(max_degree=1, endspan_alpha=0.1) # endspan_alpha > 0 to trigger specific knot logic
    passer = ForwardPasser(earth_model)
    passer.run(X,y) # This sets up initial intercept model

    # After run (with intercept), current_basis_functions = [ConstantBasisFunction]
    # We expect candidates generated by splitting on X0 (the only feature)
    # Potential knots for X0 with parent=intercept & endspan_alpha>0: [1,2,3,4]

    candidates = passer._generate_candidates()
    assert len(candidates) == 4 # 4 knots * 1 parent * 1 var (since only 1 var)
                                # Each knot produces one pair (left/right)

    for bf_left, bf_right in candidates:
        assert isinstance(bf_left, HingeBasisFunction)
        assert isinstance(bf_right, HingeBasisFunction)
        assert bf_left.parent1 == passer.current_basis_functions[0] # Parent is intercept
        assert bf_right.parent1 == passer.current_basis_functions[0]
        assert bf_left.variable_idx == 0
        assert bf_right.variable_idx == 0
        assert bf_left.knot_val == bf_right.knot_val
        assert bf_left.knot_val in [1.0, 2.0, 3.0, 4.0]
        assert bf_left.is_right_hinge is False
        assert bf_right.is_right_hinge is True

    # Test max_degree = 0 (should generate no candidates from intercept)
    earth_model_max0 = MockEarth(max_degree=0)
    passer_max0 = ForwardPasser(earth_model_max0)
    passer_max0.run(X,y)
    candidates_max0 = passer_max0._generate_candidates()
    assert len(candidates_max0) == 0


def test_find_best_candidate_pair_simple(simple_data):
    X, y = simple_data # X = [[1],[2],[3],[4],[5]], y = [2,4,5.5,8.5,10]
    earth_model = MockEarth(max_degree=1, endspan_alpha=0.1, max_terms=10) # Default max_terms
    passer = ForwardPasser(earth_model)

    # Manually set up the state after intercept term is added
    passer.X_train = X
    passer.y_train = y.ravel()
    passer.n_samples, passer.n_features = X.shape
    intercept_bf = ConstantBasisFunction()
    passer.current_basis_functions = [intercept_bf]
    passer.current_B_matrix = passer._build_basis_matrix(passer.X_train, passer.current_basis_functions)
    rss, coeffs = passer._calculate_rss_and_coeffs(passer.current_B_matrix, passer.y_train)
    passer.current_coefficients = coeffs
    passer.current_rss = rss
    # At this point, passer.current_rss is for the intercept-only model.
    # passer.run(X,y) # Original line that caused the issue by running the full loop.
                    # mean(y) = 6
                    # (2-6)^2 + (4-6)^2 + (5.5-6)^2 + (8.5-6)^2 + (10-6)^2
                    # 16 + 4 + 0.25 + 6.25 + 16 = 42.5

    initial_rss = passer.current_rss
    assert np.isclose(initial_rss, 42.5)

    passer._find_best_candidate_pair()

    assert passer._best_candidate_pair is not None
    bf_l, bf_r = passer._best_candidate_pair
    assert isinstance(bf_l, HingeBasisFunction)
    assert isinstance(bf_r, HingeBasisFunction)
    assert passer._min_candidate_rss < initial_rss # RSS should have reduced
    assert passer._best_new_B_matrix is not None
    assert passer._best_new_coeffs is not None
    assert passer._best_new_B_matrix.shape == (X.shape[0], 1 + 2) # Intercept + 2 new hinges
    assert len(passer._best_new_coeffs) == 3

    # Check if the selected knot is reasonable.
    # For y = approx 2*X, a knot around the midpoint might be chosen.
    # Knots available: [1,2,3,4].
    # Let's manually check RSS for a few knots:
    # Knot = 3.0: bf_l = max(0, 3-x), bf_r = max(0, x-3)
    # X:     [1,  2,  3,  4,  5]
    # bf_l:  [2,  1,  0,  0,  0]
    # bf_r:  [0,  0,  0,  1,  2]
    # B_k3 = [[1,2,0],[1,1,0],[1,0,0],[1,0,1],[1,0,2]]
    # Solve B_k3 @ c = y. RSS_k3 = sum((y - B_k3@c)^2)
    # This is tedious to do by hand, but the test ensures *some* pair is chosen that reduces RSS.


def test_run_main_loop_simple_case(simple_data):
    """Test the main run loop for a few iterations on a simple case."""
    X, y = simple_data # X = [[1],[2],[3],[4],[5]], y = [2,4,5.5,8.5,10]

    # Max_terms = 3 means intercept + one pair of hinges
    earth_model = MockEarth(max_degree=1, max_terms=3, endspan_alpha=0.1)
    passer = ForwardPasser(earth_model)

    final_bfs, final_coeffs = passer.run(X, y)

    assert len(final_bfs) == 3 # Intercept + 1 pair
    assert isinstance(final_bfs[0], ConstantBasisFunction)
    assert isinstance(final_bfs[1], HingeBasisFunction)
    assert isinstance(final_bfs[2], HingeBasisFunction)
    assert len(final_coeffs) == 3

    # Check that RSS reduced from intercept-only model
    rss_intercept_only = np.sum((y - np.mean(y))**2)
    assert passer.current_rss < rss_intercept_only - 1e-9 # EPSILON check

    # Test max_terms stopping
    # Max_terms = 1 means only intercept
    earth_model_mt1 = MockEarth(max_degree=1, max_terms=1, endspan_alpha=0.1)
    passer_mt1 = ForwardPasser(earth_model_mt1)
    final_bfs_mt1, final_coeffs_mt1 = passer_mt1.run(X,y)
    assert len(final_bfs_mt1) == 1
    assert isinstance(final_bfs_mt1[0], ConstantBasisFunction)

    # Max_terms = 2 (should also be only intercept, as we add 2 terms at a time)
    # The condition is `len(self.current_basis_functions) + 2 > max_terms_for_loop`
    # If max_terms_for_loop = 2. Intercept is len=1. 1+2 > 2 is true. So stops.
    earth_model_mt2 = MockEarth(max_degree=1, max_terms=2, endspan_alpha=0.1)
    passer_mt2 = ForwardPasser(earth_model_mt2)
    final_bfs_mt2, final_coeffs_mt2 = passer_mt2.run(X,y)
    assert len(final_bfs_mt2) == 1

# More detailed tests will require simulating the forward pass steps:
# (Old comment, new tests are more detailed)

@pytest.fixture
def interaction_data():
    # Data where y = x0 * x1
    X = np.array([[1,1], [1,2], [1,3],
                  [2,1], [2,2], [2,3],
                  [3,1], [3,2], [3,3]])
    y = X[:,0] * X[:,1]
    return X, y

def test_generate_candidates_for_interaction(interaction_data):
    X, y = interaction_data
    # Allow up to degree 2 interactions
    earth_model = MockEarth(max_degree=2, max_terms=5, endspan_alpha=0.0)
    passer = ForwardPasser(earth_model)

    # Manually set up a state with one hinge function (degree 1)
    passer.X_train = X
    passer.y_train = y
    passer.n_samples, passer.n_features = X.shape

    # Add an initial intercept and one hinge term: h(x0-1.5)
    bf_intercept = ConstantBasisFunction()
    bf_hinge_x0 = HingeBasisFunction(variable_idx=0, knot_val=1.5, is_right_hinge=True, variable_name="x0")
    passer.current_basis_functions = [bf_intercept, bf_hinge_x0]

    # Generate candidates. Now parent_bf can be bf_hinge_x0 (degree 1).
    # Candidates should be of the form: bf_hinge_x0 * Hinge(x1, knot)
    candidates = passer._generate_candidates()

    found_interaction_candidate = False
    for bf_left, bf_right in candidates:
        if bf_left.parent1 == bf_hinge_x0 and bf_left.variable_idx == 1: # Interacting with x1
            assert bf_left.degree() == 2
            assert bf_right.degree() == 2
            assert bf_left.get_involved_variables() == {0, 1}
            assert str(bf_left).startswith(f"({str(bf_hinge_x0)}) * max(0, ")
            found_interaction_candidate = True
            break
    assert found_interaction_candidate, "Should generate degree 2 interaction candidates."

    # Test that no degree 3 candidates are generated if max_degree is 2
    # Add a degree 2 term manually and try to generate candidates
    # bf_inter_x0_x1 = HingeBasisFunction(variable_idx=1, knot_val=1.5, is_right_hinge=True, parent_bf=bf_hinge_x0, variable_name="x1")
    # passer.current_basis_functions = [bf_intercept, bf_hinge_x0, bf_inter_x0_x1] # bf_inter_x0_x1 is degree 2
    # candidates_deg3_attempt = passer._generate_candidates() # Should only try to split intercept and bf_hinge_x0 further if possible

    # This check is tricky as _generate_candidates iterates through all current_basis_functions.
    # A simpler check: if a parent is already max_degree, it's skipped.
    # If a parent is max_degree-1, it can form max_degree interactions.

def test_run_with_interaction(interaction_data):
    X, y = interaction_data
    # Expecting a model like: intercept + x0*h1(x1) + x0*h2(x1) or h1(x0)*h1(x1) + ...
    # Max_terms needs to be sufficient for intercept + at least one interaction pair (3 terms) or two pairs (5 terms)
    # Increasing max_terms to give more room for interactions.
    earth_model = MockEarth(max_degree=2, max_terms=7, penalty=0, endspan_alpha=0.0)
    passer = ForwardPasser(earth_model)

    final_bfs, final_coeffs = passer.run(X,y)

    assert len(final_bfs) > 1 # Should be more than just intercept
    assert len(final_bfs) <= 7 # Max_terms is 7

    has_interaction_term = any(bf.degree() == 2 for bf in final_bfs)
    assert has_interaction_term, "Expected at least one interaction term to be selected."

    # Check RSS is very low, as data is perfectly x0*x1
    # This requires the model to actually find something close to x0*x1
    # For instance, x0 * (c1*max(0, x1-k) + c2*max(0, k-x1)) can approximate x0*x1
    # Or (ax0+b)*()*...
    # A perfect fit might be hard to achieve with limited terms and simple hinge choices.
    # We just check if RSS is significantly lower than intercept-only RSS
    rss_intercept_only = np.sum((y - np.mean(y))**2)
    final_B = passer._build_basis_matrix(X, final_bfs)
    final_y_pred = final_B @ final_coeffs
    final_rss = np.sum((y - final_y_pred)**2)

    assert final_rss < rss_intercept_only * 0.1, "Interaction model should significantly reduce RSS."


if __name__ == '__main__':
    # pytest.main([__file__])
    print("Run tests using 'pytest tests/test_forward.py'")
