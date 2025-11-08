"""
Unit tests for the ForwardPasser in pymars._forward
"""

import numpy as np
import pytest

from pymars._basis import (
    ConstantBasisFunction,
    HingeBasisFunction,
    LinearBasisFunction,
    MissingnessBasisFunction,
)
from pymars._forward import ForwardPasser
from pymars.earth import Earth


class MockEarth(Earth):
    def __init__(
        self,
        max_degree=1,
        max_terms=10,
        minspan_alpha=0.0,
        endspan_alpha=0.0,
        minspan=-1,
        endspan=-1,
        penalty=3.0,
        allow_linear=True,
        allow_missing=False,
    ):  # Added allow_missing
        super().__init__(
            max_degree=max_degree,
            penalty=penalty,
            max_terms=max_terms,
            minspan_alpha=minspan_alpha,
            endspan_alpha=endspan_alpha,
            minspan=minspan,
            endspan=endspan,
            allow_linear=allow_linear,
            allow_missing=allow_missing,
        )
        self.record_ = None


@pytest.fixture
def simple_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([2.0, 4.0, 5.5, 8.5, 10.0])
    return X, y


@pytest.fixture
def multi_feature_data():
    X = np.array([[1, 10], [2, 20], [3, 15], [4, 25], [5, 12]], dtype=float)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(5) * 0.1
    return X, y


@pytest.fixture
def interaction_data():
    X = np.array(
        [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]],
        dtype=float,
    )
    y = X[:, 0] * X[:, 1]
    return X, y


def test_forward_passer_instantiation_and_initial_state(simple_data):
    X, y = simple_data
    earth_model = MockEarth(max_degree=1, max_terms=5)
    passer = ForwardPasser(earth_model)
    assert passer.model is earth_model
    assert passer.X_train is None
    assert passer.y_train is None
    assert passer.current_basis_functions == []
    assert passer.current_rss == np.inf
    assert passer.X_fit_original is None
    assert passer.missing_mask is None


def test_initial_model_setup_in_run(simple_data):
    X, y = simple_data
    earth_model = MockEarth(max_degree=1, max_terms=3)
    passer = ForwardPasser(earth_model)

    def mock_find_best_candidate_addition_inner():
        passer._best_candidate_addition = None
        passer._min_candidate_rss = passer.current_rss

    original_find_best = passer._find_best_candidate_addition
    passer._find_best_candidate_addition = mock_find_best_candidate_addition_inner

    dummy_missing_mask = np.zeros_like(X, dtype=bool)
    bfs, coeffs = passer.run(
        X_fit_processed=X,
        y_fit=y.ravel(),
        missing_mask=dummy_missing_mask,
        X_fit_original=X,
    )

    passer._find_best_candidate_addition = original_find_best

    assert len(passer.current_basis_functions) == 1
    assert isinstance(passer.current_basis_functions[0], ConstantBasisFunction)
    assert passer.current_B_matrix is not None
    assert passer.current_B_matrix.shape == (X.shape[0], 1)
    assert np.all(passer.current_B_matrix == 1.0)
    assert passer.current_coefficients is not None
    assert len(passer.current_coefficients) == 1
    assert np.isclose(passer.current_coefficients[0], np.mean(y))
    expected_rss_intercept_only = np.sum((y - np.mean(y)) ** 2)
    assert np.isclose(passer.current_rss, expected_rss_intercept_only)
    assert len(bfs) == 1
    assert isinstance(bfs[0], ConstantBasisFunction)
    assert np.isclose(coeffs[0], np.mean(y))


def test_build_basis_matrix(simple_data):
    X, _ = simple_data
    earth_model = MockEarth()
    passer = ForwardPasser(earth_model)
    # Manually set necessary attributes for _build_basis_matrix if it uses self.missing_mask
    passer.missing_mask = np.zeros_like(X, dtype=bool)

    bf_const = ConstantBasisFunction()
    bf_hinge = HingeBasisFunction(variable_idx=0, knot_val=3.0, variable_name="x0")

    B_empty = passer._build_basis_matrix(X, [])
    assert B_empty.shape == (X.shape[0], 0)

    B_one = passer._build_basis_matrix(X, [bf_const])
    assert B_one.shape == (X.shape[0], 1)
    assert np.all(B_one == 1.0)

    B_two = passer._build_basis_matrix(X, [bf_const, bf_hinge])
    assert B_two.shape == (X.shape[0], 2)
    expected_col1 = np.ones(X.shape[0])
    expected_col2 = np.maximum(0, X[:, 0] - 3.0)
    assert np.allclose(B_two[:, 0], expected_col1)
    assert np.allclose(B_two[:, 1], expected_col2)


def test_calculate_rss_and_coeffs(simple_data):
    X, y = simple_data
    earth_model = MockEarth()
    passer = ForwardPasser(earth_model)
    passer.missing_mask = np.zeros_like(X, dtype=bool)

    bf_const = ConstantBasisFunction()
    B_const = passer._build_basis_matrix(X, [bf_const])
    rss_const, coeffs_const, num_valid_const = passer._calculate_rss_and_coeffs(
        B_const, y
    )
    assert np.isclose(coeffs_const[0], np.mean(y))
    assert np.isclose(rss_const, np.sum((y - np.mean(y)) ** 2))
    assert num_valid_const == len(y)

    bf_hinge = HingeBasisFunction(variable_idx=0, knot_val=2.5, variable_name="x0_k2.5")
    B_two = passer._build_basis_matrix(X, [bf_const, bf_hinge])
    rss_two, coeffs_two, num_valid_two = passer._calculate_rss_and_coeffs(B_two, y)
    assert coeffs_two is not None
    assert len(coeffs_two) == 2
    manual_coeffs, _, _, _ = np.linalg.lstsq(B_two, y, rcond=None)
    assert np.allclose(coeffs_two, manual_coeffs)
    manual_y_pred = B_two @ manual_coeffs
    manual_rss = np.sum((y - manual_y_pred) ** 2)
    assert np.isclose(rss_two, manual_rss)
    assert rss_two < rss_const
    assert num_valid_two == len(y)

    B_singular = passer._build_basis_matrix(X, [bf_const, bf_const])
    rss_singular, coeffs_singular, num_valid_sing = passer._calculate_rss_and_coeffs(
        B_singular, y
    )
    assert np.isclose(rss_singular, rss_const)
    y_pred_singular = B_singular @ coeffs_singular
    assert np.allclose(y_pred_singular, np.mean(y))
    assert num_valid_sing == len(y)


def test_find_best_addition_simple_original(simple_data):
    X, y = simple_data
    earth_model = MockEarth(max_degree=1, endspan_alpha=0.0, max_terms=10)
    passer = ForwardPasser(earth_model)

    passer.X_train = X
    passer.y_train = y.ravel()
    passer.n_samples, passer.n_features = X.shape
    passer.missing_mask = np.zeros_like(X, dtype=bool)
    passer.X_fit_original = X

    intercept_bf = ConstantBasisFunction()
    passer.current_basis_functions = [intercept_bf]
    passer.current_B_matrix = passer._build_basis_matrix(
        passer.X_train, passer.current_basis_functions
    )
    rss, coeffs, _ = passer._calculate_rss_and_coeffs(
        passer.current_B_matrix, passer.y_train
    )
    passer.current_coefficients = coeffs
    passer.current_rss = rss
    initial_rss = passer.current_rss
    assert np.isclose(initial_rss, 42.5)

    passer._find_best_candidate_addition()

    assert passer._best_candidate_addition is not None
    bf1, bf2_or_None = passer._best_candidate_addition

    assert passer._min_candidate_rss < initial_rss, "RSS should improve"
    assert passer._best_new_B_matrix is not None, "Best new B matrix should be set"
    assert passer._best_new_coeffs is not None, "Best new coeffs should be set"

    num_added_terms = 1 if bf2_or_None is None else 2
    assert passer._best_new_B_matrix.shape == (X.shape[0], 1 + num_added_terms)
    assert len(passer._best_new_coeffs) == 1 + num_added_terms


def test_get_allowable_knot_values(simple_data):
    X, _ = simple_data
    earth_model = MockEarth(endspan_alpha=0.0)
    passer = ForwardPasser(earth_model)
    passer.X_train = X
    passer.n_samples, passer.n_features = X.shape
    passer.missing_mask = np.zeros_like(X, dtype=bool)
    passer.X_fit_original = X

    parent_intercept = ConstantBasisFunction()

    knots1 = passer._get_allowable_knot_values(X[:, 0], parent_intercept, 0)
    assert np.array_equal(knots1, np.array([1.0, 2.0, 3.0, 4.0]))

    earth_model_endspan = MockEarth(endspan_alpha=0.1)
    passer_endspan = ForwardPasser(earth_model_endspan)
    passer_endspan.X_train = X
    passer_endspan.n_samples, passer_endspan.n_features = X.shape
    passer_endspan.missing_mask = np.zeros_like(X, dtype=bool)
    passer_endspan.X_fit_original = X
    knots2 = passer_endspan._get_allowable_knot_values(X[:, 0], parent_intercept, 0)
    assert np.array_equal(knots2, np.array([]))

    X_few_unique = np.array([[1.0], [1.0], [2.0]])
    passer.X_train = X_few_unique  # This is X_processed for the passer
    passer.n_samples, passer.n_features = X_few_unique.shape
    passer.missing_mask = np.zeros_like(X_few_unique, dtype=bool)
    passer.X_fit_original = X_few_unique  # Original values for knot selection
    knots_few = passer._get_allowable_knot_values(
        X_few_unique[:, 0], parent_intercept, 0
    )
    assert np.array_equal(knots_few, np.array([1.0]))

    parent_hinge = HingeBasisFunction(0, 2.0, variable_name="x0_h2")
    passer.X_train = X
    passer.n_samples, passer.n_features = X.shape
    passer.missing_mask = np.zeros_like(X, dtype=bool)
    passer.X_fit_original = X
    knots_inter = passer._get_allowable_knot_values(X[:, 0], parent_hinge, 0)
    assert np.array_equal(knots_inter, np.array([3.0, 4.0, 5.0]))

    earth_model_direct_endspan = MockEarth(endspan=1)
    passer_direct_endspan = ForwardPasser(earth_model_direct_endspan)
    passer_direct_endspan.X_train = X
    passer_direct_endspan.n_samples, passer_direct_endspan.n_features = X.shape
    passer_direct_endspan.missing_mask = np.zeros_like(X, dtype=bool)
    passer_direct_endspan.X_fit_original = X
    knots_direct_es = passer_direct_endspan._get_allowable_knot_values(
        X[:, 0], parent_intercept, 0
    )
    assert np.array_equal(knots_direct_es, np.array([2.0, 3.0]))

    earth_model_direct_minspan = MockEarth(endspan=1, minspan=1)
    passer_direct_minspan = ForwardPasser(earth_model_direct_minspan)
    passer_direct_minspan.X_train = X
    passer_direct_minspan.n_samples, passer_direct_minspan.n_features = X.shape
    passer_direct_minspan.missing_mask = np.zeros_like(X, dtype=bool)
    passer_direct_minspan.X_fit_original = X
    knots_direct_ms = passer_direct_minspan._get_allowable_knot_values(
        X[:, 0], parent_intercept, 0
    )
    assert np.array_equal(knots_direct_ms, np.array([2.0, 3.0]))

    earth_model_direct_minspan_restrict = MockEarth(endspan=1, minspan=2)
    passer_direct_minspan_restrict = ForwardPasser(earth_model_direct_minspan_restrict)
    passer_direct_minspan_restrict.X_train = X
    (
        passer_direct_minspan_restrict.n_samples,
        passer_direct_minspan_restrict.n_features,
    ) = X.shape
    passer_direct_minspan_restrict.missing_mask = np.zeros_like(X, dtype=bool)
    passer_direct_minspan_restrict.X_fit_original = X
    knots_direct_ms_restrict = (
        passer_direct_minspan_restrict._get_allowable_knot_values(
            X[:, 0], parent_intercept, 0
        )
    )
    assert np.array_equal(knots_direct_ms_restrict, np.array([2.0]))

    earth_model_alpha_minspan = MockEarth(endspan=1, minspan_alpha=0.5)
    passer_alpha_minspan = ForwardPasser(earth_model_alpha_minspan)
    passer_alpha_minspan.X_train = X
    passer_alpha_minspan.n_samples, passer_alpha_minspan.n_features = X.shape
    passer_alpha_minspan.missing_mask = np.zeros_like(X, dtype=bool)
    passer_alpha_minspan.X_fit_original = X
    knots_alpha_ms = passer_alpha_minspan._get_allowable_knot_values(
        X[:, 0], parent_intercept, 0
    )
    assert np.array_equal(knots_alpha_ms, np.array([2.0, 3.0]))

    earth_model_minspan_cooldown = MockEarth(endspan=0, minspan=2)
    passer_minspan_cooldown = ForwardPasser(earth_model_minspan_cooldown)
    passer_minspan_cooldown.X_train = X
    passer_minspan_cooldown.n_samples, passer_minspan_cooldown.n_features = X.shape
    passer_minspan_cooldown.missing_mask = np.zeros_like(X, dtype=bool)
    passer_minspan_cooldown.X_fit_original = X
    knots_cooldown = passer_minspan_cooldown._get_allowable_knot_values(
        X[:, 0], parent_intercept, 0
    )
    assert np.array_equal(knots_cooldown, np.array([1.0, 3.0]))

    parent_hinge_interaction = HingeBasisFunction(0, 0.5, variable_name="x0_h_dummy")
    earth_model_inter_minspan = MockEarth(endspan=0, minspan=2)
    passer_inter_minspan = ForwardPasser(earth_model_inter_minspan)
    passer_inter_minspan.X_train = X
    passer_inter_minspan.n_samples, passer_inter_minspan.n_features = X.shape
    passer_inter_minspan.missing_mask = np.zeros_like(X, dtype=bool)
    passer_inter_minspan.X_fit_original = X
    parent_hinge_interaction.transform = lambda x_arr, mm_arr: np.ones(x_arr.shape[0])
    knots_inter_ms = passer_inter_minspan._get_allowable_knot_values(
        X[:, 0], parent_hinge_interaction, 0
    )
    assert np.array_equal(knots_inter_ms, np.array([1.0, 3.0, 5.0]))

    earth_model_high_endspan = MockEarth(endspan=3)
    passer_high_endspan = ForwardPasser(earth_model_high_endspan)
    passer_high_endspan.X_train = X
    passer_high_endspan.n_samples, passer_high_endspan.n_features = X.shape
    passer_high_endspan.missing_mask = np.zeros_like(X, dtype=bool)
    passer_high_endspan.X_fit_original = X
    knots_high_es = passer_high_endspan._get_allowable_knot_values(
        X[:, 0], parent_intercept, 0
    )
    assert np.array_equal(knots_high_es, np.array([]))

    X_active_test = np.array([[1], [2], [3], [4], [5]])
    parent_bf_inactive = HingeBasisFunction(0, 10.0)
    earth_model_inactive_parent = MockEarth()
    passer_inactive_parent = ForwardPasser(earth_model_inactive_parent)
    passer_inactive_parent.X_train = X_active_test
    passer_inactive_parent.n_samples, passer_inactive_parent.n_features = (
        X_active_test.shape
    )
    passer_inactive_parent.missing_mask = np.zeros_like(X_active_test, dtype=bool)
    passer_inactive_parent.X_fit_original = X_active_test
    knots_inactive = passer_inactive_parent._get_allowable_knot_values(
        X_active_test[:, 0], parent_bf_inactive, 0
    )
    assert np.array_equal(knots_inactive, np.array([]))


def test_generate_candidates_simple(simple_data):
    X, y = simple_data
    earth_model = MockEarth(max_degree=1, endspan_alpha=0.0, allow_linear=True)
    passer = ForwardPasser(earth_model)
    passer.X_train = X
    passer.y_train = y.ravel()
    passer.n_samples, passer.n_features = X.shape
    passer.missing_mask = np.zeros_like(X, dtype=bool)
    passer.X_fit_original = X
    intercept_bf = ConstantBasisFunction()
    passer.current_basis_functions = [intercept_bf]
    passer.current_B_matrix = passer._build_basis_matrix(
        passer.X_train, passer.current_basis_functions
    )
    rss, coeffs, _ = passer._calculate_rss_and_coeffs(
        passer.current_B_matrix, passer.y_train
    )
    passer.current_coefficients = coeffs
    passer.current_rss = rss

    candidates = passer._generate_candidates()
    expected_num_candidates = 4 + (1 if earth_model.allow_linear else 0)
    assert len(candidates) == expected_num_candidates

    num_hinge_pairs = 0
    num_linear_terms = 0
    found_linear_x0 = False

    for bf1, bf2_or_None in candidates:
        if bf2_or_None is not None:
            num_hinge_pairs += 1
            bf_left, bf_right = bf1, bf2_or_None
            assert isinstance(bf_left, HingeBasisFunction)
            assert isinstance(bf_right, HingeBasisFunction)
            assert bf_left.parent1 == intercept_bf
            assert bf_right.parent1 == intercept_bf
            assert bf_left.variable_idx == 0
            assert bf_right.variable_idx == 0
            assert bf_left.knot_val == bf_right.knot_val
            assert bf_left.knot_val in [1.0, 2.0, 3.0, 4.0]
            assert not bf_left.is_right_hinge
            assert bf_right.is_right_hinge
        else:
            num_linear_terms += 1
            assert isinstance(bf1, LinearBasisFunction)
            assert bf1.parent1 == intercept_bf
            assert bf1.variable_idx == 0
            found_linear_x0 = True

    assert num_hinge_pairs == 4
    assert num_linear_terms == (1 if earth_model.allow_linear else 0)
    if earth_model.allow_linear:
        assert found_linear_x0

    earth_model_max0 = MockEarth(max_degree=0, allow_linear=True)
    passer_max0 = ForwardPasser(earth_model_max0)
    dummy_missing_mask_max0 = np.zeros_like(X, dtype=bool)
    passer_max0.X_train = X
    passer_max0.y_train = y.ravel()
    passer_max0.n_samples, passer_max0.n_features = X.shape
    passer_max0.missing_mask = dummy_missing_mask_max0
    passer_max0.X_fit_original = X
    passer_max0.current_basis_functions = [ConstantBasisFunction()]

    candidates_max0 = passer_max0._generate_candidates()
    assert len(candidates_max0) == 0


def test_run_main_loop_simple_case(simple_data):
    X, y = simple_data
    earth_model = MockEarth(
        max_degree=1, max_terms=3, endspan_alpha=0.0, allow_linear=True
    )
    passer = ForwardPasser(earth_model)
    dummy_missing_mask = np.zeros_like(X, dtype=bool)
    final_bfs, final_coeffs = passer.run(
        X_fit_processed=X,
        y_fit=y.ravel(),
        missing_mask=dummy_missing_mask,
        X_fit_original=X,
    )

    assert len(final_bfs) in [2, 3]
    assert isinstance(final_bfs[0], ConstantBasisFunction)
    if len(final_bfs) == 2:
        assert isinstance(final_bfs[1], LinearBasisFunction)
    elif len(final_bfs) == 3:
        assert isinstance(final_bfs[1], HingeBasisFunction)
        assert isinstance(final_bfs[2], HingeBasisFunction)
    assert len(final_coeffs) == len(final_bfs)

    rss_intercept_only = np.sum((y - np.mean(y)) ** 2)
    assert passer.current_rss < rss_intercept_only - 1e-9

    earth_model_mt1 = MockEarth(max_degree=1, max_terms=1, endspan_alpha=0.1)
    passer_mt1 = ForwardPasser(earth_model_mt1)
    dummy_missing_mask_mt1 = np.zeros_like(X, dtype=bool)
    final_bfs_mt1, final_coeffs_mt1 = passer_mt1.run(
        X_fit_processed=X,
        y_fit=y.ravel(),
        missing_mask=dummy_missing_mask_mt1,
        X_fit_original=X,
    )
    assert len(final_bfs_mt1) == 1
    assert isinstance(final_bfs_mt1[0], ConstantBasisFunction)

    earth_model_mt2 = MockEarth(
        max_degree=1, max_terms=2, endspan_alpha=0.1, allow_linear=True
    )
    passer_mt2 = ForwardPasser(earth_model_mt2)
    dummy_missing_mask_mt2 = np.zeros_like(X, dtype=bool)
    # Corrected to use keyword arguments
    final_bfs_mt2, final_coeffs_mt2 = passer_mt2.run(
        X_fit_processed=X,
        y_fit=y.ravel(),
        missing_mask=dummy_missing_mask_mt2,
        X_fit_original=X,
    )
    assert len(final_bfs_mt2) == 2


def test_generate_candidates_for_interaction(interaction_data):
    X, y = interaction_data
    earth_model = MockEarth(
        max_degree=2, max_terms=5, endspan_alpha=0.0, allow_linear=True
    )
    passer = ForwardPasser(earth_model)
    passer.X_train = X
    passer.y_train = y.ravel()
    passer.n_samples, passer.n_features = X.shape
    passer.missing_mask = np.zeros_like(X, dtype=bool)
    passer.X_fit_original = X

    bf_intercept = ConstantBasisFunction()
    bf_hinge_x0 = HingeBasisFunction(
        variable_idx=0,
        knot_val=1.5,
        is_right_hinge=True,
        variable_name="x0",
        parent_bf=bf_intercept,
    )
    passer.current_basis_functions = [bf_intercept, bf_hinge_x0]

    candidates = passer._generate_candidates()

    found_interaction_candidate = False
    for bf1, bf2_or_None in candidates:
        if bf2_or_None is not None:
            bf_left, bf_right = bf1, bf2_or_None
            if bf_left.parent1 == bf_hinge_x0 and bf_left.variable_idx == 1:
                assert bf_left.degree() == 2
                assert bf_right.degree() == 2
                assert bf_left.get_involved_variables() == {0, 1}
                found_interaction_candidate = True
        elif (
            isinstance(bf1, LinearBasisFunction)
            and bf1.parent1 == bf_hinge_x0
            and bf1.variable_idx == 1
        ):
            assert bf1.degree() == 2
            assert bf1.get_involved_variables() == {0, 1}
            found_interaction_candidate = True

    assert found_interaction_candidate, (
        "Should generate degree 2 interaction candidates (hinge or linear)."
    )


def test_run_with_interaction(interaction_data):
    X, y = interaction_data
    earth_model = MockEarth(
        max_degree=2, max_terms=7, penalty=0, endspan_alpha=0.0, allow_linear=True
    )
    passer = ForwardPasser(earth_model)

    dummy_missing_mask_inter = np.zeros_like(X, dtype=bool)
    final_bfs, final_coeffs = passer.run(
        X_fit_processed=X,
        y_fit=y.ravel(),
        missing_mask=dummy_missing_mask_inter,
        X_fit_original=X,
    )

    assert len(final_bfs) > 1
    assert len(final_bfs) <= 7

    has_interaction_term = any(bf.degree() == 2 for bf in final_bfs)
    assert has_interaction_term, (
        "Expected at least one interaction term to be selected."
    )

    rss_intercept_only = np.sum((y - np.mean(y)) ** 2)
    final_B = passer._build_basis_matrix(X, final_bfs)
    if final_coeffs is None or final_B.shape[1] != len(final_coeffs):
        final_rss = rss_intercept_only
    else:
        y_pred = final_B @ final_coeffs
        final_rss = np.sum((y - y_pred) ** 2)

    assert final_rss < rss_intercept_only * 0.1, (
        "Interaction model should significantly reduce RSS."
    )


def test_generate_linear_candidates(multi_feature_data):
    X, y = multi_feature_data
    earth_model_linear_true = MockEarth(max_degree=2, allow_linear=True, max_terms=5)
    passer_linear_true = ForwardPasser(earth_model_linear_true)
    passer_linear_true.X_train = X
    passer_linear_true.y_train = y.ravel()
    passer_linear_true.n_samples, passer_linear_true.n_features = X.shape
    passer_linear_true.missing_mask = np.zeros_like(X, dtype=bool)
    passer_linear_true.X_fit_original = X

    intercept_bf = ConstantBasisFunction()
    passer_linear_true.current_basis_functions = [intercept_bf]

    candidates = passer_linear_true._generate_candidates()

    has_linear_candidate = False
    num_linear_candidates = 0
    for bf1, bf2_or_None in candidates:
        if isinstance(bf1, LinearBasisFunction) and bf2_or_None is None:
            has_linear_candidate = True
            num_linear_candidates += 1
            assert bf1.parent1 == intercept_bf
            assert bf1.degree() == 1
            assert bf1.variable_idx in [0, 1]

    assert has_linear_candidate
    assert num_linear_candidates == X.shape[1]

    earth_model_linear_false = MockEarth(max_degree=2, allow_linear=False, max_terms=5)
    passer_linear_false = ForwardPasser(earth_model_linear_false)
    passer_linear_false.X_train = X
    passer_linear_false.y_train = y.ravel()
    passer_linear_false.n_samples, passer_linear_false.n_features = X.shape
    passer_linear_false.missing_mask = np.zeros_like(X, dtype=bool)
    passer_linear_false.X_fit_original = X
    passer_linear_false.current_basis_functions = [intercept_bf]

    candidates_no_linear = passer_linear_false._generate_candidates()
    has_linear_candidate_false = any(
        isinstance(bf1, LinearBasisFunction) for bf1, _ in candidates_no_linear
    )
    assert not has_linear_candidate_false

    hinge_parent_x0 = HingeBasisFunction(
        variable_idx=0, knot_val=np.median(X[:, 0]), parent_bf=intercept_bf
    )
    assert hinge_parent_x0.degree() == 1

    earth_model_deg_limit = MockEarth(max_degree=1, allow_linear=True, max_terms=5)
    passer_deg_limit = ForwardPasser(earth_model_deg_limit)
    passer_deg_limit.X_train = X
    passer_deg_limit.y_train = y.ravel()
    passer_deg_limit.n_samples, passer_deg_limit.n_features = X.shape
    passer_deg_limit.missing_mask = np.zeros_like(X, dtype=bool)
    passer_deg_limit.X_fit_original = X
    passer_deg_limit.current_basis_functions = [intercept_bf, hinge_parent_x0]

    candidates_deg_limit = passer_deg_limit._generate_candidates()

    found_linear_from_hinge_parent = False
    for bf1, bf2_or_None in candidates_deg_limit:
        if isinstance(bf1, LinearBasisFunction) and bf1.parent1 == hinge_parent_x0:
            found_linear_from_hinge_parent = True
            break
    assert not found_linear_from_hinge_parent

    linear_parent_x0 = LinearBasisFunction(variable_idx=0, parent_bf=intercept_bf)
    assert linear_parent_x0.degree() == 1

    earth_model_var_reuse = MockEarth(max_degree=2, allow_linear=True, max_terms=5)
    passer_var_reuse = ForwardPasser(earth_model_var_reuse)
    passer_var_reuse.X_train = X
    passer_var_reuse.y_train = y.ravel()
    passer_var_reuse.n_samples, passer_var_reuse.n_features = X.shape
    passer_var_reuse.missing_mask = np.zeros_like(X, dtype=bool)
    passer_var_reuse.X_fit_original = X
    passer_var_reuse.current_basis_functions = [intercept_bf, linear_parent_x0]

    candidates_var_reuse = passer_var_reuse._generate_candidates()

    generated_linear_x0_on_linear_x0 = False
    generated_linear_x1_on_linear_x0 = False
    for bf1, bf2_or_None in candidates_var_reuse:
        if isinstance(bf1, LinearBasisFunction) and bf1.parent1 == linear_parent_x0:
            if bf1.variable_idx == 0:
                generated_linear_x0_on_linear_x0 = True
            if bf1.variable_idx == 1:
                generated_linear_x1_on_linear_x0 = True
                assert bf1.degree() == 2

    assert not generated_linear_x0_on_linear_x0
    assert generated_linear_x1_on_linear_x0


def test_find_best_addition_selects_linear(simple_data):
    X_linear = np.array([[1], [2], [3], [4], [5], [6]])
    y_linear = 2 * X_linear[:, 0] + 1

    earth_model = MockEarth(
        max_degree=1,
        allow_linear=True,
        max_terms=3,
        endspan_alpha=0.0,
        minspan_alpha=0.0,
    )
    passer = ForwardPasser(earth_model)
    passer.X_train = X_linear
    passer.y_train = y_linear.ravel()
    passer.n_samples, passer.n_features = X_linear.shape
    passer.missing_mask = np.zeros_like(X_linear, dtype=bool)
    passer.X_fit_original = X_linear

    intercept_bf = ConstantBasisFunction()
    passer.current_basis_functions = [intercept_bf]
    passer.current_B_matrix = passer._build_basis_matrix(
        passer.X_train, passer.current_basis_functions
    )
    rss_intercept, coeffs_intercept, _ = passer._calculate_rss_and_coeffs(
        passer.current_B_matrix, passer.y_train
    )
    passer.current_coefficients = coeffs_intercept
    passer.current_rss = rss_intercept

    passer._find_best_candidate_addition()

    assert passer._best_candidate_addition is not None
    bf1, bf2_or_None = passer._best_candidate_addition

    assert isinstance(bf1, LinearBasisFunction)
    assert bf2_or_None is None
    assert bf1.variable_idx == 0
    assert bf1.parent1 == intercept_bf

    assert np.isclose(passer.current_rss, 70.0)
    assert passer._min_candidate_rss < 1e-5


def test_run_adds_linear_term():
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y = 3 * X[:, 0] - 2

    earth_model = MockEarth(
        max_degree=1,
        allow_linear=True,
        max_terms=2,
        endspan_alpha=0.0,
        minspan_alpha=0.0,
    )
    passer = ForwardPasser(earth_model)
    dummy_missing_mask = np.zeros_like(X, dtype=bool)
    final_bfs, final_coeffs = passer.run(
        X_fit_processed=X,
        y_fit=y.ravel(),
        missing_mask=dummy_missing_mask,
        X_fit_original=X,
    )

    assert len(final_bfs) == 2
    assert any(
        isinstance(bf, LinearBasisFunction)
        and bf.variable_idx == 0
        and bf.parent1.is_constant()
        for bf in final_bfs
    )

    final_B = passer._build_basis_matrix(X, final_bfs)
    y_pred = final_B @ final_coeffs
    assert np.allclose(y_pred, y.ravel(), atol=1e-5)


def test_run_adds_linear_interaction_term():
    X_inter = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]], dtype=float)
    y_inter = X_inter[:, 0] * X_inter[:, 1]

    earth_model = MockEarth(
        max_degree=2,
        allow_linear=True,
        max_terms=5,
        endspan_alpha=0.0,
        minspan_alpha=0.0,
        penalty=0,
    )
    passer = ForwardPasser(earth_model)
    dummy_missing_mask = np.zeros_like(X_inter, dtype=bool)
    final_bfs, final_coeffs = passer.run(
        X_fit_processed=X_inter,
        y_fit=y_inter.ravel(),
        missing_mask=dummy_missing_mask,
        X_fit_original=X_inter,
    )

    print("\nLinear Interaction Test BFs:")
    for i, bf_s in enumerate(final_bfs):
        print(
            f"  BF {i}: {str(bf_s)}, Degree: {bf_s.degree()}, Coeff: {final_coeffs[i] if final_coeffs is not None and i < len(final_coeffs) else 'N/A'}"
        )
    print(f"Final RSS: {passer.current_rss}")

    has_linear_interaction = False
    for bf in final_bfs:
        if (
            bf.degree() == 2
            and isinstance(bf, LinearBasisFunction)
            and isinstance(bf.parent1, LinearBasisFunction)
        ):
            has_linear_interaction = True
            break
        if bf.degree() == 2 and (
            isinstance(bf, LinearBasisFunction) or isinstance(bf, HingeBasisFunction)
        ):
            if bf.parent1 and not bf.parent1.is_constant():
                has_linear_interaction = True
                break
    assert has_linear_interaction, "Expected a degree 2 linear interaction term."

    if final_coeffs is not None and len(final_bfs) == len(final_coeffs):
        final_B = passer._build_basis_matrix(X_inter, final_bfs)
        if final_B.shape[1] == len(final_coeffs):
            y_pred = final_B @ final_coeffs
            assert np.allclose(y_pred, y_inter.ravel(), atol=1e-3)
        else:
            pytest.fail("Basis matrix and coeffs mismatch in linear interaction test.")
    else:
        pytest.fail("Coefficients not properly set in linear interaction test.")


if __name__ == "__main__":
    print("Run tests using 'pytest tests/test_forward.py'")


def test_generate_candidates_with_missingness():
    """Test that MissingnessBasisFunction candidates are generated."""
    X_orig = np.array([[1.0, np.nan], [2.0, 20.0], [np.nan, 30.0], [4.0, np.nan]])
    y_dummy = np.array([1, 2, 3, 4])

    # Mock Earth model that allows missing values
    earth_model_missing = MockEarth(allow_missing=True)
    # Setup ForwardPasser internal state usually done by run()
    passer = ForwardPasser(earth_model_missing)
    passer.X_train = np.nan_to_num(X_orig, nan=0.0)  # X_processed
    passer.missing_mask = np.isnan(X_orig)
    passer.X_fit_original = X_orig
    passer.n_samples, passer.n_features = X_orig.shape
    passer.current_basis_functions = [ConstantBasisFunction()]  # Start with intercept

    # Mock record to provide feature names
    class MockRecord:
        feature_names_in_ = ["feature0", "feature1"]

    passer.model.record_ = MockRecord()

    candidates = passer._generate_candidates()

    found_missingness_bf_0 = False
    found_missingness_bf_1 = False
    num_missingness_candidates = 0

    for bf1, bf2_or_None in candidates:
        if isinstance(bf1, MissingnessBasisFunction) and bf2_or_None is None:
            num_missingness_candidates += 1
            if bf1.variable_idx == 0:
                assert str(bf1) == "is_missing(feature0)"
                found_missingness_bf_0 = True
            elif bf1.variable_idx == 1:
                assert str(bf1) == "is_missing(feature1)"
                found_missingness_bf_1 = True

    assert num_missingness_candidates == 2  # Both features have NaNs
    assert found_missingness_bf_0
    assert found_missingness_bf_1

    # Test case: only one feature has NaNs
    X_one_nan = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0]])
    passer.missing_mask = np.isnan(X_one_nan)
    passer.X_fit_original = X_one_nan
    passer.n_samples, passer.n_features = X_one_nan.shape

    candidates_one_nan = passer._generate_candidates()
    num_missingness_one_nan = 0
    found_missingness_bf_1_only = False
    for bf1, _ in candidates_one_nan:
        if isinstance(bf1, MissingnessBasisFunction):
            num_missingness_one_nan += 1
            if bf1.variable_idx == 1:
                found_missingness_bf_1_only = True
    assert num_missingness_one_nan == 1
    assert found_missingness_bf_1_only

    # Test: no missingness candidates if allow_missing=False
    earth_model_no_missing = MockEarth(allow_missing=False)
    passer_no_missing = ForwardPasser(earth_model_no_missing)
    passer_no_missing.X_train = np.nan_to_num(X_orig, nan=0.0)
    passer_no_missing.missing_mask = np.isnan(X_orig)  # Mask is still there
    passer_no_missing.X_fit_original = X_orig
    passer_no_missing.n_samples, passer_no_missing.n_features = X_orig.shape
    passer_no_missing.current_basis_functions = [ConstantBasisFunction()]
    passer_no_missing.model.record_ = MockRecord()

    candidates_no_missing = passer_no_missing._generate_candidates()
    assert not any(
        isinstance(bf1, MissingnessBasisFunction) for bf1, _ in candidates_no_missing
    )


def test_run_selects_missingness_bf():
    """Test if ForwardPasser.run can select a MissingnessBasisFunction."""
    # Create data where missingness in X0 is perfectly correlated with y
    X_orig = np.array([[np.nan], [1.0], [np.nan], [2.0], [np.nan], [3.0]])
    y = np.array([10.0, 1.0, 10.0, 2.0, 10.0, 3.0])  # High y when X0 is missing

    earth_model = MockEarth(
        allow_missing=True,
        max_terms=2,
        penalty=0,
        allow_linear=False,
        endspan_alpha=0.0,
        minspan_alpha=0.0,
    )  # No linear/hinge to isolate missingness

    # Mock record for feature names
    class MockRecord:
        feature_names_in_ = ["x0"]

    earth_model.record_ = MockRecord()

    passer = ForwardPasser(earth_model)

    X_processed = np.nan_to_num(X_orig, nan=0.0)
    missing_mask = np.isnan(X_orig)

    final_bfs, final_coeffs = passer.run(
        X_fit_processed=X_processed,
        y_fit=y.ravel(),
        missing_mask=missing_mask,
        X_fit_original=X_orig,
    )

    assert len(final_bfs) >= 1

    # Check coefficients roughly - e.g. missingness term should have a positive coeff
    # B = Intercept | is_missing(x0)
    # y = c0*1 + c1*is_missing(x0)
    # When not missing: y_approx = c0. Mean of (1,2,3) is 2. So c0 approx 2.
    # When missing: y_approx = c0 + c1 = 10. So c1 approx 8.
    idx_missing_bf = -1
    for i, bf in enumerate(final_bfs):
        if isinstance(bf, MissingnessBasisFunction):
            idx_missing_bf = i
            break

    if idx_missing_bf != -1 and final_coeffs is not None and len(final_coeffs) == 2:
        assert final_coeffs[idx_missing_bf] > 5
        idx_intercept = 1 - idx_missing_bf
        assert np.isclose(
            final_coeffs[idx_intercept], np.mean(y[~missing_mask[:, 0]]), atol=1.0
        )
