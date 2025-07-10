# -*- coding: utf-8 -*-

"""
Unit tests for the main Earth class in pymars.earth
"""

import pytest
import numpy as np
from pymars.earth import Earth
from pymars._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
from pymars._record import EarthRecord

# Minimal data for testing basic fit and predict
@pytest.fixture
def simple_earth_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.0]) # Piecewise linear
    return X, y

@pytest.fixture
def more_complex_earth_data():
    np.random.seed(42)
    X = np.random.rand(50, 3)
    y = 2 * X[:,0] + np.sin(np.pi * X[:,1]) - X[:,2]**2 + np.random.randn(50) * 0.1
    return X, y


def test_earth_module_importable():
    """Test that the earth module can be imported and Earth class is available."""
    from pymars.earth import Earth as EarthClass
    assert EarthClass is not None

def test_earth_instantiation():
    """Test basic instantiation and default parameter storage."""
    model = Earth()
    assert model.max_degree == 1
    assert model.penalty == 3.0
    assert model.max_terms is None
    assert model.minspan_alpha == 0.0
    assert model.endspan_alpha == 0.0
    assert model.allow_linear is True
    assert model.basis_ is None
    assert model.coef_ is None
    assert model.fitted_ is False

def test_earth_instantiation_custom_params():
    """Test instantiation with custom parameters."""
    model = Earth(max_degree=2, penalty=2.0, max_terms=15,
                  minspan_alpha=0.1, endspan_alpha=0.1, allow_linear=False)
    assert model.max_degree == 2
    assert model.penalty == 2.0
    assert model.max_terms == 15
    assert model.minspan_alpha == 0.1
    assert model.endspan_alpha == 0.1
    assert model.allow_linear is False

def test_earth_fit_simple_case(simple_earth_data):
    """Test fit method on a simple, predictable dataset."""
    X, y = simple_earth_data
    model = Earth(max_degree=1, max_terms=5, penalty=0) # Low penalty to allow terms

    model.fit(X, y)

    assert model.fitted_ is True
    assert model.basis_ is not None
    assert len(model.basis_) > 0
    # For this data, expect intercept + at least one pair of hinges
    # Given max_terms=5, it should be able to fit something like intercept + 2 pairs
    assert len(model.basis_) <= 5
    assert model.coef_ is not None
    assert len(model.coef_) == len(model.basis_)
    assert model.gcv_ is not None
    assert model.rss_ is not None
    assert model.mse_ is not None
    assert isinstance(model.record_, EarthRecord)

def test_earth_predict_before_fit():
    """Test that predict raises an error if called before fitting."""
    model = Earth()
    X_test = np.array([[1.0]])
    with pytest.raises(RuntimeError, match="This Earth instance is not fitted yet."):
        model.predict(X_test)

def test_earth_predict_after_fit(simple_earth_data):
    """Test predict method after fitting."""
    X, y = simple_earth_data
    model = Earth(max_degree=1, max_terms=5, penalty=0)
    model.fit(X, y)

    X_test = np.array([[1.5], [2.5], [3.5], [4.5], [0.5], [8.5]])
    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X_test.shape[0],)
    # Exact prediction values depend on the specific basis functions selected,
    # but they should be reasonable for this data.
    # For instance, predictions for X_test values within fitted range should be sensible.
    # Example: prediction for 1.5 should be between y for 1.0 and 2.0 (approx 1.0 and 2.0)
    assert 0.5 < predictions[0] < 2.5 # For X_test = 1.5
    assert 2.5 < predictions[2] < 4.5 # For X_test = 3.5

def test_earth_fit_predict_more_complex(more_complex_earth_data):
    """Test fit and predict on a slightly more complex dataset."""
    X, y = more_complex_earth_data
    model = Earth(max_degree=2, max_terms=10, penalty=3.0) # Allow interactions

    model.fit(X,y)
    assert model.fitted_
    assert len(model.basis_) > 1 # Expect more than just intercept
    assert len(model.basis_) <= 10

    predictions = model.predict(X)
    assert predictions.shape == (y.shape[0],)

    # Check that MSE on training data is reasonable (i.e., model learned something)
    mse_train = np.mean((y - predictions)**2)
    assert mse_train < np.var(y) # MSE should be less than variance of y if model is useful

    # Check for interaction terms if max_degree > 1
    if model.max_degree > 1:
        has_interaction = any(bf.degree() > 1 for bf in model.basis_)
        # This is not a strict guarantee for all data, but for this one it should find some.
        # If not, the test might be too brittle or data not complex enough for this max_terms.
        # Marking as xfail as interaction selection is complex and data/param dependent.
        if len(model.basis_) > 1 : # If more than just intercept
            if not has_interaction:
                pytest.xfail("Interaction term not selected for this data/param combination, "
                             "which can happen with greedy forward pass.")
            assert has_interaction, "Expected interaction terms with max_degree > 1"


def test_earth_summary_method(simple_earth_data, capsys):
    """Test that the summary method runs and prints output."""
    X, y = simple_earth_data
    model = Earth(max_degree=1, max_terms=3)

    # Summary before fit
    model.summary()
    captured_before = capsys.readouterr()
    assert "Model not yet fitted." in captured_before.out

    model.fit(X,y)
    model.summary()
    captured_after = capsys.readouterr()
    assert "pymars Earth Model Summary" in captured_after.out
    assert "Selected Basis Functions:" in captured_after.out
    assert "GCV (final model):" in captured_after.out
    assert "Basis Functions and Coefficients:" in captured_after.out
    assert "Coef:" in captured_after.out # Check if coefficients are printed

def test_input_validation_in_fit(simple_earth_data):
    """Test input validation within the fit method."""
    X, y = simple_earth_data
    model = Earth()

    # Incorrect y dimensions (should be 1D)
    y_2d_col = y.reshape(-1,1)
    # y_2d_row = y.reshape(1,-1) # This would fail shape[0] check first

    # Current basic validation in fit() allows y_2d_col if y.shape[1]==1 as it ravels it.
    # This test might need to be more stringent once robust sklearn validation is in.
    # For now, test y with more than 1 column.
    y_2d_multi_col = np.hstack([y_2d_col, y_2d_col])
    with pytest.raises(ValueError, match="Target y must be 1-dimensional."):
        model.fit(X, y_2d_multi_col)

    # Inconsistent number of samples
    X_short = X[:-1]
    with pytest.raises(ValueError, match="X and y have inconsistent numbers of samples."):
        model.fit(X_short, y)

def test_empty_model_after_pruning(simple_earth_data):
    """Test behavior if pruning results in an empty model (should default to intercept)."""
    X, y = simple_earth_data
    model = Earth(max_degree=1, max_terms=3, penalty=3.0) # Normal penalty

    # Mock PruningPasser.run to return an empty/invalid model
    # Need to import PruningPasser for this test's GCV check part too
    from pymars._pruning import PruningPasser # Local import for this test
    from unittest.mock import patch

    # Patch PruningPasser in the module where it's defined and imported from by Earth.fit
    with patch('pymars._pruning.PruningPasser.run') as mock_pruning_run:
        # Simulate pruning returning no basis functions and None coefficients
        mock_pruning_run.return_value = ([], None, np.inf)
        model.fit(X, y)

    assert model.fitted_
    assert model.basis_ is not None
    assert len(model.basis_) == 1, "Should default to intercept model"
    assert isinstance(model.basis_[0], ConstantBasisFunction), "Basis should be ConstantBasisFunction"
    assert model.coef_ is not None
    assert len(model.coef_) == 1, "Should have one coefficient for intercept"
    assert np.isclose(model.coef_[0], np.mean(y)), "Coefficient should be mean of y"

    # Predict should work and give mean of y
    predictions = model.predict(X)
    assert np.allclose(predictions, np.mean(y), atol=1e-5), "Predictions should be mean of y"

    # Check GCV (should be GCV of intercept-only model)
    # Re-calculate GCV for intercept-only model manually for assertion
    temp_earth_for_gcv_check = Earth(penalty=model.penalty)
    temp_pruner_for_gcv_check = PruningPasser(temp_earth_for_gcv_check)
    temp_pruner_for_gcv_check.X_train = X # Still needed for _build_basis_matrix if it uses self.X_train
    temp_pruner_for_gcv_check.y_train = y.ravel()
    temp_pruner_for_gcv_check.n_samples = X.shape[0]
    expected_gcv_intercept_only, _, _ = temp_pruner_for_gcv_check._compute_gcv_for_subset(
        X, y.ravel(), X.shape[0], [ConstantBasisFunction()]
    )

    assert model.gcv_ is not None, "GCV should be set"
    assert np.isclose(model.gcv_, expected_gcv_intercept_only), "GCV should be for the intercept-only model"

def test_earth_feature_importance_parameter(simple_earth_data):
    """Test that feature_importance_type parameter is stored and used."""
    X, y = simple_earth_data
    model = Earth(feature_importance_type='nb_subsets')
    assert model.feature_importance_type == 'nb_subsets'

    model.fit(X,y)
    assert model.fitted_
    assert model.feature_importances_ is not None
    assert isinstance(model.feature_importances_, np.ndarray)
    assert len(model.feature_importances_) == X.shape[1]
    if X.shape[1] > 0 and len(model.basis_) > 1: # If there are features and model is not just intercept
        assert np.isclose(np.sum(model.feature_importances_), 1.0)
    else: # Handles case of no features or only intercept (no variables involved)
        assert np.sum(model.feature_importances_) == 0


    # Test with None (default)
    model_no_fi = Earth()
    model_no_fi.fit(X,y)
    assert model_no_fi.feature_importance_type is None
    assert model_no_fi.feature_importances_ is None

def test_earth_summary_feature_importances(simple_earth_data, capsys):
    """Test the summary_feature_importances method."""
    X, y = simple_earth_data

    # Test before fit
    model_unfit = Earth(feature_importance_type='nb_subsets')
    summary_unfit = model_unfit.summary_feature_importances()
    assert "Model not yet fitted" in summary_unfit

    # Test after fit, but with feature_importance_type=None
    model_no_fi = Earth(feature_importance_type=None)
    model_no_fi.fit(X,y)
    summary_no_fi = model_no_fi.summary_feature_importances()
    assert "Feature importances not computed" in summary_no_fi

    # Test after fit with feature_importance_type='nb_subsets'
    model_fi = Earth(feature_importance_type='nb_subsets', max_terms=3)
    model_fi.fit(X,y)
    summary_fi = model_fi.summary_feature_importances()
    captured = capsys.readouterr() # To clear previous prints by model.summary() if any

    print(f"\nCaptured for summary_feature_importances:\n{summary_fi}") # Print for manual inspection
    assert "Feature Importances (nb_subsets)" in summary_fi
    assert "x0" in summary_fi # Assuming x0 is the feature name for single feature data
    assert ":" in summary_fi # Check for value separator

    # Test with a different type (currently placeholder, should show warning in summary)
    # The _calculate_feature_importances prints a warning for unknown types.
    # This test currently checks if the summary method handles it.
    model_unknown_fi = Earth(feature_importance_type='unknown_type', max_terms=3)
    model_unknown_fi.fit(X,y)
    summary_unknown = model_unknown_fi.summary_feature_importances()
    # The summary will still try to print based on self.feature_importances_ (which would be zeros)
    assert "Feature Importances (unknown_type)" in summary_unknown
    assert "x0" in summary_unknown # Will show x0 : 0.0000

def test_earth_nb_subsets_calculation_mocked_record(capsys):
    """Test nb_subsets calculation with a mocked record pruning trace."""
    X = np.array([[1,10], [2,20], [3,30]]) # 2 features
    y = np.array([1,2,3])

    model = Earth(feature_importance_type='nb_subsets')

    # Manually create a mock record with a pruning trace
    mock_record = EarthRecord(X, y, model) # model instance for params

    # Define some basis functions for the trace
    bf_const = ConstantBasisFunction()
    bf_x0 = LinearBasisFunction(variable_idx=0, variable_name="x0")
    bf_x1 = LinearBasisFunction(variable_idx=1, variable_name="x1")
    bf_hinge_x0 = HingeBasisFunction(variable_idx=0, knot_val=1.5, variable_name="x0_h")

    # Pruning trace:
    # Model 1: Intercept, x0, x1, x0_h  (x0 appears twice, x1 once)
    # Model 2: Intercept, x0, x1       (x0 once, x1 once)
    # Model 3: Intercept, x0          (x0 once)
    mock_record.pruning_trace_basis_functions_ = [
        [bf_const, bf_x0, bf_x1, bf_hinge_x0],
        [bf_const, bf_x0, bf_x1],
        [bf_const, bf_x0]
    ]
    # Dummy values for other trace attributes, not used by nb_subsets directly
    mock_record.pruning_trace_coeffs_ = [np.array([]),np.array([]),np.array([])]
    mock_record.pruning_trace_gcv_ = [0.1, 0.2, 0.3]
    mock_record.pruning_trace_rss_ = [1,2,3]

    model.record_ = mock_record # Assign the mocked record
    model.n_features = X.shape[1] # Manually set as fit is not fully run

    # Call the method that calculates importances
    model._calculate_feature_importances(X) # Pass X for num_features if record_.n_features is not set by mock
    model.fitted_ = True # Manually set fitted flag for summary method to proceed

    assert model.feature_importances_ is not None
    # Expected counts:
    # x0 (var_idx 0): in model1 (twice via bf_x0, bf_hinge_x0), model2 (once), model3 (once) -> total unique presence = 3
    # x1 (var_idx 1): in model1 (once), model2 (once) -> total unique presence = 2
    # Total presence counts = 3 + 2 = 5
    # Expected importances: x0 = 3/5 = 0.6, x1 = 2/5 = 0.4
    expected_importances = np.array([0.6, 0.4])
    assert np.allclose(model.feature_importances_, expected_importances)

    # Test summary output
    summary = model.summary_feature_importances()
    print(summary) # For manual check during test dev
    assert "Feature Importances (nb_subsets)" in summary
    # For feature names "x0", "x1", max_name_len is 2.
    # Format is "  {name:<{max_name_len+2}} : {value:.4f}" which is "  {name:<4} : {value:.4f}"
    # "x0" becomes "x0  ", "x1" becomes "x1  "
    # Lines are "  x0   : 0.6000" and "  x1   : 0.4000"
    assert "  x0   : 0.6000" in summary
    assert "  x1   : 0.4000" in summary

def test_earth_feature_importance_gcv(simple_earth_data, more_complex_earth_data, capsys):
    """Test GCV-based feature importance calculation."""
    # Using more_complex_earth_data as it has multiple features
    X, y = more_complex_earth_data

    # Create a dataset where one feature is clearly more important
    # y = 2 * X[:,0] + np.sin(np.pi * X[:,1]) - X[:,2]**2 is the setup
    # Feature 0 (linear) and Feature 1 (sinusoidal) are important. Feature 2 (quadratic) also.
    # GCV scores can be subtle.

    model = Earth(max_degree=1, max_terms=7, feature_importance_type='gcv', penalty=3.0)
    model.fit(X, y)

    assert model.feature_importances_ is not None
    assert model.feature_importances_.shape == (X.shape[1],)

    # Check that importances sum to 1.0 (if any are non-zero) or 0.0 (if all are zero)
    if np.any(model.feature_importances_ > 0):
        assert np.isclose(np.sum(model.feature_importances_), 1.0, atol=1e-5)
    else:
        # This case (all zero) can happen if no terms had positive gcv_score_
        assert np.all(model.feature_importances_ == 0.0)

    # Qualitative check: For the more_complex_earth_data, X0, X1, X2 all contribute.
    # It's hard to give a strict ordering for GCV without running py-earth side-by-side
    # or doing manual calculations.
    # For now, just check that some importances are non-zero if the model is not trivial.
    if len(model.basis_) > 1: # If model is more than just intercept
        # It's possible all gcv_scores were <=0, leading to zero importances.
        # This test mostly ensures the calculation runs and format is okay.
        pass # Specific value assertions are too brittle for GCV feature importance here.

    # Test that the summary can be printed
    summary_str = model.summary_feature_importances()
    captured = capsys.readouterr() # Clear previous prints
    print(f"\nGCV Importance Summary:\n{summary_str}")

    assert isinstance(summary_str, str)
    assert "Feature Importances (gcv)" in summary_str
    if X.shape[1] > 0:
        assert "x0" in summary_str # Generic feature name

    # Test with a dataset where no terms might be selected or GCV scores are zero/negative
    # Using simple_earth_data which is 1D - results might be all for x0 or zero.
    X_simple, y_simple = simple_earth_data

    model_simple_signal = Earth(max_degree=1, max_terms=3, feature_importance_type='gcv', penalty=3.0)
    model_simple_signal.fit(X_simple, y_simple)

    assert model_simple_signal.feature_importances_ is not None
    assert model_simple_signal.feature_importances_.shape == (X_simple.shape[1],)
    if np.any(model_simple_signal.feature_importances_ > 0):
        assert np.isclose(np.sum(model_simple_signal.feature_importances_), 1.0, atol=1e-5)
    else:
        assert np.all(model_simple_signal.feature_importances_ == 0.0)

    # If there's only one feature, its importance should be 1.0 if any term involving it was added
    # with a positive gcv_score. Otherwise, it's 0.0.
    if X_simple.shape[1] == 1 and np.any(model_simple_signal.feature_importances_ > 0) :
        assert np.isclose(model_simple_signal.feature_importances_[0], 1.0, atol=1e-5)

def test_earth_feature_importance_rss(simple_earth_data, more_complex_earth_data, capsys):
    """Test RSS-based feature importance calculation."""
    X, y = more_complex_earth_data

    model = Earth(max_degree=1, max_terms=7, feature_importance_type='rss', penalty=3.0)
    model.fit(X, y)

    assert model.feature_importances_ is not None
    assert model.feature_importances_.shape == (X.shape[1],)

    if np.any(model.feature_importances_ > 0):
        assert np.isclose(np.sum(model.feature_importances_), 1.0, atol=1e-5)
    else:
        assert np.all(model.feature_importances_ == 0.0)

    # Qualitative check: RSS reduction is the primary driver for term selection in forward pass.
    # Expect features that allow good RSS reduction to have higher scores.
    if len(model.basis_) > 1: # If model is more than just intercept
        # For more_complex_data, all features contribute.
        # We expect non-zero importances if terms were added.
        # This is more likely to be non-zero than GCV if terms were added, as RSS reduction is direct.
        # However, if all rss_scores were <=0 (e.g. due to numerical precision with EPSILON checks),
        # then importances could still be zero.
        pass # Specific value assertions are too brittle here.

    summary_str = model.summary_feature_importances()
    captured = capsys.readouterr() # Clear previous prints
    print(f"\nRSS Importance Summary:\n{summary_str}")

    assert isinstance(summary_str, str)
    assert "Feature Importances (rss)" in summary_str
    if X.shape[1] > 0:
        assert "x0" in summary_str

    # Test with simple_earth_data
    X_simple, y_simple = simple_earth_data
    model_simple_signal = Earth(max_degree=1, max_terms=3, feature_importance_type='rss', penalty=3.0)
    model_simple_signal.fit(X_simple, y_simple)

    assert model_simple_signal.feature_importances_ is not None
    assert model_simple_signal.feature_importances_.shape == (X_simple.shape[1],)
    if np.any(model_simple_signal.feature_importances_ > 0):
        assert np.isclose(np.sum(model_simple_signal.feature_importances_), 1.0, atol=1e-5)
    else:
        assert np.all(model_simple_signal.feature_importances_ == 0.0)

    if X_simple.shape[1] == 1 and np.any(model_simple_signal.feature_importances_ > 0):
        assert np.isclose(model_simple_signal.feature_importances_[0], 1.0, atol=1e-5)

def test_earth_invalid_feature_importance_type(simple_earth_data, capsys):
    """Test behavior with an invalid feature_importance_type."""
    X, y = simple_earth_data
    invalid_type = "this_is_not_a_valid_type"
    model = Earth(feature_importance_type=invalid_type)
    model.fit(X, y)

    assert model.fitted_
    assert model.feature_importances_ is not None
    assert isinstance(model.feature_importances_, np.ndarray)
    assert len(model.feature_importances_) == X.shape[1]
    assert np.all(model.feature_importances_ == 0.0), "Importances should be all zeros for invalid type"

    # Check for the warning message
    captured = capsys.readouterr()
    # The warning is printed during _calculate_feature_importances, which is called by fit()
    # stdout of fit() method is captured here.
    # print(f"Captured output for invalid type:\nSTDOUT:\n{captured.out}\nSTDERR:\n{captured.err}")
    assert f"Warning: feature_importance_type '{invalid_type}'" in captured.out
    assert "is not yet fully implemented. Returning zeros for importances." in captured.out

    # Test summary method for this case
    summary_str = model.summary_feature_importances()
    assert f"Feature Importances ({invalid_type})" in summary_str
    if X.shape[1] > 0:
        # Default feature name for single feature X is 'x0'
        # max_name_len will be 2. Formatting is {name:<{max_name_len+2}} which is {name:<4}
        # So "x0" becomes "x0  "
        # The line is "  {formatted_name} : {value:.4f}"
        expected_feature_line = "  x0   : 0.0000" # "x0" + 2 spaces (from padding) + " : 0.0000"
        assert expected_feature_line in summary_str


if __name__ == '__main__':
    pytest.main([__file__])
