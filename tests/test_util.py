"""
Unit tests for utility functions in pymars._util
"""

import numpy as np
import pytest


def test_util_module_importable():
    """Test that the _util module can be imported."""
    try:
        from pymars import _util

        assert _util is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pymars._util: {e}")


def test_check_array_numpy_conversion():
    """check_array should convert list input to a NumPy array."""
    from pymars._util import check_array

    data = [[1, 2], [3, 4]]
    arr = check_array(data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)


def test_check_array_ensure_2d():
    """check_array should reject 1D input when 2D is required."""
    from pymars._util import check_array

    data_1d = [1, 2, 3]
    with pytest.raises(ValueError, match="Expected 2D array"):
        check_array(data_1d, ensure_2d=True)

    data_2d = [[1], [2], [3]]
    arr = check_array(data_2d, ensure_2d=True)
    assert arr.ndim == 2


def test_check_array_allow_nd():
    """check_array should optionally preserve higher-dimensional input."""
    from pymars._util import check_array

    data_3d = np.random.rand(2, 2, 2)
    arr = check_array(data_3d, allow_nd=True)
    assert arr.ndim == 3
    with pytest.raises(ValueError, match="Expected 1D or 2D array"):
        check_array(data_3d, allow_nd=False)


def test_check_array_nan_handling():
    """check_array should enforce missing-value handling rules."""
    from pymars._util import check_array

    data_with_nan = [[1, np.nan], [3, 4]]
    with pytest.raises(ValueError, match="Input contains NaN values"):
        check_array(data_with_nan, allow_missing=False)
    arr = check_array(data_with_nan, allow_missing=True)
    assert np.isnan(arr[0, 1])


def test_check_array_conversion_failure(monkeypatch):
    """Conversion errors should surface as ValueError."""
    from pymars._util import check_array

    def raise_conversion_error(_value):
        raise RuntimeError("cannot convert")

    monkeypatch.setattr(np, "asarray", raise_conversion_error)

    with pytest.raises(ValueError, match="could not be converted to a NumPy array"):
        check_array(object())


def test_gcv_penalty_cost():
    """gcv_penalty_cost_effective_parameters should follow the expected formula."""
    from pymars._util import gcv_penalty_cost_effective_parameters
    assert (
        gcv_penalty_cost_effective_parameters(
            num_terms=1, num_hinge_terms=0, penalty=3.0, num_samples=10
        )
        == 1.0
    )
    assert (
        gcv_penalty_cost_effective_parameters(
            num_terms=2, num_hinge_terms=0, penalty=3.0, num_samples=10
        )
        == 2.0
    )
    assert gcv_penalty_cost_effective_parameters(
        num_terms=2, num_hinge_terms=1, penalty=3.0, num_samples=10
    ) == (2.0 + 3.0 * 1.0)

    expected_complex = 5.0 + 3.0 * 2.0
    assert np.isclose(
        gcv_penalty_cost_effective_parameters(
            num_terms=5, num_hinge_terms=2, penalty=3.0, num_samples=20
        ),
        expected_complex,
    )
    assert (
        gcv_penalty_cost_effective_parameters(
            num_terms=0, num_hinge_terms=0, penalty=3.0, num_samples=10
        )
        == 0.0
    )
    expected_penalty_2 = 3.0 + 2.0 * 1.0
    assert np.isclose(
        gcv_penalty_cost_effective_parameters(
            num_terms=3, num_hinge_terms=1, penalty=2.0, num_samples=10
        ),
        expected_penalty_2,
    )


def test_calculate_gcv():
    """calculate_gcv should match the closed-form expression."""
    from pymars._util import calculate_gcv

    rss = 100.0
    n_samples = 50
    eff_params1 = 10
    gcv1_denom = (1.0 - eff_params1 / n_samples) ** 2
    expected_gcv1 = rss / (n_samples * gcv1_denom)
    assert np.isclose(calculate_gcv(rss, n_samples, eff_params1), expected_gcv1)
    eff_params2 = n_samples - 1
    gcv2_denom = (1.0 - eff_params2 / n_samples) ** 2
    expected_gcv2 = rss / (n_samples * gcv2_denom)
    assert np.isclose(calculate_gcv(rss, n_samples, eff_params2), expected_gcv2)
    assert calculate_gcv(rss, n_samples, n_samples) == np.inf
    assert calculate_gcv(rss, n_samples, n_samples + 1) == np.inf
    assert calculate_gcv(rss, 0, 5) == np.inf
    eff_params_tight = n_samples * (1 - 1e-7)
    assert calculate_gcv(rss, n_samples, eff_params_tight) > 1e9
