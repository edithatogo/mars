"""
Unit tests for utility functions in pymars._util
"""

import numpy as np
import pytest

# from pymars._util import check_array, check_X_y, calculate_gcv, gcv_penalty_ κόστος_ কার্যকর_ παραμέτρων


def test_util_module_importable():
    """Test that the _util module can be imported."""
    try:
        from pymars import _util

        assert _util is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pymars._util: {e}")


# Tests for check_array (examples, more can be added)
def test_check_array_numpy_conversion():
    from pymars._util import check_array

    data = [[1, 2], [3, 4]]
    arr = check_array(data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)


def test_check_array_ensure_2d():
    from pymars._util import check_array

    data_1d = [1, 2, 3]
    with pytest.raises(ValueError, match="Expected 2D array"):
        check_array(data_1d, ensure_2d=True)

    data_2d = [[1], [2], [3]]
    arr = check_array(data_2d, ensure_2d=True)
    assert arr.ndim == 2


def test_check_array_allow_nd():
    from pymars._util import check_array

    data_3d = np.random.rand(2, 2, 2)
    arr = check_array(data_3d, allow_nd=True)
    assert arr.ndim == 3
    with pytest.raises(ValueError, match="Expected 1D or 2D array"):
        check_array(data_3d, allow_nd=False)


def test_check_array_nan_handling():
    from pymars._util import check_array

    data_with_nan = [[1, np.nan], [3, 4]]
    with pytest.raises(ValueError, match="Input contains NaN values"):
        check_array(data_with_nan, allow_missing=False)
    arr = check_array(data_with_nan, allow_missing=True)
    assert np.isnan(arr[0, 1])


# Tests for GCV calculation components
def test_gcv_penalty_cost():
    from pymars._util import gcv_penalty_cost_effective_parameters
    # Test cases based on the new formula: num_terms + penalty * num_hinge_terms
    # num_samples is required by signature but not used for capping in this specific function.

    # Case 1: Intercept only
    # num_terms=1 (intercept), num_hinge_terms=0, penalty=3
    assert (
        gcv_penalty_cost_effective_parameters(
            num_terms=1, num_hinge_terms=0, penalty=3.0, num_samples=10
        )
        == 1.0
    )

    # Case 2: Intercept + 1 linear term
    # num_terms=2 (intercept, 1 linear), num_hinge_terms=0, penalty=3
    assert (
        gcv_penalty_cost_effective_parameters(
            num_terms=2, num_hinge_terms=0, penalty=3.0, num_samples=10
        )
        == 2.0
    )

    # Case 3: Intercept + 1 hinge term (or a pair, counted as 1 effective hinge for penalty if it's per knot)
    # Assuming num_hinge_terms is the count of HingeBasisFunction objects.
    # If a "knot" implies two HingeBasisFunctions but one penalty unit, the interpretation might differ.
    # Sticking to "num_hinge_terms is count of HingeBasisFunction objects".
    # num_terms=2 (intercept, 1 hinge), num_hinge_terms=1, penalty=3
    assert gcv_penalty_cost_effective_parameters(
        num_terms=2, num_hinge_terms=1, penalty=3.0, num_samples=10
    ) == (2.0 + 3.0 * 1.0)  # 5.0

    # Case 4: Intercept + 2 linear terms + 2 hinge terms
    # num_terms=5 (1 intercept + 2 linear + 2 hinge), num_hinge_terms=2, penalty=3
    expected_complex = 5.0 + 3.0 * 2.0  # 11.0
    assert np.isclose(
        gcv_penalty_cost_effective_parameters(
            num_terms=5, num_hinge_terms=2, penalty=3.0, num_samples=20
        ),
        expected_complex,
    )

    # Case 5: Zero terms
    # num_terms=0, num_hinge_terms=0, penalty=3
    assert (
        gcv_penalty_cost_effective_parameters(
            num_terms=0, num_hinge_terms=0, penalty=3.0, num_samples=10
        )
        == 0.0
    )

    # Test with different penalty
    # num_terms=3 (e.g. intercept, 1 linear, 1 hinge), num_hinge_terms=1, penalty=2
    expected_penalty_2 = 3.0 + 2.0 * 1.0  # 5.0
    assert np.isclose(
        gcv_penalty_cost_effective_parameters(
            num_terms=3, num_hinge_terms=1, penalty=2.0, num_samples=10
        ),
        expected_penalty_2,
    )


def test_calculate_gcv():
    from pymars._util import calculate_gcv

    rss = 100.0
    n_samples = 50

    # Effective params < n_samples
    eff_params1 = 10
    gcv1_denom = (1.0 - eff_params1 / n_samples) ** 2  # (1 - 0.2)^2 = 0.8^2 = 0.64
    expected_gcv1 = rss / (
        n_samples * gcv1_denom
    )  # 100 / (50 * 0.64) = 100 / 32 = 3.125
    assert np.isclose(calculate_gcv(rss, n_samples, eff_params1), expected_gcv1)

    # Effective params near n_samples (e.g. N-1)
    eff_params2 = n_samples - 1  # 49
    gcv2_denom = (
        1.0 - eff_params2 / n_samples
    ) ** 2  # (1 - 49/50)^2 = (1/50)^2 = 1/2500
    expected_gcv2 = rss / (
        n_samples * gcv2_denom
    )  # 100 / (50 * 1/2500) = 100 / (1/50) = 5000
    assert np.isclose(calculate_gcv(rss, n_samples, eff_params2), expected_gcv2)

    # Effective params >= n_samples (should be inf)
    assert calculate_gcv(rss, n_samples, n_samples) == np.inf
    assert calculate_gcv(rss, n_samples, n_samples + 1) == np.inf

    # Zero samples
    assert calculate_gcv(rss, 0, 5) == np.inf

    # Denominator very close to zero
    eff_params_tight = n_samples * (1 - 1e-7)  # (1 - C/N) is 1e-7
    assert calculate_gcv(rss, n_samples, eff_params_tight) > 1e9  # Should be very large


if __name__ == "__main__":
    # pytest.main([__file__])
    print("Run tests using 'pytest tests/test_util.py'")
