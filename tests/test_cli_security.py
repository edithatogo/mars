"""Test CLI security features."""

import pytest

from pymars.cli import _load_model


def test_load_model_pickle_security(tmp_path):
    """Test that loading pickle files requires the --allow-pickle flag."""
    import pickle

    dummy_model_path = tmp_path / "dummy.pkl"
    with open(dummy_model_path, "wb") as f:
        pickle.dump({"dummy": "data"}, f)

    # Should raise error without allow_pickle flag
    with pytest.raises(
        ValueError,
        match=r"Loading pickle files is not allowed unless --allow-pickle is specified\.",
    ):
        _load_model(str(dummy_model_path))

    # Should work with allow_pickle flag
    model = _load_model(str(dummy_model_path), allow_pickle=True)
    assert model == {"dummy": "data"}
