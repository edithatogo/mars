from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pymars import Earth, runtime
from pymars._model_spec import spec_from_json, validate_model_spec
from pymars.cli import _load_model, _save_model

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MODEL_SPEC_V1_PATH = FIXTURES_DIR / "model_spec_v1.json"


def _fit_sample_model() -> Earth:
    X = np.array(
        [
            [-1.0, 0.0, 0.5],
            [-0.2, 0.3, 0.0],
            [0.1, -0.4, 0.2],
            [0.5, 0.2, -0.3],
            [1.0, -0.7, 0.8],
            [1.4, -1.1, 1.1],
        ],
        dtype=float,
    )
    y = np.array([0.8, 1.3, 2.0, 2.6, 4.9, 6.2], dtype=float)
    model = Earth(max_terms=8, max_degree=1, feature_importance_type=None)
    model.fit(X, y)
    return model


def _runtime_portability_fixture_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for model_spec_path in sorted(FIXTURES_DIR.glob("model_spec_*.json")):
        suffix = model_spec_path.stem.removeprefix("model_spec_")
        runtime_fixture_path = FIXTURES_DIR / f"runtime_portability_fixture_{suffix}.json"
        if runtime_fixture_path.exists():
            pairs.append((model_spec_path, runtime_fixture_path))
    return pairs


def _load_runtime_portability_fixture(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_model_spec_json_roundtrip_preserves_predictions(tmp_path: Path):
    model = _fit_sample_model()
    probe = np.array([[0.0, 0.0, 0.1], [0.8, -0.5, 0.6]], dtype=float)
    expected = model.predict(probe)

    target = tmp_path / "model.json"
    runtime.save_model(model, target)

    payload = json.loads(target.read_text())
    validate_model_spec(payload)
    restored = runtime.load_model_spec(target)
    clone = runtime.load_model(restored)

    np.testing.assert_allclose(clone.predict(probe), expected)
    np.testing.assert_allclose(runtime.predict(restored, probe), expected)


def test_spec_from_json_accepts_json_string():
    payload = _fit_sample_model().export_model(format="json")
    restored = spec_from_json(payload)
    validate_model_spec(restored)
    assert restored["spec_version"] == "1.0"


def test_cli_load_model_reads_json_artifact(tmp_path: Path):
    model = _fit_sample_model()
    probe = np.array([[0.0, 0.0, 0.1], [0.8, -0.5, 0.6]], dtype=float)
    target = tmp_path / "model.json"
    runtime.save_model(model, target)

    restored = _load_model(str(target))

    np.testing.assert_allclose(restored.predict(probe), model.predict(probe))


def test_cli_save_model_writes_json_and_pickle(tmp_path: Path):
    model = _fit_sample_model()
    json_target = tmp_path / "model.json"
    pickle_target = tmp_path / "model.pkl"

    assert _save_model(model, str(json_target)) == "json"
    assert _save_model(model, str(pickle_target)) == "pickle"
    assert json_target.exists()
    assert pickle_target.exists()


@pytest.mark.parametrize(
    ("model_spec_path", "runtime_fixture_path"),
    _runtime_portability_fixture_pairs(),
    ids=lambda path: path.stem,
)
def test_checked_in_model_spec_fixture_roundtrip(
    model_spec_path: Path, runtime_fixture_path: Path
):
    spec = runtime.load_model_spec(model_spec_path)
    model = runtime.load_model(spec)
    fixture = _load_runtime_portability_fixture(runtime_fixture_path)
    probe = np.asarray(fixture["probe"], dtype=float)
    expected_design_matrix = np.asarray(fixture["design_matrix"], dtype=float)
    expected_predict = np.asarray(fixture["predict"], dtype=float)

    np.testing.assert_allclose(
        runtime.design_matrix(spec, probe), expected_design_matrix, equal_nan=True
    )
    np.testing.assert_allclose(model.predict(probe), expected_predict, equal_nan=True)
    np.testing.assert_allclose(
        runtime.predict(spec, probe), expected_predict, equal_nan=True
    )


def test_validate_model_spec_rejects_missing_required_field():
    payload = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    payload.pop("basis_terms")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_spec(payload)


def test_validate_model_spec_rejects_mismatched_basis_and_coefficients():
    payload = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    payload["coefficients"] = payload["coefficients"][:-1]

    with pytest.raises(ValueError, match="one coefficient per basis term"):
        validate_model_spec(payload)


def test_validate_model_spec_rejects_empty_basis_term_kind():
    payload = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    payload["basis_terms"][0]["kind"] = ""

    with pytest.raises(ValueError, match="kind"):
        validate_model_spec(payload)


def test_runtime_validate_accepts_path_and_json_string():
    via_path = runtime.validate(MODEL_SPEC_V1_PATH)
    via_json = runtime.validate(MODEL_SPEC_V1_PATH.read_text())

    assert via_path == via_json
    assert via_path["spec_version"] == "1.0"


def test_runtime_validate_rejects_invalid_payload():
    payload = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    payload.pop("basis_terms")

    with pytest.raises(ValueError, match="missing required fields"):
        runtime.validate(payload)


def test_runtime_uses_rust_backend_for_supported_specs(monkeypatch):
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    probe = np.array([[0.0, 0.0, 0.1], [0.8, -0.5, 0.6]], dtype=float)
    calls: list[tuple[str, object]] = []

    class DummyRustBackend:
        def validate_model_spec_json(self, spec_json: str) -> None:
            calls.append(("validate", spec_json))

        def design_matrix_json(
            self, spec_json: str, rows: list[list[float]]
        ) -> list[list[float]]:
            calls.append(("design_matrix", rows))
            return [[1.0, 2.0], [3.0, 4.0]]

        def predict_json(
            self, spec_json: str, rows: list[list[float]]
        ) -> list[float]:
            calls.append(("predict", rows))
            return [5.0, 6.0]

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())
    monkeypatch.setattr(runtime, "_spec_is_rust_runtime_compatible", lambda spec: True)

    assert runtime.validate(spec) == spec
    np.testing.assert_allclose(runtime.design_matrix(spec, probe), [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(runtime.predict(spec, probe), [5.0, 6.0])

    assert [name for name, _payload in calls] == [
        "validate",
        "design_matrix",
        "predict",
    ]


def test_runtime_falls_back_to_python_when_rust_backend_is_incompatible(monkeypatch):
    spec = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    probe = np.array([[0.0, 0.0, 0.1], [0.8, -0.5, 0.6]], dtype=float)
    model = runtime.load_model(spec)
    X_processed, missing_mask = model._prepare_prediction_data(probe)
    expected_design_matrix = model._build_basis_matrix(
        X_processed, model.basis_, missing_mask
    )
    expected_prediction = model.predict(probe)
    calls: list[tuple[str, object]] = []

    class DummyRustBackend:
        def validate_model_spec_json(self, spec_json: str) -> None:
            calls.append(("validate", spec_json))

        def design_matrix_json(
            self, spec_json: str, rows: list[list[float]]
        ) -> list[list[float]]:
            calls.append(("design_matrix", rows))
            return [[999.0]]

        def predict_json(
            self, spec_json: str, rows: list[list[float]]
        ) -> list[float]:
            calls.append(("predict", rows))
            return [999.0]

    monkeypatch.setattr(runtime, "_rust_backend", DummyRustBackend())
    monkeypatch.setattr(runtime, "_spec_is_rust_runtime_compatible", lambda spec: False)

    assert runtime.validate(spec) == spec
    np.testing.assert_allclose(
        runtime.design_matrix(spec, probe),
        expected_design_matrix,
        equal_nan=True,
    )
    np.testing.assert_allclose(runtime.predict(spec, probe), expected_prediction)

    assert calls == []


def test_validate_model_spec_rejects_unsupported_major_version():
    payload = runtime.load_model_spec(MODEL_SPEC_V1_PATH)
    payload["spec_version"] = "2.0"

    with pytest.raises(ValueError, match="Unsupported model spec version"):
        validate_model_spec(payload)
