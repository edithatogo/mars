import argparse
import logging
import pickle
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pymars import cli, explain, plot
from pymars._missing import handle_missing_X, handle_missing_y
from pymars._record import EarthRecord
from pymars._util import (
    calculate_gcv,
    check_array,
    gcv_penalty_cost_effective_parameters,
)
from pymars.demos import basic_classification_demo, basic_regression_demo


class FakeSeries:
    def __init__(self, values):
        self.values = np.asarray(values)


class FakeFrame:
    def __init__(self, data):
        self._data = {key: np.asarray(value) for key, value in data.items()}
        self.columns = list(self._data)

    def drop(self, columns):
        return FakeFrame({k: v for k, v in self._data.items() if k not in columns})

    def __getitem__(self, key):
        return FakeSeries(self._data[key])

    @property
    def values(self):
        return np.column_stack(list(self._data.values()))


class FakePredFrame:
    def __init__(self, data):
        self.data = data
        self.saved = None

    def to_csv(self, path, index=False):
        self.saved = (path, index)


class FakePandasModule:
    def __init__(self, frame):
        self._frame = frame
        self.pred_frames = []

    def read_csv(self, _path):
        return self._frame

    def DataFrame(self, data):
        frame = FakePredFrame(data)
        self.pred_frames.append(frame)
        return frame


class FakeEarth:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.basis_ = [object(), object()]
        self.gcv_ = 1.2345

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] if X.ndim > 1 else np.asarray([X[0]])

    def score(self, X, y):
        del X, y
        return 0.5


class FakeInternalEarth:
    def _scrub_input_data(self, X, y):
        X = np.asarray(X)
        del y
        mask = np.zeros_like(X, dtype=bool)
        return X, mask, None

    def _build_basis_matrix(self, X_proc, basis, mask):
        X_proc = np.asarray(X_proc)
        del basis, mask
        return np.column_stack([np.ones(X_proc.shape[0]), X_proc[:, 0]])


class FakePlotModel:
    def __init__(self):
        self.basis_ = [object()]
        self.earth_ = FakeInternalEarth()

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0]


class FakeExplainModel:
    def __init__(self):
        self.fitted_ = True
        self.basis_ = [SimpleNamespace(name="b0"), SimpleNamespace(name="b1")]
        self.coef_ = np.array([1.0, 2.0])
        self.gcv_ = 0.42
        self.feature_importances_ = np.array([0.7, 0.3])

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + X[:, 1]

    def score(self, X, y):
        del X, y
        return 0.9


class FakeCliEarth(FakeEarth):
    pass


class FakeCliModel:
    def __init__(self):
        self.basis_ = [object()]

    def predict(self, X):
        X = np.asarray(X)
        return np.full(X.shape[0], 3.14)

    def score(self, X, y):
        del X, y
        return 0.75


class FakeDemoModel:
    def __init__(self, *args, **kwargs):
        del args
        self.kwargs = kwargs
        self.earth_ = SimpleNamespace(summary=lambda: None)

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.zeros(X.shape[0])

    def score(self, X, y):
        del X, y
        return 0.25


class FakeClassifierDemoModel(FakeDemoModel):
    def predict(self, X):
        X = np.asarray(X)
        return np.zeros(X.shape[0], dtype=int)


class FakeAdvancedExampleModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.basis_ = [object(), object(), object()]
        self.gcv_ = 0.321
        self.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1, 0.0])

    def fit(self, X, y):
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] * 0.0 + 1.0

    def score(self, X, y):
        del X, y
        return 0.88


class DummyBasisFunction:
    def __init__(
        self,
        involved: tuple[int, ...] = (0,),
        constant: bool = False,
        gcv_score: float = 0.0,
        rss_score: float = 0.0,
    ):
        self._involved = involved
        self._constant = constant
        self.gcv_score_ = gcv_score
        self.rss_score_ = rss_score

    def is_constant(self):
        return self._constant

    def get_involved_variables(self):
        return list(self._involved)

    def transform(self, X, missing_mask):
        X = np.asarray(X)
        del missing_mask
        return np.ones(X.shape[0])

    def __str__(self):
        return "DummyBasisFunction"


def test_utilities_cover_edge_cases():
    arr = check_array([[1, 2], [3, 4]], ensure_min_samples=2, ensure_min_features=2)
    assert arr.shape == (2, 2)
    with pytest.raises(ValueError):
        check_array([[1]], ensure_min_samples=2)
    with pytest.raises(ValueError):
        check_array([[1]], ensure_2d=True, ensure_min_features=2)

    assert gcv_penalty_cost_effective_parameters(2, 1, 3.0, 10) == 5.0
    assert calculate_gcv(10.0, 5, 2.0) > 0


def test_earth_gcv_feature_importance_empty_basis():
    from pymars import Earth

    model = Earth(feature_importance_type="gcv")
    model.fitted_ = True
    model.basis_ = []

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    model._calculate_feature_importances(X)

    assert model.feature_importances_ is not None
    assert model.feature_importances_.size == 0


def test_missing_helpers_cover_strategies():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    assert np.allclose(
        handle_missing_X(X, strategy="mean"), np.array([[1.0, 4.0], [3.0, 4.0]])
    )
    assert np.allclose(
        handle_missing_X(X, strategy="median"),
        np.array([[1.0, 4.0], [3.0, 4.0]]),
    )
    assert np.allclose(
        handle_missing_X(X, strategy="most_frequent"),
        np.array([[1.0, 4.0], [3.0, 4.0]]),
    )
    assert np.allclose(
        handle_missing_X(
            X, strategy="pass_through", allow_missing_for_some_strategies=True
        ),
        X,
        equal_nan=True,
    )
    with pytest.raises(ValueError):
        handle_missing_X(X, strategy="error")


def test_missing_helpers_cover_remaining_branches():
    X_object = np.array([1.0, 2.0], dtype=object)
    with pytest.raises(TypeError):
        handle_missing_X(X_object, strategy="mean")
    X_object_with_nan = np.array([1.0, np.nan], dtype=float)
    assert np.allclose(
        handle_missing_X(X_object_with_nan, strategy="mean"), np.array([1.0, 1.0])
    )
    assert np.allclose(
        handle_missing_X(X_object_with_nan, strategy="median"), np.array([1.0, 1.0])
    )
    assert np.allclose(
        handle_missing_X(X_object_with_nan, strategy="most_frequent"),
        np.array([1.0, 1.0]),
    )
    assert np.allclose(
        handle_missing_X(
            X_object_with_nan,
            strategy="pass_through",
            allow_missing_for_some_strategies=True,
        ),
        X_object_with_nan,
        equal_nan=True,
    )
    with pytest.raises(ValueError):
        handle_missing_X(X_object_with_nan, strategy="pass_through")
    with pytest.raises(ValueError):
        handle_missing_X(X_object_with_nan, strategy="unknown")

    y = np.array([1.0, np.nan, 3.0])
    processed, mask = handle_missing_y(y, strategy=None, problem_type="regression")
    assert np.allclose(processed, np.array([1.0, 2.0, 3.0]))
    assert mask.sum() == 1

    with pytest.raises(ValueError):
        handle_missing_y(y, strategy=None, problem_type="classification")

    y_object = np.array([np.nan, np.nan], dtype=float)
    processed_reg, mask_reg = handle_missing_y(
        y_object, strategy="most_frequent", problem_type="regression"
    )
    assert np.allclose(processed_reg, np.array([0.0, 0.0]))
    assert mask_reg.sum() == 2

    processed_removed, mask_removed = handle_missing_y(y, strategy="remove_samples")
    assert processed_removed.shape == (2,)
    assert mask_removed.sum() == 1

    with pytest.raises(ValueError):
        handle_missing_y(y, strategy="mean", problem_type="classification")
    with pytest.raises(ValueError):
        handle_missing_y(y, strategy="median", problem_type="classification")
    with pytest.raises(ValueError):
        handle_missing_y(y, strategy="unknown")

    y = np.array([1.0, np.nan, 3.0])
    processed, mask = handle_missing_y(y, strategy="mean")
    assert processed.shape == y.shape
    assert mask.dtype == bool
    processed, mask = handle_missing_y(y, strategy="remove_samples")
    assert processed.shape == (2,)
    assert mask.sum() == 1
    with pytest.raises(ValueError):
        handle_missing_y(y, strategy="error")


def test_record_logs_and_string_representation():
    X = np.ones((3, 2))
    y = np.array([1.0, 2.0, 3.0])
    record = EarthRecord(X, y, earth_model_instance=SimpleNamespace(alpha=1))
    record.log_forward_pass_step([], np.array([1.0]), 0.5)
    record.log_pruning_step([], np.array([1.0]), 0.4, 0.3)
    record.set_final_model([], np.array([1.0]), 0.4, 0.3, 0.1)
    text = str(record)
    assert "Earth Model Fit Record" in text
    assert "Final Selected Model" in text


def test_plot_helpers_cover_both_paths():
    model = FakePlotModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    axes = plot.plot_basis_functions(model, X)
    assert axes is not None
    axes = plot.plot_residuals(model, X, np.array([1.0, 2.0]))
    assert axes is not None


def test_explain_helpers_cover_both_paths(monkeypatch):
    model = FakeExplainModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(
        explain,
        "partial_dependence",
        lambda *_args, **_kwargs: {
            "values": [np.array([0.0, 1.0])],
            "average": [np.array([1.0, 2.0])],
        },
    )
    fig, axes = explain.plot_partial_dependence(model, X, [0])
    assert fig is not None and axes is not None

    def raise_value_error(*args, **kwargs):
        del args, kwargs
        raise ValueError("boom")

    monkeypatch.setattr(explain, "partial_dependence", raise_value_error)
    fig, axes = explain.plot_partial_dependence(model, X, [0])
    assert fig is not None and axes is not None

    fig, axes = explain.plot_individual_conditional_expectation(model, X, [0])
    assert fig is not None and axes is not None
    explanation = explain.get_model_explanation(model, X, feature_names=["a", "b"])
    assert explanation["model_summary"]["n_basis_functions"] == 2
    assert explanation["feature_importance"]["values"] == [0.7, 0.3]


def test_explain_helpers_cover_remaining_paths(monkeypatch):
    del monkeypatch
    fitted_model = FakeExplainModel()
    large_x = np.ones((10000, 2))
    unfitted_model = FakeExplainModel()
    unfitted_model.fitted_ = False

    with pytest.raises(ValueError):
        explain.plot_partial_dependence(
            unfitted_model,
            large_x,
            [0],
        )

    with pytest.raises(ValueError):
        explain.plot_individual_conditional_expectation(
            unfitted_model,
            large_x,
            [0],
        )

    fitted_model.feature_importances_ = None
    explanation = explain.get_model_explanation(fitted_model, large_x)
    assert explanation["model_summary"]["r2_score"] == "Too large to compute"
    assert explanation["feature_importance"] == {}


def test_cli_helpers_cover_success_and_error_paths(tmp_path, monkeypatch, capsys):
    frame = FakeFrame({"x1": [1.0, 2.0], "target": [3.0, 4.0]})
    fake_pd = FakePandasModule(frame)
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: fake_pd)
    monkeypatch.setattr(cli, "Earth", FakeCliEarth)

    output_model = tmp_path / "model.pkl"
    cli.fit_model(
        argparse.Namespace(
            input="ignored.csv",
            target="target",
            output_model=str(output_model),
            max_degree=1,
            penalty=3.0,
            max_terms=None,
        )
    )
    assert output_model.exists()

    with pytest.raises(ValueError):
        cli.fit_model(
            argparse.Namespace(
                input="ignored.csv",
                target="missing",
                output_model=str(output_model),
                max_degree=1,
                penalty=3.0,
                max_terms=None,
            )
        )

    model_path = tmp_path / "pred.pkl"
    with model_path.open("wb") as f:
        pickle.dump(FakeCliModel(), f)

    output_pred = tmp_path / "pred.csv"
    cli.make_predictions(
        argparse.Namespace(
            model=str(model_path),
            input="ignored.csv",
            output=str(output_pred),
        )
    )
    assert output_pred.exists() or fake_pd.pred_frames

    cli.score_model(
        argparse.Namespace(
            model=str(model_path),
            input="ignored.csv",
            target="target",
        )
    )
    out = capsys.readouterr().out
    assert "Model fitted" in out or "Predictions made" in out or "Model R² score" in out


def test_demo_entrypoints(monkeypatch):
    monkeypatch.setattr(basic_regression_demo, "EarthRegressor", FakeDemoModel)
    monkeypatch.setattr(
        basic_classification_demo, "EarthClassifier", FakeClassifierDemoModel
    )
    basic_regression_demo.main()
    basic_classification_demo.main()


def test_advanced_example_path(monkeypatch):
    from pymars.demos import advanced_example

    monkeypatch.setattr(advanced_example.earth, "Earth", FakeAdvancedExampleModel)
    monkeypatch.setattr(
        advanced_example.earth,
        "plot_partial_dependence",
        lambda *_args, **_kwargs: (object(), [object(), object(), object()]),
    )
    monkeypatch.setattr(
        advanced_example.earth,
        "get_model_explanation",
        lambda *_args, **_kwargs: {
            "model_summary": {
                "n_features": 5,
                "n_basis_functions": 3,
                "gcv_score": 0.321,
            }
        },
    )
    monkeypatch.setattr(advanced_example.plt, "savefig", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(advanced_example.plt, "close", lambda *_args, **_kwargs: None)

    model, score = advanced_example.advanced_example()
    assert isinstance(model, FakeAdvancedExampleModel)
    assert isinstance(score, float)


def test_earth_internal_branches(monkeypatch):
    del monkeypatch
    from pymars import Earth

    earth_model = Earth()
    earth_model.feature_importance_type = "nb_subsets"
    earth_model.record_ = None
    earth_model._calculate_feature_importances(np.zeros((2, 3)))
    assert earth_model.feature_importances_.size == 0

    earth_model.record_ = SimpleNamespace(
        pruning_trace_basis_functions_=[
            [DummyBasisFunction(involved=(0, 1)), DummyBasisFunction(involved=(1,))]
        ]
    )
    earth_model._calculate_feature_importances(np.zeros((2, 3)))
    assert earth_model.feature_importances_.tolist() == [0.5, 0.5, 0.0]

    earth_model.feature_importance_type = "gcv"
    earth_model.record_ = SimpleNamespace(n_features=2)
    earth_model.basis_ = [
        DummyBasisFunction(constant=True),
        DummyBasisFunction(involved=(0, 1), gcv_score=2.0),
    ]
    earth_model._calculate_feature_importances(np.zeros((2, 2)))
    assert earth_model.feature_importances_.tolist() == [0.5, 0.5]

    earth_model.feature_importance_type = "rss"
    earth_model.basis_ = [
        DummyBasisFunction(constant=True),
        DummyBasisFunction(involved=(0,), rss_score=3.0),
    ]
    earth_model._calculate_feature_importances(np.zeros((2, 2)))
    assert earth_model.feature_importances_.tolist() == [1.0, 0.0]

    earth_model.feature_importance_type = "mystery"
    earth_model._calculate_feature_importances(np.zeros((2, 2)))
    assert earth_model.feature_importances_.tolist() == [0.0, 0.0]


def test_earth_predict_and_summary_branches(monkeypatch):
    from pymars import Earth
    from pymars._basis import ConstantBasisFunction

    earth_model = Earth()
    with pytest.raises(RuntimeError):
        earth_model.predict(np.zeros((1, 1)))

    earth_model.fitted_ = True
    earth_model.basis_ = None
    earth_model.coef_ = np.array([1.0])
    earth_model.record_ = SimpleNamespace(n_features=1)
    with pytest.raises(ValueError):
        earth_model.predict(np.zeros((1, 1)))

    earth_model.basis_ = [ConstantBasisFunction()]
    earth_model.coef_ = None
    with pytest.raises(ValueError):
        earth_model.predict(np.zeros((1, 1)))

    earth_model.basis_ = [DummyBasisFunction(constant=True)]
    earth_model.coef_ = np.array([1.0])
    earth_model.record_ = None
    with pytest.raises(ValueError):
        earth_model.predict(np.zeros((1, 1)))

    earth_model.record_ = SimpleNamespace(n_features=2, y_mean_=4.2)
    earth_model.basis_ = []
    earth_model.coef_ = np.array([])
    assert np.allclose(earth_model.predict(np.zeros((2, 2))), np.array([4.2, 4.2]))

    earth_model.record_ = SimpleNamespace(n_features=1)
    earth_model.basis_ = [ConstantBasisFunction()]
    earth_model.coef_ = np.array([1.0])
    monkeypatch.setattr(
        earth_model,
        "_build_basis_matrix",
        lambda *_args, **_kwargs: np.ones((2, 2)),
    )
    assert np.allclose(earth_model.predict(np.zeros((2, 1))), np.array([1.0, 1.0]))

    earth_model.basis_ = [DummyBasisFunction()]
    earth_model.coef_ = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        earth_model.predict(np.zeros((2, 1)))

    earth_model.basis_ = [DummyBasisFunction()]
    earth_model.coef_ = np.array([1.0])
    earth_model.record_ = SimpleNamespace(n_samples=2, n_features=1)
    earth_model.gcv_ = 0.5
    earth_model.rss_ = 1.0
    earth_model.mse_ = 0.5
    earth_model.feature_importances_ = np.array([0.4])
    assert earth_model.summary() is None
    assert "Model not yet fitted." in Earth().summary()
    assert Earth().summary_feature_importances() == (
        "Model not yet fitted. Feature importances not available."
    )
    earth_model.fitted_ = True
    earth_model.feature_importances_ = None
    assert (
        "Feature importances not computed" in earth_model.summary_feature_importances()
    )
    earth_model.feature_importances_ = np.array([])
    assert (
        "No features or importances available."
        in earth_model.summary_feature_importances()
    )
    earth_model.feature_importances_ = np.array([0.4])
    assert "x0" in earth_model.summary_feature_importances()


def test_earth_fit_fallback_and_summary_paths(monkeypatch, caplog):
    from pymars import Earth

    class FakeForwardPasser:
        def __init__(self, model):
            self.model = model

        def run(self, **kwargs):
            del kwargs
            return [DummyBasisFunction()], np.array([1.0])

    class FakePruningPasser:
        def __init__(self, model):
            self.model = model

        def run(self, **kwargs):
            del kwargs
            return [DummyBasisFunction()], None, np.inf

    monkeypatch.setattr("pymars._forward.ForwardPasser", FakeForwardPasser)
    monkeypatch.setattr("pymars._pruning.PruningPasser", FakePruningPasser)

    model = Earth(max_terms=3, penalty=0.0, feature_importance_type=None)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model.fit(X, y)

    assert model.fitted_
    assert len(model.basis_) == 1
    assert np.isclose(model.coef_[0], np.mean(y))

    model.summary()
    model.record_ = None
    model.fitted_ = True
    with caplog.at_level(logging.INFO, logger="pymars.earth"):
        assert model.summary() is None
    assert "Number of samples: N/A" in caplog.text
    assert "Number of features: N/A" in caplog.text


def test_earth_fit_forward_empty_branch(monkeypatch):
    from pymars import Earth

    class FakeForwardPasser:
        def __init__(self, model):
            self.model = model

        def run(self, **kwargs):
            del kwargs
            return [], np.array([1.0])

    monkeypatch.setattr("pymars._forward.ForwardPasser", FakeForwardPasser)

    model = Earth(max_terms=1, penalty=0.0)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    model.fit(X, y)

    assert model.fitted_
    assert len(model.basis_) == 1
    assert np.isclose(model.coef_[0], np.mean(y))


def test_cli_main_dispatch_and_help(monkeypatch, capsys):
    monkeypatch.setattr(cli, "fit_model", lambda _args: None)
    monkeypatch.setattr(cli, "make_predictions", lambda _args: None)
    monkeypatch.setattr(cli, "score_model", lambda _args: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pymars",
            "fit",
            "--input",
            "in.csv",
            "--target",
            "target",
            "--output-model",
            "model.pkl",
        ],
    )
    cli.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pymars",
            "predict",
            "--model",
            "model.pkl",
            "--input",
            "in.csv",
            "--output",
            "pred.csv",
        ],
    )
    cli.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pymars",
            "score",
            "--model",
            "model.pkl",
            "--input",
            "in.csv",
            "--target",
            "target",
        ],
    )
    cli.main()

    monkeypatch.setattr(sys, "argv", ["pymars"])
    cli.main()
    out = capsys.readouterr().out
    assert "Available commands" in out


def test_cli_fit_and_score_branches(tmp_path, monkeypatch, capsys):
    import pandas as pd

    class FakeCliModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.basis_ = [DummyBasisFunction()]
            self.gcv_ = 0.1234

        def fit(self, X, y):
            self.X_ = X
            self.y_ = y
            return self

        def score(self, X, y):
            del X, y
            return 0.75

    monkeypatch.setattr(cli, "Earth", FakeCliModel)
    monkeypatch.setattr(cli.pickle, "dump", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.pickle, "load", lambda *_args, **_kwargs: FakeCliModel())

    frame = pd.DataFrame({"x0": [1.0, 2.0], "target": [3.0, 4.0]})
    input_path = tmp_path / "input.csv"
    frame.to_csv(input_path, index=False)
    model_path = tmp_path / "model.pkl"
    output_path = tmp_path / "pred.csv"

    cli.fit_model(
        SimpleNamespace(
            input=str(input_path),
            target="target",
            output_model=str(model_path),
            max_degree=2,
            penalty=1.5,
            max_terms=5,
        )
    )

    cli.score_model(
        SimpleNamespace(
            model=str(model_path),
            input=str(input_path),
            target="target",
        )
    )
    out = capsys.readouterr().out
    assert "Number of selected basis functions" in out
    assert "Model R² score" in out

    with pytest.raises(ValueError):
        cli.fit_model(
            SimpleNamespace(
                input=str(input_path),
                target="missing",
                output_model=str(model_path),
                max_degree=1,
                penalty=3.0,
                max_terms=None,
            )
        )
    with pytest.raises(ValueError):
        cli.score_model(
            SimpleNamespace(
                model=str(model_path),
                input=str(input_path),
                target="missing",
            )
        )
