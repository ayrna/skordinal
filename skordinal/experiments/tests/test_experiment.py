"""Tests for the experiment runner module."""

import math

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skordinal.experiments import Experiment, ExperimentResult, ModelConfig

_MINIMAL_CONF: ModelConfig = ModelConfig(SVC())
_CONF_CV: ModelConfig = ModelConfig(SVC(), param_grid={"C": [0.1, 1.0]})


@pytest.fixture
def split_with_test():
    """Return (X_train, y_train, X_test, y_test) with three balanced classes."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((30, 4))
    y_train = np.tile([0, 1, 2], 10)
    X_test = rng.standard_normal((15, 4))
    y_test = np.tile([0, 1, 2], 5)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def split_train_only():
    """Return (X_train, y_train, None, None) for train-only partition tests."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((30, 4))
    y_train = np.tile([0, 1, 2], 10)
    return X_train, y_train, None, None


def _make_experiment(model=None, **kwargs):
    """Build an Experiment with default eval_metrics for run() seam tests."""
    if model is None:
        model = _MINIMAL_CONF
    kwargs.setdefault("eval_metrics", ["mean_absolute_error"])
    return Experiment(model, **kwargs)


def _call_run(experiment, X_train, y_train, X_test, y_test):
    """Invoke run() with fixed identity kwargs; returns the ExperimentResult."""
    return experiment.run(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="ds",
        classifier_name="cfg",
        resample_id=0,
    )


@pytest.mark.parametrize(
    "overrides, exc_type, match",
    [
        pytest.param(
            {"model": SVC()},
            TypeError,
            "'model' must be a ModelConfig instance",
            id="bare-estimator",
        ),
        pytest.param(
            {"eval_metrics": []},
            ValueError,
            "'eval_metrics' must be a non-empty list",
            id="empty-eval-metrics",
        ),
        pytest.param(
            {"eval_metrics": ["ranked_probability_score"]},
            ValueError,
            "ranked_probability_score",
            id="proba-only-eval-metric",
        ),
        pytest.param(
            {"tuning_metric": "mean_absolute_error"},
            ValueError,
            "only registered as 'neg_",
            id="bare-loss-tuning-metric",
        ),
        pytest.param(
            {"input_preprocessing": "std"},
            TypeError,
            "'input_preprocessing' must be None",
            id="string-input-preprocessing",
        ),
    ],
)
def test_constructor_rejects_invalid_arguments(overrides, exc_type, match):
    """Each invalid argument raises at construction, before anything is fitted."""
    kwargs = {
        "model": _MINIMAL_CONF,
        "eval_metrics": ["mean_absolute_error"],
        **overrides,
    }
    with pytest.raises(exc_type, match=match):
        Experiment(kwargs.pop("model"), **kwargs)


@pytest.mark.parametrize("kwargs, expected", [({}, None), ({"random_state": 42}, 42)])
def test_random_state_stored(kwargs, expected):
    """random_state defaults to None and is stored as given on the instance."""
    exp = Experiment(_MINIMAL_CONF, eval_metrics=["mean_absolute_error"], **kwargs)
    assert exp.random_state == expected


@pytest.mark.parametrize("with_indices", [True, False], ids=["indices", "defaults"])
def test_run_forwards_identity_and_indices(split_with_test, with_indices):
    """The result echoes the identity kwargs; indices verbatim or None."""
    X_train, y_train, X_test, y_test = split_with_test
    train_index = np.arange(X_train.shape[0]) if with_indices else None
    test_index = np.arange(X_test.shape[0]) if with_indices else None

    result = _make_experiment().run(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="my_dataset",
        classifier_name="my_conf",
        resample_id=42,
        train_index=train_index,
        test_index=test_index,
    )

    assert isinstance(result, ExperimentResult)
    assert result.train_predicted_y.shape == (30,)
    assert result.dataset_name == "my_dataset"
    assert result.classifier_name == "my_conf"
    assert result.resample_id == 42
    if with_indices:
        npt.assert_array_equal(result.train_index, train_index)
        npt.assert_array_equal(result.test_index, test_index)
    else:
        assert result.train_index is None
        assert result.test_index is None


@pytest.mark.parametrize("has_test", [True, False])
def test_run_test_present_vs_absent(split_with_test, split_train_only, has_test):
    """With test data: test_predicted_y is an array and metrics are finite; without: None and NaN."""
    X_train, y_train, X_test, y_test = split_with_test if has_test else split_train_only
    result = _call_run(_make_experiment(), X_train, y_train, X_test, y_test)

    metric_test_key = "mean_absolute_error_test"
    if has_test:
        assert result.test_predicted_y is not None
        assert result.test_predicted_y.shape == (15,)
        assert math.isfinite(result.test_metrics[metric_test_key])
        assert math.isfinite(result.test_metrics["time_test"])
    else:
        assert result.test_predicted_y is None
        assert math.isnan(result.test_metrics[metric_test_key])


def test_run_timing_no_cv(split_with_test):
    """Singleton param grid produces NaN cv_time_* and best_params echoes the config."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(_make_experiment(), X_train, y_train, X_test, y_test)

    assert math.isnan(result.train_metrics["cv_time_train"])
    assert math.isnan(result.test_metrics["cv_time_test"])
    assert math.isfinite(result.train_metrics["time_train"])
    assert math.isfinite(result.test_metrics["time_test"])
    assert result.best_params == {}


def test_run_timing_with_cv(split_with_test):
    """Multi-value param grid produces finite cv_time_* and best_params reflects a searched value."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(_make_experiment(_CONF_CV), X_train, y_train, X_test, y_test)

    assert math.isfinite(result.train_metrics["cv_time_train"])
    assert not math.isnan(result.train_metrics["cv_time_train"])
    assert math.isfinite(result.test_metrics["cv_time_test"])
    assert math.isfinite(result.train_metrics["time_train"])
    assert math.isfinite(result.test_metrics["time_test"])
    assert result.best_params.get("C") in _CONF_CV.param_grid["C"]


def test_run_preprocessing_train_only_does_not_raise(split_train_only):
    """input_preprocessing with X_test=None runs without transforming None."""
    exp = _make_experiment(input_preprocessing=StandardScaler())
    X_train, y_train, X_test, y_test = split_train_only

    result = _call_run(exp, X_train, y_train, X_test, y_test)

    assert result.test_predicted_y is None
    assert math.isnan(result.test_metrics["mean_absolute_error_test"])


def test_run_search_accepts_array_grid_and_scalar_entry(split_with_test):
    """np.array candidates trigger a search; a scalar entry rides along."""
    X_train, y_train, X_test, y_test = split_with_test
    # "linear" is not SVC's default, so dropping the scalar would show up
    conf = ModelConfig(
        SVC(), param_grid={"C": np.array([0.1, 1.0]), "kernel": "linear"}
    )
    assert conf.needs_search

    result = _call_run(_make_experiment(conf), X_train, y_train, X_test, y_test)

    assert result.best_params["C"] in (0.1, 1.0)
    assert result.best_model.kernel == "linear"


def test_run_rejects_y_test_without_x_test(split_train_only):
    """y_test without X_test raises ValueError, not a bare assert."""
    X_train, y_train, _, _ = split_train_only
    with pytest.raises(ValueError, match="'y_test' was given without 'X_test'"):
        _call_run(_make_experiment(), X_train, y_train, None, y_train)


def test_run_best_model_carries_no_refit_metadata(split_with_test):
    """The direct-fit path leaves no GridSearchCV-shaped attributes on best_model."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(_make_experiment(), X_train, y_train, X_test, y_test)

    assert not hasattr(result.best_model, "best_estimator_")
    assert not hasattr(result.best_model, "best_params_")
    assert not hasattr(result.best_model, "refit_time_")


def test_run_preprocessing_does_not_mutate_inputs(split_with_test):
    """Preprocessing operates on copies; the caller's arrays are unchanged."""
    exp = _make_experiment(input_preprocessing=StandardScaler())
    X_train, y_train, X_test, y_test = split_with_test
    train_before = X_train.copy()
    test_before = X_test.copy()

    _call_run(exp, X_train, y_train, X_test, y_test)

    npt.assert_array_equal(X_train, train_before)
    npt.assert_array_equal(X_test, test_before)


def test_run_metric_keys_for_each_eval_metric(split_with_test):
    """Each eval metric yields a _train and a _test key, reported unnegated."""
    X_train, y_train, X_test, y_test = split_with_test
    exp = _make_experiment(eval_metrics=["mean_absolute_error", "accuracy_score"])
    result = _call_run(exp, X_train, y_train, X_test, y_test)

    for name in ("mean_absolute_error", "accuracy_score"):
        # mean_absolute_error is a loss, but reporting must not negate it
        assert result.train_metrics[name + "_train"] >= 0
        assert result.test_metrics[name + "_test"] >= 0


def test_run_proba_absent_without_predict_proba(split_with_test):
    """Both proba fields are None when the estimator has no predict_proba."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(_make_experiment(), X_train, y_train, X_test, y_test)

    assert result.y_proba is None
    assert result.train_y_proba is None


def test_run_proba_present_with_predict_proba(split_with_test):
    """Both proba fields are populated when predict_proba is supported."""
    X_train, y_train, X_test, y_test = split_with_test
    conf = ModelConfig(SVC(), param_grid={"probability": [True]})
    result = _call_run(_make_experiment(conf), X_train, y_train, X_test, y_test)

    n_classes = np.unique(y_train).size
    assert result.y_proba is not None
    assert result.y_proba.shape == (X_test.shape[0], n_classes)
    assert result.train_y_proba is not None
    assert result.train_y_proba.shape == (X_train.shape[0], n_classes)


def test_run_no_test_labels_skips_test_proba(split_with_test):
    """y_proba stays None when y_test is None even if X_test is given."""
    X_train, y_train, X_test, _ = split_with_test
    conf = ModelConfig(SVC(), param_grid={"probability": [True]})
    result = _call_run(_make_experiment(conf), X_train, y_train, X_test, None)

    assert result.y_proba is None
    assert result.train_y_proba is not None


def test_run_proba_none_when_predict_proba_raises(split_with_test, monkeypatch):
    """A call-time AttributeError from predict_proba warns and yields None."""
    X_train, y_train, X_test, y_test = split_with_test

    def raise_attribute_error(self, X):
        """Raise as meta-estimators without probabilistic members do."""
        raise AttributeError("predict_proba is unavailable")

    monkeypatch.setattr(SVC, "predict_proba", raise_attribute_error)
    with pytest.warns(RuntimeWarning, match="probabilities are omitted"):
        result = _call_run(_make_experiment(), X_train, y_train, X_test, y_test)

    assert result.train_y_proba is None
    assert result.y_proba is None


@pytest.mark.parametrize(
    "input_preprocessing,expected_type",
    [(StandardScaler(), StandardScaler), (None, type(None))],
    ids=["scaler", "none"],
)
def test_run_records_the_fitted_scaler(
    split_with_test, input_preprocessing, expected_type
):
    """run() hands back the scaler it fitted, so save can persist it."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(
        _make_experiment(input_preprocessing=input_preprocessing),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    assert isinstance(result.scaler, expected_type)
    if input_preprocessing is not None:
        npt.assert_allclose(
            result.scaler.transform(X_train)[0],
            expected_type().fit(X_train).transform(X_train)[0],
        )


def test_transformer_instance_is_cloned_and_seeded(split_with_test):
    """The caller's transformer is never fitted; its clone gets the run seed."""
    X_train, y_train, X_test, y_test = split_with_test
    transformer = PCA(n_components=2)
    exp = _make_experiment(input_preprocessing=transformer, random_state=7)
    result = _call_run(exp, X_train, y_train, X_test, y_test)

    assert result.scaler is not transformer
    assert not hasattr(transformer, "components_")
    assert result.scaler.random_state == 7
    assert transformer.random_state is None
    # The estimator saw the reduced features, so both splits were transformed
    assert result.best_model.n_features_in_ == 2


def test_run_scores_on_the_fitted_scale(split_with_test):
    """A split missing an intermediate class keeps the gaps the model was fitted on."""

    class Extremes(DummyClassifier):
        """Predicts only the two extreme classes, never the middle one."""

        def predict(self, X):
            return np.resize(np.array([0, 20]), len(X))

    X_train = np.zeros((30, 2))
    y_train = np.repeat([0, 10, 20], 10)
    X_test = np.zeros((6, 2))
    y_test = np.array([0, 0, 20, 20, 0, 20])
    exp = _make_experiment(
        ModelConfig(Extremes(strategy="most_frequent")),
        eval_metrics=["accuracy_off1_score"],
    )

    result = _call_run(exp, X_train, y_train, X_test, y_test)

    # Without the fitted scale the two extremes look adjacent, scoring 1.0
    npt.assert_array_equal(result.best_model.classes_, [0, 10, 20])
    assert result.test_metrics["accuracy_off1_score_test"] == pytest.approx(2 / 3)
