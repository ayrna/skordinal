"""Tests for the experiment runner module."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from skordinal.experiments import Experiment, ExperimentResult

_MINIMAL_CONF: dict = {"classifier": "SVC", "parameters": {}}
_CONF_CV: dict = {"classifier": "SVC", "parameters": {"C": [0.1, 1.0]}}


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


def _make_experiment(configuration=None, **kwargs):
    """Build an Experiment with default eval_metrics for run() seam tests."""
    if configuration is None:
        configuration = _MINIMAL_CONF
    kwargs.setdefault("eval_metrics", ["mean_absolute_error"])
    return Experiment(configuration, **kwargs)


def _call_run(experiment, X_train, y_train, X_test, y_test):
    """Invoke run() with fixed identity kwargs; returns the ExperimentResult."""
    return experiment.run(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="ds",
        classifier_name="cfg",
        resample_id="0",
    )


def test_empty_eval_metrics_raises():
    """An empty eval_metrics list raises ValueError."""
    with pytest.raises(ValueError, match="'eval_metrics' must be a non-empty list"):
        Experiment(_MINIMAL_CONF, eval_metrics=[])


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("std", "std"),
        ("norm", "norm"),
        (" STD ", "std"),
        ("NORM", "norm"),
    ],
)
def test_input_preprocessing_accepted_and_normalized(raw, expected):
    """Valid input_preprocessing values are accepted and lower-stripped."""
    exp = Experiment(
        _MINIMAL_CONF, eval_metrics=["mean_absolute_error"], input_preprocessing=raw
    )
    assert exp.input_preprocessing == expected


@pytest.mark.parametrize("bad_value", ["minmax", ""])
def test_input_preprocessing_invalid_raises(bad_value):
    """Unrecognised input_preprocessing values raise ValueError."""
    with pytest.raises(ValueError, match="'input_preprocessing' must be one of"):
        Experiment(
            _MINIMAL_CONF,
            eval_metrics=["mean_absolute_error"],
            input_preprocessing=bad_value,
        )


@pytest.mark.parametrize("kwargs, expected", [({}, None), ({"random_state": 42}, 42)])
def test_random_state_stored(kwargs, expected):
    """random_state defaults to None and is stored as given on the instance."""
    exp = Experiment(_MINIMAL_CONF, eval_metrics=["mean_absolute_error"], **kwargs)
    assert exp.random_state == expected


def test_configuration_is_deep_copied():
    """Mutating the original configuration dict does not affect the stored copy."""
    original = {"classifier": "SVC", "parameters": {"C": [1]}}
    exp = Experiment(original, eval_metrics=["mean_absolute_error"])
    original["parameters"]["C"].append(10)
    assert exp.configuration["parameters"]["C"] == [1]


def test_run_returns_experiment_result(split_with_test):
    """run returns a populated ExperimentResult."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(_make_experiment(), X_train, y_train, X_test, y_test)

    assert isinstance(result, ExperimentResult)
    assert result.train_predicted_y.shape == (30,)


def test_run_identity_passthrough(split_with_test):
    """dataset_name, classifier_name, and resample_id are forwarded verbatim."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _make_experiment().run(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="my_dataset",
        classifier_name="my_conf",
        resample_id="42",
    )

    assert result.dataset_name == "my_dataset"
    assert result.classifier_name == "my_conf"
    assert result.resample_id == "42"


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
    assert result.best_params == _MINIMAL_CONF["parameters"]


def test_run_timing_with_cv(split_with_test):
    """Multi-value param grid produces finite cv_time_* and best_params reflects a searched value."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(_make_experiment(_CONF_CV), X_train, y_train, X_test, y_test)

    assert math.isfinite(result.train_metrics["cv_time_train"])
    assert not math.isnan(result.train_metrics["cv_time_train"])
    assert math.isfinite(result.test_metrics["cv_time_test"])
    assert math.isfinite(result.train_metrics["time_train"])
    assert math.isfinite(result.test_metrics["time_test"])
    assert result.best_params.get("C") in _CONF_CV["parameters"]["C"]


@pytest.mark.parametrize("preprocessing", ["std", "norm"])
def test_run_preprocessing_train_only_raises(split_train_only, preprocessing):
    """input_preprocessing with X_test=None raises ValueError."""
    exp = _make_experiment(input_preprocessing=preprocessing)
    X_train, y_train, X_test, y_test = split_train_only
    with pytest.raises(ValueError):
        _call_run(exp, X_train, y_train, X_test, y_test)


def test_run_preprocessing_does_not_mutate_inputs(split_with_test):
    """input_preprocessing='std' operates on copies; caller's arrays are unchanged."""
    exp = _make_experiment(input_preprocessing="std")
    X_train, y_train, X_test, y_test = split_with_test
    train_before = X_train.copy()
    test_before = X_test.copy()

    _call_run(exp, X_train, y_train, X_test, y_test)

    npt.assert_array_equal(X_train, train_before)
    npt.assert_array_equal(X_test, test_before)


def test_run_metric_keys_for_each_eval_metric(split_with_test):
    """Each eval metric yields a _train and a _test key in the result."""
    X_train, y_train, X_test, y_test = split_with_test
    exp = _make_experiment(eval_metrics=["mean_absolute_error", "accuracy_score"])
    result = _call_run(exp, X_train, y_train, X_test, y_test)

    for name in ("mean_absolute_error", "accuracy_score"):
        assert name + "_train" in result.train_metrics
        assert name + "_test" in result.test_metrics


def test_run_y_proba_absent_without_predict_proba(split_with_test):
    """y_proba is None when the fitted estimator has no predict_proba."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run(_make_experiment(), X_train, y_train, X_test, y_test)

    assert result.y_proba is None


def test_run_y_proba_present_with_predict_proba(split_with_test):
    """y_proba is populated when the estimator supports predict_proba."""
    X_train, y_train, X_test, y_test = split_with_test
    conf = {"classifier": "SVC", "parameters": {"probability": [True]}}
    result = _call_run(_make_experiment(conf), X_train, y_train, X_test, y_test)

    assert result.y_proba is not None
    assert result.y_proba.shape[0] == X_test.shape[0]
