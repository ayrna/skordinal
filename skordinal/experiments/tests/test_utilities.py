"""Tests for the experiment utilities module."""

import math
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from skordinal.experiments import ExperimentResult, Utilities
from skordinal.experiments._utilities import _load_dataset

_MINIMAL_CONF = {"cfg": {"classifier": "SVC", "parameters": {}}}
_CONF_CV: dict = {"classifier": "SVC", "parameters": {"C": [0.1, 1.0]}}


def create_csv(path, filename):
    """Create a csv file with sample data."""
    sample_data = "1,2,3,0\n4,5,6,1"
    (path / filename).write_text(sample_data)


def _write_partition_csv(directory, filename, n_per_class=10):
    rng = np.random.default_rng(0)
    n_rows = n_per_class * 3
    features = rng.integers(1, 6, size=(n_rows, 4))
    labels = np.repeat([0, 1, 2], n_per_class).reshape(-1, 1)
    data = np.hstack([features, labels])
    np.savetxt(directory / filename, data, delimiter=",", fmt="%d")


def _from_general_conf(general_conf: dict, configurations: dict, **kwargs) -> Utilities:
    """Construct Utilities from an old-style ``general_conf`` dict.

    Keeps test call-sites readable without duplicating the key mapping.
    """
    return Utilities(
        configurations,
        data_path=general_conf["basedir"],
        datasets=general_conf["datasets"],
        eval_metrics=general_conf["metrics"],
        results_path=general_conf["output_folder"],
        tuning_metric=general_conf.get("cv_metric", "neg_mean_absolute_error"),
        cv=general_conf.get("hyperparam_cv_nfolds", 3),
        n_jobs=general_conf.get("jobs", 1),
        input_preprocessing=general_conf.get("input_preprocessing"),
        **kwargs,
    )


@pytest.fixture
def partition_dataset(tmp_path):
    dataset_dir = tmp_path / "data" / "balance"
    dataset_dir.mkdir(parents=True)
    for i in range(2):
        _write_partition_csv(dataset_dir, f"train_balance_{i}.csv")
        _write_partition_csv(dataset_dir, f"test_balance_{i}.csv")
    return tmp_path / "data"


@pytest.fixture
def experiment_conf(tmp_path, partition_dataset):
    return {
        "basedir": partition_dataset,
        "datasets": ["balance"],
        "input_preprocessing": "std",
        "hyperparam_cv_nfolds": 3,
        "jobs": 1,
        "output_folder": str(tmp_path / "runs"),
        "metrics": [
            "accuracy_score",
            "mean_absolute_error",
            "average_mean_absolute_error",
            "mean_zero_one_error",
        ],
        "cv_metric": "mean_absolute_error",
    }


@pytest.fixture
def svm_conf():
    return {
        "SVM": {
            "classifier": "SVC",
            "parameters": {"C": [0.1, 1.0], "gamma": [0.1]},
        },
    }


def test_run_experiment(tmp_path, experiment_conf, svm_conf):
    """run_experiment and write_report produce the expected on-disk layout."""
    util = _from_general_conf(experiment_conf, svm_conf, verbose=False)
    util.run_experiment()
    util.write_report()

    runs_dir = Path(experiment_conf["output_folder"])
    assert runs_dir.exists()

    svm_dir = runs_dir / "SVM" / "balance"
    assert svm_dir.exists()

    df = pd.read_csv(svm_dir / "report.csv", index_col=0)
    assert df.shape == (2, 12)
    assert all(df[c].dtype == np.float64 for c in df.columns)

    assert len(list((svm_dir / "models").iterdir())) == 2
    assert len(list((svm_dir / "predictions").iterdir())) == 4

    train_summary = pd.read_csv(runs_dir / "train_summary.csv")
    assert train_summary.shape == (1, 15)
    assert all(
        train_summary[c].dtype == np.float64 for c in train_summary.columns[2:-1]
    )

    test_summary = pd.read_csv(runs_dir / "test_summary.csv")
    assert test_summary.shape == (1, 15)
    assert all(test_summary[c].dtype == np.float64 for c in test_summary.columns[2:-1])


def test_load_complete_dataset(tmp_path):
    """Load a dataset of 5 partitions, each with a train and a test file."""
    dataset_path = tmp_path / "complete"
    dataset_path.mkdir()

    for i in range(5):
        create_csv(dataset_path, f"train_complete.{i}")
        create_csv(dataset_path, f"test_complete.{i}")

    partition_list = _load_dataset(dataset_path)

    # Every partition holds train and test inputs and outputs (4 entries).
    assert len(partition_list) == len(list(dataset_path.iterdir())) / 2
    assert all(len(partition[1]) == 4 for partition in partition_list)


def test_load_partitionless_dataset(tmp_path):
    """Load a dataset of a single train and test file."""
    dataset_path = tmp_path / "partitionless"
    dataset_path.mkdir()

    create_csv(dataset_path, "train_partitionless.csv")
    create_csv(dataset_path, "test_partitionless.csv")

    partition_list = _load_dataset(dataset_path)

    assert len(partition_list) == 1
    assert all(len(partition[1]) == 4 for partition in partition_list)


def test_load_nontestfile_dataset(tmp_path):
    """Load a dataset of five train files with no test files."""
    dataset_path = tmp_path / "nontestfile"
    dataset_path.mkdir()

    for i in range(5):
        create_csv(dataset_path, f"train_nontestfile.{i}")

    partition_list = _load_dataset(dataset_path)

    assert len(partition_list) == len(list(dataset_path.iterdir()))
    assert all(len(partition[1]) == 2 for partition in partition_list)


def test_load_nontrainfile_dataset(tmp_path):
    """A partition lacking its train file raises RuntimeError."""
    dataset_path = tmp_path / "nontrainfile"
    dataset_path.mkdir()

    for i in range(2):
        create_csv(dataset_path, f"test_nontrainfile.{i}")

    with pytest.raises(RuntimeError):
        _load_dataset(dataset_path)


def test_empty_configurations_raises(tmp_path):
    """An empty configurations dict raises ValueError."""
    with pytest.raises(ValueError, match="'configurations' must be a non-empty dict"):
        Utilities(
            {},
            data_path=".",
            datasets=["x"],
            eval_metrics=["mean_absolute_error"],
            results_path=str(tmp_path),
        )


def test_none_configurations_raises(tmp_path):
    """None configurations is rejected by the non-empty check (ValueError)."""
    with pytest.raises(ValueError, match="'configurations' must be a non-empty dict"):
        Utilities(
            None,  # type: ignore[arg-type]
            data_path=".",
            datasets=["x"],
            eval_metrics=["mean_absolute_error"],
            results_path=str(tmp_path),
        )


def test_empty_datasets_raises(tmp_path):
    """An empty datasets list raises ValueError."""
    with pytest.raises(ValueError, match="'datasets' must be a non-empty list"):
        Utilities(
            _MINIMAL_CONF,
            data_path=".",
            datasets=[],
            eval_metrics=["mean_absolute_error"],
            results_path=str(tmp_path),
        )


def test_empty_eval_metrics_raises(tmp_path):
    """An empty eval_metrics list raises ValueError."""
    with pytest.raises(ValueError, match="'eval_metrics' must be a non-empty list"):
        Utilities(
            _MINIMAL_CONF,
            data_path=".",
            datasets=["x"],
            eval_metrics=[],
            results_path=str(tmp_path),
        )


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
def test_input_preprocessing_accepted_and_normalized(tmp_path, raw, expected):
    """Valid input_preprocessing values are accepted and lower-stripped."""
    util = Utilities(
        _MINIMAL_CONF,
        data_path=".",
        datasets=["x"],
        eval_metrics=["mean_absolute_error"],
        results_path=str(tmp_path),
        input_preprocessing=raw,
    )
    assert util.input_preprocessing == expected


@pytest.mark.parametrize("bad_value", ["minmax", ""])
def test_input_preprocessing_invalid_raises(tmp_path, bad_value):
    """Unrecognised input_preprocessing values raise ValueError."""
    with pytest.raises(ValueError, match="'input_preprocessing' must be one of"):
        Utilities(
            _MINIMAL_CONF,
            data_path=".",
            datasets=["x"],
            eval_metrics=["mean_absolute_error"],
            results_path=str(tmp_path),
            input_preprocessing=bad_value,
        )


@pytest.mark.parametrize("kwargs, expected", [({}, None), ({"random_state": 42}, 42)])
def test_random_state_stored(tmp_path, kwargs, expected):
    """random_state defaults to None and is stored as given on the instance."""
    util = Utilities(
        _MINIMAL_CONF,
        data_path=".",
        datasets=["x"],
        eval_metrics=["mean_absolute_error"],
        results_path=str(tmp_path),
        **kwargs,
    )
    assert util.random_state == expected


def test_configurations_is_deep_copied(tmp_path):
    """Mutating the original configurations dict does not affect the stored copy."""
    original = {"cfg": {"classifier": "SVC", "parameters": {"C": [1]}}}
    util = Utilities(
        original,
        data_path=".",
        datasets=["x"],
        eval_metrics=["mean_absolute_error"],
        results_path=str(tmp_path),
    )
    original["cfg"]["parameters"]["C"].append(10)
    assert util.configurations["cfg"]["parameters"]["C"] == [1]


@pytest.fixture
def run_single_util(tmp_path):
    """Minimal Utilities instance configured for _run_single seam tests."""
    return Utilities(
        _MINIMAL_CONF,
        data_path=".",
        datasets=["x"],
        eval_metrics=["mean_absolute_error"],
        results_path=str(tmp_path),
    )


@pytest.fixture
def run_single_util_std(tmp_path):
    """Utilities instance with input_preprocessing='std' for mutation tests."""
    return Utilities(
        _MINIMAL_CONF,
        data_path=".",
        datasets=["x"],
        eval_metrics=["mean_absolute_error"],
        results_path=str(tmp_path),
        input_preprocessing="std",
    )


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


def _call_run_single(util, X_train, y_train, X_test, y_test, conf=None):
    """Invoke _run_single with fixed identity kwargs; returns the ExperimentResult."""
    if conf is None:
        conf = _MINIMAL_CONF["cfg"]
    return util._run_single(
        X_train,
        y_train,
        X_test,
        y_test,
        conf,
        dataset_name="ds",
        conf_name="cfg",
        resample_id="0",
    )


def test_run_single_returns_experiment_result_and_does_not_persist(
    tmp_path, run_single_util, split_with_test
):
    """_run_single returns an ExperimentResult without writing any files."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run_single(run_single_util, X_train, y_train, X_test, y_test)

    assert isinstance(result, ExperimentResult)
    written = list(tmp_path.iterdir())
    assert written == [], f"_run_single wrote files unexpectedly: {written}"


def test_run_single_identity_passthrough(run_single_util, split_with_test):
    """dataset_name, classifier_name, and resample_id are forwarded verbatim."""
    X_train, y_train, X_test, y_test = split_with_test
    result = run_single_util._run_single(
        X_train,
        y_train,
        X_test,
        y_test,
        _MINIMAL_CONF["cfg"],
        dataset_name="my_dataset",
        conf_name="my_conf",
        resample_id="42",
    )

    assert result.dataset_name == "my_dataset"
    assert result.classifier_name == "my_conf"
    assert result.resample_id == "42"


@pytest.mark.parametrize("has_test", [True, False])
def test_run_single_test_present_vs_absent(
    run_single_util, split_with_test, split_train_only, has_test
):
    """With test data: test_predicted_y is an array and metrics are finite; without: None and NaN."""
    X_train, y_train, X_test, y_test = split_with_test if has_test else split_train_only
    result = _call_run_single(run_single_util, X_train, y_train, X_test, y_test)

    metric_test_key = "mean_absolute_error_test"
    if has_test:
        assert result.test_predicted_y is not None
        assert result.test_predicted_y.shape == (15,)
        assert math.isfinite(result.test_metrics[metric_test_key])
        assert math.isfinite(result.test_metrics["time_test"])
    else:
        assert result.test_predicted_y is None
        assert math.isnan(result.test_metrics[metric_test_key])


def test_run_single_timing_no_cv(run_single_util, split_with_test):
    """Singleton param grid produces NaN cv_time_* and best_params echoes the config."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run_single(run_single_util, X_train, y_train, X_test, y_test)

    assert math.isnan(result.train_metrics["cv_time_train"])
    assert math.isnan(result.test_metrics["cv_time_test"])
    assert math.isfinite(result.train_metrics["time_train"])
    assert math.isfinite(result.test_metrics["time_test"])
    assert result.best_params == _MINIMAL_CONF["cfg"]["parameters"]


def test_run_single_timing_with_cv(run_single_util, split_with_test):
    """Multi-value param grid produces finite cv_time_* and best_params reflects a searched value."""
    X_train, y_train, X_test, y_test = split_with_test
    result = _call_run_single(
        run_single_util, X_train, y_train, X_test, y_test, conf=_CONF_CV
    )

    assert math.isfinite(result.train_metrics["cv_time_train"])
    assert not math.isnan(result.train_metrics["cv_time_train"])
    assert math.isfinite(result.test_metrics["cv_time_test"])
    assert not math.isnan(result.test_metrics["cv_time_test"])
    assert math.isfinite(result.train_metrics["time_train"])
    assert math.isfinite(result.test_metrics["time_test"])
    assert result.best_params.get("C") in _CONF_CV["parameters"]["C"]


@pytest.mark.parametrize("preprocessing", ["std", "norm"])
def test_run_single_preprocessing_train_only_raises(
    tmp_path, split_train_only, preprocessing
):
    """input_preprocessing with X_test=None raises ValueError."""
    util = Utilities(
        _MINIMAL_CONF,
        data_path=".",
        datasets=["x"],
        eval_metrics=["mean_absolute_error"],
        results_path=str(tmp_path),
        input_preprocessing=preprocessing,
    )
    X_train, y_train, X_test, y_test = split_train_only
    with pytest.raises(ValueError):
        _call_run_single(util, X_train, y_train, X_test, y_test)


def test_run_single_preprocessing_does_not_mutate_inputs(
    run_single_util_std, split_with_test
):
    """input_preprocessing='std' operates on copies; caller's arrays are unchanged."""
    X_train, y_train, X_test, y_test = split_with_test
    train_before = X_train.copy()
    test_before = X_test.copy()

    _call_run_single(run_single_util_std, X_train, y_train, X_test, y_test)

    npt.assert_array_equal(X_train, train_before)
    npt.assert_array_equal(X_test, test_before)
