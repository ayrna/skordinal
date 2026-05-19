"""Tests for the experiment utilities module."""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from skordinal.utilities import Utilities
from skordinal.utils._testing import make_balance_scale_split


@pytest.fixture
def util():
    general_conf = {}
    configurations = {}
    return Utilities(general_conf, configurations)


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
    """End-to-end test: run_experiment and write_report complete without error
    and produce the expected output structure and metrics files.
    """
    util = Utilities(experiment_conf, svm_conf, verbose=False)
    util.run_experiment()
    util.write_report()

    runs_dir = Path(experiment_conf["output_folder"])
    assert runs_dir.exists()

    exp_dirs = list(runs_dir.iterdir())
    npt.assert_equal(len(exp_dirs), 1)
    exp_dir = exp_dirs[0]

    svm_dir = exp_dir / "balance-SVM"
    assert svm_dir.exists()

    metrics_csv = svm_dir / "balance-SVM.csv"
    df = pd.read_csv(metrics_csv, index_col=0)
    npt.assert_equal(df.shape[0], 2)
    metric_block = df.iloc[:, -12:]
    npt.assert_equal(metric_block.shape, (2, 12))
    npt.assert_equal(
        all(metric_block[c].dtype == np.float64 for c in metric_block.columns), True
    )

    models = list((svm_dir / "models").iterdir())
    npt.assert_equal(len(models), 2)

    predictions = list((svm_dir / "predictions").iterdir())
    npt.assert_equal(len(predictions), 4)

    train_summary = pd.read_csv(exp_dir / "train_summary.csv")
    npt.assert_equal(train_summary.shape, (1, 13))
    npt.assert_equal(
        all(train_summary[c].dtype == np.float64 for c in train_summary.columns[1:]),
        True,
    )

    test_summary = pd.read_csv(exp_dir / "test_summary.csv")
    npt.assert_equal(test_summary.shape, (1, 13))
    npt.assert_equal(
        all(test_summary[c].dtype == np.float64 for c in test_summary.columns[1:]),
        True,
    )


def test_load_complete_dataset(tmp_path, util):
    """Loading dataset composed of 5 partitions, each one of them composed of
    a train and test file.

    """
    dataset_path = tmp_path / "complete"
    dataset_path.mkdir()

    for i in range(5):
        create_csv(dataset_path, f"train_complete.{i}")
        create_csv(dataset_path, f"test_complete.{i}")

    partition_list = util._load_dataset(dataset_path)

    # Check all partitions have been loaded
    npt.assert_equal(len(partition_list), (len(list(dataset_path.iterdir())) / 2))
    # Check if every partition has train and test inputs and outputs (4 diferent dictionaries)
    npt.assert_equal(
        all([len(partition[1]) == 4 for partition in partition_list]), True
    )


def test_load_partitionless_dataset(tmp_path, util):
    """Loading dataset composed of only two csv files (train and test files)"""
    dataset_path = tmp_path / "partitionless"
    dataset_path.mkdir()

    create_csv(dataset_path, "train_partitionless.csv")
    create_csv(dataset_path, "test_partitionless.csv")

    partition_list = util._load_dataset(dataset_path)

    npt.assert_equal(len(partition_list), 1)
    npt.assert_equal(
        all([len(partition[1]) == 4 for partition in partition_list]), True
    )


def test_load_nontestfile_dataset(tmp_path, util):
    """Loading dataset composed of five train files."""
    dataset_path = tmp_path / "nontestfile"
    dataset_path.mkdir()

    for i in range(5):
        create_csv(dataset_path, f"train_nontestfile.{i}")

    partition_list = util._load_dataset(dataset_path)

    npt.assert_equal(len(partition_list), len(list(dataset_path.iterdir())))
    npt.assert_equal(
        all([len(partition[1]) == 2 for partition in partition_list]), True
    )


def test_load_nontrainfile_dataset(tmp_path, util):
    """Loading dataset with 2 partitions, one of them lacking its train file.
    This should raise an exception.

    """
    dataset_path = tmp_path / "nontrainfile"
    dataset_path.mkdir()

    for i in range(2):
        create_csv(dataset_path, f"test_nontrainfile.{i}")

    with pytest.raises(RuntimeError):
        util._load_dataset(dataset_path)


def test_normalize_data(tmp_path, util):
    # Test preparation
    X_train, X_test, _, _ = make_balance_scale_split()

    # Test execution
    norm_X_train, _ = util._normalize_data(X_train, X_test)

    # Test verification
    result = (norm_X_train >= 0).all() and (norm_X_train <= 1).all()
    npt.assert_equal(result, True)


def test_standardize_data(util):
    # Test preparation
    X_train, X_test, _, _ = make_balance_scale_split()

    # Test execution
    std_X_train, _ = util._standardize_data(X_train, X_test)

    # Test verification
    npt.assert_almost_equal(np.mean(std_X_train), 0)
    npt.assert_almost_equal(np.std(std_X_train), 1)
