"""Tests for the experiment utilities module."""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from skordinal.experiments import Utilities


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

    svm_dir = runs_dir / "SVM" / "balance"
    assert svm_dir.exists()

    metrics_csv = svm_dir / "report.csv"
    df = pd.read_csv(metrics_csv, index_col=0)
    npt.assert_equal(df.shape[0], 2)
    npt.assert_equal(df.shape[1], 12)
    npt.assert_equal(all(df[c].dtype == np.float64 for c in df.columns), True)

    models = list((svm_dir / "models").iterdir())
    npt.assert_equal(len(models), 2)

    predictions = list((svm_dir / "predictions").iterdir())
    npt.assert_equal(len(predictions), 4)
