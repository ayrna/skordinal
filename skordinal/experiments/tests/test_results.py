"""Tests for the Results class."""

from collections import OrderedDict
from pathlib import Path
from pickle import load
from shutil import rmtree

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.svm import SVC

from skordinal.experiments import ExperimentResult, Results


def _make_result(
    partition: str,
    dataset: str,
    configuration: str,
    best_params: dict,
    train_metrics: dict,
    test_metrics: dict,
    train_predicted_y: np.ndarray,
    test_predicted_y: np.ndarray,
    estimator=None,
) -> ExperimentResult:
    if estimator is None:
        estimator = SVC()
    return ExperimentResult(
        dataset_name=dataset,
        classifier_name=configuration,
        resample_id=partition,
        train_predicted_y=train_predicted_y,
        test_predicted_y=test_predicted_y,
        y_proba=None,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        best_params=best_params,
        best_model=estimator,
    )


@pytest.fixture
def results():
    return Results(Path("my_runs/"))


def test_save(results):
    """Checking behavior of save method.

    Two partitions for the same dataset and configuration will be added and
    retreived later on to check if they are similar.

    """
    estimator = SVC()

    # Saving first partition results to DataFrame
    result_0 = _make_result(
        partition="0",
        dataset="toy",
        configuration="conf_1",
        best_params=OrderedDict([("C", 0.1), ("gamma", 1)]),
        train_metrics=OrderedDict(
            [("ccr_train", 0.7222222222), ("mae_train", 0.2777777777)]
        ),
        test_metrics=OrderedDict(
            [("ccr_test", 0.6666666666), ("mae_test", 0.3333333333)]
        ),
        train_predicted_y=np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]),
        test_predicted_y=np.array([1, 1, 2, 2, 2, 3, 3]),
        estimator=estimator,
    )
    results.save(result_0)

    # Saving second partition to DataFrame
    result_1 = _make_result(
        partition="1",
        dataset="toy",
        configuration="conf_1",
        best_params=OrderedDict([("C", 1), ("gamma", 1)]),
        train_metrics=OrderedDict(
            [("ccr_train", 0.9333333333), ("mae_train", 0.2777777777)]
        ),
        test_metrics=OrderedDict([("ccr_test", 1.0), ("mae_test", 0.3333333333)]),
        train_predicted_y=np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 3]),
        test_predicted_y=np.array([1, 1, 2, 1, 2, 3, 3]),
        estimator=estimator,
    )
    results.save(result_1)

    # Saving first partition for a different configuration
    result_conf2 = _make_result(
        partition="0",
        dataset="toy",
        configuration="conf_2",
        best_params=OrderedDict([("C", 1), ("gamma", 0.1)]),
        train_metrics=OrderedDict(
            [("ccr_train", 0.8333333333), ("mae_train", 0.2777777777)]
        ),
        test_metrics=OrderedDict([("ccr_test", 1.0), ("mae_test", 0.3333333333)]),
        train_predicted_y=np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 3]),
        test_predicted_y=np.array([1, 1, 2, 1, 2, 3, 3]),
        estimator=estimator,
    )
    results.save(result_conf2)

    # Checking if everything has been saved correctly
    experiment_folder = Path(results._experiment_folder)

    # Data for toy-conf_1
    expected_data_conf_1 = [
        OrderedDict(
            [
                ("C", 0.1),
                ("gamma", 1),
                ("ccr_train", 0.7222222222),
                ("ccr_test", 0.6666666666),
                ("mae_train", 0.2777777777),
                ("mae_test", 0.3333333333),
            ]
        ),
        OrderedDict(
            [
                ("C", 1),
                ("gamma", 1),
                ("ccr_train", 0.9333333333),
                ("ccr_test", 1.0),
                ("mae_train", 0.2777777777),
                ("mae_test", 0.3333333333),
            ]
        ),
    ]
    expected_data_conf_1 = pd.DataFrame(data=expected_data_conf_1, index=[0, 1])
    conf_1_path = experiment_folder / "toy-conf_1"

    # Check inconsistencies in CSV for toy-conf_1
    actual_data_conf_1 = pd.read_csv(conf_1_path / "toy-conf_1.csv", index_col=[0])
    pdt.assert_frame_equal(actual_data_conf_1, expected_data_conf_1)

    # Data for toy-conf_2
    expected_data_conf_2 = [
        OrderedDict(
            [
                ("C", 1),
                ("gamma", 0.1),
                ("ccr_train", 0.8333333333),
                ("ccr_test", 1.0),
                ("mae_train", 0.2777777777),
                ("mae_test", 0.3333333333),
            ]
        )
    ]
    expected_data_conf_2 = pd.DataFrame(data=expected_data_conf_2, index=[0])
    conf_2_path = experiment_folder / "toy-conf_2"

    # Check inconsistencies in CSV for toy-conf_2
    actual_data_conf_2 = pd.read_csv(conf_2_path / "toy-conf_2.csv", index_col=[0])
    pdt.assert_frame_equal(actual_data_conf_2, expected_data_conf_2)

    # Checking if models have been saved successfully
    with (
        open(conf_1_path / "models" / "toy-conf_1.0", "rb") as model_0,
        open(conf_1_path / "models" / "toy-conf_1.1", "rb") as model_1,
    ):
        actual_data = [load(model_0), load(model_1)]
        npt.assert_equal(all(isinstance(model, SVC) for model in actual_data), True)

    # Checking if actual and expected predictions are the same
    expected_data = {
        "0": {
            "train": np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]),
            "test": np.array([1, 1, 2, 2, 2, 3, 3]),
        },
        "1": {
            "train": np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 3]),
            "test": np.array([1, 1, 2, 1, 2, 3, 3]),
        },
    }

    with (
        open(conf_1_path / "predictions" / "train_toy-conf_1.0", "rb") as train_0,
        open(conf_1_path / "predictions" / "test_toy-conf_1.0", "rb") as test_0,
        open(conf_1_path / "predictions" / "train_toy-conf_1.1", "rb") as train_1,
        open(conf_1_path / "predictions" / "test_toy-conf_1.1", "rb") as test_1,
    ):
        actual_data = {
            "0": {"train": np.loadtxt(train_0), "test": np.loadtxt(test_0)},
            "1": {"train": np.loadtxt(train_1), "test": np.loadtxt(test_1)},
        }

        npt.assert_equal(actual_data, expected_data)

    # Deleting temporary directories
    rmtree("my_runs/")


def test_save_model_false(results):
    """save_model=False must not create a models/ folder."""
    result = _make_result(
        partition="0",
        dataset="toy",
        configuration="conf_1",
        best_params=OrderedDict([("C", 1)]),
        train_metrics=OrderedDict([("ccr_train", 0.9)]),
        test_metrics=OrderedDict([("ccr_test", 0.8)]),
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=np.array([1, 2]),
    )
    results.save(result, save_model=False)

    folder = Path(results._experiment_folder) / "toy-conf_1"
    assert not (folder / "models").exists()
    assert (folder / "predictions").exists()

    # Deleting temporary directories
    rmtree("my_runs/")


def test_save_proba(results):
    """y_proba is persisted when provided."""
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    result = ExperimentResult(
        dataset_name="toy",
        classifier_name="conf_1",
        resample_id="0",
        train_predicted_y=np.array([1, 2]),
        test_predicted_y=np.array([1, 2]),
        y_proba=y_proba,
        train_metrics=OrderedDict([("ccr_train", 0.9)]),
        test_metrics=OrderedDict([("ccr_test", 0.8)]),
        best_params={"C": 1},
        best_model=SVC(),
    )
    results.save(result, save_model=False)

    proba_path = (
        Path(results._experiment_folder)
        / "toy-conf_1"
        / "predictions"
        / "proba_toy-conf_1.0"
    )
    assert proba_path.exists()
    loaded = np.loadtxt(proba_path)
    npt.assert_allclose(loaded, y_proba)

    # Deleting temporary directories
    rmtree("my_runs/")


def test_create_summary(results):
    """Tests create_summary method."""
    estimator = SVC()

    # Adding two identical rows as two partitions
    for partition in ("0", "1"):
        result = _make_result(
            partition=partition,
            dataset="toy",
            configuration="conf_1",
            best_params=OrderedDict([("C", 0.1), ("gamma", 1)]),
            train_metrics=OrderedDict(
                [("ccr_train", 0.7222222222), ("mae_train", 0.2777777777)]
            ),
            test_metrics=OrderedDict(
                [("ccr_test", 0.6666666666), ("mae_test", 0.3333333333)]
            ),
            train_predicted_y=np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 3]),
            test_predicted_y=np.array([1, 1, 2, 1, 2, 3, 3]),
            estimator=estimator,
        )
        results.save(result)

    mean_index = ["ccr_mean", "mae_mean"]
    std_index = ["ccr_std", "mae_std"]

    experiment_folder = results._experiment_folder

    # Getting actual summaries
    df = pd.read_csv(experiment_folder / "toy-conf_1" / "toy-conf_1.csv")
    train_row, test_row = results._create_summary(df, mean_index, std_index)

    # Desired row values and indexes
    desired_train_row = pd.Series(
        data=OrderedDict(
            [
                ("ccr_mean", 0.7222222222),
                ("ccr_std", 0.0),
                ("mae_mean", 0.2777777777),
                ("mae_std", 0.0),
            ]
        ),
        index=["ccr_mean", "ccr_std", "mae_mean", "mae_std"],
    )
    desired_test_row = pd.Series(
        data=OrderedDict(
            [
                ("ccr_mean", 0.6666666666),
                ("ccr_std", 0.0),
                ("mae_mean", 0.3333333333),
                ("mae_std", 0.0),
            ]
        ),
        index=["ccr_mean", "ccr_std", "mae_mean", "mae_std"],
    )

    # Check series similarity
    pdt.assert_series_equal(train_row, desired_train_row)
    pdt.assert_series_equal(test_row, desired_test_row)

    # Deleting temporary directories
    rmtree("my_runs/")
