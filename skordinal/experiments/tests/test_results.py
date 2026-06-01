"""Tests for the Results class."""

import json
from pathlib import Path

import joblib
import numpy as np
import numpy.testing as npt
import pandas as pd
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
    train_true_y=None,
    test_true_y=None,
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
        train_true_y=train_true_y,
        test_true_y=test_true_y,
    )


def _make_pair_csv(base: Path, classifier: str, dataset: str, rows: list[dict]) -> Path:
    """Write a minimal report.csv under base/classifier/dataset/."""
    pair_dir = base / classifier / dataset
    pair_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = pair_dir / "report.csv"
    df.to_csv(csv_path)
    return csv_path


@pytest.fixture
def two_pair_results(tmp_path):
    """Two classifiers x two datasets with 2 partitions each."""
    for clf in ("A", "B"):
        for ds in ("d1", "d2"):
            _make_pair_csv(
                tmp_path,
                clf,
                ds,
                [
                    {"mae_train": 0.2, "mae_test": 0.3},
                    {"mae_train": 0.4, "mae_test": 0.5},
                ],
            )
    return Results(tmp_path)


def test_save(tmp_path):
    """Two partitions produce the expected on-disk layout: report.csv, params.json, models, predictions."""
    estimator = SVC()
    results = Results(tmp_path)

    result_0 = _make_result(
        partition="0",
        dataset="toy",
        configuration="conf_1",
        best_params={"C": 0.1, "gamma": 1},
        train_metrics={"ccr_train": 0.7222, "mae_train": 0.2778},
        test_metrics={"ccr_test": 0.6667, "mae_test": 0.3333},
        train_predicted_y=np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]),
        test_predicted_y=np.array([1, 1, 2, 2, 2, 3, 3]),
        estimator=estimator,
    )
    results.save(result_0)

    result_1 = _make_result(
        partition="1",
        dataset="toy",
        configuration="conf_1",
        best_params={"C": 1, "gamma": 1},
        train_metrics={"ccr_train": 0.9333, "mae_train": 0.2778},
        test_metrics={"ccr_test": 1.0, "mae_test": 0.3333},
        train_predicted_y=np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 3, 3]),
        test_predicted_y=np.array([1, 1, 2, 1, 2, 3, 3]),
        estimator=estimator,
    )
    results.save(result_1)

    pair_dir = tmp_path / "conf_1" / "toy"

    df = pd.read_csv(pair_dir / "report.csv", index_col=0)
    assert df.shape == (2, 4)
    assert list(df.columns) == ["ccr_train", "mae_train", "ccr_test", "mae_test"]

    params = json.loads((pair_dir / "params.json").read_text())
    assert params["0"] == {"C": 0.1, "gamma": 1}
    assert params["1"] == {"C": 1, "gamma": 1}

    models_dir = pair_dir / "models"
    assert (models_dir / "0.joblib").is_file()
    assert (models_dir / "1.joblib").is_file()
    assert isinstance(joblib.load(models_dir / "0.joblib"), SVC)

    pred_dir = pair_dir / "predictions"
    train_0 = pd.read_csv(pred_dir / "train_0.csv")
    assert list(train_0.columns) == ["y_pred"]
    npt.assert_array_equal(train_0["y_pred"].values, result_0.train_predicted_y)

    test_0 = pd.read_csv(pred_dir / "test_0.csv")
    assert list(test_0.columns) == ["y_pred"]
    npt.assert_array_equal(test_0["y_pred"].values, result_0.test_predicted_y)


def test_save_with_true_labels(tmp_path):
    """When true labels are provided, y_true appears first in prediction CSVs."""
    train_true = np.array([1, 2, 3, 1, 2])
    test_true = np.array([1, 2, 3])
    result = _make_result(
        partition="0",
        dataset="toy",
        configuration="clf",
        best_params={},
        train_metrics={"acc_train": 0.8},
        test_metrics={"acc_test": 0.7},
        train_predicted_y=np.array([1, 2, 2, 1, 2]),
        test_predicted_y=np.array([1, 2, 2]),
        train_true_y=train_true,
        test_true_y=test_true,
    )
    Results(tmp_path).save(result, save_model=False)

    pred_dir = tmp_path / "clf" / "toy" / "predictions"

    train_df = pd.read_csv(pred_dir / "train_0.csv")
    assert list(train_df.columns) == ["y_true", "y_pred"]
    npt.assert_array_equal(train_df["y_true"].values, train_true)

    test_df = pd.read_csv(pred_dir / "test_0.csv")
    assert list(test_df.columns) == ["y_true", "y_pred"]
    npt.assert_array_equal(test_df["y_true"].values, test_true)


def test_save_model_false(tmp_path):
    """save_model=False must not create a models/ folder."""
    result = _make_result(
        partition="0",
        dataset="toy",
        configuration="conf_1",
        best_params={"C": 1},
        train_metrics={"ccr_train": 0.9},
        test_metrics={"ccr_test": 0.8},
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=np.array([1, 2]),
    )
    Results(tmp_path).save(result, save_model=False)

    pair_dir = tmp_path / "conf_1" / "toy"
    assert not (pair_dir / "models").exists()
    assert (pair_dir / "predictions").exists()


def test_save_proba_not_written_to_disk(tmp_path):
    """y_proba is stored in ExperimentResult but not written to disk."""
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    result = ExperimentResult(
        dataset_name="toy",
        classifier_name="conf_1",
        resample_id="0",
        train_predicted_y=np.array([1, 2]),
        test_predicted_y=np.array([1, 2]),
        y_proba=y_proba,
        train_metrics={"ccr_train": 0.9},
        test_metrics={"ccr_test": 0.8},
        best_params={"C": 1},
        best_model=SVC(),
    )
    Results(tmp_path).save(result, save_model=False)

    pred_dir = tmp_path / "conf_1" / "toy" / "predictions"
    assert list(pred_dir.glob("proba*")) == []


def test_save_no_test_partition(tmp_path):
    """When test_predicted_y is None, no test CSV is written."""
    result = ExperimentResult(
        dataset_name="toy",
        classifier_name="clf",
        resample_id="0",
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=None,
        y_proba=None,
        train_metrics={"ccr_train": 0.9},
        test_metrics={},
        best_params={},
        best_model=SVC(),
    )
    Results(tmp_path).save(result, save_model=False)

    pred_dir = tmp_path / "clf" / "toy" / "predictions"
    assert (pred_dir / "train_0.csv").is_file()
    assert not (pred_dir / "test_0.csv").exists()


def test_save_multiple_partitions_and_params_upsert(tmp_path):
    """Saving multiple partitions accumulates rows in report.csv; saving the same id twice overwrites its params entry."""
    r = Results(tmp_path)
    for i in range(3):
        r.save(
            _make_result(
                partition=str(i),
                dataset="ds",
                configuration="clf",
                best_params={},
                train_metrics={"mae_train": float(i)},
                test_metrics={"mae_test": float(i)},
                train_predicted_y=np.array([1]),
                test_predicted_y=np.array([1]),
            ),
            save_model=False,
        )

    df = pd.read_csv(tmp_path / "clf" / "ds" / "report.csv", index_col=0)
    assert df.shape[0] == 3

    base_result = dict(
        dataset="ds",
        configuration="clf",
        train_metrics={"mae_train": 0.1},
        test_metrics={"mae_test": 0.1},
        train_predicted_y=np.array([1]),
        test_predicted_y=np.array([1]),
    )
    r.save(
        _make_result(partition="0", best_params={"C": 0.1}, **base_result),
        save_model=False,
    )
    r.save(
        _make_result(partition="0", best_params={"C": 1.0}, **base_result),
        save_model=False,
    )

    params = json.loads((tmp_path / "clf" / "ds" / "params.json").read_text())
    assert params["0"]["C"] == 1.0


def test_load(tmp_path):
    """Results.load() returns a Results pointing at the given folder; does not raise if folder is absent."""
    r = Results.load(tmp_path)
    assert isinstance(r, Results)
    assert r._experiment_folder == tmp_path

    Results.load("/nonexistent/path/that/does/not/exist")


def test_exists(tmp_path):
    """exists() returns False when CSV is absent, False when resample is missing, True after save."""
    r = Results(tmp_path)
    assert r.exists("SVC", "toy", "0") is False

    _make_pair_csv(tmp_path, "SVC", "toy", [{"mae_test": 0.3}])
    assert r.exists("SVC", "toy", "99") is False

    r.save(
        _make_result(
            partition="0",
            dataset="toy",
            configuration="SVC",
            best_params={},
            train_metrics={"mae_train": 0.1},
            test_metrics={"mae_test": 0.2},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )
    assert r.exists("SVC", "toy", "0") is True
    assert r.exists("SVC", "toy", "1") is False


def test_summarize(two_pair_results, tmp_path):
    """summarize() returns a MultiIndex DataFrame; split filtering works; mean/std values are correct."""
    # split="test" returns MultiIndex DataFrame with correct shape
    df_test = two_pair_results.summarize(split="test")
    assert isinstance(df_test.index, pd.MultiIndex)
    assert isinstance(df_test.columns, pd.MultiIndex)
    assert df_test.shape == (4, 3)

    # split="train" includes only _train columns
    df_train = two_pair_results.summarize(split="train")
    metric_cols = [c for c in df_train.columns if c[0] != "n_completed"]
    assert all(c[0].endswith("_train") for c in metric_cols)

    # split="both" includes both _test and _train columns
    df_both = two_pair_results.summarize(split="both")
    metric_cols_both = [c[0] for c in df_both.columns if c[0] != "n_completed"]
    assert any(c.endswith("_test") for c in metric_cols_both)
    assert any(c.endswith("_train") for c in metric_cols_both)

    # mean and std are computed correctly from 2 partitions
    _make_pair_csv(
        tmp_path / "mean_check", "clf", "ds", [{"mae_test": 0.2}, {"mae_test": 0.4}]
    )
    df_vals = Results(tmp_path / "mean_check").summarize(split="test")
    assert df_vals.loc[("clf", "ds"), ("mae_test", "mean")] == pytest.approx(0.3)
    assert df_vals.loc[("clf", "ds"), ("mae_test", "std")] == pytest.approx(
        0.1414, rel=1e-3
    )
    assert df_vals.loc[("clf", "ds"), ("n_completed", "")] == 2


def test_summarize_labels_filter(two_pair_results):
    """summarize(labels=[...]) restricts results to the given classifiers."""
    df = two_pair_results.summarize(labels=["A"])
    assert all(clf == "A" for clf, _ in df.index)


def test_summarize_labels_string_raises(two_pair_results):
    """Passing a bare string to labels raises TypeError."""
    with pytest.raises(TypeError, match="iterable"):
        two_pair_results.summarize(labels="A")


def test_summarize_invalid_split(two_pair_results):
    with pytest.raises(ValueError, match="split must be"):
        two_pair_results.summarize(split="bad")


def test_tabulate(two_pair_results):
    """tabulate() returns a pivot DataFrame; missing metric yields n/a; invalid split raises."""
    df = two_pair_results.tabulate(metric="mae", split="test")
    assert isinstance(df, pd.DataFrame)
    assert set(df.index) == {"A", "B"}
    assert set(df.columns) == {"d1", "d2"}

    df_missing = two_pair_results.tabulate(metric="nonexistent", split="test")
    assert (df_missing == "n/a").all().all()

    with pytest.raises(ValueError, match="split must be"):
        two_pair_results.tabulate(split="both")


def test_summarize_tabulate_empty_folder(tmp_path):
    """Both summarize() and tabulate() return an empty DataFrame when the folder has no results."""
    r = Results(tmp_path)
    assert r.summarize().empty
    assert r.tabulate().empty


def test_save_summary_writes_csv(two_pair_results):
    """save_summary() writes {split}_summary.csv and returns its path."""
    path = two_pair_results.save_summary(split="test")
    assert path.is_file()
    assert path.name == "test_summary.csv"
    df = pd.read_csv(path)
    assert df.shape[0] == 4


def test_save_summary_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No results"):
        Results(tmp_path).save_summary()
