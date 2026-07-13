"""Tests for the Results class."""

import os

import joblib
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from skordinal.experiments import ExperimentResult, Results
from skordinal.experiments._results import (
    _TEMP_PREFIX,
    _atomic_dump,
    _atomic_write,
    _check_path_component,
    _format_proba_column,
    _write_split_files,
)


def _fitted_svc(classes=(1, 2, 3), probability=False):
    """Return an SVC fitted so that ``classes_`` equals ``classes``."""
    labels = np.asarray(classes)
    rng = np.random.default_rng(0)
    features = rng.standard_normal((labels.size * 4, 4))
    targets = np.tile(labels, 4)
    return SVC(probability=probability).fit(features, targets)


def _make_result(
    partition: int,
    dataset: str,
    configuration: str,
    best_params: dict,
    train_metrics: dict,
    test_metrics: dict,
    train_predicted_y: np.ndarray,
    test_predicted_y: np.ndarray | None,
    estimator=None,
    train_true_y=None,
    test_true_y=None,
    train_index=None,
    test_index=None,
) -> ExperimentResult:
    if estimator is None:
        estimator = _fitted_svc()
    if train_true_y is None:
        train_true_y = train_predicted_y
    if test_true_y is None and test_predicted_y is not None:
        test_true_y = test_predicted_y
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
        train_index=train_index,
        test_index=test_index,
    )


def _make_pair_csv(base, classifier, dataset, rows):
    """Write a minimal report.csv under base/classifier/dataset/."""
    pair_dir = base / classifier / dataset
    pair_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = pair_dir / "report.csv"
    df.to_csv(csv_path)
    return csv_path


def test_save(tmp_path):
    """Two partitions produce the full per-seed on-disk layout."""
    estimator = _fitted_svc()
    results = Results(tmp_path)

    result_0 = _make_result(
        partition=0,
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
        partition=1,
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

    assert not (pair_dir / "params.json").exists()
    hyper = pd.read_csv(pair_dir / "hyperparameter_configuration.csv")
    assert list(hyper.columns) == ["Seed", "C", "gamma"]
    row_0 = hyper[hyper["Seed"] == 0].iloc[0]
    assert row_0["C"] == 0.1
    assert row_0["gamma"] == 1
    row_1 = hyper[hyper["Seed"] == 1].iloc[0]
    assert row_1["C"] == 1
    assert row_1["gamma"] == 1

    models_dir = pair_dir / "models"
    assert (models_dir / "0.joblib").is_file()
    assert (models_dir / "1.joblib").is_file()
    assert isinstance(joblib.load(models_dir / "0.joblib"), SVC)

    # Check the old flat predictions/ directory is gone in the new layout
    assert not (pair_dir / "predictions").exists()

    seed_dir = pair_dir / "predictions_by_seed" / "seed_0"
    train_0 = pd.read_csv(seed_dir / "train_predictions.csv")
    assert list(train_0.columns) == ["Pattern ID", "Target", "Prediction"]

    test_0 = pd.read_csv(seed_dir / "test_predictions.csv")
    assert list(test_0.columns) == ["Pattern ID", "Target", "Prediction"]


def test_save_encodes_labels_zero_based(tmp_path):
    """Target/Prediction are searchsorted indices into 1-based classes_."""
    estimator = _fitted_svc()
    npt.assert_array_equal(estimator.classes_, np.array([1, 2, 3]))

    test_true = np.array([1, 2, 3, 1])
    test_pred = np.array([2, 2, 3, 1])
    result = _make_result(
        partition=0,
        dataset="toy",
        configuration="clf",
        best_params={},
        train_metrics={"acc_train": 1.0},
        test_metrics={"acc_test": 1.0},
        train_predicted_y=np.array([2, 2, 3]),
        test_predicted_y=test_pred,
        estimator=estimator,
        train_true_y=np.array([1, 2, 3]),
        test_true_y=test_true,
    )
    Results(tmp_path).save(result, save_model=False)

    seed_dir = tmp_path / "clf" / "toy" / "predictions_by_seed" / "seed_0"
    train_df = pd.read_csv(seed_dir / "train_predictions.csv")
    npt.assert_array_equal(train_df["Target"].values, np.array([0, 1, 2]))
    npt.assert_array_equal(train_df["Prediction"].values, np.array([1, 1, 2]))

    test_df = pd.read_csv(seed_dir / "test_predictions.csv")
    npt.assert_array_equal(test_df["Target"].values, np.array([0, 1, 2, 0]))
    npt.assert_array_equal(test_df["Prediction"].values, np.array([1, 1, 2, 0]))


def test_save_model_false(tmp_path):
    """save_model=False writes the seed dir but no models/ folder."""
    result = _make_result(
        partition=0,
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
    assert (pair_dir / "predictions_by_seed" / "seed_0").exists()


def test_save_proba_written_to_disk(tmp_path):
    """Probability columns are persisted in both prediction files."""
    rng = np.random.default_rng(0)
    estimator = _fitted_svc(probability=True)
    q = estimator.classes_.size

    raw_train = rng.random((3, q))
    raw_test = rng.random((4, q))
    train_y_proba = raw_train / raw_train.sum(axis=1, keepdims=True)
    y_proba = raw_test / raw_test.sum(axis=1, keepdims=True)
    result = ExperimentResult(
        dataset_name="toy",
        classifier_name="conf_1",
        resample_id=0,
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=np.array([1, 2, 3, 1]),
        y_proba=y_proba,
        train_metrics={"ccr_train": 0.9},
        test_metrics={"ccr_test": 0.8},
        best_params={"C": 1},
        best_model=estimator,
        train_true_y=np.array([1, 2, 3]),
        test_true_y=np.array([1, 2, 3, 1]),
        train_y_proba=train_y_proba,
    )
    Results(tmp_path).save(result, save_model=False)

    seed_dir = tmp_path / "conf_1" / "toy" / "predictions_by_seed" / "seed_0"
    for name, proba in (
        ("train_predictions.csv", train_y_proba),
        ("test_predictions.csv", y_proba),
    ):
        df = pd.read_csv(seed_dir / name)
        assert "Prediction probabilities" in df.columns
        assert set(df["Target"]) <= set(range(q))
        # Check Prediction equals the argmax index of the probabilities
        npt.assert_array_equal(df["Prediction"].values, np.argmax(proba, axis=1))
        for cell, source in zip(df["Prediction probabilities"], proba):
            parsed = np.fromstring(cell.strip("[]"), sep=",")
            assert parsed.shape == (q,)
            npt.assert_allclose(parsed, source)
            npt.assert_allclose(parsed.sum(), 1.0, atol=1e-8)


def test_save_requires_true_labels(tmp_path):
    """save raises ValueError when a split lacks its true labels."""
    base = dict(
        dataset_name="toy",
        classifier_name="clf",
        resample_id=0,
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=np.array([1, 2]),
        y_proba=None,
        train_metrics={},
        test_metrics={},
        best_params={},
        best_model=_fitted_svc(),
    )

    no_train = ExperimentResult(**base, test_true_y=np.array([1, 2]))
    with pytest.raises(ValueError, match="train_true_y"):
        Results(tmp_path).save(no_train, save_model=False)

    no_test = ExperimentResult(**base, train_true_y=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="test_true_y"):
        Results(tmp_path).save(no_test, save_model=False)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"train_true_y": np.array([1, 2, 4])}, "'train' true labels"),
        ({"test_true_y": np.array([1, 2, 4])}, "'test' true labels"),
        ({"test_predicted_y": np.array([1, 2, 4])}, "'test' predicted labels"),
    ],
)
def test_save_rejects_unknown_labels(tmp_path, overrides, match):
    """A label absent from classes_ raises instead of mis-encoding."""
    kwargs = dict(
        partition=0,
        dataset="toy",
        configuration="clf",
        best_params={},
        train_metrics={"acc_train": 1.0},
        test_metrics={"acc_test": 1.0},
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=np.array([1, 2, 3]),
        train_true_y=np.array([1, 2, 3]),
        test_true_y=np.array([1, 2, 3]),
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        Results(tmp_path).save(_make_result(**kwargs))
    # Check the failed save left no model behind
    assert not (tmp_path / "clf" / "toy" / "models" / "0.joblib").exists()


def test_save_rejects_misshaped_proba(tmp_path):
    """A probability matrix without one column per class raises."""
    result = ExperimentResult(
        dataset_name="toy",
        classifier_name="clf",
        resample_id=0,
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=None,
        y_proba=None,
        train_metrics={},
        test_metrics={},
        best_params={},
        best_model=_fitted_svc(),
        train_true_y=np.array([1, 2, 3]),
        train_y_proba=np.full((3, 2), 0.5),
    )
    with pytest.raises(ValueError, match="probabilities have shape"):
        Results(tmp_path).save(result, save_model=False)


def test_format_proba_column_cells_round_trip():
    """Wide probability cells stay single-line and parse back exactly."""
    rng = np.random.default_rng(0)
    raw = rng.random((3, 20))
    proba = raw / raw.sum(axis=1, keepdims=True)

    for cell, row in zip(_format_proba_column(proba), proba):
        assert "\n" not in cell
        npt.assert_array_equal(np.fromstring(cell.strip("[]"), sep=","), row)


def test_save_no_test_partition(tmp_path):
    """When test_predicted_y is None, no test files are written."""
    result = _make_result(
        partition=0,
        dataset="toy",
        configuration="clf",
        best_params={},
        train_metrics={"ccr_train": 0.9},
        test_metrics={},
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=None,
    )
    Results(tmp_path).save(result, save_model=False)

    seed_dir = tmp_path / "clf" / "toy" / "predictions_by_seed" / "seed_0"
    assert (seed_dir / "train_predictions.csv").is_file()
    assert (seed_dir / "train_confusion_matrix.txt").is_file()
    assert not (seed_dir / "test_predictions.csv").exists()
    assert not (seed_dir / "test_confusion_matrix.txt").exists()


def test_save_writes_confusion_matrices(tmp_path):
    """Each seed dir gets train and test confusion-matrix text files."""
    estimator = _fitted_svc()
    test_true = np.array([1, 1, 2, 2, 3, 3])
    test_pred = np.array([1, 2, 2, 3, 3, 3])
    result = _make_result(
        partition=0,
        dataset="toy",
        configuration="clf",
        best_params={},
        train_metrics={"acc_train": 1.0},
        test_metrics={"acc_test": 1.0},
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=test_pred,
        estimator=estimator,
        train_true_y=np.array([1, 2, 3]),
        test_true_y=test_true,
    )
    Results(tmp_path).save(result, save_model=False)

    seed_dir = tmp_path / "clf" / "toy" / "predictions_by_seed" / "seed_0"
    for name in ("train_confusion_matrix.txt", "test_confusion_matrix.txt"):
        lines = (seed_dir / name).read_text().split("\n")
        assert lines[0] == "Seed 0"
        assert lines[1] == "=" * 21

    # Check the body matches the labelled confusion matrix of the saved CSV
    q = estimator.classes_.size
    test_df = pd.read_csv(seed_dir / "test_predictions.csv")
    expected = confusion_matrix(
        test_df["Target"], test_df["Prediction"], labels=np.arange(q)
    )
    text = (seed_dir / "test_confusion_matrix.txt").read_text()
    assert text.endswith("\n")
    body = text.split("\n", 2)[2].rstrip("\n")
    assert body == np.array2string(expected, separator=", ")


def test_confusion_matrix_not_elided_for_many_classes(tmp_path):
    """A large confusion matrix is written in full, without summarising."""
    labels = np.arange(40)
    _write_split_files(
        tmp_path,
        "train",
        index=None,
        true_y=labels,
        predicted_y=labels,
        proba=None,
        classes=labels,
        resample_id=0,
    )

    body = (
        (tmp_path / "train_confusion_matrix.txt")
        .read_text()
        .split("\n", 2)[2]
        .rstrip("\n")
    )
    assert "..." not in body
    # Check the matrix keeps one physical line per row (no wrapping)
    assert body.count("\n") == 39


@pytest.mark.parametrize("with_indices", [False, True])
def test_save_pattern_id_fallback(tmp_path, with_indices):
    """Pattern ID echoes the given indices and falls back to range(n)."""
    train_index = np.array([10, 11, 12, 13]) if with_indices else None
    test_index = np.array([40, 41]) if with_indices else None
    Results(tmp_path).save(
        _make_result(
            partition=0,
            dataset="toy",
            configuration="clf",
            best_params={},
            train_metrics={"acc_train": 1.0},
            test_metrics={"acc_test": 1.0},
            train_predicted_y=np.array([1, 2, 3, 1]),
            test_predicted_y=np.array([1, 2]),
            train_index=train_index,
            test_index=test_index,
        ),
        save_model=False,
    )

    seed_dir = tmp_path / "clf" / "toy" / "predictions_by_seed" / "seed_0"
    train_df = pd.read_csv(seed_dir / "train_predictions.csv")
    test_df = pd.read_csv(seed_dir / "test_predictions.csv")
    expected_train = train_index if with_indices else np.arange(4)
    expected_test = test_index if with_indices else np.arange(2)
    npt.assert_array_equal(train_df["Pattern ID"].values, expected_train)
    npt.assert_array_equal(test_df["Pattern ID"].values, expected_test)


def test_save_multiple_partitions_and_hyperparameter_upsert(tmp_path):
    """Repeated saves accumulate report rows and upsert hyperparameters."""
    r = Results(tmp_path)
    for i in range(3):
        r.save(
            _make_result(
                partition=i,
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
        _make_result(partition=0, best_params={"C": 0.1}, **base_result),
        save_model=False,
    )
    r.save(
        _make_result(partition=0, best_params={"C": 1.0}, **base_result),
        save_model=False,
    )

    hyper = pd.read_csv(tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv")
    seed_0 = hyper[hyper["Seed"] == 0]
    assert len(seed_0) == 1
    assert seed_0.iloc[0]["C"] == 1.0
    # Check rows stay sorted by Seed after the out-of-order upsert
    npt.assert_array_equal(hyper["Seed"].values, [0, 1, 2])


def test_hyperparameters_prefix_strip_and_union(tmp_path):
    """The clf__ prefix is stripped and missing params union to NaN."""
    r = Results(tmp_path)
    r.save(
        _make_result(
            partition=0,
            dataset="ds",
            configuration="clf",
            best_params={"clf__gamma": 1},
            train_metrics={"mae_train": 0.1},
            test_metrics={"mae_test": 0.1},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )
    r.save(
        _make_result(
            partition=1,
            dataset="ds",
            configuration="clf",
            best_params={"clf__gamma": 0.3, "clf__alpha": 2},
            train_metrics={"mae_train": 0.2},
            test_metrics={"mae_test": 0.2},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )

    hyper = pd.read_csv(tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv")
    # Check columns are Seed first, then alphabetical (not insertion order)
    assert list(hyper.columns) == ["Seed", "alpha", "gamma"]
    npt.assert_array_equal(hyper["Seed"].values, [0, 1])
    assert pd.isna(hyper[hyper["Seed"] == 0].iloc[0]["alpha"])
    assert hyper[hyper["Seed"] == 1].iloc[0]["alpha"] == 2

    # Check integer values survive the NaN column union unquoted
    raw = (tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv").read_text()
    assert "2.0" not in raw


def test_hyperparameters_empty_params(tmp_path):
    """Empty best_params yields a one-column CSV holding only Seed."""
    Results(tmp_path).save(
        _make_result(
            partition=0,
            dataset="ds",
            configuration="clf",
            best_params={},
            train_metrics={"mae_train": 0.1},
            test_metrics={"mae_test": 0.1},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )

    hyper = pd.read_csv(tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv")
    assert list(hyper.columns) == ["Seed"]


def test_hyperparameters_seed_param_cannot_shadow_key(tmp_path):
    """A parameter literally named Seed cannot overwrite the Seed column."""
    Results(tmp_path).save(
        _make_result(
            partition=5,
            dataset="ds",
            configuration="clf",
            best_params={"Seed": 999},
            train_metrics={"mae_train": 0.1},
            test_metrics={"mae_test": 0.1},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )

    hyper = pd.read_csv(tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv")
    npt.assert_array_equal(hyper["Seed"].values, [5])


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
            partition=0,
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


def test_atomic_write_success_leaves_no_temp(tmp_path):
    """_atomic_write writes full content and leaves no temp file."""
    target = tmp_path / "f.csv"
    _atomic_write(target, "a,b\n1,2\n")
    assert target.read_text() == "a,b\n1,2\n"
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_atomic_write_cleans_up_on_failure(tmp_path, monkeypatch):
    """A failing os.replace unlinks the temp file and re-raises."""
    monkeypatch.setattr(
        "skordinal.experiments._results.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(OSError, match="boom"):
        _atomic_write(tmp_path / "f.csv", "data")
    assert not (tmp_path / "f.csv").exists()
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_atomic_write_fsyncs(tmp_path, monkeypatch):
    """_atomic_write calls os.fsync on the temp file descriptor."""
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(
        "skordinal.experiments._results.os.fsync",
        lambda fd: calls.append(fd) or real_fsync(fd),
    )
    _atomic_write(tmp_path / "f.txt", "x")
    assert len(calls) == 1


def test_atomic_dump_cleans_up_on_failure(tmp_path, monkeypatch):
    """A failing os.replace unlinks the dump's temp file and re-raises."""
    monkeypatch.setattr(
        "skordinal.experiments._results.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(OSError, match="boom"):
        _atomic_dump(tmp_path / "m.joblib", {"k": 1})
    assert not (tmp_path / "m.joblib").exists()
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_atomic_dump_round_trips_no_temp(tmp_path):
    """_atomic_dump serialises an object recoverable by joblib.load."""
    target = tmp_path / "m.joblib"
    _atomic_dump(target, {"k": [1, 2, 3]})
    assert joblib.load(target) == {"k": [1, 2, 3]}
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", f"a{os.sep}b"])
def test_check_path_component_rejects_bad_strings(bad):
    """_check_path_component rejects empty, dotted or separator names."""
    with pytest.raises(ValueError):
        _check_path_component(bad, "classifier_name")


@pytest.mark.parametrize("bad", [3, None, ("x",)])
def test_check_path_component_rejects_non_str(bad):
    """_check_path_component rejects a non-string component."""
    with pytest.raises(TypeError):
        _check_path_component(bad, "classifier_name")


def test_save_rejects_traversal_before_writing(tmp_path):
    """save raises on a traversal component and writes nothing."""
    result = _make_result(
        partition=0,
        dataset="..",
        configuration="clf",
        best_params={},
        train_metrics={"mae_train": 0.1},
        test_metrics={"mae_test": 0.1},
        train_predicted_y=np.array([1]),
        test_predicted_y=np.array([1]),
    )
    with pytest.raises(ValueError):
        Results(tmp_path).save(result, save_model=False)
    assert list(tmp_path.iterdir()) == []


def test_report_row_round_trip_precision(tmp_path):
    """A high-precision metric survives a later save's report read."""
    value = 0.12345678901234566
    r = Results(tmp_path)
    r.save(
        _make_result(
            partition=0,
            dataset="ds",
            configuration="clf",
            best_params={},
            train_metrics={"mae_train": value},
            test_metrics={"mae_test": 0.1},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )
    # Second save reads report.csv back through the round-trip reader
    r.save(
        _make_result(
            partition=1,
            dataset="ds",
            configuration="clf",
            best_params={},
            train_metrics={"mae_train": 0.2},
            test_metrics={"mae_test": 0.2},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )
    df = pd.read_csv(
        tmp_path / "clf" / "ds" / "report.csv",
        index_col=0,
        float_precision="round_trip",
    )
    assert df.loc[0, "mae_train"] == value


def test_resave_is_idempotent_no_duplicate_row(tmp_path):
    """Re-saving the same resample keeps exactly one report row."""
    r = Results(tmp_path)
    for _ in range(2):
        r.save(
            _make_result(
                partition=0,
                dataset="ds",
                configuration="clf",
                best_params={},
                train_metrics={"mae_train": 0.1},
                test_metrics={"mae_test": 0.1},
                train_predicted_y=np.array([1]),
                test_predicted_y=np.array([1]),
            ),
            save_model=False,
        )
    df = pd.read_csv(tmp_path / "clf" / "ds" / "report.csv", index_col=0)
    assert df.shape[0] == 1


def test_orphan_temp_file_swept_on_save(tmp_path):
    """A leftover temp file under the pair dir is removed by save."""
    pair = tmp_path / "clf" / "ds"
    (pair / "predictions_by_seed" / "seed_0").mkdir(parents=True)
    stale = pair / f"{_TEMP_PREFIX}leftover.tmp"
    stale.write_text("junk")
    Results(tmp_path).save(
        _make_result(
            partition=0,
            dataset="ds",
            configuration="clf",
            best_params={},
            train_metrics={"mae_train": 0.1},
            test_metrics={"mae_test": 0.1},
            train_predicted_y=np.array([1]),
            test_predicted_y=np.array([1]),
        ),
        save_model=False,
    )
    assert not stale.exists()


def test_crash_between_predictions_and_report_not_committed(tmp_path, monkeypatch):
    """A crash before the report write leaves exists() False; rerun fixes it."""
    r = Results(tmp_path)
    kwargs = dict(
        partition=0,
        dataset="ds",
        configuration="clf",
        best_params={},
        train_metrics={"mae_train": 0.1},
        test_metrics={"mae_test": 0.1},
        train_predicted_y=np.array([1]),
        test_predicted_y=np.array([1]),
    )
    monkeypatch.setattr(
        Results,
        "_append_report_row",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        r.save(_make_result(**kwargs), save_model=False)
    # Predictions were written but no report row was committed
    seed = tmp_path / "clf" / "ds" / "predictions_by_seed" / "seed_0"
    assert (seed / "train_predictions.csv").is_file()
    assert r.exists("clf", "ds", "0") is False
    # A normal rerun commits the row
    monkeypatch.undo()
    r.save(_make_result(**kwargs), save_model=False)
    assert r.exists("clf", "ds", "0") is True


def test_stale_report_row_uncommitted_on_crash_resave(tmp_path, monkeypatch):
    """Re-saving a resample uncommits its row; a crash leaves it uncommitted."""
    r = Results(tmp_path)
    kwargs = dict(
        partition=0,
        dataset="ds",
        configuration="clf",
        best_params={},
        train_metrics={"mae_train": 0.1},
        test_metrics={"mae_test": 0.1},
        train_predicted_y=np.array([1]),
        test_predicted_y=np.array([1]),
    )
    r.save(_make_result(**kwargs), save_model=False)
    assert r.exists("clf", "ds", "0") is True
    # Crash on re-save after the uncommit but before the recommit
    monkeypatch.setattr(
        Results,
        "_append_report_row",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        r.save(_make_result(**kwargs), save_model=False)
    assert r.exists("clf", "ds", "0") is False
    monkeypatch.undo()
    r.save(_make_result(**kwargs), save_model=False)
    assert r.exists("clf", "ds", "0") is True
