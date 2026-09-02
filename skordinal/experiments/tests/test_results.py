"""Tests for the Results class."""

import joblib
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skordinal.classifiers import LogisticAT
from skordinal.experiments import (
    Experiment,
    ExperimentResult,
    ModelConfig,
    Results,
)
from skordinal.experiments._io import _TEMP_PREFIX
from skordinal.metrics import accuracy_score, mean_absolute_error


def _fitted_svc(classes=(1, 2, 3), probability=False):
    """Return an SVC fitted so that ``classes_`` equals ``classes``."""
    labels = np.asarray(classes)
    rng = np.random.default_rng(0)
    features = rng.standard_normal((labels.size * 4, 4))
    targets = np.tile(labels, 4)
    return SVC(probability=probability).fit(features, targets)


_KEEP = object()


def _make_result(
    partition: int = 0,
    dataset: str = "ds",
    configuration: str = "clf",
    best_params: dict | None = None,
    train_metrics: dict | None = None,
    test_metrics: dict | None = None,
    train_predicted_y: np.ndarray | None = None,
    test_predicted_y=_KEEP,
    estimator=None,
    train_true_y=None,
    test_true_y=None,
    train_index=None,
    test_index=None,
    y_proba=None,
    scaler=None,
) -> ExperimentResult:
    if best_params is None:
        best_params = {}
    if train_metrics is None:
        train_metrics = {"mae_train": 0.1}
    if test_metrics is None:
        test_metrics = {"mae_test": 0.1}
    if train_predicted_y is None:
        train_predicted_y = np.array([1])
    if test_predicted_y is _KEEP:
        test_predicted_y = np.array([1])
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
        y_proba=y_proba,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        best_params=best_params,
        best_model=estimator,
        train_true_y=train_true_y,
        test_true_y=test_true_y,
        train_index=train_index,
        test_index=test_index,
        scaler=scaler,
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
        dataset="toy",
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
    estimator = _fitted_svc(probability=True)
    q = estimator.classes_.size

    # predicted_y encodes to [0, 2, 1] while the proba argmax is [0, 0, 2]
    train_predicted_y = np.array([1, 3, 2])
    train_y_proba = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.5, 0.3, 0.2],
            [0.2, 0.3, 0.5],
        ]
    )
    # predicted_y encodes to [0, 1, 2, 0] while the proba argmax is [2, 1, 2, 1]
    test_predicted_y = np.array([1, 2, 3, 1])
    y_proba = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.3, 0.4, 0.3],
        ]
    )
    result = ExperimentResult(
        dataset_name="toy",
        classifier_name="conf_1",
        resample_id=0,
        train_predicted_y=train_predicted_y,
        test_predicted_y=test_predicted_y,
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
    for name, proba, expected in (
        ("train_predictions.csv", train_y_proba, [0, 2, 1]),
        ("test_predictions.csv", y_proba, [0, 1, 2, 0]),
    ):
        df = pd.read_csv(seed_dir / name)
        assert "Prediction probabilities" in df.columns
        assert set(df["Target"]) <= set(range(q))
        # Check Prediction records predict(), not the argmax of the probabilities
        npt.assert_array_equal(df["Prediction"].values, expected)
        for cell, source in zip(df["Prediction probabilities"], proba):
            parsed = np.fromstring(cell.strip("[]"), sep=",")
            assert parsed.shape == (q,)
            npt.assert_allclose(parsed, source)
            npt.assert_allclose(parsed.sum(), 1.0, atol=1e-8)


def test_csv_reproduces_report_metrics(tmp_path):
    """Report metrics recompute from the saved CSV for rank-encoded labels."""
    rng = np.random.default_rng(0)
    X_train, X_test = rng.standard_normal((30, 4)), rng.standard_normal((15, 4))
    y_train, y_test = np.tile([0, 1, 2], 10), np.tile([0, 1, 2], 5)
    metrics = {
        "accuracy_score": accuracy_score,
        "mean_absolute_error": mean_absolute_error,
    }
    result = Experiment(ModelConfig(LogisticAT()), eval_metrics=list(metrics)).run(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="ds",
        classifier_name="clf",
        resample_id=0,
    )
    assert not np.array_equal(
        np.argmax(result.y_proba, axis=1), result.test_predicted_y
    )
    Results(tmp_path).save(result, save_model=False)

    pair_dir = tmp_path / "clf" / "ds"
    predictions = pd.read_csv(
        pair_dir / "predictions_by_seed" / "seed_0" / "test_predictions.csv"
    )
    report = pd.read_csv(
        pair_dir / "report.csv", index_col=0, float_precision="round_trip"
    )
    for name, metric in metrics.items():
        recomputed = metric(predictions["Target"], predictions["Prediction"])
        assert recomputed == pytest.approx(report.loc[0, f"{name}_test"])


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
        (
            {
                "test_predicted_y": np.array([1, 2, 4]),
                "y_proba": np.full((3, 3), 1 / 3),
            },
            "'test' predicted labels",
        ),
    ],
)
def test_save_rejects_unknown_labels(tmp_path, overrides, match):
    """A label absent from classes_ raises instead of mis-encoding."""
    kwargs = dict(
        train_predicted_y=np.array([1, 2, 3]),
        test_predicted_y=np.array([1, 2, 3]),
        train_true_y=np.array([1, 2, 3]),
        test_true_y=np.array([1, 2, 3]),
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        Results(tmp_path).save(_make_result(**kwargs))
    # Check the failed save left no model behind
    assert not (tmp_path / "clf" / "ds" / "models" / "0.joblib").exists()


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


def test_save_no_test_partition(tmp_path):
    """When test_predicted_y is None, no test files are written."""
    result = _make_result(
        dataset="toy",
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
        dataset="toy",
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


@pytest.mark.parametrize("with_indices", [False, True])
def test_save_pattern_id_fallback(tmp_path, with_indices):
    """Pattern ID echoes the given indices and falls back to range(n)."""
    train_index = np.array([10, 11, 12, 13]) if with_indices else None
    test_index = np.array([40, 41]) if with_indices else None
    Results(tmp_path).save(
        _make_result(
            dataset="toy",
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
                train_metrics={"mae_train": float(i)},
                test_metrics={"mae_test": float(i)},
            ),
            save_model=False,
        )

    df = pd.read_csv(tmp_path / "clf" / "ds" / "report.csv", index_col=0)
    assert df.shape[0] == 3

    r.save(_make_result(best_params={"C": 0.1}), save_model=False)
    r.save(_make_result(best_params={"C": 1.0}), save_model=False)

    # Re-saving seed 0 keeps one report row and one upserted parameter row
    df = pd.read_csv(tmp_path / "clf" / "ds" / "report.csv", index_col=0)
    assert df.shape[0] == 3
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
        _make_result(best_params={"clf__gamma": 1}),
        save_model=False,
    )
    r.save(
        _make_result(
            partition=1,
            best_params={"clf__gamma": 0.3, "clf__alpha": 2},
            train_metrics={"mae_train": 0.2},
            test_metrics={"mae_test": 0.2},
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


def test_hyperparameters_keep_a_searched_float_a_float(tmp_path):
    """A float grid value stays 10.0; the int column beside it stays an int."""
    r = Results(tmp_path)
    for seed, params in (
        (0, {"clf__C": 10.0, "clf__degree": 2}),
        (1, {"clf__C": 1.0, "clf__degree": 3}),
    ):
        r.save(
            _make_result(partition=seed, best_params=params),
            save_model=False,
        )

    raw = (tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv").read_text()
    # convert_dtypes() used to rewrite an all-integral float column as ints
    assert "10.0" in raw
    assert "3.0" not in raw


def test_hyperparameters_empty_params(tmp_path):
    """Empty best_params yields a one-column CSV holding only Seed."""
    Results(tmp_path).save(
        _make_result(),
        save_model=False,
    )

    hyper = pd.read_csv(tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv")
    assert list(hyper.columns) == ["Seed"]


def test_hyperparameters_seed_param_cannot_shadow_key(tmp_path):
    """A parameter literally named Seed cannot overwrite the Seed column."""
    Results(tmp_path).save(
        _make_result(partition=5, best_params={"Seed": 999}),
        save_model=False,
    )

    hyper = pd.read_csv(tmp_path / "clf" / "ds" / "hyperparameter_configuration.csv")
    npt.assert_array_equal(hyper["Seed"].values, [5])


def test_load(tmp_path):
    """Results.load() returns a Results pointing at the given folder; does not raise if folder is absent."""
    r = Results.load(tmp_path)
    assert isinstance(r, Results)
    assert r.path == tmp_path

    Results.load("/nonexistent/path/that/does/not/exist")


def test_path_is_expanded_and_absolute(monkeypatch, tmp_path):
    """Results.path expands a leading ~ and resolves a relative path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Results("~/my-run").path == tmp_path / "my-run"
    monkeypatch.chdir(tmp_path)
    assert Results("my-run").path == tmp_path / "my-run"


def test_report_and_hyperparameters_round_trip(tmp_path):
    """report and hyperparameters read back what save wrote, without precision loss."""
    results = Results(tmp_path)
    results.save(
        _make_result(
            partition=7,
            dataset="toy",
            configuration="conf_1",
            best_params={"clf__C": 0.12345678901234566},
            train_metrics={"mae_train": 0.12345678901234566},
            test_metrics={"mae_test": 0.25},
            train_predicted_y=np.array([1, 2, 3]),
            test_predicted_y=np.array([1, 2, 2]),
        ),
        save_model=False,
    )

    report = results.report("conf_1", "toy")
    assert list(report.index.astype(str)) == ["7"]
    assert report.loc[7, "mae_train"] == 0.12345678901234566
    assert report.loc[7, "mae_test"] == 0.25

    hyper = results.hyperparameters("conf_1", "toy")
    assert list(hyper["Seed"]) == [7]
    assert hyper.loc[0, "C"] == 0.12345678901234566


def test_predictions_reads_the_requested_split(tmp_path):
    """predictions returns the split's own rows, not the other split's."""
    results = Results(tmp_path)
    results.save(
        _make_result(
            dataset="toy",
            configuration="conf_1",
            train_metrics={},
            test_metrics={},
            train_predicted_y=np.array([1, 2, 3, 3]),
            test_predicted_y=np.array([1, 1]),
            train_true_y=np.array([1, 2, 3, 1]),
            test_true_y=np.array([1, 3]),
        ),
        save_model=False,
    )

    test_df = results.predictions("conf_1", "toy", 0)
    npt.assert_array_equal(test_df["Target"], [0, 2])
    npt.assert_array_equal(test_df["Prediction"], [0, 0])

    train_df = results.predictions("conf_1", "toy", 0, split="train")
    npt.assert_array_equal(train_df["Target"], [0, 1, 2, 0])
    npt.assert_array_equal(train_df["Prediction"], [0, 1, 2, 2])


def test_predictions_parse_proba(tmp_path):
    """parse_proba expands the stored cells into rows; the default leaves strings."""
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    results = Results(tmp_path)
    results.save(
        _make_result(
            dataset="toy",
            configuration="conf_1",
            train_metrics={},
            test_metrics={},
            train_predicted_y=np.array([1, 2]),
            test_predicted_y=np.array([1, 3]),
            y_proba=proba,
        ),
        save_model=False,
    )

    parsed = results.predictions("conf_1", "toy", 0, parse_proba=True)
    npt.assert_allclose(np.vstack(parsed["Prediction probabilities"]), proba)

    raw = results.predictions("conf_1", "toy", 0)
    assert raw["Prediction probabilities"].iloc[0].startswith("[")


def test_model_round_trips_the_bare_estimator(tmp_path):
    """Without a scaler, model loads back the estimator itself, not a Pipeline."""
    estimator = _fitted_svc()
    results = Results(tmp_path)
    results.save(
        _make_result(
            dataset="toy",
            configuration="conf_1",
            train_metrics={},
            test_metrics={},
            train_predicted_y=np.array([1, 2]),
            test_predicted_y=None,
            estimator=estimator,
        )
    )

    loaded = results.model("conf_1", "toy", 0)
    assert not isinstance(loaded, Pipeline)
    npt.assert_array_equal(loaded.classes_, estimator.classes_)


@pytest.mark.parametrize(
    "reader,args",
    [
        ("report", ("conf_1", "toy")),
        ("hyperparameters", ("conf_1", "toy")),
        ("predictions", ("conf_1", "toy", 0)),
        ("model", ("conf_1", "toy", 0)),
    ],
)
def test_readers_raise_for_a_missing_artefact(tmp_path, reader, args):
    """Every reader reports the absent path rather than an empty result."""
    with pytest.raises(FileNotFoundError):
        getattr(Results(tmp_path), reader)(*args)


@pytest.mark.parametrize(
    "reader,args",
    [
        ("report", (1, "toy")),
        ("hyperparameters", ("conf_1", 1)),
        ("predictions", (1, "toy", 0)),
        ("model", ("conf_1", 1, 0)),
    ],
)
def test_readers_reject_non_string_names(tmp_path, reader, args):
    """Every reader validates its path components the way exists does."""
    with pytest.raises(TypeError, match="must be a str"):
        getattr(Results(tmp_path), reader)(*args)


@pytest.mark.parametrize(
    "reader,args",
    [
        ("report", ("..", "toy")),
        ("hyperparameters", ("conf_1", "a/b")),
        ("predictions", ("conf_1", "toy", "..")),
        ("model", ("conf_1", "toy", "a/b")),
    ],
)
def test_readers_reject_escaping_names(tmp_path, reader, args):
    """Every reader rejects components that would escape the results tree."""
    with pytest.raises(ValueError):
        getattr(Results(tmp_path), reader)(*args)


def test_predictions_rejects_unknown_split(tmp_path):
    """predictions has no 'both' split to read."""
    with pytest.raises(ValueError, match="split must be"):
        Results(tmp_path).predictions("conf_1", "toy", 0, split="both")


def test_iter_experiments_yields_readable_pairs_sorted(tmp_path):
    """A report.csv or a predictions directory qualifies a pair; nothing else does."""
    _make_pair_csv(tmp_path, "clf_b", "ds_2", [{"mae_test": 0.1}])
    _make_pair_csv(tmp_path, "clf_a", "ds_2", [{"mae_test": 0.2}])
    _make_pair_csv(tmp_path, "clf_a", "ds_1", [{"mae_test": 0.3}])
    # Predictions without a report.csv, as in a tree written by another tool
    (tmp_path / "clf_a" / "ds_3" / "predictions_by_seed").mkdir(parents=True)
    (tmp_path / "clf_a" / "ds_0").mkdir()
    (tmp_path / "stray.txt").write_text("noise")

    assert list(Results(tmp_path).iter_experiments()) == [
        ("clf_a", "ds_1"),
        ("clf_a", "ds_2"),
        ("clf_a", "ds_3"),
        ("clf_b", "ds_2"),
    ]


def test_iter_experiments_on_a_missing_folder_is_empty(tmp_path):
    """iter_experiments yields nothing rather than raising on an absent root."""
    assert list(Results(tmp_path / "absent").iter_experiments()) == []


def test_exists(tmp_path):
    """exists() returns False when CSV is absent, False when resample is missing, True after save."""
    r = Results(tmp_path)
    assert r.exists("SVC", "toy", "0") is False

    _make_pair_csv(tmp_path, "SVC", "toy", [{"mae_test": 0.3}])
    assert r.exists("SVC", "toy", "99") is False

    r.save(
        _make_result(
            dataset="toy", configuration="SVC", test_metrics={"mae_test": 0.2}
        ),
        save_model=False,
    )
    assert r.exists("SVC", "toy", "0") is True
    assert r.exists("SVC", "toy", "1") is False
    # An int id must match the stringified CSV index just like its str form
    assert r.exists("SVC", "toy", 0) is True
    assert r.exists("SVC", "toy", 1) is False


def test_exists_rejects_traversal_resample_id(tmp_path):
    """exists() raises on a traversal-capable resample_id."""
    with pytest.raises(ValueError, match="resample_id"):
        Results(tmp_path).exists("SVC", "toy", "../../evil")


def test_report_csv_index_label_and_numeric_order(tmp_path):
    """report.csv's index column is labelled and rows sort numerically."""
    r = Results(tmp_path)
    for partition in (10, 2, 1):
        r.save(
            _make_result(partition=partition),
            save_model=False,
        )
    csv_path = tmp_path / "clf" / "ds" / "report.csv"
    assert csv_path.read_text().splitlines()[0].split(",")[0] == "resample_id"
    df = pd.read_csv(csv_path, index_col=0)
    assert list(df.index) == [1, 2, 10]


def test_report_rows_fall_back_to_lexicographic_order(tmp_path):
    """Non-integer resample ids sort as strings instead of failing the save."""
    r = Results(tmp_path)
    for partition in ("fold_b", "fold_a"):
        r.save(_make_result(partition=partition), save_model=False)
    df = pd.read_csv(tmp_path / "clf" / "ds" / "report.csv", index_col=0)
    assert list(df.index) == ["fold_a", "fold_b"]


def test_report_row_sort_preserves_pre_existing_duplicates(tmp_path):
    """Sorting a report that already holds a duplicate row must not multiply it."""
    _make_pair_csv(
        tmp_path,
        "clf",
        "ds",
        [{"mae_test": 0.1}, {"mae_test": 0.2}],
    )
    csv_path = tmp_path / "clf" / "ds" / "report.csv"
    # Rewrite the index so row "1" is duplicated, as an older writer could leave it
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.Index(["1", "1"])
    df.to_csv(csv_path)

    Results(tmp_path).save(
        _make_result(
            partition=2,
            train_metrics={"mae_train": 0.3},
            test_metrics={"mae_test": 0.3},
        ),
        save_model=False,
    )
    assert pd.read_csv(csv_path, index_col=0).shape[0] == 3


def test_save_removes_stale_artefacts(tmp_path):
    """Re-saving without a test split drops the prior run's test files and model."""
    r = Results(tmp_path)
    r.save(
        _make_result(),
        save_model=True,
    )
    seed_dir = tmp_path / "clf" / "ds" / "predictions_by_seed" / "seed_0"
    model_path = tmp_path / "clf" / "ds" / "models" / "0.joblib"
    assert (seed_dir / "test_predictions.csv").is_file()
    assert (seed_dir / "test_confusion_matrix.txt").is_file()
    assert model_path.is_file()

    r.save(
        _make_result(
            train_metrics={"mae_train": 0.2}, test_metrics={}, test_predicted_y=None
        ),
        save_model=False,
    )
    assert not (seed_dir / "test_predictions.csv").exists()
    assert not (seed_dir / "test_confusion_matrix.txt").exists()
    assert not model_path.exists()
    assert (seed_dir / "train_predictions.csv").is_file()
    df = pd.read_csv(tmp_path / "clf" / "ds" / "report.csv", index_col=0)
    assert df.shape[0] == 1


def test_save_wraps_a_failing_stale_artefact_removal(tmp_path, monkeypatch):
    """An OSError while clearing a prior run is wrapped with the failing dir."""
    r = Results(tmp_path)
    r.save(_make_result(), save_model=False)
    monkeypatch.setattr(
        "skordinal.experiments._results.shutil.rmtree",
        lambda *a, **k: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(OSError, match="Could not remove stale results"):
        r.save(_make_result(), save_model=False)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"dataset": ".."}, "dataset_name"),
        ({"partition": "../../evil"}, "resample_id"),
    ],
)
def test_save_rejects_traversal_before_writing(tmp_path, overrides, match):
    """save raises on a traversal name or resample id and writes nothing."""
    result = _make_result(**overrides)
    with pytest.raises(ValueError, match=match):
        Results(tmp_path).save(result, save_model=False)
    assert list(tmp_path.iterdir()) == []


def test_report_row_round_trip_precision(tmp_path):
    """A high-precision metric survives a later save's report read."""
    value = 0.12345678901234566
    r = Results(tmp_path)
    r.save(
        _make_result(train_metrics={"mae_train": value}),
        save_model=False,
    )
    # Second save reads report.csv back through the round-trip reader
    r.save(
        _make_result(
            partition=1,
            train_metrics={"mae_train": 0.2},
            test_metrics={"mae_test": 0.2},
        ),
        save_model=False,
    )
    df = pd.read_csv(
        tmp_path / "clf" / "ds" / "report.csv",
        index_col=0,
        float_precision="round_trip",
    )
    assert df.loc[0, "mae_train"] == value


def test_orphan_temp_file_swept_on_save(tmp_path):
    """A leftover temp file under the pair dir is removed by save."""
    pair = tmp_path / "clf" / "ds"
    (pair / "predictions_by_seed" / "seed_0").mkdir(parents=True)
    stale = pair / f"{_TEMP_PREFIX}leftover.tmp"
    stale.write_text("junk")
    Results(tmp_path).save(
        _make_result(),
        save_model=False,
    )
    assert not stale.exists()


def test_crash_between_predictions_and_report_not_committed(tmp_path, monkeypatch):
    """A crash before the report write leaves exists() False; rerun fixes it."""
    r = Results(tmp_path)
    monkeypatch.setattr(
        Results,
        "_append_report_row",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        r.save(_make_result(), save_model=False)
    # Predictions were written but no report row was committed
    seed = tmp_path / "clf" / "ds" / "predictions_by_seed" / "seed_0"
    assert (seed / "train_predictions.csv").is_file()
    assert r.exists("clf", "ds", "0") is False
    # A normal rerun commits the row
    monkeypatch.undo()
    r.save(_make_result(), save_model=False)
    assert r.exists("clf", "ds", "0") is True


def test_stale_report_row_uncommitted_on_crash_resave(tmp_path, monkeypatch):
    """Re-saving a resample uncommits its row; a crash leaves it uncommitted."""
    r = Results(tmp_path)
    r.save(_make_result(), save_model=False)
    assert r.exists("clf", "ds", "0") is True
    # Crash on re-save after the uncommit but before the recommit
    monkeypatch.setattr(
        Results,
        "_append_report_row",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        r.save(_make_result(), save_model=False)
    assert r.exists("clf", "ds", "0") is False
    monkeypatch.undo()
    r.save(_make_result(), save_model=False)
    assert r.exists("clf", "ds", "0") is True


def test_resave_crash_before_uncommit_leaves_prior_run_intact(tmp_path, monkeypatch):
    """A re-save that crashes at the uncommit deletes nothing from the prior run."""
    r = Results(tmp_path)
    r.save(_make_result(), save_model=False)
    monkeypatch.setattr(
        Results,
        "_uncommit_report_row",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("crash")),
    )
    with pytest.raises(RuntimeError):
        r.save(_make_result(), save_model=False)
    # The committed row must never outlive the files it describes
    assert r.exists("clf", "ds", "0") is True
    seed = tmp_path / "clf" / "ds" / "predictions_by_seed" / "seed_0"
    assert (seed / "train_predictions.csv").is_file()


def test_save_composes_the_scaler_into_the_model_artefact(tmp_path):
    """A scaled run persists a Pipeline that accepts raw, unscaled features."""
    raw = np.repeat([[1.0], [100.0], [10000.0]], 4, axis=0)
    labels = np.repeat([1, 2, 3], 4)
    scaler = StandardScaler().fit(raw)
    estimator = SVC().fit(scaler.transform(raw), labels)

    results = Results(tmp_path)
    results.save(
        _make_result(
            dataset="toy",
            configuration="conf_1",
            train_metrics={},
            test_metrics={},
            train_predicted_y=estimator.predict(scaler.transform(raw)),
            test_predicted_y=None,
            estimator=estimator,
            train_true_y=labels,
            scaler=scaler,
        )
    )

    artefact = results.model("conf_1", "toy", 0)
    assert isinstance(artefact, Pipeline)
    npt.assert_array_equal(
        artefact.predict(raw), estimator.predict(scaler.transform(raw))
    )
