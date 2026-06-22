"""Tests for the benchmark runner module."""

import json

import numpy as np
import pandas as pd
import pytest

from skordinal.experiments import Benchmark

_SVC_CONF: dict = {"SVM": {"classifier": "SVC", "parameters": {"C": [1]}}}
_MINIMAL_CONF: dict = {"cfg": {"classifier": "SVC", "parameters": {}}}
_BUNDLED_DS = "balance_scale"

_INVALID_CONSTRUCTOR_CASES = [
    pytest.param(
        {},
        [_BUNDLED_DS],
        ["mean_absolute_error"],
        "non-empty",
        id="empty-configurations",
    ),
    pytest.param(
        _MINIMAL_CONF,
        [],
        ["mean_absolute_error"],
        "non-empty",
        id="empty-datasets",
    ),
    pytest.param(
        _MINIMAL_CONF,
        [_BUNDLED_DS],
        [],
        "non-empty",
        id="empty-eval-metrics",
    ),
]


@pytest.fixture
def csv_ds_dir(tmp_path):
    """Write a small 3-class CSV plus a 2-entry masks file under tmp_path."""
    rng = np.random.default_rng(7)
    n = 60
    X = rng.standard_normal((n, 4))
    y = np.repeat([0, 1, 2], n // 3)
    rows = np.hstack([X, y.reshape(-1, 1)])
    np.savetxt(tmp_path / "smallds.csv", rows, delimiter=",", fmt="%.6f")
    mask0 = [True] * (n // 2) + [False] * (n // 2)
    mask1 = [False] * (n // 2) + [True] * (n // 2)
    (tmp_path / "smallds.masks.json").write_text(
        json.dumps([mask0, mask1]), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize(
    "configurations, datasets, eval_metrics, match",
    _INVALID_CONSTRUCTOR_CASES,
)
def test_benchmark_constructor_validation(
    tmp_path, configurations, datasets, eval_metrics, match
):
    """Constructor raises ValueError for each category of invalid argument."""
    with pytest.raises(ValueError, match=match):
        Benchmark(
            configurations,
            datasets=datasets,
            eval_metrics=eval_metrics,
            results_path=tmp_path,
        )


@pytest.mark.parametrize("bad_value", ["minmax", ""])
def test_input_preprocessing_invalid_raises(tmp_path, bad_value):
    """Unrecognised input_preprocessing values raise ValueError."""
    with pytest.raises(ValueError, match="'input_preprocessing' must be one of"):
        Benchmark(
            _MINIMAL_CONF,
            datasets=[_BUNDLED_DS],
            eval_metrics=["mean_absolute_error"],
            results_path=tmp_path,
            input_preprocessing=bad_value,
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
def test_input_preprocessing_accepted_and_normalised(tmp_path, raw, expected):
    """Valid input_preprocessing values are accepted and lower-stripped."""
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
        input_preprocessing=raw,
    )
    assert b.input_preprocessing == expected


def test_configurations_deep_copied(tmp_path):
    """Mutating the original configurations dict does not affect the stored copy."""
    original = {"cfg": {"classifier": "SVC", "parameters": {"C": [1]}}}
    b = Benchmark(
        original,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
    )
    original["cfg"]["parameters"]["C"].append(10)
    assert b.configurations["cfg"]["parameters"]["C"] == [1]


def test_data_home_str_stays_str(tmp_path):
    """A str data_home is stored verbatim, not converted to Path."""
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
        data_home=str(tmp_path),
    )
    assert isinstance(b.data_home, str)
    assert b.data_home == str(tmp_path)


def test_data_home_none_stays_none(tmp_path):
    """data_home=None is stored as None."""
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
    )
    assert b.data_home is None


def test_resamples_stored_verbatim(tmp_path):
    """resamples is stored as-is (int, not coerced)."""
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
        resamples=7,
    )
    assert b.resamples == 7


def test_protocol_attrs_stored(tmp_path):
    """cv, tuning_metric, eval_metrics, random_state, n_jobs, verbose are stored."""
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error", "accuracy_score"],
        results_path=tmp_path,
        resamples=3,
        cv=4,
        tuning_metric="accuracy_score",
        n_jobs=2,
        random_state=99,
        verbose=False,
    )
    assert b.cv == 4
    assert b.tuning_metric == "accuracy_score"
    assert b.eval_metrics == ["mean_absolute_error", "accuracy_score"]
    assert b.n_jobs == 2
    assert b.random_state == 99
    assert b.verbose is False


def test_run_and_summarize_bundled_dataset(tmp_path):
    """run() + summarize() over a bundled dataset write the expected on-disk layout."""
    results_dir = tmp_path / "runs"
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=results_dir,
        resamples=3,
        cv=2,
        verbose=False,
        random_state=0,
    )
    b.run()
    b.summarize()

    pair_dir = results_dir / "SVM" / _BUNDLED_DS
    assert pair_dir.is_dir()

    # report.csv uses resample_id as its index; one row per resample
    df = pd.read_csv(pair_dir / "report.csv", index_col=0)
    assert df.shape[0] == 3

    assert (pair_dir / "params.json").is_file()

    pred_dir = pair_dir / "predictions"
    train_preds = sorted(pred_dir.glob("train_*.csv"))
    test_preds = sorted(pred_dir.glob("test_*.csv"))
    assert len(train_preds) == 3
    assert len(test_preds) == 3

    # resample_id stems on prediction filenames are ints (0, 1, 2)
    ids_from_files = sorted(int(f.stem.split("_")[1]) for f in train_preds)
    assert ids_from_files == [0, 1, 2]

    # report.csv must contain one <metric>_train and one <metric>_test column
    assert "mean_absolute_error_train" in df.columns
    assert "mean_absolute_error_test" in df.columns

    assert (results_dir / "train_summary.csv").is_file()
    assert (results_dir / "test_summary.csv").is_file()

    # test_summary.csv must be well-formed with the expected aggregated columns
    summary = pd.read_csv(results_dir / "test_summary.csv")
    assert "SVM" in summary["classifier"].values
    assert "mean_absolute_error_test_mean" in summary.columns
    assert "n_completed" in summary.columns


def test_run_resamples_count_matches_requested(tmp_path):
    """run() with resamples=N produces exactly N rows in report.csv."""
    results_dir = tmp_path / "out"
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=results_dir,
        resamples=4,
        cv=2,
        verbose=False,
        random_state=1,
    )
    b.run()

    df = pd.read_csv(results_dir / "SVM" / _BUNDLED_DS / "report.csv", index_col=0)
    assert df.shape[0] == 4


def test_run_mask_path_correct_partition_count(tmp_path, csv_ds_dir):
    """run() with a masks file consumes exactly the mask-defined number of partitions."""
    results_dir = tmp_path / "mask_runs"
    b = Benchmark(
        _SVC_CONF,
        data_home=csv_ds_dir,
        datasets=["smallds"],
        eval_metrics=["mean_absolute_error"],
        results_path=results_dir,
        resamples=2,
        cv=2,
        verbose=False,
        random_state=0,
    )
    b.run()

    df = pd.read_csv(results_dir / "SVM" / "smallds" / "report.csv", index_col=0)
    assert df.shape[0] == 2


def test_run_mask_path_train_test_sizes_match_masks(tmp_path, csv_ds_dir):
    """Prediction file row counts match the mask-defined train/test sizes."""
    results_dir = tmp_path / "mask_runs2"
    b = Benchmark(
        _SVC_CONF,
        data_home=csv_ds_dir,
        datasets=["smallds"],
        eval_metrics=["mean_absolute_error"],
        results_path=results_dir,
        resamples=2,
        cv=2,
        verbose=False,
        random_state=0,
    )
    b.run()

    pred_dir = results_dir / "SVM" / "smallds" / "predictions"
    # Each mask splits n=60 half-and-half: 30 train / 30 test
    train_0 = pd.read_csv(pred_dir / "train_0.csv")
    test_0 = pd.read_csv(pred_dir / "test_0.csv")
    assert len(train_0) == 30
    assert len(test_0) == 30


def test_run_resamples_below_two_raises(tmp_path):
    """resamples=1 on the CV-fallback path raises ValueError at run time."""
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
        resamples=1,
        cv=2,
        verbose=False,
    )
    with pytest.raises(ValueError, match="resamples must be >= 2"):
        b.run()


def test_run_unresolvable_dataset_raises(tmp_path):
    """run() propagates FileNotFoundError for an unknown dataset name."""
    b = Benchmark(
        _SVC_CONF,
        datasets=["this_dataset_does_not_exist_xyz"],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
        resamples=3,
        cv=2,
        verbose=False,
    )
    with pytest.raises(FileNotFoundError):
        b.run()


def test_run_returns_none(tmp_path):
    """run() returns None."""
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path / "out",
        resamples=2,
        cv=2,
        verbose=False,
        random_state=0,
    )
    assert b.run() is None


def test_verbose_false_no_stdout(tmp_path, capsys):
    """verbose=False produces no stdout output during run()."""
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path / "out",
        resamples=2,
        cv=2,
        verbose=False,
        random_state=0,
    )
    b.run()
    captured = capsys.readouterr()
    assert captured.out == ""
