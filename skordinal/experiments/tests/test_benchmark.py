"""Tests for the benchmark runner module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

from skordinal.experiments import Benchmark, ModelConfig
from skordinal.experiments._experiment import Experiment

_SVC_CONF: dict[str, ModelConfig] = {"SVM": ModelConfig(SVC(), param_grid={"C": [1]})}
_MINIMAL_CONF: dict[str, ModelConfig] = {"cfg": ModelConfig(SVC())}
_BUNDLED_DS = "balance_scale"

_INVALID_CONSTRUCTOR_CASES = [
    pytest.param({"models": {}}, ValueError, "non-empty", id="empty-configurations"),
    pytest.param({"datasets": []}, ValueError, "non-empty", id="empty-datasets"),
    pytest.param(
        {"eval_metrics": []}, ValueError, "non-empty", id="empty-eval-metrics"
    ),
    pytest.param(
        {"eval_metrics": "accuracy_score"},
        TypeError,
        r"not a bare string; pass \['accuracy_score'\]",
        id="bare-string-eval-metrics",
    ),
    pytest.param(
        {"eval_metrics": [3]},
        TypeError,
        r"must contain only metric name strings; got 'int'",
        id="non-str-eval-metric",
    ),
    # Eval metrics use the label registry, where a loss keeps its plain name
    pytest.param(
        {"eval_metrics": ["neg_mean_absolute_error"]},
        ValueError,
        r"Unknown metric name: 'neg_mean_absolute_error'\.",
        id="scorer-name-as-eval-metric",
    ),
    # Tuning metrics use the scorer registry, where a loss exists only as neg_
    pytest.param(
        {"tuning_metric": "mean_absolute_error"},
        ValueError,
        r"a loss is only registered as 'neg_mean_absolute_error'",
        id="bare-loss-as-tuning-metric",
    ),
    # Both name a directory in the results tree, so they fail eagerly
    pytest.param(
        {"datasets": ["dir/era"]},
        ValueError,
        "dataset name must not contain a path separator",
        id="path-dataset-name",
    ),
    pytest.param(
        {"models": {"a/b": ModelConfig(SVC())}},
        ValueError,
        "model label must not contain a path separator",
        id="path-model-label",
    ),
    pytest.param(
        {"tuning_metric": 5},
        TypeError,
        "scoring",
        id="non-string-tuning-metric",
    ),
    pytest.param(
        {"models": {"cfg": SVC()}},
        TypeError,
        "must be ModelConfig instances",
        id="non-modelconfig-value",
    ),
    pytest.param(
        {"models": [ModelConfig(SVC())]},
        TypeError,
        "'models' must be a dict",
        id="models-not-a-dict",
    ),
    # A bare string would otherwise become one dataset per character
    pytest.param(
        {"datasets": _BUNDLED_DS},
        TypeError,
        r"not a bare string; pass \['balance_scale'\]",
        id="bare-string-datasets",
    ),
]


@pytest.fixture
def csv_ds_dir(tmp_path):
    """Write a small 3-class CSV plus a 2-entry masks file under tmp_path."""
    rng = np.random.default_rng(7)
    n = 60
    X = rng.standard_normal((n, 4))
    # Interleave classes so both mask halves contain every class
    y = np.tile([0, 1, 2], n // 3)
    rows = np.hstack([X, y.reshape(-1, 1)])
    np.savetxt(tmp_path / "smallds.csv", rows, delimiter=",", fmt="%.6f")
    mask0 = [True] * (n // 2) + [False] * (n // 2)
    mask1 = [False] * (n // 2) + [True] * (n // 2)
    (tmp_path / "smallds.masks.json").write_text(
        json.dumps([mask0, mask1]), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize("overrides, exc_type, match", _INVALID_CONSTRUCTOR_CASES)
def test_benchmark_constructor_validation(tmp_path, overrides, exc_type, match):
    """Each invalid constructor argument raises at construction."""
    kwargs = {
        "models": _MINIMAL_CONF,
        "datasets": [_BUNDLED_DS],
        "eval_metrics": ["mean_absolute_error"],
        "results_path": tmp_path,
        **overrides,
    }
    with pytest.raises(exc_type, match=match):
        Benchmark(**kwargs)


def test_names_stored_stripped(tmp_path):
    """Metric and dataset names are stripped; a one-shot iterable is kept."""
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[f" {_BUNDLED_DS} "],
        eval_metrics=(m for m in [" mean_absolute_error "]),
        results_path=tmp_path,
    )
    assert b.eval_metrics == ["mean_absolute_error"]
    assert b.datasets == [_BUNDLED_DS]


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


def test_run_default_seed_reproducible_across_constructions(tmp_path):
    """Two separately-constructed runs with the default random_state match."""

    def _run(results_dir):
        b = Benchmark(
            _SVC_CONF,
            datasets=[_BUNDLED_DS],
            eval_metrics=["mean_absolute_error"],
            results_path=results_dir,
            resamples=3,
            cv=2,
            verbose=False,
        )
        b.run()
        df = pd.read_csv(results_dir / "SVM" / _BUNDLED_DS / "report.csv", index_col=0)
        return df.drop(columns=[c for c in df.columns if c.startswith("time_")])

    df_a = _run(tmp_path / "a")
    df_b = _run(tmp_path / "b")
    pd.testing.assert_frame_equal(df_a, df_b)


def test_run_all_models_see_identical_partitions_with_random_state_none(tmp_path):
    """random_state=None resolves once, so every model sees identical partitions."""
    configs = {
        "SVM1": ModelConfig(SVC(), param_grid={"C": [1]}),
        "SVM2": ModelConfig(SVC(), param_grid={"C": [1]}),
    }
    results_dir = tmp_path / "out"
    b = Benchmark(
        configs,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=results_dir,
        resamples=3,
        cv=2,
        verbose=False,
        random_state=None,
    )
    assert isinstance(b.random_state, int)
    b.run()

    def _pattern_ids(label):
        seed_dir = results_dir / label / _BUNDLED_DS / "predictions_by_seed" / "seed_0"
        return pd.read_csv(seed_dir / "test_predictions.csv")["Pattern ID"].values

    np.testing.assert_array_equal(_pattern_ids("SVM1"), _pattern_ids("SVM2"))


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

    assert (pair_dir / "hyperparameter_configuration.csv").is_file()
    assert not (pair_dir / "params.json").exists()

    pred_dir = pair_dir / "predictions_by_seed"
    train_preds = sorted(pred_dir.glob("seed_*/train_predictions.csv"))
    test_preds = sorted(pred_dir.glob("seed_*/test_predictions.csv"))
    assert len(train_preds) == 3
    assert len(test_preds) == 3

    # Check per-seed directory names carry the resample_id (0, 1, 2)
    ids_from_dirs = sorted(int(f.parent.name.split("_")[1]) for f in train_preds)
    assert ids_from_dirs == [0, 1, 2]

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


@pytest.mark.parametrize("overwrite, reruns", [(False, 0), (True, 3)])
def test_run_overwrite_controls_rerun(tmp_path, monkeypatch, overwrite, reruns):
    """A rerun recomputes already-saved resamples only when overwrite is True."""
    kwargs = dict(
        models=_SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path / "runs",
        resamples=3,
        cv=2,
        verbose=False,
        random_state=0,
        overwrite=overwrite,
    )
    Benchmark(**kwargs).run()

    calls = []
    original = Experiment.run

    def _counting_run(self, *args, **kw):
        calls.append(1)
        return original(self, *args, **kw)

    monkeypatch.setattr(Experiment, "run", _counting_run)
    Benchmark(**kwargs).run()
    assert len(calls) == reruns


@pytest.mark.parametrize("resamples", [1, 4], ids=["one", "several"])
def test_run_resamples_count_matches_requested(tmp_path, resamples):
    """run() with resamples=N produces exactly N rows in report.csv."""
    results_dir = tmp_path / "out"
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=results_dir,
        resamples=resamples,
        cv=2,
        verbose=False,
        random_state=1,
    )
    b.run()

    df = pd.read_csv(results_dir / "SVM" / _BUNDLED_DS / "report.csv", index_col=0)
    assert df.shape[0] == resamples


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

    seed_dir = results_dir / "SVM" / "smallds" / "predictions_by_seed" / "seed_0"
    # Each mask splits n=60 half-and-half: 30 train / 30 test
    train_0 = pd.read_csv(seed_dir / "train_predictions.csv")
    test_0 = pd.read_csv(seed_dir / "test_predictions.csv")
    assert len(train_0) == 30
    assert len(test_0) == 30

    # Check Pattern ID carries the original row positions defined by mask 0
    np.testing.assert_array_equal(train_0["Pattern ID"].values, np.arange(30))
    np.testing.assert_array_equal(test_0["Pattern ID"].values, np.arange(30, 60))


def test_run_forwards_test_size(tmp_path):
    """test_size reaches load_partitions: balance_scale at 0.5 tests 313 rows."""
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
        resamples=1,
        test_size=0.5,
        cv=2,
        verbose=False,
    )
    b.run()

    seed_dir = tmp_path / "SVM" / _BUNDLED_DS / "predictions_by_seed" / "seed_0"
    assert len(pd.read_csv(seed_dir / "test_predictions.csv")) == 313


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


def test_results_path_resolved_once_across_chdir(tmp_path, monkeypatch):
    """A relative results_path is anchored at construction, not at run."""
    work_a = tmp_path / "a"
    work_b = tmp_path / "b"
    work_a.mkdir()
    work_b.mkdir()
    monkeypatch.chdir(work_a)
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path="runs",
        resamples=3,
        cv=2,
        verbose=False,
        random_state=0,
    )
    # results_path is now absolute and anchored under work_a
    assert Path(b.results_path).is_absolute()
    assert Path(b.results_path) == (work_a / "runs").resolve()
    monkeypatch.chdir(work_b)
    b.run()
    # Results land under the construction-time root, not the new cwd
    assert (work_a / "runs" / "SVM" / _BUNDLED_DS / "report.csv").is_file()
    assert not (work_b / "runs").exists()


def test_summarize_reraises_a_corrupt_report(tmp_path):
    """An unreadable report.csv is data loss, so it must not read as 'nothing'."""
    pair = tmp_path / "out" / "cfg" / _BUNDLED_DS
    pair.mkdir(parents=True)
    (pair / "report.csv").write_text("", encoding="utf-8")
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path / "out",
        verbose=True,
    )
    with pytest.raises(pd.errors.EmptyDataError):
        b.summarize()


@pytest.mark.parametrize("make_dir", [True, False], ids=["empty-dir", "missing-dir"])
def test_summarize_without_results_reports_instead_of_raising(
    tmp_path, capsys, make_dir
):
    """summarize() over an empty or never-created folder warns, never raises."""
    results_path = tmp_path / "out"
    if make_dir:
        results_path.mkdir()
    b = Benchmark(
        _SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=results_path,
        resamples=2,
        cv=2,
        verbose=True,
        random_state=0,
    )
    b.summarize()
    assert "No metrics to summarise" in capsys.readouterr().out
