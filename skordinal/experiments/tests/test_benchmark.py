"""Tests for the benchmark runner module."""

import json
import logging
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
    pytest.param(
        {"input_preprocessing": "std"},
        TypeError,
        "'input_preprocessing' must be None",
        id="string-input-preprocessing",
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


def _make_benchmark(results_path, **overrides):
    """Build a small, quiet, seeded Benchmark on the bundled dataset."""
    kwargs = dict(
        models=_SVC_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=results_path,
        resamples=2,
        cv=2,
        random_state=0,
        verbose=False,
    )
    kwargs.update(overrides)
    return Benchmark(**kwargs)


@pytest.fixture
def unconfigured_logging(monkeypatch):
    """Pretend no logging is configured; pytest's capture handlers would hide it."""
    monkeypatch.setattr(
        Benchmark, "_has_real_handler", staticmethod(lambda logger: False)
    )


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
def test_constructor_validation(tmp_path, overrides, exc_type, match):
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
    b = _make_benchmark(
        tmp_path,
        datasets=[f" {_BUNDLED_DS} "],
        eval_metrics=(m for m in [" mean_absolute_error "]),
    )
    assert b.eval_metrics == ["mean_absolute_error"]
    assert b.datasets == [_BUNDLED_DS]


def test_protocol_attrs_stored(tmp_path):
    """Every protocol argument is stored verbatim on the instance."""
    b = _make_benchmark(
        tmp_path,
        eval_metrics=["mean_absolute_error", "accuracy_score"],
        data_home=str(tmp_path),
        resamples=7,
        cv=4,
        tuning_metric="accuracy_score",
        n_jobs=2,
        random_state=99,
        verbose=False,
    )
    assert b.data_home == str(tmp_path)
    assert isinstance(b.data_home, str)
    assert b.resamples == 7
    assert b.cv == 4
    assert b.tuning_metric == "accuracy_score"
    assert b.eval_metrics == ["mean_absolute_error", "accuracy_score"]
    assert b.n_jobs == 2
    assert b.random_state == 99
    assert b.verbose is False


def test_protocol_defaults(tmp_path):
    """The documented defaults land unchanged on the instance."""
    b = Benchmark(
        _MINIMAL_CONF,
        datasets=[_BUNDLED_DS],
        eval_metrics=["mean_absolute_error"],
        results_path=tmp_path,
    )
    assert b.data_home is None
    assert b.resamples == 30
    assert b.tuning_metric == "neg_mean_absolute_error"
    assert b.random_state == 0
    assert b.overwrite is False
    assert b.verbose is True


def test_run_default_seed_is_reproducible(tmp_path):
    """Two separately-constructed runs with the same seed match exactly."""

    def _run(results_dir):
        _make_benchmark(results_dir, resamples=3).run()
        df = pd.read_csv(results_dir / "SVM" / _BUNDLED_DS / "report.csv", index_col=0)
        return df.drop(columns=[c for c in df.columns if c.startswith("time_")])

    df_a = _run(tmp_path / "a")
    df_b = _run(tmp_path / "b")
    pd.testing.assert_frame_equal(df_a, df_b)


def test_run_random_state_none_shares_partitions(tmp_path):
    """random_state=None resolves once, so every model sees identical partitions."""
    configs = {
        "SVM1": ModelConfig(SVC(), param_grid={"C": [1]}),
        "SVM2": ModelConfig(SVC(), param_grid={"C": [1]}),
    }
    results_dir = tmp_path / "out"
    b = _make_benchmark(results_dir, models=configs, resamples=3, random_state=None)
    assert isinstance(b.random_state, int)
    b.run()

    def _pattern_ids(label):
        seed_dir = results_dir / label / _BUNDLED_DS / "predictions_by_seed" / "seed_0"
        return pd.read_csv(seed_dir / "test_predictions.csv")["Pattern ID"].values

    np.testing.assert_array_equal(_pattern_ids("SVM1"), _pattern_ids("SVM2"))


def test_run_and_summarize_layout(tmp_path):
    """run() + summarize() over a bundled dataset write the expected on-disk layout."""
    results_dir = tmp_path / "runs"
    b = _make_benchmark(results_dir, resamples=3)
    assert b.run() is None
    b.summarize()

    pair_dir = results_dir / "SVM" / _BUNDLED_DS
    assert pair_dir.is_dir()

    # report.csv uses resample_id as its index; one row per resample
    df = pd.read_csv(pair_dir / "report.csv", index_col=0)
    assert df.shape[0] == 3
    assert "mean_absolute_error_train" in df.columns
    assert "mean_absolute_error_test" in df.columns

    assert (pair_dir / "hyperparameter_configuration.csv").is_file()
    assert not (pair_dir / "params.json").exists()

    pred_dir = pair_dir / "predictions_by_seed"
    train_preds = sorted(pred_dir.glob("seed_*/train_predictions.csv"))
    test_preds = sorted(pred_dir.glob("seed_*/test_predictions.csv"))
    assert len(train_preds) == 3
    assert len(test_preds) == 3
    ids_from_dirs = sorted(int(f.parent.name.split("_")[1]) for f in train_preds)
    assert ids_from_dirs == [0, 1, 2]

    assert (results_dir / "train_summary.csv").is_file()
    summary = pd.read_csv(results_dir / "test_summary.csv")
    assert "SVM" in summary["classifier"].values
    assert "mean_absolute_error_test_mean" in summary.columns
    assert "n_completed" in summary.columns


@pytest.mark.parametrize("overwrite, reruns", [(False, 0), (True, 3)])
def test_run_overwrite_controls_rerun(tmp_path, monkeypatch, overwrite, reruns):
    """A rerun recomputes already-saved resamples only when overwrite is True."""
    kwargs = dict(results_path=tmp_path / "runs", resamples=3, overwrite=overwrite)
    _make_benchmark(**kwargs).run()

    calls = []
    original = Experiment.run

    def _counting_run(self, *args, **kw):
        calls.append(1)
        return original(self, *args, **kw)

    monkeypatch.setattr(Experiment, "run", _counting_run)
    _make_benchmark(**kwargs).run()
    assert len(calls) == reruns


def test_run_with_a_masks_file(tmp_path, csv_ds_dir):
    """A masks file defines the partition count, sizes and Pattern IDs."""
    results_dir = tmp_path / "mask_runs"
    b = _make_benchmark(results_dir, data_home=csv_ds_dir, datasets=["smallds"])
    b.run()

    df = pd.read_csv(results_dir / "SVM" / "smallds" / "report.csv", index_col=0)
    assert df.shape[0] == 2

    seed_dir = results_dir / "SVM" / "smallds" / "predictions_by_seed" / "seed_0"
    # Mask 0 splits n=60 half-and-half: 30 train / 30 test, in file order
    train_0 = pd.read_csv(seed_dir / "train_predictions.csv")
    test_0 = pd.read_csv(seed_dir / "test_predictions.csv")
    np.testing.assert_array_equal(train_0["Pattern ID"].values, np.arange(30))
    np.testing.assert_array_equal(test_0["Pattern ID"].values, np.arange(30, 60))


def test_run_forwards_test_size(tmp_path):
    """test_size reaches load_partitions: balance_scale at 0.5 tests 313 rows."""
    b = _make_benchmark(tmp_path, resamples=1, test_size=0.5)
    b.run()

    seed_dir = tmp_path / "SVM" / _BUNDLED_DS / "predictions_by_seed" / "seed_0"
    assert len(pd.read_csv(seed_dir / "test_predictions.csv")) == 313


def test_run_unresolvable_dataset_raises(tmp_path):
    """run() propagates FileNotFoundError for an unknown dataset name."""
    b = _make_benchmark(tmp_path, datasets=["this_dataset_does_not_exist_xyz"])
    with pytest.raises(FileNotFoundError):
        b.run()


def test_run_not_verbose_logs_without_stdout(tmp_path, capsys, caplog):
    """verbose=False writes nothing to stdout but still emits the log records."""
    b = _make_benchmark(tmp_path / "out")
    with caplog.at_level("INFO", logger="skordinal.experiments"):
        b.run()
    assert capsys.readouterr().out == ""
    assert any("Running the" in m for m in caplog.messages)


def test_run_verbose_without_tqdm_logs_to_stdout(
    tmp_path, capsys, monkeypatch, unconfigured_logging
):
    """With verbose=True and no tqdm, progress falls back to stdout messages."""
    monkeypatch.setattr("skordinal.experiments._benchmark.tqdm", None)
    b = _make_benchmark(tmp_path / "out", resamples=1, verbose=True)
    b.run()
    out = capsys.readouterr().out
    assert f"Running the {_BUNDLED_DS} dataset" in out
    # The label line attributes the resample lines that follow it
    assert "Running SVM" in out
    # Without a live bar the per-resample lines surface at INFO, as on stdout
    assert "Running resample 0" in out
    assert "1 resample(s) run, 0 skipped" in out


def test_run_verbose_with_configured_logging_emits_once(tmp_path, capsys):
    """An app-configured handler wins: no stdout duplicate is attached."""
    records = []

    class Recorder(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("skordinal.experiments")
    handler = Recorder()
    logger.addHandler(handler)
    try:
        _make_benchmark(tmp_path / "out", resamples=1, verbose=True).run()
    finally:
        logger.removeHandler(handler)
    assert capsys.readouterr().out == ""
    assert any("resample(s) run" in m for m in records)


def test_run_wraps_resamples_in_a_progress_bar(tmp_path, monkeypatch, caplog):
    """The bar gets the total and latest score, and supersedes the INFO lines."""
    calls = {"postfixes": []}

    class FakeBar:
        def __init__(self, iterable, **kwargs):
            calls["kwargs"] = kwargs
            self._iterable = iterable
            self.disable = False

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, postfix, refresh=True):
            calls["postfixes"].append(postfix)
            calls["refresh"] = refresh

    monkeypatch.setattr("skordinal.experiments._benchmark.tqdm", FakeBar)
    b = _make_benchmark(tmp_path / "out", resamples=[0, 2], verbose=True)
    with caplog.at_level(logging.INFO, logger="skordinal.experiments"):
        b.run()
    assert calls["kwargs"]["total"] == 2
    assert calls["kwargs"]["desc"] == "  SVM"
    assert "mean_absolute_error_test" in calls["postfixes"][-1]
    # Refreshing on the spot would redraw the counter before it advances
    assert calls["refresh"] is False
    # The bar already names the model and counts resamples: no INFO duplicates
    assert not [m for m in caplog.messages if "Running SVM" in m]
    assert not [m for m in caplog.messages if "Running resample" in m]
    assert any("resample(s) run" in m for m in caplog.messages)


def test_run_leaves_nan_scores_out_of_the_summary(tmp_path, monkeypatch, caplog):
    """A NaN test score is not averaged into the per-model summary line."""
    original = Experiment.run

    def _nan_run(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.test_metrics["mean_absolute_error_test"] = float("nan")
        return result

    monkeypatch.setattr(Experiment, "run", _nan_run)
    b = _make_benchmark(tmp_path / "out", resamples=1)
    with caplog.at_level(logging.INFO, logger="skordinal.experiments"):
        b.run()
    line = next(m for m in caplog.messages if "resample(s) run" in m)
    assert "mean" not in line


def test_results_path_resolved_once_across_chdir(tmp_path, monkeypatch):
    """A relative results_path is anchored at construction, not at run."""
    work_a = tmp_path / "a"
    work_b = tmp_path / "b"
    work_a.mkdir()
    work_b.mkdir()
    monkeypatch.chdir(work_a)
    b = _make_benchmark("runs", resamples=3)
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
    pair = tmp_path / "out" / "SVM" / _BUNDLED_DS
    pair.mkdir(parents=True)
    (pair / "report.csv").write_text("", encoding="utf-8")
    with pytest.raises(pd.errors.EmptyDataError):
        _make_benchmark(tmp_path / "out").summarize()


def test_has_real_handler_walks_the_logger_chain():
    """The walk stops at a non-propagating logger and at the end of the chain."""
    orphan = logging.Logger("orphan-with-no-parent")
    assert Benchmark._has_real_handler(orphan) is False

    blocked = logging.getLogger("skordinal.experiments.tests.non-propagating")
    blocked.propagate = False
    try:
        assert Benchmark._has_real_handler(blocked) is False
    finally:
        blocked.propagate = True


@pytest.mark.parametrize("make_dir", [True, False], ids=["empty-dir", "missing-dir"])
def test_summarize_reports_when_nothing_to_do(
    tmp_path, capsys, unconfigured_logging, make_dir
):
    """summarize() over an empty or never-created folder warns, never raises."""
    results_path = tmp_path / "out"
    if make_dir:
        results_path.mkdir()
    b = _make_benchmark(results_path, verbose=True)
    b.summarize()
    assert "No metrics to summarise" in capsys.readouterr().out
