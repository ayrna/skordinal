"""Tests for the _evaluation read-back/reporting functions."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

from skordinal.experiments import (
    ExperimentResult,
    Results,
    evaluate,
    save_summary,
    summarize,
    tabulate_results,
)
from skordinal.metrics import mean_absolute_error


def _write_report(base, classifier, dataset, rows):
    """Write report.csv from a {resample_id: metrics} mapping."""
    pair_dir = base / classifier / dataset
    pair_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pair_dir / "report.csv"
    frame = pd.DataFrame(list(rows.values()), index=pd.Index(list(rows)))
    frame.to_csv(csv_path, index_label="resample_id")
    return csv_path


def _make_pair_csv(base, classifier, dataset, rows):
    """Write a report.csv whose resample ids are 0..len(rows) - 1."""
    return _write_report(base, classifier, dataset, dict(enumerate(rows)))


@pytest.fixture
def two_pair_folder(tmp_path):
    """Two classifiers by two datasets, each with two resamples."""
    for clf in ("A", "B"):
        for ds in ("d1", "d2"):
            _make_pair_csv(
                tmp_path,
                clf,
                ds,
                [
                    {
                        "ccr_train": 0.8,
                        "mae_train": 0.2,
                        "ccr_test": 0.7,
                        "mae_test": 0.3,
                    },
                    {
                        "ccr_train": 0.6,
                        "mae_train": 0.4,
                        "ccr_test": 0.5,
                        "mae_test": 0.5,
                    },
                ],
            )
    return tmp_path


def _write_seed(
    base, classifier, dataset, resample_id, split, target=None, prediction=None
):
    """Create one seed directory, writing its {split}_predictions.csv when given."""
    seed_dir = (
        base / classifier / dataset / "predictions_by_seed" / f"seed_{resample_id}"
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    if target is not None:
        pd.DataFrame(
            {
                "Pattern ID": range(len(target)),
                "Target": target,
                "Prediction": prediction,
            }
        ).to_csv(seed_dir / f"{split}_predictions.csv", index=False)
    return seed_dir


@pytest.fixture
def seed_folder(tmp_path):
    """One pair with resamples 2 and 10 committed, each split differing."""
    _write_report(
        tmp_path,
        "A",
        "d1",
        {
            2: {"mean_absolute_error_train": 0.5, "mean_absolute_error_test": 0.25},
            10: {"mean_absolute_error_train": 0.5, "mean_absolute_error_test": 1.0},
        },
    )
    _write_seed(tmp_path, "A", "d1", 2, "test", [0, 1, 2, 2], [0, 1, 1, 2])
    _write_seed(tmp_path, "A", "d1", 10, "test", [0, 1, 2, 2], [2, 1, 0, 2])
    _write_seed(tmp_path, "A", "d1", 2, "train", [0, 1, 2], [2, 1, 2])
    _write_seed(tmp_path, "A", "d1", 10, "train", [0, 1, 2], [0, 1, 2])
    return tmp_path


def test_evaluate_recomputes_metrics_per_seed(seed_folder):
    """evaluate scores every committed seed, in numeric resample order."""
    df = evaluate(seed_folder, "A", "d1")
    assert list(df.columns) == ["mean_absolute_error", "accuracy_score"]
    assert df.index.name == "resample_id"
    # 10 sorts after 2 numerically but before it lexicographically
    assert list(df.index) == [2, 10]
    assert list(df["mean_absolute_error"]) == pytest.approx([0.25, 1.0])
    assert list(df["accuracy_score"]) == pytest.approx([0.75, 0.5])


def test_evaluate_reads_the_requested_split(seed_folder):
    """evaluate scores the train files when split='train'."""
    df = evaluate(seed_folder, "A", "d1", split="train")
    assert list(df["mean_absolute_error"]) == pytest.approx([2 / 3, 0.0])


def test_evaluate_includes_string_resample_ids(tmp_path):
    """A committed non-integer resample id is evaluated, not silently dropped."""
    _write_report(
        tmp_path,
        "A",
        "d1",
        {
            2: {"mean_absolute_error_test": 0.0},
            "fold": {"mean_absolute_error_test": 1.0},
        },
    )
    _write_seed(tmp_path, "A", "d1", 2, "test", [0, 1], [0, 1])
    _write_seed(tmp_path, "A", "d1", "fold", "test", [0, 2], [0, 0])
    df = evaluate(tmp_path, "A", "d1")
    assert list(df.index) == [2, "fold"]
    assert df.loc["fold", "mean_absolute_error"] == pytest.approx(1.0)


def test_evaluate_skips_uncommitted_seed(seed_folder):
    """A seed directory with no report.csv row is not evaluated."""
    _write_seed(seed_folder, "A", "d1", 99, "test", [0, 1, 2, 2], [2, 2, 2, 2])
    df = evaluate(seed_folder, "A", "d1")
    assert list(df.index) == [2, 10]


def test_evaluate_skips_non_seed_entries(tmp_path, recwarn):
    """Only a directory named ``seed_<id>`` counts, even when the id is committed."""
    _write_report(
        tmp_path,
        "A",
        "d1",
        {2: {"mean_absolute_error_test": 0.0}, 10: {"mean_absolute_error_test": 0.0}},
    )
    _write_seed(tmp_path, "A", "d1", 2, "test", [0, 1], [0, 1])
    seeds_dir = tmp_path / "A" / "d1" / "predictions_by_seed"
    # Taken for a seed, a file with no predictions file inside warns
    (seeds_dir / "seed_10").write_text("noise")
    # Taken for a seed, an unprefixed directory contributes foreign predictions
    unprefixed = seeds_dir / "10"
    unprefixed.mkdir()
    pd.DataFrame({"Pattern ID": [0], "Target": [0], "Prediction": [2]}).to_csv(
        unprefixed / "test_predictions.csv", index=False
    )

    df = evaluate(tmp_path, "A", "d1")
    assert list(df.index) == [2]
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_evaluate_warns_on_a_missing_committed_file(tmp_path):
    """A committed row with metrics for the split but no file is corruption."""
    _write_report(tmp_path, "A", "d1", {3: {"mean_absolute_error_test": 0.5}})
    _write_seed(tmp_path, "A", "d1", 3, "test")
    with pytest.warns(RuntimeWarning, match="is missing"):
        df = evaluate(tmp_path, "A", "d1")
    assert df.empty


def test_evaluate_is_silent_for_a_train_only_row(tmp_path, recwarn):
    """A committed row with no test metrics is a train-only run, not corruption."""
    _write_report(
        tmp_path,
        "A",
        "d1",
        {
            3: {
                "mean_absolute_error_train": 0.5,
                "mean_absolute_error_test": float("nan"),
            }
        },
    )
    _write_seed(tmp_path, "A", "d1", 3, "train", [0, 1], [0, 1])
    assert evaluate(tmp_path, "A", "d1").empty
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_evaluate_reads_a_pair_without_a_report(seed_folder):
    """With no commit marker at all, every seed on disk scores as it would with one."""
    (seed_folder / "A" / "d1" / "report.csv").unlink()
    df = evaluate(seed_folder, "A", "d1")
    assert list(df.index) == [2, 10]
    assert list(df["mean_absolute_error"]) == pytest.approx([0.25, 1.0])


def test_summarize_and_tabulate_skip_a_pair_without_a_report(tmp_path):
    """A pair holding only predictions stores no metrics, so both readers skip it."""
    _make_pair_csv(tmp_path, "A", "d1", [{"mae_test": 0.3}])
    _write_seed(tmp_path, "B", "d1", 0, "test", [0, 1], [0, 1])

    assert list(summarize(tmp_path).index) == [("A", "d1")]
    table = tabulate_results(tmp_path, metric="mae")
    assert list(table.index) == ["A"]
    assert list(table.columns) == ["d1"]


@pytest.mark.parametrize(
    "args,kwargs,exception,match",
    [
        ((1, "d1"), {}, TypeError, "must be a str"),
        ((("..", "d1")), {}, ValueError, "dot segment"),
        (("A", "a/b"), {}, ValueError, "path separator"),
        (("A", "d1"), {"split": "both"}, ValueError, "split must be"),
        (
            ("A", "d1"),
            {"metrics": "mean_absolute_error"},
            TypeError,
            "not a bare string",
        ),
        (("A", "d1"), {"metrics": []}, ValueError, "non-empty"),
        (
            ("A", "d1"),
            {"metrics": [mean_absolute_error]},
            TypeError,
            "only metric name strings",
        ),
        (("A", "d1"), {"metrics": ["not_a_metric"]}, ValueError, "Unknown metric name"),
    ],
    ids=[
        "name-type",
        "name-dot",
        "name-separator",
        "split",
        "metrics-string",
        "metrics-empty",
        "metrics-element",
        "metrics-unknown",
    ],
)
def test_evaluate_rejects_invalid_arguments(tmp_path, args, kwargs, exception, match):
    """Each guard fires before any read: tmp_path holds no tree to fall back on."""
    with pytest.raises(exception, match=match):
        evaluate(tmp_path, *args, **kwargs)


def test_evaluate_missing_predictions_directory_raises(tmp_path):
    """evaluate raises FileNotFoundError when the pair was never written."""
    with pytest.raises(FileNotFoundError, match="predictions_by_seed"):
        evaluate(tmp_path, "A", "d1")


def test_evaluate_computes_exactly_the_requested_metrics(seed_folder):
    """Requested names become columns once each, stripped and deduplicated."""
    df = evaluate(seed_folder, "A", "d1", metrics=[" accuracy_score "] * 2)
    assert list(df.columns) == ["accuracy_score"]
    assert list(df["accuracy_score"]) == pytest.approx([0.75, 0.5])


def test_evaluate_reports_a_malformed_predictions_file(tmp_path):
    """A committed predictions file without the label columns fails loud."""
    _write_report(tmp_path, "A", "d1", {0: {"mean_absolute_error_test": 0.5}})
    seed_dir = _write_seed(tmp_path, "A", "d1", 0, "test")
    pd.DataFrame({"Pattern ID": [0, 1]}).to_csv(
        seed_dir / "test_predictions.csv", index=False
    )
    with pytest.raises(ValueError, match="Malformed predictions file"):
        evaluate(tmp_path, "A", "d1")


def test_evaluate_scores_in_rank_space_not_raw_labels(tmp_path):
    """Recomputed values use ranks, so a gapped label set diverges from report.csv."""
    classes = np.array([0, 5, 10])
    features = np.repeat(classes, 4).reshape(-1, 1).astype(float)
    estimator = SVC().fit(features, np.repeat(classes, 4))
    true_y = np.array([0, 5, 10])
    predicted_y = np.array([0, 0, 10])
    Results(tmp_path).save(
        ExperimentResult(
            dataset_name="d1",
            classifier_name="A",
            resample_id=0,
            train_predicted_y=predicted_y,
            test_predicted_y=predicted_y,
            y_proba=None,
            train_metrics={},
            test_metrics={"mean_absolute_error_test": 5 / 3},
            best_params={},
            best_model=estimator,
            train_true_y=true_y,
            test_true_y=true_y,
        ),
        save_model=False,
    )
    df = evaluate(tmp_path, "A", "d1")
    # Ranks make 0 and 5 adjacent, so the recomputed error is 1/3, not 5/3
    assert df.loc[0, "mean_absolute_error"] == pytest.approx(1 / 3)
    assert mean_absolute_error(true_y, predicted_y) == pytest.approx(5 / 3)


def test_summarize_aggregates_metrics(two_pair_folder):
    """summarize returns a MultiIndex frame with mean, std, and count."""
    df = summarize(two_pair_folder, split="test")
    assert isinstance(df.index, pd.MultiIndex)
    assert isinstance(df.columns, pd.MultiIndex)
    # ccr_test and mae_test each contribute mean and std, plus n_completed
    assert df.shape == (4, 5)
    assert df.loc[("A", "d1"), ("mae_test", "mean")] == pytest.approx(0.4)
    assert df.loc[("A", "d1"), ("mae_test", "std")] == pytest.approx(0.1414, abs=1e-3)
    assert df.loc[("A", "d1"), ("n_completed", "")] == 2


@pytest.mark.parametrize(
    "split,expected",
    [
        ("test", {"ccr_test", "mae_test"}),
        ("train", {"ccr_train", "mae_train"}),
        ("both", {"ccr_train", "mae_train", "ccr_test", "mae_test"}),
    ],
)
def test_summarize_selects_columns_by_split(two_pair_folder, split, expected):
    """summarize keeps only the metric columns for the requested split."""
    df = summarize(two_pair_folder, split=split)
    metrics = {c[0] for c in df.columns if c[0] != "n_completed"}
    assert metrics == expected


@pytest.mark.parametrize(
    "classifiers,expected_clfs",
    [
        (["A"], {"A"}),
        (["nonexistent"], set()),
    ],
    ids=["subset", "absent"],
)
def test_summarize_filters_by_classifiers(two_pair_folder, classifiers, expected_clfs):
    """summarize with classifiers keeps only matching pairs or empty."""
    df = summarize(two_pair_folder, split="test", classifiers=classifiers)
    if expected_clfs:
        assert {clf for clf, _ in df.index} == expected_clfs
    else:
        assert df.empty


@pytest.mark.parametrize(
    "rows,mean,std,n_completed",
    [
        ([{"mae_test": float("nan")}], float("nan"), float("nan"), 1),
        ([{"mae_test": 0.3}], 0.3, 0.0, 1),
        ([{"mae_test": 0.2}, {"mae_test": 0.4}], 0.3, 0.1414, 2),
    ],
    ids=["all-nan", "single-value", "two-values"],
)
def test_summarize_std_rule_by_n(tmp_path, rows, mean, std, n_completed):
    """summarize std: nan/0.0/ddof=1 by n; n_completed counts all rows."""
    _make_pair_csv(tmp_path, "clf", "ds", rows)
    df = summarize(tmp_path, split="test")
    got_mean = df.loc[("clf", "ds"), ("mae_test", "mean")]
    got_std = df.loc[("clf", "ds"), ("mae_test", "std")]
    if math.isnan(mean):
        assert math.isnan(got_mean)
        assert math.isnan(got_std)
    else:
        assert got_mean == pytest.approx(mean)
        assert got_std == pytest.approx(std, abs=1e-3)
    # n_completed counts total rows written, including any with NaN values
    assert df.loc[("clf", "ds"), ("n_completed", "")] == n_completed


def test_summarize_skips_non_pair_entries(tmp_path):
    """summarize ignores stray files and dataset dirs without a report."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": 0.1}])
    (tmp_path / "stray.txt").write_text("noise")
    (tmp_path / "clf" / "stray.txt").write_text("noise")
    (tmp_path / "clf" / "empty_ds").mkdir()
    df = summarize(tmp_path, split="test")
    assert {(clf, ds) for clf, ds in df.index} == {("clf", "ds")}


@pytest.mark.parametrize(
    "kwargs, exc_type, match",
    [
        ({"classifiers": "A"}, TypeError, "iterable"),
        ({"split": "bad"}, ValueError, "split must be"),
    ],
    ids=["bare-string-classifiers", "unknown-split"],
)
def test_summarize_rejects_invalid_arguments(two_pair_folder, kwargs, exc_type, match):
    """A bare-string classifiers filter and an unknown split both raise."""
    with pytest.raises(exc_type, match=match):
        summarize(two_pair_folder, **kwargs)


@pytest.mark.parametrize(
    "split,expected_cell",
    [
        ("test", "0.4000 +/- 0.1414"),
        ("train", "0.3000 +/- 0.1414"),
    ],
)
def test_tabulate_results_pivots_formatted_cells(two_pair_folder, split, expected_cell):
    """tabulate_results pivots classifiers by datasets with mean +/- std."""
    df = tabulate_results(two_pair_folder, metric="mae", split=split)
    assert set(df.index) == {"A", "B"}
    assert set(df.columns) == {"d1", "d2"}
    assert df.loc["A", "d1"] == expected_cell


@pytest.mark.parametrize(
    "rows, metric, expected",
    [
        ([{"mae_test": 0.25}], "mae", "0.2500 +/- 0.0000"),
        ([{"mae_test": 0.3}], "nonexistent", "n/a"),
        ([{"mae_test": float("nan")}], "mae", "n/a"),
        ([{"mae_test": float("inf")}], "mae", "n/a"),
    ],
    ids=["single-resample", "absent", "all-nan", "non-finite"],
)
def test_tabulate_results_cell_rule(tmp_path, rows, metric, expected):
    """A single resample gets zero std; absent or non-finite metrics get n/a."""
    _make_pair_csv(tmp_path, "clf", "ds", rows)
    df = tabulate_results(tmp_path, metric=metric, split="test")
    assert df.loc["clf", "ds"] == expected


def test_tabulate_results_fills_missing_pairs_with_na(tmp_path):
    """tabulate_results fills a ragged classifier/dataset grid with n/a."""
    _make_pair_csv(tmp_path, "A", "d1", [{"mae_test": 0.2}])
    _make_pair_csv(tmp_path, "A", "d2", [{"mae_test": 0.4}])
    _make_pair_csv(tmp_path, "B", "d1", [{"mae_test": 0.6}])
    df = tabulate_results(tmp_path, metric="mae", split="test")
    assert df.loc["A", "d2"] == "0.4000 +/- 0.0000"
    assert df.loc["B", "d2"] == "n/a"


def test_tabulate_results_rejects_both_split(two_pair_folder):
    """tabulate_results rejects split=both with ValueError."""
    with pytest.raises(ValueError, match="split must be"):
        tabulate_results(two_pair_folder, split="both")


def test_aggregators_return_empty_for_an_empty_folder(tmp_path):
    """An existing root with no pairs yields an empty frame, not an error."""
    assert summarize(tmp_path).empty
    assert tabulate_results(tmp_path).empty


@pytest.mark.parametrize(
    "split,expected_metric_cols",
    [
        ("test", {"ccr_test_mean", "ccr_test_std", "mae_test_mean", "mae_test_std"}),
        (
            "train",
            {"ccr_train_mean", "ccr_train_std", "mae_train_mean", "mae_train_std"},
        ),
        (
            "both",
            {
                "ccr_train_mean",
                "ccr_train_std",
                "mae_train_mean",
                "mae_train_std",
                "ccr_test_mean",
                "ccr_test_std",
                "mae_test_mean",
                "mae_test_std",
            },
        ),
    ],
)
def test_save_summary_writes_csv_and_returns_path(
    two_pair_folder, split, expected_metric_cols
):
    """save_summary writes ``<split>_summary.csv`` with flat metric columns."""
    path = save_summary(two_pair_folder, split=split)
    assert isinstance(path, Path)
    assert path.name == f"{split}_summary.csv"
    flat = pd.read_csv(path)
    assert flat.shape[0] == 4
    # flat columns are the reset MultiIndex levels plus the metric columns
    assert (
        set(flat.columns)
        == {"classifier", "dataset", "n_completed"} | expected_metric_cols
    )


def test_save_summary_writes_under_the_expanded_root(monkeypatch, tmp_path):
    """A ~ root is expanded for the write too, not turned into a literal dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _make_pair_csv(tmp_path / "run", "A", "d1", [{"mae_test": 0.1}])
    out_path = save_summary("~/run")
    assert out_path == tmp_path / "run" / "test_summary.csv"
    assert out_path.is_file()
    assert not (tmp_path / "~").exists()


def test_save_summary_empty_folder_raises(tmp_path):
    """save_summary raises ValueError when there are no results."""
    with pytest.raises(ValueError, match="No results"):
        save_summary(tmp_path)


@pytest.mark.parametrize(
    "call",
    [
        lambda root: evaluate(root, "A", "d1"),
        summarize,
        tabulate_results,
        save_summary,
    ],
    ids=["evaluate", "summarize", "tabulate_results", "save_summary"],
)
def test_missing_results_root_raises(tmp_path, call):
    """Every entry point fails loud on an absent root, never with an empty result."""
    with pytest.raises(FileNotFoundError, match="No results directory"):
        call(tmp_path / "absent")


def test_summarize_round_trip_precision(tmp_path):
    """summarize reports a single high-precision metric bit-exact."""
    value = 0.12345678901234566
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": value}])
    df = summarize(tmp_path, split="test")
    assert df.loc[("clf", "ds"), ("mae_test", "mean")] == value


def test_evaluate_recovers_the_scale_from_the_confusion_matrix(tmp_path):
    """K comes from the stored K x K matrix, so a missing class keeps its gap."""
    _write_report(tmp_path, "A", "d1", {0: {"accuracy_off1_score_test": 2 / 3}})
    seed = _write_seed(
        tmp_path, "A", "d1", 0, "test", [0, 0, 2, 2, 0, 2], [0, 2, 0, 2, 0, 2]
    )
    (seed / "test_confusion_matrix.txt").write_text(
        "Seed 0\n=====\n[[3, 0, 0],\n [0, 0, 0],\n [0, 0, 3]]\n"
    )

    scored = evaluate(tmp_path, "A", "d1", metrics=["accuracy_off1_score"])

    # Ranks 0 and 2 are two apart on the recovered scale, not adjacent
    assert scored.loc[0, "accuracy_off1_score"] == pytest.approx(2 / 3)
