"""Tests for the _evaluation read-back/reporting functions."""

import math
from pathlib import Path

import pandas as pd
import pytest

from skordinal.experiments import save_summary, summarize, tabulate_results


def _make_pair_csv(base, classifier, dataset, rows):
    """Write a minimal report.csv under base/classifier/dataset/."""
    pair_dir = base / classifier / dataset
    pair_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = pair_dir / "report.csv"
    df.to_csv(csv_path)
    return csv_path


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
    "labels,expected_clfs",
    [
        (["A"], {"A"}),
        (["nonexistent"], set()),
    ],
    ids=["subset", "absent"],
)
def test_summarize_filters_by_labels(two_pair_folder, labels, expected_clfs):
    """summarize with labels keeps only matching classifiers or empty."""
    df = summarize(two_pair_folder, split="test", labels=labels)
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


def test_summarize_empty_folder_returns_empty(tmp_path):
    """summarize returns an empty frame when no pairs are present."""
    assert summarize(tmp_path).empty


def test_summarize_rejects_string_labels(two_pair_folder):
    """summarize raises TypeError when labels is a bare string."""
    with pytest.raises(TypeError, match="iterable"):
        summarize(two_pair_folder, labels="A")


def test_summarize_rejects_unknown_split(two_pair_folder):
    """summarize raises ValueError on an unrecognised split."""
    with pytest.raises(ValueError, match="split must be"):
        summarize(two_pair_folder, split="bad")


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
    "rows,metric",
    [
        ([{"mae_test": 0.3}], "nonexistent"),
        ([{"mae_test": float("nan")}], "mae"),
        ([{"mae_test": float("inf")}], "mae"),
    ],
    ids=["absent", "all-nan", "non-finite"],
)
def test_tabulate_results_renders_na(tmp_path, rows, metric):
    """tabulate_results shows n/a for absent or non-finite metrics."""
    _make_pair_csv(tmp_path, "clf", "ds", rows)
    df = tabulate_results(tmp_path, metric=metric, split="test")
    assert df.loc["clf", "ds"] == "n/a"


def test_tabulate_results_single_resample_zero_std(tmp_path):
    """tabulate_results reports zero std for a single resample."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": 0.25}])
    df = tabulate_results(tmp_path, metric="mae", split="test")
    assert df.loc["clf", "ds"] == "0.2500 +/- 0.0000"


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


def test_tabulate_results_empty_folder_returns_empty(tmp_path):
    """tabulate_results returns an empty frame when no pairs are present."""
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


def test_save_summary_empty_folder_raises(tmp_path):
    """save_summary raises ValueError when there are no results."""
    with pytest.raises(ValueError, match="No results"):
        save_summary(tmp_path)
