"""Tests for the _evaluation module (read-back/reporting functions)."""

import math
from pathlib import Path

import pandas as pd
import pytest

import skordinal.experiments as exp_pkg
from skordinal.experiments import Results, save_summary, summarize, tabulate_results
from skordinal.experiments._evaluation import _check_split, _iter_pairs


def _make_pair_csv(base: Path, classifier: str, dataset: str, rows: list) -> Path:
    """Write a minimal report.csv under base/classifier/dataset/."""
    pair_dir = base / classifier / dataset
    pair_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = pair_dir / "report.csv"
    df.to_csv(csv_path)
    return csv_path


@pytest.fixture
def two_pair_folder(tmp_path):
    """Two classifiers x two datasets with 2 partitions each under tmp_path."""
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
    return tmp_path


def test_summarize_importable_from_package():
    """``summarize`` is importable from ``skordinal.experiments``."""
    from skordinal.experiments import summarize as fn  # noqa: F401

    assert callable(fn)


def test_tabulate_results_importable_from_package():
    """``tabulate_results`` is importable from ``skordinal.experiments``."""
    from skordinal.experiments import tabulate_results as fn  # noqa: F401

    assert callable(fn)


def test_save_summary_importable_from_package():
    """``save_summary`` is importable from ``skordinal.experiments``."""
    from skordinal.experiments import save_summary as fn  # noqa: F401

    assert callable(fn)


def test_summarize_importable_from_evaluation_module():
    """summarize is importable from skordinal.experiments._evaluation."""
    from skordinal.experiments._evaluation import summarize as fn  # noqa: F401

    assert callable(fn)


def test_tabulate_results_importable_from_evaluation_module():
    """tabulate_results is importable from the _evaluation module."""
    from skordinal.experiments._evaluation import tabulate_results as fn  # noqa: F401

    assert callable(fn)


def test_save_summary_importable_from_evaluation_module():
    """save_summary is importable from skordinal.experiments._evaluation."""
    from skordinal.experiments._evaluation import save_summary as fn  # noqa: F401

    assert callable(fn)


def test_three_publics_in_dunder_all():
    """summarize, tabulate_results, and save_summary appear in __all__."""
    assert "summarize" in exp_pkg.__all__
    assert "tabulate_results" in exp_pkg.__all__
    assert "save_summary" in exp_pkg.__all__


def test_evaluate_not_in_package():
    """evaluate must not be importable from skordinal.experiments."""
    assert not hasattr(exp_pkg, "evaluate")


def test_results_has_no_summarize():
    """Results must not have a summarize attribute after the extraction."""
    assert not hasattr(Results, "summarize")


def test_results_has_no_tabulate():
    """Results must not have a tabulate attribute after the extraction."""
    assert not hasattr(Results, "tabulate")


def test_results_has_no_save_summary():
    """Results must not have a save_summary attribute after the extraction."""
    assert not hasattr(Results, "save_summary")


def test_summarize_shape_split_test(two_pair_folder):
    """summarize split=test returns a (4, 3) MultiIndex DataFrame."""
    df = summarize(two_pair_folder, split="test")
    assert isinstance(df.index, pd.MultiIndex)
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.shape == (4, 3)


def test_summarize_split_train_columns(two_pair_folder):
    """summarize split=train selects only _train-suffixed columns."""
    df = summarize(two_pair_folder, split="train")
    metric_cols = [c for c in df.columns if c[0] != "n_completed"]
    assert all(c[0].endswith("_train") for c in metric_cols)


def test_summarize_split_both_includes_test_and_train(two_pair_folder):
    """summarize split=both includes both _test and _train columns."""
    df = summarize(two_pair_folder, split="both")
    metric_names = [c[0] for c in df.columns if c[0] != "n_completed"]
    assert any(c.endswith("_test") for c in metric_names)
    assert any(c.endswith("_train") for c in metric_names)


def test_summarize_mean_std_arithmetic(tmp_path):
    """Mean and std are computed correctly for 2-resample pairs."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": 0.2}, {"mae_test": 0.4}])
    df = summarize(tmp_path, split="test")
    assert df.loc[("clf", "ds"), ("mae_test", "mean")] == pytest.approx(0.3)
    assert df.loc[("clf", "ds"), ("mae_test", "std")] == pytest.approx(0.1414, rel=1e-3)
    assert df.loc[("clf", "ds"), ("n_completed", "")] == 2


def test_summarize_labels_filter(two_pair_folder):
    """``summarize(labels=["A"])`` restricts rows to classifier A."""
    df = summarize(two_pair_folder, labels=["A"])
    assert all(clf == "A" for clf, _ in df.index)


def test_tabulate_results_pivot_structure(two_pair_folder):
    """tabulate_results returns a classifiers x datasets pivot DataFrame."""
    df = tabulate_results(two_pair_folder, metric="mae", split="test")
    assert isinstance(df, pd.DataFrame)
    assert set(df.index) == {"A", "B"}
    assert set(df.columns) == {"d1", "d2"}


def test_tabulate_results_missing_metric_returns_na(two_pair_folder):
    """tabulate_results returns all n/a when the metric column is absent."""
    df = tabulate_results(two_pair_folder, metric="nonexistent", split="test")
    assert (df == "n/a").all().all()


def test_summarize_empty_folder(tmp_path):
    """summarize returns an empty DataFrame when the folder has no results."""
    assert summarize(tmp_path).empty


def test_tabulate_results_empty_folder(tmp_path):
    """tabulate_results returns an empty DataFrame when no results exist."""
    assert tabulate_results(tmp_path).empty


def test_save_summary_writes_csv(two_pair_folder):
    """``save_summary`` writes ``test_summary.csv`` and returns its path."""
    path = save_summary(two_pair_folder, split="test")
    assert path.is_file()
    assert path.name == "test_summary.csv"
    df = pd.read_csv(path)
    assert df.shape[0] == 4


def test_save_summary_returns_path_object(two_pair_folder):
    """``save_summary`` return value is a ``Path`` to the written file."""
    path = save_summary(two_pair_folder, split="test")
    assert isinstance(path, Path)


def test_summarize_labels_string_raises(two_pair_folder):
    """Bare string for labels raises TypeError mentioning iterable."""
    with pytest.raises(TypeError, match="iterable"):
        summarize(two_pair_folder, labels="A")


def test_summarize_invalid_split_raises(two_pair_folder):
    """Invalid split value raises ValueError mentioning split must be."""
    with pytest.raises(ValueError, match="split must be"):
        summarize(two_pair_folder, split="bad")


def test_tabulate_results_split_both_raises(two_pair_folder):
    """split=both is rejected by tabulate_results with ValueError."""
    with pytest.raises(ValueError, match="split must be"):
        tabulate_results(two_pair_folder, split="both")


def test_save_summary_empty_folder_raises(tmp_path):
    """save_summary raises ValueError when the folder has no results."""
    with pytest.raises(ValueError, match="No results"):
        save_summary(tmp_path)


@pytest.mark.parametrize(
    "split,expected_suffix",
    [
        ("test", "_test"),
        ("train", "_train"),
    ],
)
def test_summarize_split_column_suffix(two_pair_folder, split, expected_suffix):
    """summarize selects only metric columns matching the suffix for split."""
    df = summarize(two_pair_folder, split=split)
    metric_cols = [c[0] for c in df.columns if c[0] != "n_completed"]
    assert all(c.endswith(expected_suffix) for c in metric_cols)


def test_summarize_split_both_is_union(two_pair_folder):
    """summarize split=both yields the union of test and train columns."""
    df_test = summarize(two_pair_folder, split="test")
    df_train = summarize(two_pair_folder, split="train")
    df_both = summarize(two_pair_folder, split="both")

    test_metric_cols = {c[0] for c in df_test.columns if c[0] != "n_completed"}
    train_metric_cols = {c[0] for c in df_train.columns if c[0] != "n_completed"}
    both_metric_cols = {c[0] for c in df_both.columns if c[0] != "n_completed"}

    assert both_metric_cols == test_metric_cols | train_metric_cols
    # test and train column sets are disjoint
    assert not test_metric_cols & train_metric_cols


def test_iter_pairs_skips_dir_without_report_csv(tmp_path):
    """_iter_pairs does not yield a dataset dir that lacks a report.csv."""
    # Valid pair
    _make_pair_csv(tmp_path, "clf", "ds_valid", [{"mae_test": 0.1}])
    # Dir without report.csv
    stray = tmp_path / "clf" / "ds_stray"
    stray.mkdir(parents=True)

    pairs = list(_iter_pairs(tmp_path))
    ds_names = {ds for _, ds, _ in pairs}
    assert "ds_valid" in ds_names
    assert "ds_stray" not in ds_names


def test_iter_pairs_skips_files_at_clf_level(tmp_path):
    """``_iter_pairs`` skips non-directory entries at the classifier level."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": 0.1}])
    # Place a stray file at the root level (not inside any clf dir)
    (tmp_path / "stray_file.txt").write_text("noise")

    pairs = list(_iter_pairs(tmp_path))
    # Only the one valid pair must appear
    assert len(pairs) == 1
    assert pairs[0][0] == "clf"


def test_iter_pairs_skips_files_at_dataset_level(tmp_path):
    """_iter_pairs skips non-directory entries inside a classifier dir."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": 0.1}])
    # Place a stray file inside the clf dir (at the dataset level)
    (tmp_path / "clf" / "stray.txt").write_text("noise")

    pairs = list(_iter_pairs(tmp_path))
    assert len(pairs) == 1
    assert pairs[0][1] == "ds"


def test_iter_pairs_nonexistent_folder_raises_file_not_found(tmp_path):
    """_iter_pairs raises FileNotFoundError when root folder does not exist."""
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        list(_iter_pairs(missing))


def test_tabulate_results_single_resample_std_zero(tmp_path):
    """A single-resample pair produces std == 0.0000 in the formatted cell."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": 0.25}])
    df = tabulate_results(tmp_path, metric="mae", split="test")
    cell = df.loc["clf", "ds"]
    assert "0.2500 +/- 0.0000" == cell


def test_tabulate_results_all_nan_column_returns_na(tmp_path):
    """An all-NaN metric column produces ``"n/a"`` in the pivot cell."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": float("nan")}])
    df = tabulate_results(tmp_path, metric="mae", split="test")
    assert df.loc["clf", "ds"] == "n/a"


def test_tabulate_results_infinite_mean_returns_na(tmp_path):
    """A metric column containing inf values produces n/a in the cell."""
    _make_pair_csv(tmp_path, "clf", "ds", [{"mae_test": float("inf")}])
    df = tabulate_results(tmp_path, metric="mae", split="test")
    assert df.loc["clf", "ds"] == "n/a"


@pytest.mark.parametrize(
    "rows,expected_std",
    [
        # n==0: column present but all-NaN; after dropna, n==0 → std is nan
        ([{"mae_test": float("nan")}], float("nan")),
        # n==1: single finite value → std falls back to 0.0
        ([{"mae_test": 0.3}], 0.0),
        # n>1: two finite values → ddof=1 std
        ([{"mae_test": 0.2}, {"mae_test": 0.4}], pytest.approx(0.1414, rel=1e-3)),
    ],
    ids=["n-zero", "n-one", "n-gt-one"],
)
def test_summarize_std_behaviour_by_n(tmp_path, rows, expected_std):
    """``summarize`` std is nan for n==0, 0.0 for n==1, and ddof=1 for n>1."""
    _make_pair_csv(tmp_path, "clf", "ds", rows)
    df = summarize(tmp_path, split="test")
    std_val = df.loc[("clf", "ds"), ("mae_test", "std")]
    if isinstance(expected_std, float) and math.isnan(expected_std):
        assert math.isnan(std_val)
    else:
        assert std_val == expected_std


def test_summarize_all_nan_metric_column_mean_is_nan(tmp_path):
    """An all-NaN metric column yields a NaN mean in the summary."""
    _make_pair_csv(
        tmp_path, "clf", "ds", [{"mae_test": float("nan")}, {"mae_test": float("nan")}]
    )
    df = summarize(tmp_path, split="test")
    mean_val = df.loc[("clf", "ds"), ("mae_test", "mean")]
    assert math.isnan(mean_val)


def test_check_split_valid_values_do_not_raise():
    """``_check_split`` does not raise for valid split values."""
    _check_split("test", allow_both=True)
    _check_split("train", allow_both=True)
    _check_split("both", allow_both=True)
    _check_split("test", allow_both=False)
    _check_split("train", allow_both=False)


def test_check_split_both_rejected_when_allow_both_false():
    """_check_split raises ValueError for both when allow_both=False."""
    with pytest.raises(ValueError, match="split must be"):
        _check_split("both", allow_both=False)


def test_check_split_unknown_value_raises():
    """_check_split raises ValueError for an unrecognised split value."""
    with pytest.raises(ValueError, match="split must be"):
        _check_split("unknown", allow_both=True)
