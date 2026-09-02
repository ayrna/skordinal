"""Read-back aggregation over saved experiment results."""

import math

import numpy as np
import pandas as pd

from ._base import _check_metric_names, _check_split, _compute_metric
from ._io import _atomic_write, _read_confusion_matrix_size
from ._results import Results


def _existing_results(results_path):
    """Return a Results over results_path, raising when the root is absent."""
    results = Results(results_path)
    if not results.path.is_dir():
        raise FileNotFoundError(f"No results directory at {results.path}.")
    return results


def _iter_reports(results):
    """Yield ``(classifier, dataset, report)`` for each pair that stores metrics."""
    for clf, ds in results.iter_experiments():
        try:
            report = results.report(clf, ds)
        except FileNotFoundError:
            # A pair with predictions but no report.csv stores no metrics
            continue
        yield clf, ds, report


def evaluate(
    results_path,
    classifier_name,
    dataset_name,
    *,
    metrics=None,
    split="test",
):
    """Recompute metrics from the saved per-seed predictions of one pair.

    Useful for adding a metric to a finished benchmark without re-fitting any
    estimator. Where a ``report.csv`` exists, only the resamples committed to
    it are read. Where the pair has none, every seed directory holding the
    split's predictions file is read, so a results tree written by another
    tool sharing this layout is readable too.

    Parameters
    ----------
    results_path : str or Path
        Root folder of the experiment results.

    classifier_name : str
        Name of the classifier configuration.

    dataset_name : str
        Name of the dataset.

    metrics : iterable of str or None, default=None
        Metric names, from the registry ``Experiment``'s ``eval_metrics``
        uses; scorer names like ``"neg_mean_absolute_error"`` are rejected.
        ``None`` computes ``"mean_absolute_error"`` and ``"accuracy_score"``.

    split : {"test", "train"}, default="test"
        Which per-seed predictions file to read.

    Returns
    -------
    pd.DataFrame
        One row per resample read, indexed by ``resample_id`` and sorted
        like ``report.csv``'s index, with one column per requested metric.
        Empty, with those columns, when no predictions file is readable.

    Raises
    ------
    ValueError
        If ``split`` is not ``"test"`` or ``"train"``, if ``metrics`` is
        empty or holds an unregistered name, if ``classifier_name`` or
        ``dataset_name`` is not a usable path component, or if a predictions
        file lacks the ``Target`` or ``Prediction`` column.

    TypeError
        If ``classifier_name`` or ``dataset_name`` is not a string, or if
        ``metrics`` is a bare string instead of an iterable of strings or
        contains a non-string element.

    FileNotFoundError
        If ``results_path`` does not exist, or the pair has no
        ``predictions_by_seed`` directory.

    Warns
    -----
    RuntimeWarning
        If a resample committed to ``report.csv`` with metrics for ``split``
        has no predictions file for it.

    Notes
    -----
    The stored ``Target`` and ``Prediction`` columns are zero-based ranks into
    ``best_model.classes_``, so metrics are recomputed in rank space, where
    adjacent categories are always distance 1. ``report.csv``'s own values are
    computed on the raw labels, so the two agree only while the label set is
    contiguous. The scale spans the pair's saved confusion matrix, so a tree
    written without one is scored on the ranks its predictions contain.

    Examples
    --------
    >>> from skordinal.experiments import evaluate
    >>> df = evaluate("/path/to/my-run", "LR", "era")  # doctest: +SKIP
    """
    _check_split(split, allow_both=False)

    if metrics is None:
        metrics = ("mean_absolute_error", "accuracy_score")
    # Stripped names key the columns, like Experiment's report.csv
    metric_names = tuple(_check_metric_names(metrics, param="metrics"))

    results = _existing_results(results_path)
    rows = {}
    for resample_id, path in results._readable_seed_files(
        classifier_name, dataset_name, split
    ):
        df = pd.read_csv(path)
        missing = {"Target", "Prediction"} - set(df.columns)
        if missing:
            raise ValueError(
                f"Malformed predictions file at {path}: missing the "
                f"{sorted(missing)} column(s)."
            )
        y_true = df["Target"].to_numpy()
        y_pred = df["Prediction"].to_numpy()
        # The saved confusion matrix is the only artefact that still carries K
        size = _read_confusion_matrix_size(
            path.parent / f"{split}_confusion_matrix.txt"
        )
        labels = np.arange(size) if size else None
        rows[resample_id] = {
            name: _compute_metric(name, y_true, y_pred, labels=labels)
            for name in metric_names
        }

    return pd.DataFrame.from_dict(
        rows, orient="index", columns=list(metric_names)
    ).rename_axis("resample_id")


def summarize(results_path, *, classifiers=None, split="test"):
    """Aggregate per-pair report CSVs into a multi-index summary DataFrame.

    Parameters
    ----------
    results_path : str or Path
        Root folder of the experiment results. The function descends into
        ``<results_path>/<classifier>/<dataset>/report.csv`` for each pair.

    classifiers : iterable of str or None, default=None
        When provided, only pairs whose classifier name is contained in
        ``classifiers`` are included.  Must be an iterable of strings, not
        a bare string.

    split : {"test", "train", "both"}, default="test"
        Which metric columns to include.

        - ``"test"``: columns ending with ``_test``.
        - ``"train"``: columns ending with ``_train``.
        - ``"both"``: all columns ending with ``_test`` or ``_train``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``(classifier, dataset)`` MultiIndex and
        MultiIndex columns at two levels: outer is the column name (e.g.
        ``"mae_test"``), inner is ``"mean"`` or ``"std"``.
        The ``("n_completed", "")`` column counts partitions per pair.
        Returns an empty ``DataFrame`` when no pairs are found.

    Raises
    ------
    ValueError
        If ``split`` is not ``"test"``, ``"train"``, or ``"both"``.

    TypeError
        If ``classifiers`` is a bare string instead of an iterable of
        strings.

    FileNotFoundError
        If ``results_path`` does not exist.

    Examples
    --------
    >>> from skordinal.experiments import summarize
    >>> df = summarize("/path/to/my-run", split="both")  # doctest: +SKIP
    """
    _check_split(split, allow_both=True)

    if isinstance(classifiers, str):
        raise TypeError(
            "classifiers must be an iterable of classifier name strings, not a bare "
            f"string; pass [{classifiers!r}] to filter by a single classifier."
        )

    classifier_set = set(classifiers) if classifiers is not None else None
    results = _existing_results(results_path)
    rows = []

    for clf, ds, df in _iter_reports(results):
        if classifier_set is not None and clf not in classifier_set:
            continue

        if split == "test":
            metric_cols = [c for c in df.columns if c.endswith("_test")]
        elif split == "train":
            metric_cols = [c for c in df.columns if c.endswith("_train")]
        else:
            metric_cols = [c for c in df.columns if c.endswith(("_test", "_train"))]

        row = {"classifier": clf, "dataset": ds}
        for col in metric_cols:
            series = df[col].dropna()
            n = len(series)
            row[(col, "mean")] = float(series.mean()) if n > 0 else float("nan")
            row[(col, "std")] = (
                float(series.std(ddof=1))
                if n > 1
                else (0.0 if n == 1 else float("nan"))
            )
        row[("n_completed", "")] = len(df)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows).set_index(["classifier", "dataset"])
    summary.columns = pd.MultiIndex.from_tuples(list(summary.columns))
    return summary


def tabulate_results(results_path, *, metric="mean_absolute_error", split="test"):
    """Pivot experiment results into a classifiers-by-datasets table.

    Parameters
    ----------
    results_path : str or Path
        Root folder of the experiment results.

    metric : str, default="mean_absolute_error"
        Base metric name.  The column ``{metric}_{split}`` is looked up in
        each per-pair CSV.

    split : {"test", "train"}, default="test"
        Which evaluation split to read.

    Returns
    -------
    pd.DataFrame
        Pivot DataFrame with classifiers as rows and datasets as columns.
        Each cell is a ``"mean +/- std"`` string formatted to 4 decimal
        places, or ``"n/a"`` when the column is absent or all-NaN.
        Returns an empty ``DataFrame`` when no pairs are found.

    Raises
    ------
    ValueError
        If ``split`` is not ``"test"`` or ``"train"``.

    FileNotFoundError
        If ``results_path`` does not exist.

    Examples
    --------
    >>> from skordinal.experiments import tabulate_results
    >>> table = tabulate_results(  # doctest: +SKIP
    ...     "/path/to/my-run",
    ...     metric="accuracy_score",
    ...     split="test",
    ... )
    """
    _check_split(split, allow_both=False)

    col = f"{metric}_{split}"
    results = _existing_results(results_path)
    rows = []

    for clf, ds, df in _iter_reports(results):
        # Format as "mean +/- std", or "n/a" when metric is absent or all-NaN
        if col not in df.columns or df[col].isna().all():
            cell = "n/a"
        else:
            series = df[col].dropna()
            mean = float(series.mean())
            std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            if not math.isfinite(mean):
                cell = "n/a"
            else:
                std = std if math.isfinite(std) else 0.0
                cell = f"{mean:.4f} +/- {std:.4f}"
        rows.append({"classifier": clf, "dataset": ds, "value": cell})

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .pivot(index="classifier", columns="dataset", values="value")
        .fillna("n/a")
        .rename_axis(index="classifier", columns="dataset")
    )


def save_summary(results_path, *, split="test"):
    """Write a flattened summary CSV for one split under the results folder.

    Parameters
    ----------
    results_path : str or Path
        Root folder of the experiment results.  The CSV is written as
        ``{split}_summary.csv`` directly under this directory.

    split : {"test", "train", "both"}, default="test"
        Which metric columns to include.  Forwarded to ``summarize``.

    Returns
    -------
    Path
        Path of the CSV file that was written.

    Raises
    ------
    ValueError
        If ``split`` is not a recognised value (via ``summarize`` →
        ``_check_split``) or if there are no results in ``results_path``.

    FileNotFoundError
        If ``results_path`` does not exist.

    Examples
    --------
    >>> from skordinal.experiments import save_summary
    >>> path = save_summary("/path/to/my-run", split="test")  # doctest: +SKIP
    """
    root = _existing_results(results_path).path
    df = summarize(root, split=split)
    if df.empty:
        raise ValueError("No results found to summarise.")
    flat = df.copy()
    flat.columns = [
        f"{outer}_{inner}" if inner else outer for outer, inner in flat.columns
    ]
    out_path = root / f"{split}_summary.csv"
    _atomic_write(out_path, flat.to_csv())
    return out_path
