"""Read-back aggregation over saved experiment results."""

import math
from pathlib import Path

import pandas as pd


def _check_split(split, *, allow_both):
    """Raise ValueError when split is not a recognised value."""
    valid = {"test", "train", "both"} if allow_both else {"test", "train"}
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)!r}, got {split!r}.")


def _iter_pairs(results_path):
    """Yield ``(classifier, dataset, csv_path)`` for each results pair."""
    root = Path(results_path)
    for clf_dir in sorted(root.iterdir()):
        if not clf_dir.is_dir():
            continue
        for ds_dir in sorted(clf_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            csv_path = ds_dir / "report.csv"
            if csv_path.is_file():
                yield clf_dir.name, ds_dir.name, csv_path


def summarize(results_path, *, labels=None, split="test"):
    """Aggregate per-pair report CSVs into a multi-index summary DataFrame.

    Parameters
    ----------
    results_path : str or Path
        Root folder of the experiment results. The function descends into
        ``<results_path>/<classifier>/<dataset>/report.csv`` for each pair.

    labels : iterable of str or None, default=None
        When provided, only pairs whose classifier name is contained in
        ``labels`` are included.  Must be an iterable of strings, not a
        bare string.

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
        If ``labels`` is a bare string instead of an iterable of strings.

    Examples
    --------
    >>> from skordinal.experiments import summarize
    >>> df = summarize("/path/to/my-run", split="both")  # doctest: +SKIP
    """
    _check_split(split, allow_both=True)

    if isinstance(labels, str):
        raise TypeError(
            "labels must be an iterable of classifier name strings, not a bare string; "
            f"pass [{labels!r}] to filter by a single classifier."
        )

    label_set = set(labels) if labels is not None else None
    rows = []

    for clf, ds, csv_path in _iter_pairs(results_path):
        # Skip pairs not in the requested label filter
        if label_set is not None and clf not in label_set:
            continue

        df = pd.read_csv(csv_path, index_col=0)

        # Select columns for the requested split
        if split == "test":
            metric_cols = [c for c in df.columns if c.endswith("_test")]
        elif split == "train":
            metric_cols = [c for c in df.columns if c.endswith("_train")]
        else:
            metric_cols = [c for c in df.columns if c.endswith(("_test", "_train"))]

        # Compute mean and std for each metric column
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

    # Build MultiIndex DataFrame from accumulated rows
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
    rows = []

    for clf, ds, csv_path in _iter_pairs(results_path):
        df = pd.read_csv(csv_path, index_col=0)
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

    # Pivot into a classifiers x datasets table
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

    Examples
    --------
    >>> from skordinal.experiments import save_summary
    >>> path = save_summary("/path/to/my-run", split="test")  # doctest: +SKIP
    """
    df = summarize(results_path, split=split)
    if df.empty:
        raise ValueError("No results found to summarise.")
    flat = df.copy()
    flat.columns = [
        f"{outer}_{inner}" if inner else outer for outer, inner in flat.columns
    ]
    out_path = Path(results_path) / f"{split}_summary.csv"
    flat.to_csv(out_path)
    return out_path
