"""Base IO code and shared machinery for skordinal datasets."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections.abc import Iterator
from importlib import resources
from numbers import Integral
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import Bunch
from sklearn.utils._param_validation import Interval, validate_params

DATA_MODULE = "skordinal.datasets.data"
DESCR_MODULE = "skordinal.datasets.descr"


def _read_csv_any(path):
    """Read an ordinal-classification CSV, auto-detecting the header style."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    header_class_names = None

    def _is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def _is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _is_metadata_header(r0, r1):
        """Return True if r0 is ``n_samples, n_features, *class_names``."""
        if len(r0) < 2:
            return False
        if not (_is_int(r0[0]) and _is_int(r0[1])):
            return False
        return int(r0[1]) == len(r1) - 1

    if len(rows) < 2:
        # With only one row there is no header, so treat it as data
        r0 = rows[0]
        data_rows = rows
        feature_names = [f"x{i}" for i in range(len(r0) - 1)]
    else:
        r0, r1 = rows[0], rows[1]
        if _is_metadata_header(r0, r1):
            # Metadata header holds n_samples, n_features then the class names
            n_features = int(r0[1])
            feature_names = [f"x{i}" for i in range(n_features)]
            header_class_names = np.array(r0[2:])
            data_rows = rows[1:]
        elif any(not _is_float(tok) for tok in r0):
            # Named header, with at least one non-numeric token
            feature_names = r0[:-1]
            data_rows = rows[1:]
        else:
            # No header row, so generate feature names
            feature_names = [f"x{i}" for i in range(len(r0) - 1)]
            data_rows = rows

    n_features = len(feature_names)
    # Parse every data row in one vectorised pass, the first n_features
    # columns are the features and the last column is the target
    table = np.asarray(data_rows, dtype=np.float64)
    X = np.ascontiguousarray(table[:, :n_features])
    y = table[:, -1].astype(np.int64)
    return X, y, feature_names, header_class_names


def _resolve_csv_path(name, data_home=None):
    """Resolve a dataset name or path to its CSV file path."""
    name_str = str(name)
    if data_home is not None:
        fname = name_str if name_str.endswith(".csv") else f"{name_str}.csv"
        return Path(data_home) / fname, None
    if Path(name_str).is_file():
        return Path(name_str), None
    stem = name_str.removesuffix(".csv")
    bundled_dir = Path(str(resources.files(DATA_MODULE)))
    return bundled_dir / f"{stem}.csv", DATA_MODULE


def _load_descr(csv_path):
    """Return the ``.rst`` description for a CSV path, or None."""
    sidecar = csv_path.with_suffix(".rst")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8")
    bundled = Path(str(resources.files(DESCR_MODULE))) / f"{csv_path.stem}.rst"
    if bundled.exists():
        return bundled.read_text(encoding="utf-8")
    return None


def _resolve_target_names(header_class_names, target):
    """Return header class names, or sorted unique targets as strings."""
    if header_class_names is not None:
        return header_class_names
    return np.unique(target).astype(str)


def _convert_data_dataframe(caller_name, data, target, feature_names, target_columns):
    """Build a pandas frame from ``data`` and ``target`` for ``as_frame``."""
    try:
        import pandas as pd
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised in environments without pandas
        raise ImportError(f"{caller_name} with as_frame=True requires pandas.") from exc
    data_df = pd.DataFrame(data, columns=feature_names, copy=False)
    target_df = pd.DataFrame(target, columns=target_columns)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_columns]
    if y.shape[1] == 1:  # pragma: no branch
        y = y.iloc[:, 0]
    return combined_df, X, y


def _load_keyed_masks(csv_dir):
    """Return the keyed ``train_masks.json`` dict near a CSV dir, or None."""
    for candidate in (
        csv_dir / "train_masks.json",
        csv_dir.parent / "train_masks.json",
    ):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    return None


def _resolve_train_masks(csv_path, X, y, ids, random_state):
    """Return one boolean train-mask per requested resample id."""

    # Coerce a raw mask to a boolean array and validate its length
    def _as_mask(raw, label):
        mask = np.asarray(raw, dtype=bool)
        if len(mask) != len(y):
            raise ValueError(
                f"Mask for {label} has length {len(mask)}; "
                f"expected {len(y)} (number of samples)."
            )
        return mask

    # First choice is a per-dataset masks file <stem>.masks.json with one
    # boolean train-mask per resample
    per_dataset_path = csv_path.parent / f"{csv_path.stem}.masks.json"
    if per_dataset_path.exists():
        with per_dataset_path.open("r", encoding="utf-8") as fh:
            masks_list = json.load(fh)
        masks = []
        for rid in ids:
            if not (0 <= rid < len(masks_list)):
                raise IndexError(
                    f"No mask for resample {rid}: {per_dataset_path.name} "
                    f"contains {len(masks_list)} entries."
                )
            masks.append(_as_mask(masks_list[rid], f"resample {rid}"))
        return masks

    # Otherwise, a shared train_masks.json keyed by "<stem>_seed_<rid>"
    keyed = _load_keyed_masks(csv_path.parent)
    if keyed is not None:
        masks = []
        for rid in ids:
            key = f"{csv_path.stem}_seed_{rid}"
            if key not in keyed:
                raise KeyError(f"No mask found for key {key!r} in train_masks.json.")
            masks.append(_as_mask(keyed[key], f"key {key!r}"))
        return masks

    # Otherwise, generate masks with StratifiedKFold (one fold per resample)
    if len(ids) < 2:
        raise ValueError(
            f"When generating CV masks, resamples must be >= 2; got {len(ids)}."
        )
    skf = StratifiedKFold(n_splits=len(ids), shuffle=True, random_state=random_state)
    masks = []
    for train_indices, _ in skf.split(X, y):
        mask = np.zeros(len(y), dtype=bool)
        mask[train_indices] = True
        masks.append(mask)
    return masks


@validate_params(
    {"data_home": [str, os.PathLike, None]},
    prefer_skip_nested_validation=True,
)
def get_data_home(data_home=None) -> str:
    """Return the path of the skordinal data directory.

    Mirrors ``sklearn.datasets.get_data_home``. Resolution order:
    ``data_home`` argument → ``$SKORDINAL_DATA`` environment variable →
    ``~/skordinal_data``. The directory is created if it does not exist.

    Parameters
    ----------
    data_home : str, os.PathLike, or None, default=None
        Path to the skordinal data directory; ``None`` triggers the
        resolution order described above.

    Returns
    -------
    data_home : str
        Path to the data directory.

    Examples
    --------
    >>> import os
    >>> from skordinal.datasets import get_data_home
    >>> os.path.isdir(get_data_home())
    True
    """
    if data_home is None:
        data_home = os.environ.get("SKORDINAL_DATA", Path.home() / "skordinal_data")
    data_home = Path(data_home).expanduser()
    data_home.mkdir(parents=True, exist_ok=True)
    return str(data_home)


@validate_params(
    {"data_home": [str, os.PathLike, None]},
    prefer_skip_nested_validation=True,
)
def clear_data_home(data_home=None) -> None:
    """Delete all content from the skordinal data cache.

    Parameters
    ----------
    data_home : str, os.PathLike, or None, default=None
        Path to the skordinal data directory. When ``None``, uses the
        default resolved by ``get_data_home``.

    Examples
    --------
    >>> from skordinal.datasets import clear_data_home
    >>> clear_data_home()  # doctest: +SKIP
    """
    shutil.rmtree(get_data_home(data_home))


@validate_params(
    {
        "name": [str, os.PathLike],
        "data_home": [str, os.PathLike, None],
        "return_X_y": ["boolean"],
        "as_frame": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_dataset(name, *, data_home=None, return_X_y=False, as_frame=False):
    """Load any ordinal dataset by name or path, auto-detecting CSV format.

    Resolution order:

    1. If ``data_home`` is given, look for ``<data_home>/<name>.csv`` (or
       ``<data_home>/<name>`` when ``name`` already ends with ``.csv``).
    2. Otherwise, if ``name`` is an existing file path, open it directly.
    3. Otherwise, resolve against the bundled data directory.

    Three CSV header styles are accepted automatically:

    - metadata header (bundled style): the first row contains
      ``n_samples, n_features, target_name_0, ...``.
    - named header: the first row holds column names
      with at least one non-numeric token.
    - no header: every row is a data row; feature names are generated
      as ``x0, x1, ...``.

    Parameters
    ----------
    name : str or path-like
        Dataset stem (e.g. ``"era"``), filename (``"era.csv"``), or a
        concrete file path when ``data_home`` is ``None``.

    data_home : str, path-like, or None, default=None
        Directory to search when ``name`` is a stem or filename. When
        ``None`` the bundled data directory is used.

    return_X_y : bool, default=False
        If ``True``, returns ``(data, target)`` instead of a Bunch.

    as_frame : bool, default=False
        If ``True``, ``data`` is a ``pandas.DataFrame`` and ``target``
        is a ``pandas.Series``.

    Returns
    -------
    bunch : ``sklearn.utils.Bunch``
        Object with the following attributes.

        data : ndarray of shape (n_samples, n_features)
            Feature matrix (float64). A DataFrame when ``as_frame`` is True.
        target : ndarray of shape (n_samples,)
            Integer target labels (int64). A Series when ``as_frame`` is True.
        frame : DataFrame or None
            Combined frame when ``as_frame`` is True; otherwise ``None``.
        feature_names : list of str
            One name per feature column.
        target_names : ndarray of str
            Class names from the metadata header when present; otherwise
            the sorted unique target values as strings.
        n_classes : int
            Number of distinct classes.
        DESCR : str
            Human-readable description sourced from a ``.rst`` sidecar
            file, or a generated one-line summary when none is found.
        filename : str
            Basename of the CSV file.
        data_module : str or None
            Python module path used by ``importlib.resources`` to
            locate the file when resolved from the bundled directory;
            ``None`` for external files.

    (data, target) : tuple if ``return_X_y`` is True

    Raises
    ------
    FileNotFoundError
        When the resolved path does not exist.

    Examples
    --------
    >>> from skordinal.datasets import load_dataset  # doctest: +SKIP
    >>> bunch = load_dataset("era")                   # doctest: +SKIP
    >>> bunch.data.shape                              # doctest: +SKIP
    (1000, 4)

    Load from a custom directory:

    >>> load_dataset("era", data_home="/my/data").data.shape  # doctest: +SKIP
    (1000, 4)
    """
    path, data_module_value = _resolve_csv_path(name, data_home)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data, target, feature_names, header_class_names = _read_csv_any(path)
    target_names = _resolve_target_names(header_class_names, target)
    n_classes = len(target_names)

    descr = _load_descr(path)
    if descr is None:
        descr = (
            f"Dataset '{path.stem}': {data.shape[0]} samples, "
            f"{data.shape[1]} features, {n_classes} classes."
        )

    filename = path.name
    frame = None
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_dataset", data, target, feature_names, ["target"]
        )
    if return_X_y:
        return data, target
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        feature_names=feature_names,
        target_names=target_names,
        n_classes=n_classes,
        DESCR=descr,
        filename=filename,
        data_module=data_module_value,
    )


@validate_params(
    {
        "name": [str, os.PathLike],
        "data_home": [str, os.PathLike, None],
        "resamples": [Interval(Integral, 1, None, closed="left"), list],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def load_partitions(
    name,
    *,
    data_home=None,
    resamples=30,
    random_state=0,
) -> Iterator[Bunch]:
    """Yield one train/test partition per resample of an ordinal dataset.

    The dataset is a single CSV resolved like ``load_dataset``
    (``<data_home>/<name>.csv``, a direct path, or the bundled data
    directory). Train/test splits come from boolean train-masks, resolved
    in this order:

    1. A per-dataset masks file ``<csv_dir>/<stem>.masks.json``: a JSON
       list whose k-th element is the boolean train-mask of resample k.
    2. A shared keyed masks file ``train_masks.json`` in ``<csv_dir>`` or
       its parent, keyed as ``f"{stem}_seed_{k}"``.
    3. Otherwise generated with
       ``sklearn.model_selection.StratifiedKFold``: one fold per requested
       resample, ``shuffle=True``, seeded by ``random_state``.

    Parameters
    ----------
    name : str or path-like
        Dataset stem (e.g. ``"era"``), filename, or concrete path.

    data_home : str, path-like, or None, default=None
        Directory to search when ``name`` is a stem or filename. When
        ``None``, the bundled data directory is used.

    resamples : int or list of int, default=30
        When an ``int``, resample ids are ``range(resamples)``. When a
        list, those ids are used directly. Must be >= 2 when masks are
        generated by cross-validation.

    random_state : int, RandomState instance, or None, default=0
        Seed passed to ``sklearn.model_selection.StratifiedKFold``
        when CV masks are generated. Ignored otherwise.

    Yields
    ------
    bunch : ``sklearn.utils.Bunch``
        Dictionary-like object with the following attributes.

        data_train : ndarray of shape (n_train, n_features)
            Training features (float64).
        target_train : ndarray of shape (n_train,)
            Training targets (int64).
        data_test : ndarray of shape (n_test, n_features)
            Test features (float64).
        target_test : ndarray of shape (n_test,)
            Test targets (int64).
        feature_names : list of str
            Feature column names.
        target_names : ndarray of str
            Class names from the metadata header when present; otherwise
            the sorted unique target values as strings.
        dataset_name : str
            Echo of the requested dataset name.
        resample_id : int
            Identifier of the current resample, taken from
            ``range(resamples)`` or from the supplied list of ids.
        n_classes : int
            Number of ordinal classes.
        DESCR : str
            One-line description of this resample.

    Raises
    ------
    FileNotFoundError
        When the CSV cannot be located.
    ValueError
        When ``resamples < 2`` and CV masks must be generated, or when a
        mask length does not match the number of samples.
    KeyError
        When a keyed masks file exists but does not contain the expected key.
    IndexError
        When a per-dataset masks file (``<stem>.masks.json``) contains
        fewer entries than the largest requested resample id.

    Examples
    --------
    >>> from skordinal.datasets import load_partitions  # doctest: +SKIP
    >>> for bunch in load_partitions("era", resamples=3):  # doctest: +SKIP
    ...     print(bunch.resample_id, bunch.data_train.shape[0])
    """
    csv_path, _ = _resolve_csv_path(name, data_home)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    X, y, feature_names, header_class_names = _read_csv_any(csv_path)
    target_names = _resolve_target_names(header_class_names, y)
    n_classes = len(target_names)

    ids = list(range(resamples)) if isinstance(resamples, Integral) else list(resamples)
    train_masks = _resolve_train_masks(csv_path, X, y, ids, random_state)

    def _iter():
        for resample_id, train_mask in zip(ids, train_masks):
            n_train = int(train_mask.sum())
            n_test = int((~train_mask).sum())
            yield Bunch(
                data_train=X[train_mask],
                target_train=y[train_mask],
                data_test=X[~train_mask],
                target_test=y[~train_mask],
                feature_names=feature_names,
                target_names=target_names,
                dataset_name=str(name),
                resample_id=int(resample_id),
                n_classes=n_classes,
                DESCR=(
                    f"{name} resample {resample_id}: "
                    f"{n_train}/{n_test} samples, {n_classes} classes."
                ),
            )

    return _iter()
