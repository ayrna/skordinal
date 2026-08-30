"""Base IO code and shared machinery for skordinal datasets."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections.abc import Iterator
from importlib import resources
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import Bunch, check_random_state
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.multiclass import check_classification_targets

DATA_MODULE = "skordinal.datasets.data"
DESCR_MODULE = "skordinal.datasets.descr"


def _cast_target_column(y_raw, path):
    """Return the target column as int32, or raise ValueError."""
    try:
        # The sklearn check casts internally, warning on the values it rejects
        with np.errstate(invalid="ignore"):
            check_classification_targets(y_raw)
    except ValueError as exc:
        raise ValueError(f"Target column of {path} is invalid: {exc}") from exc
    info = np.iinfo(np.int32)
    if not ((y_raw >= info.min) & (y_raw <= info.max)).all():
        raise ValueError(
            f"Target column of {path} is invalid: label outside the int32 range."
        )
    return y_raw.astype(np.int32)


def _read_csv_any(path):
    """Read an ordinal-classification CSV, auto-detecting the header style."""
    # utf-8-sig strips a leading BOM instead of folding it into the first
    # token, which would otherwise break header and integer detection
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
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
        if int(r0[1]) != len(r1) - 1:
            return False
        if len(r0) != len(r1) or not all(_is_float(tok) for tok in r0):
            # Row 0 cannot be a data row, so it is the header
            return True
        # r0 could be a data row, so demand corroborating counts
        if int(r0[0]) != len(rows) - 1:
            return False
        n_declared = len(r0) - 2
        observed = {float(row[-1]) for row in rows[1:] if row and _is_float(row[-1])}
        return n_declared == 0 or len(observed) <= n_declared

    n_samples_declared = None
    if len(rows) < 2:
        # With only one row there is no header, so treat it as data
        r0 = rows[0]
        if any(not _is_float(tok) for tok in r0):
            raise ValueError(f"CSV file {path} has a header but no data rows.")
        data_rows = rows
        feature_names = [f"x{i}" for i in range(len(r0) - 1)]
    else:
        r0, r1 = rows[0], rows[1]
        if _is_metadata_header(r0, r1):
            # Metadata header holds n_samples, n_features then the class names
            n_samples_declared = int(r0[0])
            n_features = int(r0[1])
            feature_names = [f"x{i}" for i in range(n_features)]
            # Only trust class-name tokens when at least one is present;
            # otherwise fall back to the unique targets downstream
            header_class_names = np.array(r0[2:]) if len(r0) > 2 else None
            data_rows = rows[1:]
        elif any(not _is_float(tok) for tok in r0):
            # Named header, with at least one non-numeric token
            feature_names = r0[:-1]
            data_rows = rows[1:]
        else:
            # No header row, so generate feature names
            feature_names = [f"x{i}" for i in range(len(r0) - 1)]
            data_rows = rows

    if n_samples_declared is not None and len(data_rows) != n_samples_declared:
        raise ValueError(
            f"Metadata header of {path} declares {n_samples_declared} "
            f"sample(s), but the file has {len(data_rows)} data row(s)."
        )

    n_features = len(feature_names)
    row_lengths = {len(row) for row in data_rows}
    if len(row_lengths) > 1:
        raise ValueError(
            f"Rows of {path} have inconsistent lengths {sorted(row_lengths)}."
        )
    row_width = row_lengths.pop()
    if row_width != n_features + 1:
        raise ValueError(
            f"Expected {n_features} feature column(s) plus 1 target column "
            f"({n_features + 1} total) in {path}, but found {row_width}."
        )
    # Parse every data row in one vectorised pass, the first n_features
    # columns are the features and the last column is the target
    table = np.asarray(data_rows, dtype=np.float64)
    X = np.ascontiguousarray(table[:, :n_features])
    y = _cast_target_column(table[:, -1], path)
    return X, y, feature_names, header_class_names


def _bundled_csv_path(stem):
    """Return the filesystem path of a bundled dataset CSV."""
    return Path(str(resources.files(DATA_MODULE))) / f"{stem}.csv"


def _resolve_csv_path(name, data_home=None):
    """Resolve a dataset name or path to its CSV file path."""
    name_str = str(name)
    if data_home is not None:
        fname = name_str if name_str.endswith(".csv") else f"{name_str}.csv"
        return Path(data_home) / fname, None
    if Path(name_str).is_file():
        return Path(name_str), None
    return _bundled_csv_path(name_str.removesuffix(".csv")), DATA_MODULE


def _load_descr(csv_path, data_module):
    """Return the ``.rst`` description for a CSV path, or None."""
    sidecar = csv_path.with_suffix(".rst")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8")
    if data_module is None:
        # A path not resolved from the bundled directory must not inherit
        # a bundled dataset's description just because the stem matches
        return None
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
    collision = set(feature_names) & set(target_columns)
    if collision:
        raise ValueError(
            f"{caller_name}: feature column {collision.pop()!r} collides with the "
            "target column name."
        )
    duplicates = {name for name in feature_names if feature_names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"{caller_name}: feature column {duplicates.pop()!r} appears more "
            "than once."
        )
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


def _validate_resample_ids(ids):
    """Raise unless ``ids`` holds unique, non-negative integers."""
    if not ids:
        raise ValueError("'resamples' must contain at least one id; got an empty list.")
    for rid in ids:
        if isinstance(rid, bool) or not isinstance(rid, Integral):
            raise TypeError(f"'resamples' ids must be integers; got {rid!r}.")
        if rid < 0:
            raise ValueError(f"'resamples' ids must be non-negative; got {rid}.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"'resamples' must not contain duplicate ids; got {ids}.")


def _generate_holdout_mask(X, y, resample_id, base_seed, test_size):
    """Return a stratified holdout train-mask for one resample id."""
    seed = int(np.random.SeedSequence([base_seed, resample_id]).generate_state(1)[0])
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_indices, _ = next(splitter.split(X, y))
    mask = np.zeros(len(y), dtype=bool)
    mask[train_indices] = True
    return mask


def _resolve_train_masks(csv_path, X, y, ids, test_size, random_state):
    """Return one boolean train-mask per requested resample id."""

    # Coerce a raw mask to a boolean array and validate it
    def _as_mask(raw, label):
        mask = np.asarray(raw)
        # Coercing with dtype=bool would turn both "false" and 0.4 into True
        if mask.dtype.kind not in "biu":
            raise ValueError(
                f"Mask for {label} must hold boolean or integer values; "
                f"got dtype {mask.dtype}."
            )
        mask = mask.astype(bool)
        if mask.shape != (len(y),):
            raise ValueError(
                f"Mask for {label} has shape {mask.shape}; "
                f"expected ({len(y)},) (one entry per sample)."
            )
        if mask.all() or not mask.any():
            raise ValueError(
                f"Mask for {label} is degenerate (all True or all False); "
                "both a train and a test split are required."
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

    # Otherwise, draw one stratified holdout per id, seeded from
    # (random_state, id) so an id's split never depends on its neighbours
    if isinstance(random_state, Integral):
        base_seed = int(random_state)
    else:
        base_seed = int(check_random_state(random_state).randint(0, 2**31 - 1))
    return [_generate_holdout_mask(X, y, rid, base_seed, test_size) for rid in ids]


@validate_params(
    {"data_home": [str, os.PathLike, None]},
    prefer_skip_nested_validation=True,
)
def get_data_home(data_home=None) -> str:
    """Return the path of the skordinal data directory.

    Mirrors ``sklearn.datasets.get_data_home``. Resolution order:
    ``data_home`` argument → ``$SKORDINAL_DATA`` environment variable →
    ``~/skordinal_data``. The directory is created if it does not exist.
    An empty ``$SKORDINAL_DATA`` value is treated as unset.

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
        # Empty SKORDINAL_DATA would otherwise resolve to the cwd
        data_home = os.environ.get("SKORDINAL_DATA") or (Path.home() / "skordinal_data")
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


def _bunch_from_csv(
    path, data_module, *, caller_name, feature_names=None, return_X_y, as_frame
):
    """Build the loader return value from a resolved CSV path."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data, target, csv_feature_names, header_class_names = _read_csv_any(path)
    # A caller-supplied list overrides the generated x0..xd-1 names, since
    # the metadata header carries no column names
    if feature_names is None:
        feature_names = csv_feature_names
    elif len(feature_names) != data.shape[1]:
        raise ValueError(
            f"{caller_name}: {len(feature_names)} feature name(s) declared, "
            f"but {path.name} has {data.shape[1]} feature column(s)."
        )
    target_names = _resolve_target_names(header_class_names, target)
    n_classes = len(target_names)

    descr = _load_descr(path, data_module)
    if descr is None:
        descr = (
            f"Dataset '{path.stem}': {data.shape[0]} samples, "
            f"{data.shape[1]} features, {n_classes} classes."
        )

    frame = None
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            caller_name, data, target, feature_names, ["target"]
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
        filename=path.name,
        data_module=data_module,
    )


def _load_bundled(stem, feature_names, *, return_X_y, as_frame):
    """Load a bundled dataset CSV under a fixed set of column names."""
    # Resolve against the bundled data module only, so a same-named file in
    # the working directory cannot shadow a shipped dataset
    return _bunch_from_csv(
        _bundled_csv_path(stem),
        DATA_MODULE,
        caller_name=f"load_{stem}",
        feature_names=feature_names,
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


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
            Integer target labels (int32). A Series when ``as_frame`` is True.
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
    ValueError
        When the CSV is empty, or when the target column does not hold
        integer class labels.

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
    return _bunch_from_csv(
        path,
        data_module_value,
        caller_name="load_dataset",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


@validate_params(
    {
        "name": [str, os.PathLike],
        "data_home": [str, os.PathLike, None],
        "resamples": [Interval(Integral, 1, None, closed="left"), list],
        "test_size": [Interval(Real, 0, 1, closed="neither")],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def load_partitions(
    name,
    *,
    data_home=None,
    resamples=30,
    test_size=0.3,
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
       ``sklearn.model_selection.StratifiedShuffleSplit``: one stratified
       holdout per requested id, sized by ``test_size`` and seeded from
       ``(random_state, id)``, so an id always maps to the same split
       under the same integer ``random_state``.

    Parameters
    ----------
    name : str or path-like
        Dataset stem (e.g. ``"era"``), filename, or concrete path.

    data_home : str, path-like, or None, default=None
        Directory to search when ``name`` is a stem or filename. When
        ``None``, the bundled data directory is used.

    resamples : int or list of int, default=30
        When an ``int``, resample ids are ``range(resamples)``.
        When a non-empty list, those ids are used directly and must
        be unique and non-negative.

    test_size : float, default=0.3
        Fraction of samples held out for testing in each generated
        partition; ignored when a masks file supplies them. The default
        follows the 70/30 protocol of the published reference partitions,
        whose exact memberships come only from their masks file.

    random_state : int, RandomState instance, or None, default=0
        Base seed combined with each resample id; ignored when a masks
        file supplies the splits. An integer seeds directly, so a split
        depends only on ``(random_state, id)``; an instance or ``None``
        is drawn from once per call, so those splits vary between calls.

    Yields
    ------
    bunch : ``sklearn.utils.Bunch``
        Dictionary-like object with the following attributes.

        data_train : ndarray of shape (n_train, n_features)
            Training features (float64).
        target_train : ndarray of shape (n_train,)
            Training targets (int32).
        data_test : ndarray of shape (n_test, n_features)
            Test features (float64).
        target_test : ndarray of shape (n_test,)
            Test targets (int32).
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
        train_index : ndarray of shape (n_train,)
            0-based indices of the training rows within the original
            dataset array.
        test_index : ndarray of shape (n_test,)
            0-based indices of the test rows within the original
            dataset array.
        n_classes : int
            Number of ordinal classes.
        DESCR : str
            One-line description of this resample.

    Raises
    ------
    FileNotFoundError
        When the CSV cannot be located.
    ValueError
        When ``resamples`` is an empty list, when the resample ids are
        negative or duplicated, when a mask does not hold boolean or
        integer values, when a mask does not have one entry per sample,
        when a mask is degenerate (all True or all False), or when the
        target column does not hold integer class labels. Stratification
        failures surface here too: a class with a single member, a split
        smaller than the class count, or a ``test_size`` that leaves the
        train split empty.
    TypeError
        When ``resamples`` is a bool, or a list of resample ids holds a
        non-integer.
    KeyError
        When a keyed masks file exists but does not contain the expected key.
    IndexError
        When a per-dataset masks file (``<stem>.masks.json``) contains
        fewer entries than the largest requested resample id.

    Notes
    -----
    With ``test_size <= 0.5`` every class keeps at least one training
    sample; above it a very small class may be missing from the training
    split. A class with very few members may be absent from the test
    split at any ratio.

    Examples
    --------
    >>> from skordinal.datasets import load_partitions  # doctest: +SKIP
    >>> for bunch in load_partitions("era", resamples=3):  # doctest: +SKIP
    ...     print(bunch.resample_id, bunch.data_train.shape[0])
    """
    csv_path, _ = _resolve_csv_path(name, data_home)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    # Resolve ids before reading any data, so a bad request costs no IO;
    # Interval accepts bools as Integral, hence the explicit rejection
    if isinstance(resamples, bool):
        raise TypeError(
            f"'resamples' must be an integer count or a list of ids; got {resamples!r}."
        )
    ids = list(range(resamples)) if isinstance(resamples, Integral) else list(resamples)
    _validate_resample_ids(ids)

    X, y, feature_names, header_class_names = _read_csv_any(csv_path)
    target_names = _resolve_target_names(header_class_names, y)
    n_classes = len(target_names)

    train_masks = _resolve_train_masks(csv_path, X, y, ids, test_size, random_state)

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
                train_index=np.flatnonzero(train_mask),
                test_index=np.flatnonzero(~train_mask),
                n_classes=n_classes,
                DESCR=(
                    f"{name} resample {resample_id}: "
                    f"{n_train}/{n_test} samples, {n_classes} classes."
                ),
            )

    return _iter()
