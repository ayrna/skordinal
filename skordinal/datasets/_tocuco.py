"""Remote download and dataset access for the TOC-UCO collection."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from collections import namedtuple
from numbers import Integral, Real
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from sklearn.utils._param_validation import Interval, validate_params

from ._base import get_data_home, load_dataset

_logger = logging.getLogger(__name__)

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url"])

# The only 4xx worth retrying, the rest are permanent
_TRANSIENT_HTTP_CODES = frozenset({408, 429})

_DATASET_BASE_URL = "https://www.uco.es/ayrna/tocuco/files"

# Remote because the cache renames the masks (see _CachedDataset.files)
_REMOTE_SIDECARS = ("metadata.csv", "train_masks.json")


def _fetch_remote(remote, dest_dir, n_retries=3, delay=1.0, validate_headers=None):
    """Fetch ``remote`` into ``dest_dir`` atomically, retrying transient errors."""
    file_path = dest_dir / remote.filename
    # Publishes are atomic, so an existing file is never a partial download
    if file_path.exists():
        return file_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix=remote.filename + ".part_", dir=dest_dir, delete=False
    )
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        for attempt in range(n_retries + 1):
            try:
                _logger.info("Downloading %s", remote.url)
                _, headers = urlretrieve(remote.url, tmp_path)
                if validate_headers is not None:
                    validate_headers(headers)
                break
            except (URLError, TimeoutError) as exc:
                non_transient = (
                    isinstance(exc, HTTPError)
                    and 400 <= exc.code < 500
                    and exc.code not in _TRANSIENT_HTTP_CODES
                )
                if non_transient or attempt == n_retries:
                    raise
                warnings.warn(
                    f"Retry {attempt + 1}/{n_retries}: {remote.url}", stacklevel=2
                )
                time.sleep(delay)
    except (Exception, KeyboardInterrupt):
        # KeyboardInterrupt included so Ctrl-C leaves no partial temp file
        tmp_path.unlink(missing_ok=True)
        raise
    shutil.move(tmp_path, file_path)
    return file_path


def _tocuco_root(data_home):
    """Return the TOC-UCO cache root under the resolved data home."""
    return Path(get_data_home(data_home)) / "tocuco"


def _check_dataset_name(name):
    """Reject a dataset name that would not be a plain cache subdirectory."""
    if not name:
        raise ValueError(f"name must be a non-empty string; got {name!r}.")
    if name.startswith("."):
        raise ValueError(f"name must not start with '.'; got {name!r}.")
    separators = {os.sep, "/"}
    if os.altsep is not None:
        separators.add(os.altsep)
    for sep in separators:
        if sep in name:
            raise ValueError(
                f"name must not contain a path separator ({sep!r}); got {name!r}."
            )
    if name.endswith(".csv"):
        raise ValueError(
            "name must not end with '.csv' (the extension is added "
            f"automatically when resolving the cached CSV); got {name!r}."
        )


def _is_oc(meta):
    """Return whether a metadata row is flagged as ordinal classification."""
    if meta is None:
        return False
    return (meta.get("is_oc") or "").strip().lower() in ("true", "1")


def _parse_class_distr(text):
    """Parse the metadata's ``[p0 p1 ...]`` class distribution into a float array."""
    values = np.array(text.strip().strip("[]").split(), dtype=float)
    if values.size == 0:
        raise ValueError(f"empty class distribution {text!r}")
    return values


def _make_descr(dataset_name, meta):
    """Build a short description for a TOC-UCO dataset from metadata."""
    if meta is None:
        return f"TOC-UCO dataset '{dataset_name}'."
    return (
        f"TOC-UCO dataset '{dataset_name}'.\n"
        f"Classes: {meta.get('n_classes', 'N/A')}, "
        f"Features: {meta.get('n_features', 'N/A')}, "
        f"Train samples: {meta.get('n_patterns_train', 'N/A')}, "
        f"Test samples: {meta.get('n_patterns_test', 'N/A')}."
    )


def _normalise_dataset_masks(train_masks_path, masks_path, name):
    """Replace a keyed 0..N-1 ``train_masks.json`` with an ordered-list masks file."""
    message = (
        f"TOC-UCO train_masks.json for {name!r} is malformed (it must be a "
        "non-empty JSON object keyed '0'..'N-1' mapping to equal-length, "
        "non-degenerate boolean lists); nothing was published."
    )
    with train_masks_path.open("r", encoding="utf-8") as fh:
        try:
            keyed = json.load(fh)
        except json.JSONDecodeError:
            # Backstop for a soft-404 page the content-type check missed
            raise OSError(message) from None
    if not isinstance(keyed, dict) or not keyed:
        raise OSError(message)
    if set(keyed) != {str(i) for i in range(len(keyed))}:
        raise OSError(message)
    values = list(keyed.values())
    if not all(isinstance(value, list) and value for value in values):
        raise OSError(message)
    # A mask that slipped through here would silently mis-split the data
    if not all(isinstance(x, bool) for value in values for x in value):
        raise OSError(message)
    if len({len(value) for value in values}) != 1:
        raise OSError(message)
    if any(all(value) or not any(value) for value in values):
        raise OSError(message)
    # The site's plain "0".."N-1" keys match neither form _base reads
    masks_list = [keyed[str(k)] for k in range(len(keyed))]
    with masks_path.open("w", encoding="utf-8") as fh:
        json.dump(masks_list, fh)
    train_masks_path.unlink()


class _CachedDataset:
    """Paths, state checks and publishing for one dataset's cache directory."""

    def __init__(self, dataset_dir):
        self.dir = dataset_dir
        self.name = dataset_dir.name

    @property
    def csv_path(self):
        """Path of the dataset's CSV inside the cache directory."""
        return self.dir / f"{self.name}.csv"

    @property
    def files(self):
        """The three files a complete cache directory holds, CSV first."""
        return (
            self.csv_path,
            self.dir / "metadata.csv",
            self.dir / f"{self.name}.masks.json",
        )

    def is_cached(self):
        """Return whether the directory exists under exactly this name."""
        # Path.exists() is case-insensitive on macOS, where a miscased name
        # would otherwise adopt an unrelated dataset's cache
        parent = self.dir.parent
        return parent.is_dir() and self.name in os.listdir(parent)

    def is_complete(self):
        """Return whether every file the cache needs is present."""
        return all(path.is_file() for path in self.files)

    def stale_error(self, problem):
        """Build the OSError for a cache tree that must be deleted and refetched."""
        return OSError(
            f"TOC-UCO per-dataset cache at {self.dir} {problem}; it appears "
            "stale or only partially populated. Delete this directory and "
            "call fetch_tocuco() again to rebuild it."
        )

    def validate(self):
        """Raise OSError naming the first missing cache file."""
        # A tree without its masks would make load_partitions fall back to
        # generated holdout splits in silence
        for path in self.files:
            if not path.is_file():
                raise self.stale_error(f"is missing '{path.name}'")

    def metadata_row(self):
        """Return this dataset's metadata.csv row, or None when absent."""
        path = self.dir / "metadata.csv"
        try:
            # utf-8-sig strips a BOM that would otherwise hide every row
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("dataset") == self.name:
                        return row
        except (UnicodeDecodeError, csv.Error) as err:
            raise self.stale_error(f"has an undecodable metadata.csv ({err})") from err
        return None

    def metadata_fields(self, meta):
        """Parse a metadata row into typed fields, mapping bad cells to OSError."""
        try:
            return {
                "is_oc": _is_oc(meta),
                "n_patterns_train": int(meta["n_patterns_train"]),
                "n_patterns_test": int(meta["n_patterns_test"]),
                "class_distr": _parse_class_distr(meta["class_distr"]),
                "imbalance_ratio": float(meta["imbalance_ratio"]),
            }
        except (KeyError, ValueError, TypeError) as err:
            raise self.stale_error(
                f"has a malformed metadata.csv for {self.name!r} ({err})"
            ) from err

    def publish(self, staged):
        """Publish a staged directory atomically under this dataset's name."""
        try:
            os.rename(staged, self.dir)
            return
        except OSError:
            # A complete rival tree came from the same source, so it serves
            if self.is_complete():
                return
        # Whatever squats there may be a file, which rmtree would skip
        if self.dir.is_dir():
            shutil.rmtree(self.dir, ignore_errors=True)
        else:
            self.dir.unlink(missing_ok=True)
        try:
            os.rename(staged, self.dir)
        except OSError:
            # A concurrent publish refilled the name, and its tree serves as well
            if not self.is_complete():
                raise


def _download_tocuco_dataset(name, data_home, n_retries, delay):
    """Download, normalise, and publish one TOC-UCO dataset's cache tree."""

    # An unknown name is answered with a web page, not an HTTP error
    def _reject_soft_404(headers):
        """Raise ValueError when the site served a web page instead of data."""
        content_type = headers.get_content_type() if headers is not None else ""
        if content_type == "text/html":
            raise ValueError(
                f"Dataset {name!r} was not found in the TOC-UCO "
                f"repository; the server returned a web page instead of "
                f"dataset files under {_DATASET_BASE_URL}/{name}/. Check "
                "the dataset name against the TOC-UCO website."
            )

    dataset = _CachedDataset(_tocuco_root(data_home) / name)
    dataset.dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=dataset.dir.parent))
    remotes = [
        RemoteFileMetadata(fname, f"{_DATASET_BASE_URL}/{name}/{fname}")
        for fname in (f"{name}.csv", *_REMOTE_SIDECARS)
    ]
    try:
        for remote in remotes:
            _fetch_remote(
                remote, staging, n_retries, delay, validate_headers=_reject_soft_404
            )
        _normalise_dataset_masks(
            staging / "train_masks.json", staging / f"{name}.masks.json", name
        )
        dataset.publish(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _ensure_dataset_cached(
    name, data_home, download_if_missing, n_retries, delay, caller
):
    """Return the validated cache entry for a dataset, downloading if needed."""
    _check_dataset_name(name)
    dataset = _CachedDataset(_tocuco_root(data_home) / name)
    if not dataset.is_cached():
        if not download_if_missing:
            raise OSError(
                f"TOC-UCO dataset {name!r} not found at {dataset.dir}. Call "
                f"{caller}() with download_if_missing=True."
            )
        _download_tocuco_dataset(name, data_home, n_retries, delay)
    # Validated even when just downloaded, since the publish race can adopt
    # a rival tree that is missing files
    dataset.validate()
    return dataset


@validate_params(
    {
        "name": [str],
        "data_home": [str, os.PathLike, None],
        "download_if_missing": ["boolean"],
        "return_X_y": ["boolean"],
        "as_frame": ["boolean"],
        "n_retries": [Interval(Integral, 0, None, closed="left")],
        "delay": [Interval(Real, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def fetch_tocuco(
    name,
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    n_retries=3,
    delay=1.0,
):
    """Download and load a single TOC-UCO dataset.

    Fetches the three files published for one TOC-UCO dataset (the full
    data CSV, its metadata row, and its train/test partition masks) into
    a per-dataset cache directory and returns the whole dataset (train
    and test samples together) as a Bunch. Only the requested dataset is
    downloaded, not the full collection.

    Parameters
    ----------
    name : str
        Dataset name, for example ``"oc09_era"``.

    data_home : str, os.PathLike, or None, default=None
        Root directory for the local cache. When ``None``, the value of
        the ``SKORDINAL_DATA`` environment variable is used if set;
        otherwise ``~/skordinal_data``.

    download_if_missing : bool, default=True
        If ``False`` and the dataset is not cached locally, raise
        ``OSError`` instead of downloading.

    return_X_y : bool, default=False
        If ``True``, return ``(data, target)`` instead of a Bunch. The
        TOC-UCO metadata fields are omitted in this form.

    as_frame : bool, default=False
        If ``True``, ``data`` is a ``pandas.DataFrame`` and ``target`` is
        a ``pandas.Series``.

    n_retries : int, default=3
        Number of retry attempts after the initial download fails. Must
        be non-negative; ``0`` performs a single attempt with no retries.

    delay : float, default=1.0
        Seconds to wait between retry attempts. Must be non-negative.

    Returns
    -------
    bunch : ``sklearn.utils.Bunch``
        Dictionary-like object with the following attributes.

        data : ndarray of shape (n_samples, n_features)
            Feature matrix over all samples (float64). A
            ``pandas.DataFrame`` when ``as_frame`` is True.
        target : ndarray of shape (n_samples,)
            Integer target labels (int32). A ``pandas.Series`` when
            ``as_frame`` is True.
        frame : pandas.DataFrame or None
            Combined frame when ``as_frame`` is True; otherwise ``None``.
        feature_names : list of str
            Feature column names from the CSV header.
        target_names : ndarray of str
            Sorted unique target values as strings.
        n_classes : int
            Number of ordinal classes.
        dataset_name : str
            Echo of the requested ``name``.
        is_oc : bool
            Whether the dataset is flagged as ordinal classification.
        n_patterns_train : int
            Canonical number of training samples.
        n_patterns_test : int
            Canonical number of test samples.
        class_distr : ndarray of shape (n_classes,)
            Class proportions as published in the metadata.
        imbalance_ratio : float
            Imbalance ratio as published in the metadata.
        url : str
            Location of this dataset in the TOC-UCO repository.
        DESCR : str
            Short human-readable description.
        filename : str
            Basename of the CSV file.
        data_module : None
            Always ``None`` for fetched datasets.

    (data, target) : tuple if ``return_X_y`` is True

    Raises
    ------
    ValueError
        When ``name`` is empty, starts with a dot, contains a path
        separator, or ends with ``".csv"``, when ``name`` is not found in
        the TOC-UCO repository, or when the downloaded CSV cannot be
        parsed.

    OSError
        When ``download_if_missing=False`` and the dataset is not cached,
        when a per-dataset cache is stale or only partially populated (a
        missing file, a metadata row absent for the name, or a malformed
        metadata row), or when the downloaded ``train_masks.json`` is
        malformed.

    urllib.error.URLError
        When the download fails: immediately for a permanent HTTP client
        error (any 4xx status other than 408 and 429), or after all
        retry attempts for a transient failure (a timeout, a 5xx status,
        or HTTP 408/429).

    Examples
    --------
    >>> from skordinal.datasets import fetch_tocuco     # doctest: +SKIP
    >>> data = fetch_tocuco("oc09_era")                 # doctest: +SKIP
    >>> data.data.shape[0] == (                         # doctest: +SKIP
    ...     data.n_patterns_train + data.n_patterns_test
    ... )
    True
    """
    dataset = _ensure_dataset_cached(
        name, data_home, download_if_missing, n_retries, delay, "fetch_tocuco"
    )

    bunch = load_dataset(
        name, data_home=dataset.dir, return_X_y=return_X_y, as_frame=as_frame
    )
    if return_X_y:
        return bunch

    meta = dataset.metadata_row()
    if meta is None:
        raise dataset.stale_error(f"has a metadata.csv with no entry for {name!r}")
    bunch.update(dataset.metadata_fields(meta))
    bunch.dataset_name = name
    bunch.url = f"{_DATASET_BASE_URL}/{name}"
    bunch.DESCR = _make_descr(name, meta)
    return bunch


def load_tocuco_partitions(
    name,
    *,
    resamples=30,
):
    """Yield one train/test partition per resample for 'tocuco' style datasets.
    This method downloads pre-computed train masks from a ``train_masks.json`` file
    and the corresponding CSV dataset from the remote server to a temporary directory.
    Once loaded into memory, the temporary files are automatically deleted. It then
    applies a standard scaler to the features, and yields train/test splits accordingly.

    Parameters
    ----------
    name : str or path-like
        Dataset name (e.g. ``"dr04_forestfires"``) to be downloaded.

    resamples : int or list of int, default=30
        When an ``int``, resample IDs are ``range(resamples)``. When a
        list, those IDs are used directly.

    Yields
    ------
    bunch : ``sklearn.utils.Bunch``
        Dictionary-like object with train/test datasets and metadata.

    Raises
    ------
    urllib.error.URLError
        When the dataset or mask file cannot be downloaded from the server.
    KeyError
        When a requested resample key does not exist in the mask file.
    """

    dataset_name = str(name)
    base_url = f"https://www.uco.es/ayrna/tocuco/files/{dataset_name}"

    csv_url = f"{base_url}/{dataset_name}.csv"
    masks_url = f"{base_url}/train_masks.json"

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)

        csv_path = tmp_path / f"{dataset_name}.csv"
        masks_path = tmp_path / "train_masks.json"

        try:
            urllib.request.urlretrieve(csv_url, csv_path)
            urllib.request.urlretrieve(masks_url, masks_path)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found. "
                    f"Check that the name is correct (attempted URL: {csv_url})"
                ) from None
            raise RuntimeError(
                f"HTTP error {e.code} while downloading dataset '{dataset_name}'."
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Connection error while downloading dataset '{dataset_name}': {e.reason}"
            ) from e

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as check_file:
            first_chars = check_file.read(200).lower()
            if "<html" in first_chars or "<!doctype" in first_chars:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found. "
                    f"The server returned an HTML web page instead of the dataset file. "
                    f"Check that the name is correct (attempted URL: {csv_url})"
                )

        try:
            dataset = pd.read_csv(csv_path)
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Could not parse the CSV file for '{dataset_name}'. The file might be corrupted."
            ) from e

        dataset = pd.read_csv(csv_path)

        with open(masks_path, "r", encoding="utf-8") as f:
            train_masks = json.load(f)

    feature_names = list(dataset.drop(columns=["y"]).columns)
    y = dataset["y"].values
    target_names = np.unique(y).astype(str)
    n_classes = len(target_names)

    ids = list(range(resamples)) if isinstance(resamples, Integral) else list(resamples)

    def _iter():
        for resample_id in ids:
            mask_key = str(resample_id)

            if mask_key not in train_masks:
                raise KeyError(f"Mask key '{mask_key}' not found in train_masks.json")

            dataset_seed_train_mask = np.array(train_masks[mask_key], dtype=bool)

            train = dataset.loc[dataset_seed_train_mask]
            test = dataset.loc[~dataset_seed_train_mask]

            X_train = train.drop(columns=["y"])
            X_test = test.drop(columns=["y"])
            y_train = train["y"].values
            y_test = test["y"].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            n_train = int(dataset_seed_train_mask.sum())
            n_test = int((~dataset_seed_train_mask).sum())

            yield Bunch(
                data_train=X_train_scaled,
                target_train=y_train,
                data_test=X_test_scaled,
                target_test=y_test,
                feature_names=feature_names,
                target_names=target_names,
                dataset_name=dataset_name,
                resample_id=int(resample_id),
                train_index=np.flatnonzero(dataset_seed_train_mask),
                test_index=np.flatnonzero(~dataset_seed_train_mask),
                n_classes=n_classes,
                DESCR=(
                    f"{dataset_name} resample {resample_id}: "
                    f"{n_train}/{n_test} samples, {n_classes} classes."
                ),
            )

    return _iter()
