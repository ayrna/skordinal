import json
import tempfile
import urllib.error
import urllib.request
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def _download_tocuco_dataset(dataset_name, filename, dest_dir):
    """Download and validate a file from a TOC-UCO dataset.
    A URL is constructed for a given TOC-UCO dataset and filename, downloads
    the file to the specified destination directory, and performs a validation
    to ensure the server did not return an HTML page instead of the expected data file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    filename : str
        The specific file to download.
    dest_dir : pathlib.Path
        The local directory path where the downloaded file should be saved.

    Returns
    -------
    dest_path : pathlib.Path
        The full path to the downloaded local file.

    Raises
    ------
    ValueError
        If the file is not found (404 error) or if the downloaded file appears
        to be an HTML page rather than the expected raw data.
    RuntimeError
        If an HTTP error (other than 404) or a connection error occurs during download.
    """
    url = f"https://www.uco.es/ayrna/tocuco/files/{dataset_name}/{filename}"
    dest_path = dest_dir / filename

    try:
        urllib.request.urlretrieve(url, dest_path)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(
                f"File '{filename}' for dataset '{dataset_name}' not found. "
                f"Check that the name is correct (attempted URL: {url})"
            ) from None
        raise RuntimeError(
            f"HTTP error {e.code} while downloading '{filename}' for dataset '{dataset_name}'."
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Connection error while downloading '{filename}' for dataset '{dataset_name}': {e.reason}"
        ) from e

    with open(dest_path, "r", encoding="utf-8", errors="ignore") as check_file:
        first_chars = check_file.read(200).lower()
        if "<html" in first_chars or "<!doctype" in first_chars:
            raise ValueError(
                f"File '{filename}' for dataset '{dataset_name}' not found. "
                f"The server returned an HTML web page instead of the valid file. "
                f"(attempted URL: {url})"
            )

    return dest_path


def load_tocuco_partitions(name, *, resamples=30):
    """Yield one train/test partition per resample for `TOC-UCO` datasets.
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

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)

        csv_path = _download_tocuco_dataset(
            dataset_name, f"{dataset_name}.csv", tmp_path
        )
        masks_path = _download_tocuco_dataset(
            dataset_name, "train_masks.json", tmp_path
        )

        try:
            dataset = pd.read_csv(csv_path)
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Could not parse the CSV file for '{dataset_name}'. The file might be corrupted."
            ) from e

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


def load_tocuco_dataset(name, *, return_X_y=False, as_frame=False):
    """Load a `TOC-UCO` dataset by name.

    This method downloads the corresponding CSV dataset.
    Once loaded into memory, the temporary files are automatically deleted.
    It returns the data using the exact same structure as ``load_dataset``.

    Parameters
    ----------
    name : str or path-like
        Dataset name (e.g. ``"dr04_forestfires"``) to be downloaded.

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
            Human-readable description generated automatically.
        filename : str
            Basename of the CSV file.
        data_module : str or None
            Always ``None`` for datasets downloaded from the web.

    (data, target) : tuple if ``return_X_y`` is True

    Raises
    ------
    urllib.error.URLError
        When the dataset cannot be downloaded from the server.
    ValueError
        When the server returns an HTML page (e.g. 404 page or redirect) instead of a CSV.
    """
    from ._base import _convert_data_dataframe, _read_csv_any, _resolve_target_names

    dataset_name = str(name)
    filename = f"{dataset_name}.csv"

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)

        csv_path = _download_tocuco_dataset(dataset_name, filename, tmp_path)

        data, target, feature_names, header_class_names = _read_csv_any(csv_path)
        target_names = _resolve_target_names(header_class_names, target)
        n_classes = len(target_names)

        descr = (
            f"TOC-UCO Dataset '{dataset_name}': {data.shape[0]} samples, "
            f"{data.shape[1]} features, {n_classes} classes."
        )

    frame = None
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_tocuco_dataset", data, target, feature_names, ["target"]
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
        data_module=None,
    )
