import json
import tempfile
import urllib.request
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


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

