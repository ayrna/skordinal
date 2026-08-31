from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def load_tocuco_partitions(
    name,
    *,
    data_home=None,
    resamples=30,
):
    """Yield one train/test partition per resample for 'tocuco' style datasets.

    This method loads pre-computed train masks from a ``train_masks.pkl`` file
    located within the specified data directory, loads the corresponding CSV dataset
    from the data subdirectory, applies a standard scaler to the features, and
    yields train/test splits accordingly.

    Parameters
    ----------
    name : str or path-like
        Dataset name (e.g. ``"tocuco_dataset"``) whose CSV file is stored
        in the data subdirectory.

    data_home : str, path-like, or None, default=None
        Root directory containing ``train_masks.pkl`` and the ``data/`` subdirectory.
        If ``None``, the current working directory is used.

    resamples : int or list of int, default=30
        When an ``int``, resample IDs are ``range(resamples)``. When a
        list, those IDs are used directly.

    Yields
    ------
    bunch : ``sklearn.utils.Bunch``
        Dictionary-like object with the following attributes:

        data_train : ndarray of shape (n_train, n_features)
            Scaled training features (float64).
        target_train : ndarray of shape (n_train,)
            Training targets (int64).
        data_test : ndarray of shape (n_test, n_features)
            Scaled test features (float64).
        target_test : ndarray of shape (n_test,)
            Test targets (int64).
        feature_names : list of str
            Feature column names.
        target_names : ndarray of str
            Sorted unique target values as strings.
        dataset_name : str
            Echo of the requested dataset name.
        resample_id : int
            Identifier of the current resample.
        train_index : ndarray of shape (n_train,)
            0-based indices of the training rows.
        test_index : ndarray of shape (n_test,)
            0-based indices of the test rows.
        n_classes : int
            Number of ordinal classes.
        DESCR : str
            One-line description of this resample.

    Raises
    ------
    FileNotFoundError
        When the pickle file or dataset CSV cannot be located.
    KeyError
        When a requested resample key does not exist in the mask file.
    """
    
    tocuco_path = Path(data_home) if data_home is not None else Path(".")
    
    with open(tocuco_path / "train_masks.pkl", "rb") as train_masks_binary:
        train_masks = joblib.load(train_masks_binary)

    tocuco_datasets_path = tocuco_path / "data"
    dataset_name = str(name)

    dataset = pd.read_csv(tocuco_datasets_path / f"{dataset_name}.csv")
    feature_names = list(dataset.drop(columns=["y"]).columns)
    y = dataset["y"].values
    target_names = np.unique(y).astype(str)
    n_classes = len(target_names)

    ids = list(range(resamples)) if isinstance(resamples, Integral) else list(resamples)

    def _iter():
        for resample_id in ids:
            mask_key = f"{dataset_name}_seed_{resample_id}"
            if mask_key not in train_masks:
                raise KeyError(f"Mask key '{mask_key}' not found in train_masks.pkl")
            
            dataset_seed_train_mask = train_masks[mask_key]
            if isinstance(dataset_seed_train_mask, pd.Series):
                dataset_seed_train_mask = dataset_seed_train_mask.to_numpy()
            
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