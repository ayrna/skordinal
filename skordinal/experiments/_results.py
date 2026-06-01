"""Results handling for storing and managing experiment results."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ExperimentResult:
    """Result of running a single classifier on one dataset partition.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    classifier_name : str
        Name of the classifier configuration.

    resample_id : str
        Partition identifier.

    train_predicted_y : ndarray
        Class predictions on the training partition.

    test_predicted_y : ndarray or None
        Class predictions on the test partition. ``None`` if no test partition
        was available.

    y_proba : ndarray or None
        Class probability estimates on the test partition, shape
        ``(n_samples, n_classes)``. ``None`` if the estimator does not support
        ``predict_proba``.

    train_metrics : dict
        Metric values computed on the training partition, including timing.

    test_metrics : dict
        Metric values computed on the test partition, including timing.

    best_params : dict
        Best hyper-parameter values found during cross-validation.

    best_model : estimator
        Fitted estimator selected during cross-validation or direct fit.

    train_true_y : ndarray of shape (n_train_samples,) or None, default=None
        True class labels for the training partition. When provided, predictions
        CSV files include a ``y_true`` column alongside ``y_pred``.

    test_true_y : ndarray of shape (n_test_samples,) or None, default=None
        True class labels for the test partition. When provided, the test
        predictions CSV file includes a ``y_true`` column alongside ``y_pred``.

    """

    dataset_name: str
    classifier_name: str
    resample_id: str
    train_predicted_y: np.ndarray
    test_predicted_y: np.ndarray | None
    y_proba: np.ndarray | None
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    best_params: dict[str, Any]
    best_model: BaseEstimator
    train_true_y: np.ndarray | None = None
    test_true_y: np.ndarray | None = None


class Results:
    """Handle all information from an experiment that needs to be saved.

    This information will be saved into an specified folder.

    Parameters
    ----------
    output_folder : Path
        Base directory for storing experimental results.

    Attributes
    ----------
    _experiment_folder : Path
        Path where all the information about the actual experiment will be saved. This
        folder will have the next format: 'exp-YY-MM-DD-hh-mm-ss'.

    """

    def __init__(self, output_folder: Path) -> None:
        # Getting experiment's folder name
        folder_name = (
            "exp-"
            + date.today().strftime("%y-%m-%d")
            + "-"
            + datetime.now().strftime("%H-%M-%S")
        )

        self._experiment_folder = Path(output_folder) / folder_name

    def save(
        self,
        result: ExperimentResult,
        *,
        save_model: bool = True,
    ) -> None:
        """Store information obtained from the run of one partition.

        Parameters
        ----------
        result : ExperimentResult
            All data produced by a single classifier run on one partition.

        save_model : bool, default=True
            Whether to pickle the fitted model to disk.

        Raises
        ------
        OSError
            If the folder cannot be created.

        """
        dataset_folder = self._experiment_folder / (
            result.dataset_name + "-" + result.classifier_name
        )
        models_folder = dataset_folder / "models"
        predictions_folder = dataset_folder / "predictions"

        # Creating folder for this dataset-configuration if necessary
        if not dataset_folder.exists():
            try:
                if save_model:
                    models_folder.mkdir(parents=True)
                else:
                    predictions_folder.mkdir(parents=True)
                predictions_folder.mkdir(exist_ok=True)

            except OSError:
                raise OSError(
                    f"Could not create folder {dataset_folder} (or subfolders) "
                    "to store results."
                )

        # Saving partition model
        if save_model:
            models_folder.mkdir(exist_ok=True)
            model_filename = (
                result.dataset_name
                + "-"
                + result.classifier_name
                + "."
                + result.resample_id
            )
            with open(models_folder / model_filename, "wb") as output:
                pickle.dump(result.best_model, output)

        # Saving model predictions
        pred_filename = (
            result.dataset_name
            + "-"
            + result.classifier_name
            + "."
            + result.resample_id
        )
        if result.train_predicted_y is not None:
            np.savetxt(
                predictions_folder / f"train_{pred_filename}",
                result.train_predicted_y,
                fmt="%d",
            )

        if result.test_predicted_y is not None:
            np.savetxt(
                predictions_folder / f"test_{pred_filename}",
                result.test_predicted_y,
                fmt="%d",
            )

        if result.y_proba is not None:
            np.savetxt(
                predictions_folder / f"proba_{pred_filename}",
                result.y_proba,
            )

        dataframe_row = OrderedDict()
        # Adding best parameters as first elements in row
        for p_name, p_value in result.best_params.items():
            # If some ensemble method has been used, then one of its parameters will be
            # a dictionary containing the best parameters found for the base classifier.
            if isinstance(p_value, dict):
                for k, v in p_value.items():
                    dataframe_row[k] = v
            else:
                dataframe_row[p_name] = p_value

        # Concatenating train and test metrics
        for (tm_name, tm_value), (ts_name, ts_value) in zip(
            result.train_metrics.items(), result.test_metrics.items()
        ):
            dataframe_row[tm_name] = tm_value
            dataframe_row[ts_name] = ts_value

        # Adding row to existing DataFrame or creating new one
        df_path = dataset_folder / f"{result.dataset_name}-{result.classifier_name}.csv"

        df = pd.DataFrame([dataframe_row], index=[result.resample_id])
        if df_path.is_file():
            previous_df = pd.read_csv(df_path, index_col=[0])
            df = pd.concat([previous_df, df], axis=0)

        # Saving DataFrame to file
        df.to_csv(df_path)
