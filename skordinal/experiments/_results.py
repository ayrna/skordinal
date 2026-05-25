"""Results handling for storing and managing experiment results."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


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

    def add_record(
        self,
        partition: str,
        best_params: dict[str, Any],
        best_model: BaseEstimator,
        configuration: dict[str, str],
        metrics: dict[str, dict[str, Any]],
        predictions: dict[str, np.ndarray | None],
    ) -> None:
        """Store information obtained from the run of one partition.

        Parameters
        ----------
        partition : str
            Partition's index.

        best_params : dict
            Best hyper-parameter's values found for this configuration and dataset
            during cross-validation. If an ensemble method has been used, there'll
            exist a parameter called 'parameters' that will store a dict with the best
            hyper-parameters found for the base classifier. Keys are the name of each
            parameter.

        best_model : estimator
            Best model created during cross-validation.

        configuration : dict
            Dictionary containing the name used for this pair of dataset and
            configuration. Keys are 'dataset' and 'config'.

        metrics : dict of dictionaries
            Dictionary containing the metrics for train and test for this particular
            configuration. It contains computational times for both of them as well.
            Keys are 'train' and 'test'.

        predictions : dict of lists
            Dictionary that stores train and test class predictions. Keys are 'train'
            and 'test'.

        Raises
        ------
        OSError
            If the folder cannot be created.

        """
        dataset_folder = self._experiment_folder / (
            configuration["dataset"] + "-" + configuration["config"]
        )
        models_folder = dataset_folder / "models"
        predictions_folder = dataset_folder / "predictions"

        # Creating folder for this dataset-configuration if necessary
        if not dataset_folder.exists():
            try:
                models_folder.mkdir(parents=True)
                predictions_folder.mkdir(parents=True)

            except OSError:
                raise OSError(
                    f"Could not create folder {dataset_folder} (or subfolders) to store results."
                )

        # Saving partition model
        model_filename = (
            configuration["dataset"] + "-" + configuration["config"] + "." + partition
        )
        with open(models_folder / model_filename, "wb") as output:
            pickle.dump(best_model, output)

        # Saving model predictions
        pred_filename = (
            configuration["dataset"] + "-" + configuration["config"] + "." + partition
        )
        train_pred = predictions["train"]
        if train_pred is not None:
            np.savetxt(
                predictions_folder / f"train_{pred_filename}",
                train_pred,
                fmt="%d",
            )

        test_pred = predictions["test"]
        if test_pred is not None:
            np.savetxt(
                predictions_folder / f"test_{pred_filename}",
                test_pred,
                fmt="%d",
            )

        dataframe_row = OrderedDict()
        # Adding best parameters as first elements in row
        for p_name, p_value in best_params.items():
            # If some ensemble method has been used, then one of its parameters will be
            # a dictionary containing the best parameters found for the base classifier.
            if isinstance(p_value, dict):
                for k, v in p_value.items():
                    dataframe_row[k] = v
            else:
                dataframe_row[p_name] = p_value

        # Concatenating train and test metrics
        for (tm_name, tm_value), (ts_name, ts_value) in zip(
            metrics["train"].items(), metrics["test"].items()
        ):
            dataframe_row[tm_name] = tm_value
            dataframe_row[ts_name] = ts_value

        # Adding row to existing DataFrame or creating new one
        df_path = (
            dataset_folder / f"{configuration['dataset']}-{configuration['config']}.csv"
        )

        df = pd.DataFrame([dataframe_row], index=[partition])
        if df_path.is_file():
            previous_df = pd.read_csv(df_path, index_col=[0])
            df = pd.concat([previous_df, df], axis=0)

        # Saving DataFrame to file
        df.to_csv(df_path)

    def save_summaries(self, metrics_names: list[str]) -> None:
        """Create an experiment summary.

        Each dataset-configuration will be represented as a single row of data, which
        will consist in the mean and standard deviation for the different metric's
        values across partitions.

        Parameters
        ----------
        metrics_names : list of str
            List with the names of all metrics used during the execution of the
            experiment. Includes computational times.

        """
        # Name of columns for summary dataframes
        avg_index = [mn + "_mean" for mn in metrics_names]
        std_index = [mn + "_std" for mn in metrics_names]

        train_summary = []
        test_summary = []
        summary_index = []

        for folder in self._experiment_folder.iterdir():
            df = pd.read_csv(folder / f"{folder.name}.csv")

            # Creating one entry per folder in summaries
            tr_sr, ts_sr = self._create_summary(df, avg_index, std_index)
            train_summary.append(tr_sr)
            test_summary.append(ts_sr)
            summary_index.append(folder.name)

        # Naming each row in datasets
        train_summary = pd.concat(train_summary, axis=1).transpose()
        train_summary.index = summary_index
        test_summary = pd.concat(test_summary, axis=1).transpose()
        test_summary.index = summary_index

        # Save summaries to csv
        train_summary.to_csv(self._experiment_folder / "train_summary.csv")
        test_summary.to_csv(self._experiment_folder / "test_summary.csv")

    def _create_summary(
        self,
        df: pd.DataFrame,
        avg_index: list[str],
        std_index: list[str],
    ) -> tuple[pd.Series, pd.Series]:
        """Summarize information from a DataFrame into a single row.

        Parameters
        ----------
        df : DataFrame
            Dataframe representing one Dataset-Configuration. Contains hyper-parameters,
            metric's scores and computational times.

        avg_index : list of str
            Includes all names of metrics ending with '_mean'.

        std_index : list of str
            Includes all names of metrics ending with '_std'.

        Returns
        -------
        train_summary_row : DataFrame
            DataFrame with only one row, containing mean and standard deviation for all
            metrics calculated across partitions (including computational times). Stores
            only train information.

        test_summary_row : DataFrame
            Stores only test information.

        """
        # Dissociating train and test metrics

        # Number of parameters used in this configuration
        n_parameters = len(df.columns) - len(avg_index) * 2
        # Even columns from dataframe (train metrics)
        train_df = df.iloc[:, n_parameters::2].copy()
        # Odd columns (test metrics)
        test_df = df.iloc[:, (n_parameters + 1) :: 2].copy()

        # Computing mean and standard deviation for metrics
        train_avg, train_std = train_df.mean(), train_df.std()
        test_avg, test_std = test_df.mean(), test_df.std()
        # Naming indexes for summary dataframes
        train_avg.index = avg_index
        train_std.index = std_index
        test_avg.index = avg_index
        test_std.index = std_index
        # Merging avg and std into one dataframe
        train_summary_row = pd.concat([train_avg, train_std])
        test_summary_row = pd.concat([test_avg, test_std])

        # Mixing avg and std DataFrame columns from metrics summaries
        train_summary_row = train_summary_row[
            list(
                sum(
                    zip(
                        train_summary_row.iloc[: len(avg_index)].keys(),
                        train_summary_row.iloc[len(std_index) :].keys(),
                    ),
                    (),
                )
            )
        ]

        test_summary_row = test_summary_row[
            list(
                sum(
                    zip(
                        test_summary_row.iloc[: len(avg_index)].keys(),
                        test_summary_row.iloc[len(std_index) :].keys(),
                    ),
                    (),
                )
            )
        ]

        return train_summary_row, test_summary_row
