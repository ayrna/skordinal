"""Utility class for running experiments."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from skordinal.model_selection import load_classifier

from ._results import Results


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from skordinal.metrics import get_ordinal_scorer

    scorer = cast(Any, get_ordinal_scorer(metric_name.strip()))
    return scorer._score_func(y_true, y_pred, **scorer._kwargs)


class Utilities:
    """Run experiments over N datasets with M different configurations.

    Configurations are composed of a classifier method and different parameters, where
    it may be multiple values for every one of them.

    Running the main function of this class will perform cross-validation for each
    partition per dataset-configuration pairs, obtaining the most optimal model, after
    what will be used to infer the labels for the test sets.

    Parameters
    ----------
    general_conf : dict
        Dictionary containing values needed to run the experiment. It gives this class
        information about where are located the different datasets, which one are going
        to be tested, the metrics to use, etc.

    configurations : dict
        Dictionary in which are stated the different classifiers to build methods upon
        the selected datasets, as well as the different values for the hyper-parameters
        used to optimize the model during cross-validation phase.

    verbose : bool
        Variable used for testing purposes. Silences all prints.

    Attributes
    ----------
    _results : Results
        Class used to manage and store all information obtained during the run of an
        experiment.

    """

    def __init__(
        self,
        general_conf: dict[str, Any],
        configurations: dict[str, Any],
        verbose: bool = True,
    ) -> None:
        self.general_conf = deepcopy(general_conf)
        self.configurations = deepcopy(configurations)
        self.verbose = verbose

    def run_experiment(self) -> None:
        """Run an experiment. Main method of this framework.

        Loads all datasets, which can be fragmented in partitions. Builds a model per
        partition, using cross-validation to find the optimal values among the
        hyper-parameters to compare from.

        Uses the built model to get train and test metrics, storing all the information
        into a Results object.

        Raises
        ------
        ValueError
            If the dataset list is inconsistent.

        AttributeError
            If the input preprocessing is unknown.

        TypeError
            If the parameters for base_classifier must be list.

        """
        self._results = Results(Path(self.general_conf["output_folder"]))

        self._check_dataset_list()

        if self.verbose:
            print("\n###############################")
            print("\tRunning Experiment")
            print("###############################")

        # Iterating over Datasets
        for x in self.general_conf["datasets"]:
            dataset_name = x.strip()
            dataset_path = Path(self.general_conf["basedir"]) / dataset_name

            dataset = self._load_dataset(dataset_path)

            if self.verbose:
                print("\nRunning", dataset_name, "dataset")
                print("--------------------------")

            # Iterating over Configurations
            for conf_name, configuration in self.configurations.items():
                if self.verbose:
                    print("Running", conf_name, "...")

                # Iterating over partitions
                for part_idx, partition in dataset:
                    if self.verbose:
                        print("  Running Partition", part_idx)

                    # Normalization or Standardization of the partition if requested
                    _preproc = (
                        self.general_conf.get("input_preprocessing", "").strip().lower()
                    )
                    if _preproc == "norm":
                        partition["train_inputs"], partition["test_inputs"] = (
                            self._normalize_data(
                                partition["train_inputs"], partition["test_inputs"]
                            )
                        )
                    elif _preproc == "std":
                        partition["train_inputs"], partition["test_inputs"] = (
                            self._standardize_data(
                                partition["train_inputs"], partition["test_inputs"]
                            )
                        )
                    elif _preproc != "":
                        raise AttributeError(
                            "Input preprocessing named '%s' unknown" % _preproc
                        )

                    optimal_estimator = self._get_optimal_estimator(
                        partition["train_inputs"],
                        partition["train_outputs"],
                        configuration["classifier"],
                        configuration["parameters"],
                    )

                    # Getting train and test predictions
                    train_predicted_y = optimal_estimator.predict(
                        partition["train_inputs"]
                    )

                    test_predicted_y = None
                    elapsed = np.nan
                    if "test_outputs" in partition:
                        start = time()
                        test_predicted_y = np.asarray(
                            optimal_estimator.predict(partition["test_inputs"])
                        )
                        elapsed = time() - start

                    # Obtaining train and test metrics values.
                    train_metrics = OrderedDict()
                    test_metrics = OrderedDict()
                    for metric_name in self.general_conf["metrics"]:
                        # Get train scores
                        train_score = _compute_metric(
                            metric_name,
                            partition["train_outputs"],
                            train_predicted_y,
                        )
                        train_metrics[metric_name.strip() + "_train"] = train_score

                        # Get test scores
                        test_metrics[metric_name.strip() + "_test"] = np.nan
                        if "test_outputs" in partition:
                            assert test_predicted_y is not None
                            test_score = _compute_metric(
                                metric_name, partition["test_outputs"], test_predicted_y
                            )
                            test_metrics[metric_name.strip() + "_test"] = test_score

                    # Cross-validation was performed to tune hyper-parameters
                    if isinstance(optimal_estimator, GridSearchCV):
                        train_metrics["cv_time_train"] = optimal_estimator.cv_results_[
                            "mean_fit_time"
                        ].mean()
                        test_metrics["cv_time_test"] = optimal_estimator.cv_results_[
                            "mean_score_time"
                        ].mean()
                        train_metrics["time_train"] = optimal_estimator.refit_time_
                        test_metrics["time_test"] = elapsed

                    else:
                        optimal_estimator.best_params_ = configuration["parameters"]
                        optimal_estimator.best_estimator_ = optimal_estimator

                        train_metrics["cv_time_train"] = np.nan
                        test_metrics["cv_time_test"] = np.nan
                        train_metrics["time_train"] = optimal_estimator.refit_time_
                        test_metrics["time_test"] = elapsed

                    # Saving the results for this partition
                    self._results.add_record(
                        part_idx,
                        optimal_estimator.best_params_,
                        optimal_estimator.best_estimator_,
                        {"dataset": dataset_name, "config": conf_name},
                        {"train": train_metrics, "test": test_metrics},
                        {"train": train_predicted_y, "test": test_predicted_y},
                    )

    def _load_dataset(self, dataset_path: Path) -> list[tuple[str, dict[str, Any]]]:
        """Load all dataset's files, divided into train and test.

        Parameters
        ----------
        dataset_path : Path
            Path to dataset folder.

        Returns
        -------
        partition_list : list of tuples
            List of partitions found inside a dataset folder. Each partition is stored
            into a dictionary, disjoining train and test inputs and outputs.

        Raises
        ------
        ValueError
            If the dataset path does not exist.

        RuntimeError
            If a partition is found without train files.

        """

        def get_partition_index(filename: str) -> str:
            # Extracts the index between the last "_" and ".csv"
            return filename.rsplit("_", 1)[-1].replace(".csv", "")

        try:
            partition_list: dict[str, dict[str, Any]] = {
                get_partition_index(filename.name): {}
                for filename in dataset_path.iterdir()
                if filename.name.startswith("train_")
            }

            # Loading each dataset
            for filename in dataset_path.iterdir():
                if filename.name.startswith("train_"):
                    idx = get_partition_index(filename.name)
                    train_inputs, train_outputs = self._read_file(filename)
                    partition_list[idx]["train_inputs"] = train_inputs
                    partition_list[idx]["train_outputs"] = train_outputs

                elif filename.name.startswith("test_"):
                    idx = get_partition_index(filename.name)
                    test_inputs, test_outputs = self._read_file(filename)
                    partition_list[idx]["test_inputs"] = test_inputs
                    partition_list[idx]["test_outputs"] = test_outputs

        except OSError:
            raise ValueError(f"No such file or directory: '{dataset_path}'")

        except KeyError:
            raise RuntimeError(
                f"Found partition without train files: partition {filename.name}"
            )

        # Saving partitions as a sorted list of (index, partition) tuples
        sorted_list: list[tuple[str, dict[str, Any]]] = sorted(
            partition_list.items(),
            key=lambda t: int(t[0]) if t[0].lstrip("-").isdigit() else t[0],
        )

        return sorted_list

    def _read_file(self, filename: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a CSV containing partitions, or full datasets.

        Train and test files must be previously divided for the experiment to run.

        Parameters
        ----------
        filename : str or Path
            Full path to train or test file.

        Returns
        -------
        inputs : {array-like, sparse-matrix} of shape (n_samples, n_features)
            Vector of sample's features.

        outputs : array-like of shape (n_samples)
            Target vector relative to inputs.

        """
        # Separator is automatically found
        f = pd.read_csv(filename, header=None, engine="python")

        inputs = f.values[:, 0:(-1)]
        outputs = f.values[:, (-1)]

        return inputs, outputs

    def _check_dataset_list(self) -> None:
        """Check if there is some inconsistency in the dataset list.

        It also simplifies running all datasets inside one folder.

        Raises
        ------
        ValueError
            If the dataset list is inconsistent or contains non-string values.

        """
        base_path = Path(self.general_conf["basedir"])
        dataset_list = self.general_conf["datasets"]

        # Check if home path is shortened
        if str(base_path).startswith("~"):
            base_path = Path.home() / str(base_path)[1:]

        # Check if 'all' is the only value, and if it is, expand it
        if len(dataset_list) == 1 and dataset_list[0] == "all":
            dataset_list = [item.name for item in base_path.iterdir() if item.is_dir()]

        elif not all(isinstance(item, str) for item in dataset_list):
            raise ValueError("Dataset list can only contain strings")

        self.general_conf["basedir"] = str(base_path)
        self.general_conf["datasets"] = dataset_list

    def _normalize_data(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize the data.

        Test data normalization will be based on train data.

        Parameters
        ----------
        train_data : 2d array
            Contain the train data features.

        test_data : 2d array
            Contain the test data features.

        Returns
        -------
        train_normalized : np.ndarray
            Normalized training data.

        test_normalized : np.ndarray
            Normalized test data.

        """
        mm_scaler = preprocessing.MinMaxScaler().fit(train_data)

        return mm_scaler.transform(train_data), mm_scaler.transform(test_data)

    def _standardize_data(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardize the data.

        Test data standardization will be based on train data.

        Parameters
        ----------
        train_data : 2d array
            Contain the train data features.

        test_data : 2d array
            Contain the test data features.

        Returns
        -------
        train_standardized : np.ndarray
            Standardized training data.

        test_standardized : np.ndarray
            Standardized test data.

        """
        std_scaler = preprocessing.StandardScaler().fit(train_data)

        return std_scaler.transform(train_data), std_scaler.transform(test_data)

    def _get_optimal_estimator(
        self,
        train_inputs: np.ndarray,
        train_outputs: np.ndarray,
        classifier_name: str,
        parameters: dict[str, Any],
    ) -> BaseEstimator | GridSearchCV:
        """Perform cross-validation over one dataset and configuration.

        Each configuration consists of one classifier and none, one or multiple
        hyper-parameters, that, in turn, can contain one or multiple values used to
        optimize the resulting model.

        At the end of cross-validation phase, the model with the specific combination of
        values from the hyper-parameters that achieved the best metrics from all the
        combinations will remain.

        Parameters
        ----------
        train_inputs : {array-like, sparse-matrix} of shape (n_samples, n_features)
            Vector of features for each sample for this dataset.

        train_outputs : array-like of shape (n_samples)
            Target vector relative to train_inputs.

        classifier_name : str
            Name of the classification algorithm being employed.

        parameters : dict
            Dictionary containing parameters to optimize as keys, and the list of
            values that we want to compare as values.

        Returns
        -------
        optimal: GridSearchCV object or classifier object
            An already fitted model of the given classifier, with the best found
            parameters after cross-validation. If cross-validation is not needed, it will
            return the classifier model already trained.

        Raises
        ------
        AttributeError
            If the metric name is not found or cv_metric is not a string.

        """
        estimator = load_classifier(
            classifier_name=classifier_name,
            random_state=int(cast(tuple, np.random.get_state())[1][0]),
            n_jobs=self.general_conf.get("jobs", 1),
            cv_n_folds=self.general_conf.get("hyperparam_cv_nfolds", 3),
            cv_metric=self.general_conf.get("cv_metric", "mae"),
            param_grid=parameters,
        )

        start = time()
        estimator.fit(train_inputs, train_outputs)
        elapsed = time() - start

        if not isinstance(estimator, GridSearchCV):
            estimator.refit_time_ = elapsed
            estimator.best_params_ = parameters
            estimator.best_estimator_ = estimator

        return estimator

    def write_report(self) -> None:
        """Save summarized information about experiment through Results class."""
        if self.verbose:
            print("\nSaving Results...")

        # Names of each metric used (plus computational times)
        metrics_names = [x.strip().lower() for x in self.general_conf["metrics"]] + [
            "cv_time",
            "time",
        ]

        # Saving results through Results class
        self._results.save_summaries(metrics_names)
