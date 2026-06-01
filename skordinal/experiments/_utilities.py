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

from ._results import ExperimentResult, Results


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from skordinal.metrics import get_ordinal_scorer

    scorer = cast(Any, get_ordinal_scorer(metric_name.strip()))
    return scorer._score_func(y_true, y_pred, **scorer._kwargs)


class Utilities:
    """Run experiments over N datasets with M different configurations.

    Configurations are composed of a classifier method and different parameters,
    where it may be multiple values for every one of them.

    Running the main function of this class will perform cross-validation for
    each partition per dataset-configuration pair, obtaining the most optimal
    model, which is then used to infer labels for the test sets.

    Parameters
    ----------
    configurations : dict
        Mapping of configuration labels to their settings. Each entry must have
        the form ``{label: {"classifier": str, "parameters": dict}}``, where
        ``"classifier"`` is the registered name of the classification algorithm
        and ``"parameters"`` is a dictionary of hyper-parameter grids (lists of
        values to search over) or fixed values.

    data_path : str or Path
        Base directory that contains one sub-folder per dataset.

    datasets : list of str
        Names of the dataset sub-folders to run. Pass ``["all"]`` to expand
        automatically to every directory found under ``data_path``.

    eval_metrics : list of str
        Metric names to compute for every partition (e.g.
        ``["mean_absolute_error", "average_mean_absolute_error"]``).
        Names must be recognised by ``skordinal.metrics.get_ordinal_scorer``.

    results_path : str or Path
        Directory where result files are written.

    tuning_metric : str, default="neg_mean_absolute_error"
        Metric used as the cross-validation scoring criterion when selecting the
        best hyper-parameter combination. Must be recognised by
        ``skordinal.metrics.get_ordinal_scorer``; validation is deferred to
        runtime.

    cv : int, default=3
        Number of folds used in hyper-parameter cross-validation.

    n_jobs : int, default=1
        Number of parallel jobs forwarded to ``GridSearchCV``.

    input_preprocessing : {"std", "norm"} or None, default=None
        Optional feature preprocessing applied to every partition before
        fitting: ``"norm"`` applies min-max normalisation and ``"std"`` applies
        z-score standardisation. Both scalers are fitted on the training split
        only, then applied to both train and test splits. ``None`` means no
        preprocessing.

    random_state : int or None, default=None
        Seed used for two sources of randomness: the base estimator and the
        cross-validation splitter (``StratifiedKFold``) used during
        hyper-parameter search. When ``None``, both use their own default
        random behaviour.

    verbose : bool, default=True
        If ``True``, progress messages are printed to stdout.

    Attributes
    ----------
    _results : Results
        Manages and stores all information obtained during the experiment run.

    Examples
    --------
    >>> from skordinal.experiments import Utilities  # doctest: +SKIP
    >>> u = Utilities(  # doctest: +SKIP
    ...     configurations={"SVM": {"classifier": "svc", "parameters": {"C": [1]}}},
    ...     data_path="/data/ordinal",
    ...     datasets=["balance-scale"],
    ...     eval_metrics=["mean_absolute_error"],
    ...     results_path="/tmp/results",
    ... )
    >>> u.run_experiment()  # doctest: +SKIP

    """

    def __init__(
        self,
        configurations: dict[str, Any],
        *,
        data_path: str | Path,
        datasets: list[str],
        eval_metrics: list[str],
        results_path: str | Path,
        tuning_metric: str = "neg_mean_absolute_error",
        cv: int = 3,
        n_jobs: int = 1,
        input_preprocessing: str | None = None,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> None:
        if not configurations:
            raise ValueError(
                "'configurations' must be a non-empty dict; got an empty mapping."
            )
        if not datasets:
            raise ValueError(
                "'datasets' must be a non-empty list; got an empty sequence."
            )
        if not eval_metrics:
            raise ValueError(
                "'eval_metrics' must be a non-empty list; got an empty sequence."
            )

        _allowed_preproc = {"std", "norm"}
        if input_preprocessing is not None:
            _normalized = str(input_preprocessing).strip().lower()
            if _normalized not in _allowed_preproc:
                raise ValueError(
                    f"'input_preprocessing' must be one of {None, 'std', 'norm'}; "
                    f"got '{input_preprocessing}'."
                )
            input_preprocessing = _normalized

        self.configurations = deepcopy(configurations)
        self.data_path: str | Path = data_path
        self.datasets: list[str] = list(datasets)
        self.eval_metrics: list[str] = list(eval_metrics)
        self.results_path: str | Path = results_path
        self.tuning_metric = tuning_metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.input_preprocessing = input_preprocessing
        self.random_state = random_state
        self.verbose = verbose

    def run_experiment(self) -> None:
        """Run an experiment. Main method of this framework.

        Loads all datasets, which can be fragmented in partitions. Builds a model
        per partition, using cross-validation to find the optimal values among the
        hyper-parameters to compare from.

        Uses the built model to get train and test metrics, storing all the
        information into a Results object.

        Raises
        ------
        ValueError
            If the dataset list is inconsistent, or a dataset path does not
            exist.

        RuntimeError
            If a partition is found without its train file.

        """
        self._results = Results(Path(self.results_path))

        self._check_dataset_list()

        if self.verbose:
            print("\n###############################")
            print("\tRunning Experiment")
            print("###############################")

        # Iterating over Datasets
        for x in self.datasets:
            dataset_name = x.strip()
            dataset_path = Path(self.data_path) / dataset_name

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

                    # Normalisation or standardisation of the partition if requested
                    if self.input_preprocessing == "norm":
                        partition["train_inputs"], partition["test_inputs"] = (
                            self._normalize_data(
                                partition["train_inputs"], partition["test_inputs"]
                            )
                        )
                    elif self.input_preprocessing == "std":
                        partition["train_inputs"], partition["test_inputs"] = (
                            self._standardize_data(
                                partition["train_inputs"], partition["test_inputs"]
                            )
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
                    for metric_name in self.eval_metrics:
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

                    y_proba = None
                    if "test_outputs" in partition and hasattr(
                        optimal_estimator.best_estimator_, "predict_proba"
                    ):
                        y_proba = optimal_estimator.best_estimator_.predict_proba(
                            partition["test_inputs"]
                        )

                    # Saving the results for this partition
                    self._results.save(
                        ExperimentResult(
                            dataset_name=dataset_name,
                            classifier_name=conf_name,
                            resample_id=part_idx,
                            train_predicted_y=train_predicted_y,
                            test_predicted_y=test_predicted_y,
                            y_proba=y_proba,
                            train_metrics=train_metrics,
                            test_metrics=test_metrics,
                            best_params=optimal_estimator.best_params_,
                            best_model=optimal_estimator.best_estimator_,
                            train_true_y=partition["train_outputs"],
                            test_true_y=partition.get("test_outputs"),
                        )
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
            List of partitions found inside a dataset folder. Each partition is
            stored into a dictionary, disjoining train and test inputs and
            outputs.

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
        base_path = Path(self.data_path)

        # Check if home path is shortened
        if str(base_path).startswith("~"):
            base_path = Path.home() / str(base_path)[1:]

        dataset_list = self.datasets

        # Check if 'all' is the only value, and if it is, expand it
        if len(dataset_list) == 1 and dataset_list[0] == "all":
            dataset_list = [item.name for item in base_path.iterdir() if item.is_dir()]

        elif not all(isinstance(item, str) for item in dataset_list):
            raise ValueError("Dataset list can only contain strings")

        self.data_path = str(base_path)
        self.datasets = dataset_list

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
        hyper-parameters, that, in turn, can contain one or multiple values used
        to optimize the resulting model.

        At the end of cross-validation phase, the model with the specific
        combination of values from the hyper-parameters that achieved the best
        metrics from all the combinations will remain.

        Parameters
        ----------
        train_inputs : {array-like, sparse-matrix} of shape (n_samples, n_features)
            Vector of features for each sample for this dataset.

        train_outputs : array-like of shape (n_samples)
            Target vector relative to train_inputs.

        classifier_name : str
            Name of the classification algorithm being employed.

        parameters : dict
            Dictionary containing parameters to optimize as keys, and the list
            of values that we want to compare as values.

        Returns
        -------
        optimal : GridSearchCV object or classifier object
            An already fitted model of the given classifier, with the best found
            parameters after cross-validation. If cross-validation is not needed,
            it will return the classifier model already trained.

        Raises
        ------
        ValueError
            If the classifier name is unknown or a hyper-parameter is invalid.

        """
        estimator = load_classifier(
            classifier_name=classifier_name,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv_n_folds=self.cv,
            cv_metric=self.tuning_metric,
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

        for split in ("train", "test"):
            try:
                self._results.save_summary(split=split)
            except ValueError:
                pass
