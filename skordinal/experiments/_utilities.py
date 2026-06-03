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
from sklearn.model_selection import GridSearchCV

from skordinal.model_selection import load_classifier

from ._results import ExperimentResult, Results


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from skordinal.metrics import get_ordinal_scorer

    scorer = cast(Any, get_ordinal_scorer(metric_name.strip()))
    return scorer._score_func(y_true, y_pred, **scorer._kwargs)


def _read_file(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV partition file into ``(inputs, outputs)`` arrays."""
    # Detect the separator automatically.
    f = pd.read_csv(filename, header=None, engine="python")

    inputs = f.values[:, 0:(-1)]
    outputs = f.values[:, (-1)]

    return inputs, outputs


def _check_dataset_list(
    data_path: str | Path, datasets: list[str]
) -> tuple[str | Path, list[str]]:
    """Resolve a dataset list, expanding ``["all"]`` and validating entries.

    Expands a home-shorthand ``data_path`` and raises ``ValueError`` if the
    list contains non-string entries.
    """
    base_path = Path(data_path)

    # Check if home path is shortened
    if str(base_path).startswith("~"):
        base_path = Path.home() / str(base_path)[1:]

    dataset_list = datasets

    # Check if 'all' is the only value, and if it is, expand it
    if len(dataset_list) == 1 and dataset_list[0] == "all":
        dataset_list = [item.name for item in base_path.iterdir() if item.is_dir()]

    elif not all(isinstance(item, str) for item in dataset_list):
        raise ValueError("Dataset list can only contain strings")

    return str(base_path), dataset_list


def _load_dataset(dataset_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load a dataset folder into a sorted list of ``(index, partition)`` tuples.

    Each partition dict disjoins train/test inputs and outputs. Raises
    ``ValueError`` if the folder is missing and ``RuntimeError`` if a partition
    has no train file.
    """

    def get_partition_index(filename: str) -> str:
        # Extract the index between the last "_" and ".csv".
        return filename.rsplit("_", 1)[-1].replace(".csv", "")

    try:
        partition_list: dict[str, dict[str, Any]] = {
            get_partition_index(filename.name): {}
            for filename in dataset_path.iterdir()
            if filename.name.startswith("train_")
        }

        # Load train and test arrays for each partition.
        for filename in dataset_path.iterdir():
            if filename.name.startswith("train_"):
                idx = get_partition_index(filename.name)
                train_inputs, train_outputs = _read_file(filename)
                partition_list[idx]["train_inputs"] = train_inputs
                partition_list[idx]["train_outputs"] = train_outputs

            elif filename.name.startswith("test_"):
                idx = get_partition_index(filename.name)
                test_inputs, test_outputs = _read_file(filename)
                partition_list[idx]["test_inputs"] = test_inputs
                partition_list[idx]["test_outputs"] = test_outputs

    except OSError:
        raise ValueError(f"No such file or directory: '{dataset_path}'")

    except KeyError:
        raise RuntimeError(
            f"Found partition without train files: partition {filename.name}"
        )

    # Sort partitions into a list of (index, partition) tuples.
    sorted_list: list[tuple[str, dict[str, Any]]] = sorted(
        partition_list.items(),
        key=lambda t: int(t[0]) if t[0].lstrip("-").isdigit() else t[0],
    )

    return sorted_list


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

        self.data_path, self.datasets = _check_dataset_list(
            self.data_path, self.datasets
        )

        if self.verbose:
            print("\n###############################")
            print("\tRunning Experiment")
            print("###############################")

        # Iterate over datasets.
        for x in self.datasets:
            dataset_name = x.strip()
            dataset_path = Path(self.data_path) / dataset_name

            dataset = _load_dataset(dataset_path)

            if self.verbose:
                print("\nRunning", dataset_name, "dataset")
                print("--------------------------")

            # Iterate over configurations.
            for conf_name, configuration in self.configurations.items():
                if self.verbose:
                    print("Running", conf_name, "...")

                # Iterate over partitions.
                for part_idx, partition in dataset:
                    if self.verbose:
                        print("  Running Partition", part_idx)

                    result = self._run_single(
                        partition["train_inputs"],
                        partition["train_outputs"],
                        partition.get("test_inputs"),
                        partition.get("test_outputs"),
                        configuration,
                        dataset_name=dataset_name,
                        conf_name=conf_name,
                        resample_id=part_idx,
                    )
                    self._results.save(result)

    def _run_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        configuration: dict[str, Any],
        *,
        dataset_name: str,
        conf_name: str,
        resample_id: str,
    ) -> ExperimentResult:
        """Run one classifier configuration on a single train/test partition.

        This is the single-partition execution unit. It applies optional
        preprocessing, selects and fits the best estimator, predicts on train
        and (when present) test splits, computes all evaluation metrics and
        timing keys, and returns an :class:`ExperimentResult`. It does **not**
        persist anything to disk; the caller is responsible for calling
        ``self._results.save(result)`` after this method returns.

        Parameters
        ----------
        X_train : ndarray of shape (n_train_samples, n_features)
            Training feature matrix.

        y_train : ndarray of shape (n_train_samples,)
            Training labels.

        X_test : ndarray of shape (n_test_samples, n_features) or None
            Test feature matrix. When ``None`` no test metrics are computed.

        y_test : ndarray of shape (n_test_samples,) or None
            Test labels. When ``None`` no test metrics are computed.

        configuration : dict
            Single configuration entry with the form
            ``{"classifier": str, "parameters": dict}``.

        dataset_name : str
            Name of the dataset, forwarded to the returned
            :class:`ExperimentResult`.

        conf_name : str
            Configuration label, used as ``classifier_name`` in the returned
            :class:`ExperimentResult`.

        resample_id : str
            Partition index string, forwarded to the returned
            :class:`ExperimentResult`.

        Returns
        -------
        ExperimentResult
            Fully populated result for this partition. No side effects.

        """
        # Apply preprocessing on local copies so the caller's arrays are not mutated.
        train_inputs: np.ndarray = X_train
        test_inputs: np.ndarray | None = X_test

        if self.input_preprocessing == "norm":
            scaler = preprocessing.MinMaxScaler().fit(train_inputs)
            train_inputs = scaler.transform(train_inputs)
            test_inputs = scaler.transform(cast(np.ndarray, X_test))
        elif self.input_preprocessing == "std":
            scaler = preprocessing.StandardScaler().fit(train_inputs)
            train_inputs = scaler.transform(train_inputs)
            test_inputs = scaler.transform(cast(np.ndarray, X_test))

        # Select and fit the best estimator via GridSearchCV or direct fit.
        optimal_estimator: Any = load_classifier(
            classifier_name=configuration["classifier"],
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv_n_folds=self.cv,
            cv_metric=self.tuning_metric,
            param_grid=configuration["parameters"],
        )

        _fit_start = time()
        optimal_estimator.fit(train_inputs, y_train)
        _fit_elapsed = time() - _fit_start

        if not isinstance(optimal_estimator, GridSearchCV):
            optimal_estimator.refit_time_ = _fit_elapsed
            optimal_estimator.best_params_ = configuration["parameters"]
            optimal_estimator.best_estimator_ = optimal_estimator

        # Predict on the training split.
        train_predicted_y = optimal_estimator.predict(train_inputs)

        # Predict on the test split when it is present.
        test_predicted_y = None
        elapsed = np.nan
        if y_test is not None:
            assert test_inputs is not None
            start = time()
            test_predicted_y = np.asarray(optimal_estimator.predict(test_inputs))
            elapsed = time() - start

        # Compute evaluation metrics for both splits.
        train_metrics: OrderedDict[str, Any] = OrderedDict()
        test_metrics: OrderedDict[str, Any] = OrderedDict()
        for metric_name in self.eval_metrics:
            train_score = _compute_metric(
                metric_name,
                y_train,
                train_predicted_y,
            )
            train_metrics[metric_name.strip() + "_train"] = train_score

            test_metrics[metric_name.strip() + "_test"] = np.nan
            if y_test is not None:
                assert test_predicted_y is not None
                test_score = _compute_metric(metric_name, y_test, test_predicted_y)
                test_metrics[metric_name.strip() + "_test"] = test_score

        # Assemble timing keys (GridSearchCV vs direct-fit branches).
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

        # Compute class probabilities when available.
        y_proba = None
        if y_test is not None and hasattr(
            optimal_estimator.best_estimator_, "predict_proba"
        ):
            assert test_inputs is not None
            y_proba = optimal_estimator.best_estimator_.predict_proba(test_inputs)

        # Build and return the ExperimentResult; no persistence here.
        return ExperimentResult(
            dataset_name=dataset_name,
            classifier_name=conf_name,
            resample_id=resample_id,
            train_predicted_y=train_predicted_y,
            test_predicted_y=test_predicted_y,
            y_proba=y_proba,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            best_params=optimal_estimator.best_params_,
            best_model=optimal_estimator.best_estimator_,
            train_true_y=y_train,
            test_true_y=y_test,
        )

    def write_report(self) -> None:
        """Save summarized information about experiment through Results class."""
        if self.verbose:
            print("\nSaving Results...")

        for split in ("train", "test"):
            try:
                self._results.save_summary(split=split)
            except ValueError:
                pass
