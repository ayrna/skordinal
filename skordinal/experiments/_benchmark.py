"""Benchmark runner for ordinal classification experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._experiment import Experiment
from ._results import Results


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


class Benchmark:
    """Run a benchmark of M configurations across N datasets and their partitions.

    Each configuration pairs a classifier method with one or more hyper-parameter
    values. Calling :meth:`run` performs cross-validation for every partition of
    each dataset-configuration pair, fits the selected model, predicts the test
    labels, and stores all metrics in a :class:`Results` object. :meth:`summarize`
    then writes the aggregated train and test summaries.

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
    >>> from skordinal.experiments import Benchmark  # doctest: +SKIP
    >>> benchmark = Benchmark(  # doctest: +SKIP
    ...     configurations={"SVM": {"classifier": "svc", "parameters": {"C": [1]}}},
    ...     data_path="/data/ordinal",
    ...     datasets=["balance-scale"],
    ...     eval_metrics=["mean_absolute_error"],
    ...     results_path="/tmp/results",
    ... )
    >>> benchmark.run()  # doctest: +SKIP
    >>> benchmark.summarize()  # doctest: +SKIP

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

    def run(self) -> None:
        """Run the benchmark over every dataset, configuration and partition.

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
            print("\tRunning Benchmark")
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

                experiment = Experiment(
                    configuration,
                    eval_metrics=self.eval_metrics,
                    tuning_metric=self.tuning_metric,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    input_preprocessing=self.input_preprocessing,
                    random_state=self.random_state,
                )

                # Iterate over partitions.
                for part_idx, partition in dataset:
                    if self.verbose:
                        print("  Running Partition", part_idx)

                    result = experiment.run(
                        partition["train_inputs"],
                        partition["train_outputs"],
                        partition.get("test_inputs"),
                        partition.get("test_outputs"),
                        dataset_name=dataset_name,
                        classifier_name=conf_name,
                        resample_id=part_idx,
                    )
                    self._results.save(result)

    def summarize(self) -> None:
        """Write the aggregated train and test summaries via the Results object."""
        if self.verbose:
            print("\nSaving summary...")

        for split in ("train", "test"):
            try:
                self._results.save_summary(split=split)
            except ValueError:
                pass
