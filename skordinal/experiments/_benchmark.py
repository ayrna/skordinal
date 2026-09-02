"""Benchmark runner for ordinal classification experiments."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from skordinal.datasets import load_partitions

from ._base import (
    _check_input_preprocessing,
    _check_metric_names,
    _check_path_component,
    _check_tuning_metric,
)
from ._evaluation import save_summary
from ._experiment import Experiment
from ._model_config import ModelConfig
from ._recipes import load_recipe
from ._results import Results


class Benchmark:
    """Run a benchmark of M configurations across N datasets and their resamples.

    Each configuration pairs a classifier method with one or more hyper-parameter
    values. Calling ``run`` performs cross-validation for every resample of each
    dataset-configuration pair, fits the selected model, predicts the test
    labels, and stores all metrics in a ``Results`` object. ``summarize``
    then writes the aggregated train and test summaries.

    Parameters
    ----------
    models : dict of str to ModelConfig
        Mapping of configuration labels to their ``ModelConfig`` instances.
        Each value must be a ``ModelConfig`` binding an estimator to an
        optional hyper-parameter grid.

    data_home : str, Path, or None, default=None
        Optional base directory used to locate dataset files. When ``None``,
        dataset names are resolved against the bundled collection via the
        dataset-loading layer.

    datasets : list of str
        Names of the datasets to load, resolved via the dataset-loading layer.
        Each name is stripped and checked at construction: it also names the
        per-dataset results folder, so it must be a plain name or filename
        without path separators.

    eval_metrics : list of str
        Metric names to compute for every resample (e.g.
        ``["mean_absolute_error", "average_mean_absolute_error"]``). Names
        must match a ``skordinal.metrics`` metric that scores predicted
        labels, which excludes ``ranked_probability_score``. Each name is
        stripped and resolved at construction, so a typo fails before any
        fitting starts.

    results_path : str or Path
        Directory where result files are written. Expanded and resolved to
        an absolute path at construction, so a later working-directory
        change does not affect where results are written or read.

    resamples : int or list of int, default=30
        Resamples (train/test splits) to load per dataset, forwarded verbatim
        to ``load_partitions``: a count, or the list of resample ids to use.

    test_size : float, default=0.3
        Fraction of samples held out for testing when ``load_partitions``
        generates the splits; ignored when a masks file supplies them.

    tuning_metric : str, callable or None, default="neg_mean_absolute_error"
        Cross-validation scoring criterion used to select the best
        hyper-parameter combination. A string is resolved at construction
        with ``skordinal.metrics.get_ordinal_scorer``; a callable is passed
        to ``GridSearchCV`` verbatim, and ``None`` falls back to the
        estimator's own ``score``.

    cv : int, default=3
        Number of folds used in hyper-parameter cross-validation.

    n_jobs : int, default=1
        Number of parallel jobs forwarded to ``GridSearchCV``.

    input_preprocessing : transformer or None, default=None
        Optional preprocessing applied before fitting: a transformer
        instance, e.g. a scaler or a ``Pipeline``, cloned and seeded with
        ``random_state`` for each resample. Fitted on the training split
        only, then applied to both splits. ``None`` means no
        preprocessing.

    random_state : int or None, default=0
        Seed forwarded to the base estimator, to the ``StratifiedKFold``
        splitter of a hyper-parameter search, to the ``input_preprocessing``
        clone, and to ``load_partitions`` when it generates the splits. A
        non-integer value (including ``None``) is resolved to one concrete
        integer at construction, so every ``load_partitions`` call in ``run``
        shares the same partitioning scheme.

    overwrite : bool, default=False
        If ``False``, resamples already saved are skipped, making runs
        resumable. ``True`` recomputes every resample.

    verbose : bool, default=True
        If ``True``, progress messages are printed to stdout.

    Raises
    ------
    TypeError
        If ``models`` is not a dict or any of its values is not a
        ``ModelConfig``, if a model label or dataset name is not a str, if
        ``datasets`` or ``eval_metrics`` is a bare string, if
        ``eval_metrics`` is not iterable or holds a non-string name, or if
        ``input_preprocessing`` is neither ``None`` nor a transformer
        instance.

    ValueError
        If ``models``, ``datasets`` or ``eval_metrics`` is empty, if a model
        label or dataset name is not a usable path component, if a name in
        ``eval_metrics`` is not a registered label metric, or if a string
        ``tuning_metric`` is not a registered scorer name.

    Attributes
    ----------
    _results : Results
        Manages and stores all information obtained during the experiment run.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from skordinal.experiments import Benchmark, ModelConfig  # doctest: +SKIP
    >>> benchmark = Benchmark(  # doctest: +SKIP
    ...     models={"SVM": ModelConfig(SVC(), param_grid={"C": [0.1, 1.0]})},
    ...     data_home="/data/ordinal",
    ...     datasets=["balance_scale"],
    ...     eval_metrics=["mean_absolute_error"],
    ...     results_path="/tmp/results",
    ...     resamples=30,
    ... )
    >>> benchmark.run()  # doctest: +SKIP
    >>> benchmark.summarize()  # doctest: +SKIP
    """

    def __init__(
        self,
        models: dict[str, ModelConfig],
        *,
        data_home: str | Path | None = None,
        datasets: list[str],
        eval_metrics: list[str],
        results_path: str | Path,
        resamples: int | list[int] = 30,
        test_size: float = 0.3,
        tuning_metric: str | Callable[..., float] | None = "neg_mean_absolute_error",
        cv: int = 3,
        n_jobs: int = 1,
        input_preprocessing: BaseEstimator | None = None,
        random_state: int | None = 0,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        if not isinstance(models, dict):
            raise TypeError(
                f"'models' must be a dict of label to ModelConfig; got "
                f"{type(models).__name__}."
            )
        if not models:
            raise ValueError("'models' must be a non-empty dict; got an empty mapping.")
        _bad = [k for k, v in models.items() if not isinstance(v, ModelConfig)]
        if _bad:
            raise TypeError(
                f"All values in 'models' must be ModelConfig instances; "
                f"got non-ModelConfig value(s) for key(s): {_bad}."
            )
        if isinstance(datasets, str):
            raise TypeError(
                f"'datasets' must be an iterable of dataset names, not a bare "
                f"string; pass [{datasets!r}] to use a single dataset."
            )
        if not datasets:
            raise ValueError(
                "'datasets' must be a non-empty list; got an empty sequence."
            )
        # Both name a directory in the results tree
        for label in models:
            _check_path_component(label, "model label")
        datasets = [x.strip() if isinstance(x, str) else x for x in datasets]
        for name in datasets:
            _check_path_component(name, "dataset name")
        eval_metrics = _check_metric_names(eval_metrics, param="eval_metrics")
        _check_tuning_metric(tuning_metric)
        _check_input_preprocessing(input_preprocessing)

        self.models: dict[str, ModelConfig] = dict(models)
        self.data_home: str | Path | None = data_home
        self.datasets: list[str] = list(datasets)
        self.eval_metrics: list[str] = eval_metrics
        self._results = Results(results_path)
        # Reuse the resolved root so a later chdir cannot split write from read
        self.results_path = self._results.path
        self.resamples: int | list[int] = resamples
        self.test_size: float = test_size
        self.tuning_metric = tuning_metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.input_preprocessing = input_preprocessing
        # Collapse to one concrete int shared by every load_partitions call in run()
        if not isinstance(random_state, Integral):
            random_state = int(check_random_state(random_state).randint(2**31 - 1))
        self.random_state = random_state
        self.overwrite = overwrite
        self.verbose = verbose

    @classmethod
    def from_recipe(
        cls,
        recipe_path: str | Path,
        **overrides: object,
    ) -> "Benchmark":
        """Construct a ``Benchmark`` from a recipe ``.py`` file.

        A recipe file must define a top-level ``RECIPE`` dict whose keys
        mirror the ``Benchmark`` constructor: ``models`` becomes the
        positional argument and the remaining keys are forwarded as keyword
        arguments.  Any ``**overrides`` are merged after loading, so they
        win over recipe values.

        Parameters
        ----------
        recipe_path : str or Path
            Filesystem path to the recipe file.

        **overrides : object
            Keyword arguments that override keys in the loaded recipe.

        Returns
        -------
        benchmark : Benchmark
            A fully configured ``Benchmark`` instance ready to call
            ``run`` on.

        Raises
        ------
        FileNotFoundError
            If the recipe file does not exist.

        AttributeError
            If the recipe file does not define a top-level ``RECIPE`` dict.

        TypeError
            If the recipe fails structural type validation, or if the
            resulting configuration fails ``Benchmark.__init__``'s validation.

        ValueError
            If the recipe fails structural constraint validation, or if the
            resulting configuration fails ``Benchmark.__init__``'s validation.
        """
        recipe = dict(load_recipe(recipe_path))
        recipe.update(overrides)
        models = recipe.pop("models")
        return cls(models, **recipe)

    def run(self) -> None:
        """Run the benchmark over every dataset, configuration and resample.

        Loads all datasets via the dataset-loading layer, one resample at a
        time. Builds a model per resample, using cross-validation to find the
        optimal values among the hyper-parameters to compare from.

        Uses the built model to get train and test metrics, storing all the
        information into a Results object. Resamples already saved are
        skipped unless ``overwrite`` is ``True``.

        Raises
        ------
        FileNotFoundError
            If a dataset name cannot be resolved by the dataset-loading layer
            (no matching path and not present in the bundled collection).
        """
        if self.verbose:
            print("\n###############################")
            print("\tRunning Benchmark")
            print("###############################")

        # Iterate over datasets
        for dataset_name in self.datasets:
            if self.verbose:
                print("\nRunning", dataset_name, "dataset")
                print("--------------------------")

            # Iterate over configurations
            for label, model in self.models.items():
                if self.verbose:
                    print("Running", label, "...")

                experiment = Experiment(
                    model,
                    eval_metrics=self.eval_metrics,
                    tuning_metric=self.tuning_metric,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    input_preprocessing=self.input_preprocessing,
                    random_state=self.random_state,
                )

                # Iterate over resamples via the dataset-loading layer
                for b in load_partitions(
                    dataset_name,
                    data_home=self.data_home,
                    resamples=self.resamples,
                    test_size=self.test_size,
                    random_state=self.random_state,
                ):
                    if not self.overwrite and self._results.exists(
                        label, dataset_name, b.resample_id
                    ):
                        if self.verbose:
                            print("  Skipping resample", b.resample_id)
                        continue

                    if self.verbose:
                        print("  Running resample", b.resample_id)

                    result = experiment.run(
                        b.data_train,
                        b.target_train,
                        b.data_test,
                        b.target_test,
                        dataset_name=dataset_name,
                        classifier_name=label,
                        resample_id=b.resample_id,
                        train_index=b.train_index,
                        test_index=b.test_index,
                    )
                    self._results.save(result)

    def summarize(self) -> None:
        """Write the train and test summaries to the results folder."""
        if self.verbose:
            print("\nSaving summary...")

        for split in ("train", "test"):
            try:
                save_summary(self.results_path, split=split)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # A corrupt report.csv is data loss, never "nothing to do"
                raise
            except (FileNotFoundError, ValueError):
                # Nothing run yet (the folder may not exist), or no pair reported
                if self.verbose:
                    print("  No metrics to summarise for the", split, "split")
