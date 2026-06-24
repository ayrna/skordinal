"""Benchmark runner for ordinal classification experiments."""

from __future__ import annotations

from pathlib import Path

from skordinal.datasets import load_partitions

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
        dataset names are resolved against a direct path or the bundled
        collection via the dataset-loading layer.

    datasets : list of str
        Names of the datasets to load, resolved via the dataset-loading layer.
        Each name is passed directly to ``load_partitions``.

    eval_metrics : list of str
        Metric names to compute for every resample (e.g.
        ``["mean_absolute_error", "average_mean_absolute_error"]``).
        Names must be recognised by ``skordinal.metrics.get_ordinal_scorer``.

    results_path : str or Path
        Directory where result files are written.

    resamples : int, default=30
        Number of resamples (train/test splits) to load per dataset. Forwarded
        to ``load_partitions``.

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
        Optional feature preprocessing applied to every resample before
        fitting: ``"norm"`` applies min-max normalisation and ``"std"`` applies
        z-score standardisation. Both scalers are fitted on the training split
        only, then applied to both train and test splits. ``None`` means no
        preprocessing.

    random_state : int or None, default=None
        Seed used for two sources of randomness: the base estimator and the
        cross-validation splitter (``StratifiedKFold``) used during
        hyper-parameter search. Also forwarded to ``load_partitions`` when a
        fallback split is generated. When ``None``, both use their own default
        random behaviour.

    verbose : bool, default=True
        If ``True``, progress messages are printed to stdout.

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
        resamples: int = 30,
        tuning_metric: str = "neg_mean_absolute_error",
        cv: int = 3,
        n_jobs: int = 1,
        input_preprocessing: str | None = None,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> None:
        if not models:
            raise ValueError("'models' must be a non-empty dict; got an empty mapping.")
        _bad = [k for k, v in models.items() if not isinstance(v, ModelConfig)]
        if _bad:
            raise TypeError(
                f"All values in 'models' must be ModelConfig instances; "
                f"got non-ModelConfig value(s) for key(s): {_bad}."
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

        self.models: dict[str, ModelConfig] = dict(models)
        self.data_home: str | Path | None = data_home
        self.datasets: list[str] = list(datasets)
        self.eval_metrics: list[str] = list(eval_metrics)
        self.results_path: str | Path = results_path
        self.resamples: int = resamples
        self.tuning_metric = tuning_metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.input_preprocessing = input_preprocessing
        self.random_state = random_state
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
            If the recipe fails structural type validation.

        ValueError
            If the recipe fails structural constraint validation.
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
        information into a Results object.

        Raises
        ------
        FileNotFoundError
            If a dataset name cannot be resolved by the dataset-loading layer
            (no matching path and not present in the bundled collection).

        """
        self._results = Results(Path(self.results_path))

        if self.verbose:
            print("\n###############################")
            print("\tRunning Benchmark")
            print("###############################")

        # Iterate over datasets
        for x in self.datasets:
            dataset_name = x.strip()

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
                    random_state=self.random_state,
                ):
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
                    )
                    self._results.save(result)

    def summarize(self) -> None:
        """Write the train and test summaries to the results folder."""
        if self.verbose:
            print("\nSaving summary...")

        for split in ("train", "test"):
            try:
                save_summary(self.results_path, split=split)
            except ValueError:
                pass
