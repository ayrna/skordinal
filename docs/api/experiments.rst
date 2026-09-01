Experiments
===========

The experiments module provides three coordinated concepts for running
ordinal classification studies at any scale, from a single evaluation to a
full multi-classifier × multi-dataset benchmark.

**ModelConfig — method configuration**
  :class:`~skordinal.experiments.ModelConfig` pairs a classifier instance
  with an optional hyperparameter grid.  It is immutable and carries no
  evaluation protocol::

      from skordinal.classifiers import POM, SVOREX
      from skordinal.experiments import ModelConfig

      pom = ModelConfig(POM())
      svorex = ModelConfig(SVOREX(), param_grid={"C": [0.1, 1, 10]})

**Experiment — single execution**
  :class:`~skordinal.experiments.Experiment` runs one
  :class:`~skordinal.experiments.ModelConfig` on a single dataset partition.
  The evaluation protocol (``eval_metrics``, ``tuning_metric``, ``cv``,
  ``input_preprocessing``, ``random_state``) is fixed at construction, while
  the data split and the labels identifying it are passed to
  :meth:`~skordinal.experiments.Experiment.run`::

      from skordinal.experiments import Experiment

      exp = Experiment(
          pom,
          eval_metrics=["mean_absolute_error", "weighted_kappa"],
          tuning_metric="neg_mean_absolute_error",
          cv=5,
          random_state=0,
      )
      result = exp.run(
          X_train,
          y_train,
          X_test,
          y_test,
          dataset_name="era",
          classifier_name="POM",
          resample_id=0,
      )

  ``result`` is an :class:`~skordinal.experiments.ExperimentResult` dataclass
  containing predictions, metric scores, best hyperparameters, and timing.
  Nothing is written to disk.

**Benchmark — the cross product**
  :class:`~skordinal.experiments.Benchmark` holds a labelled set of
  :class:`~skordinal.experiments.ModelConfig` objects and one shared
  evaluation protocol.  It builds and runs an
  :class:`~skordinal.experiments.Experiment` for every
  ``(model, dataset, resample)`` cell, collecting all
  :class:`~skordinal.experiments.ExperimentResult` values and writing them
  to disk::

      from skordinal.experiments import Benchmark

      bench = Benchmark(
          models={"POM": pom, "SVOREX": svorex},
          datasets=["era", "esl"],
          eval_metrics=["mean_absolute_error", "weighted_kappa"],
          results_path="./results/",
          resamples=30,
          tuning_metric="neg_mean_absolute_error",
          cv=5,
          random_state=0,
      )
      bench.run()
      bench.summarize()

  Resamples already saved are skipped unless ``overwrite=True``, so a run is
  resumable.  :meth:`~skordinal.experiments.Benchmark.summarize` writes the
  aggregated train and test summaries.

  The quickest way to run a study is via the recipe interface.  Define a
  ``RECIPE`` dict in a standalone Python file (see ``examples/recipes/`` for
  templates) and run it from the command line::

      python examples/run_recipe.py examples/recipes/full_demo.py

  :meth:`~skordinal.experiments.Benchmark.from_recipe` reads the recipe and
  constructs the :class:`~skordinal.experiments.Benchmark` automatically:
  ``models`` becomes the positional argument and the remaining keys are
  forwarded as keyword arguments.

.. currentmodule:: skordinal.experiments

.. autosummary::
   :toctree: generated/

   ModelConfig
   Experiment
   ExperimentResult
   Benchmark
   Results
   load_recipe
   validate_recipe
   summarize
   save_summary
   tabulate_results
