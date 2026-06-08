"""Experiment runner for a single classifier configuration."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from time import time
from typing import Any, cast

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from skordinal.model_selection import load_classifier

from ._results import ExperimentResult


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from skordinal.metrics import get_ordinal_scorer

    scorer = cast(Any, get_ordinal_scorer(metric_name.strip()))
    return scorer._score_func(y_true, y_pred, **scorer._kwargs)


class Experiment:
    """Run a single classifier configuration on one train/test partition.

    Wraps one configuration (a classifier name and its hyper-parameter grid)
    together with the cross-validation and preprocessing settings shared across
    partitions. Calling :meth:`run` applies optional preprocessing, selects and
    fits the best estimator via
    :func:`~skordinal.model_selection.load_classifier`, predicts on the train
    and (when present) test splits, computes all evaluation metrics and timing
    keys, and returns an :class:`ExperimentResult`. Nothing is written to disk.

    Parameters
    ----------
    configuration : dict
        Single configuration entry with the form
        ``{"classifier": str, "parameters": dict}``, where ``"classifier"`` is
        the registered name of the classification algorithm and ``"parameters"``
        is a dictionary of hyper-parameter grids (lists of values to search
        over) or fixed values. A deep copy is taken at construction time.

    eval_metrics : list of str
        Metric names to compute for every partition (e.g.
        ``["mean_absolute_error", "average_mean_absolute_error"]``). Names must
        be recognised by ``skordinal.metrics.get_ordinal_scorer``.

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

    Examples
    --------
    >>> from skordinal.experiments import Experiment  # doctest: +SKIP
    >>> exp = Experiment(  # doctest: +SKIP
    ...     {"classifier": "svc", "parameters": {"C": [1]}},
    ...     eval_metrics=["mean_absolute_error"],
    ... )
    >>> result = exp.run(  # doctest: +SKIP
    ...     X_train, y_train, X_test, y_test,
    ...     dataset_name="balance-scale",
    ...     classifier_name="SVM",
    ...     resample_id="0",
    ... )

    """

    def __init__(
        self,
        configuration: dict[str, Any],
        *,
        eval_metrics: list[str],
        tuning_metric: str = "neg_mean_absolute_error",
        cv: int = 3,
        n_jobs: int = 1,
        input_preprocessing: str | None = None,
        random_state: int | None = None,
    ) -> None:
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

        self.configuration = deepcopy(configuration)
        self.eval_metrics: list[str] = list(eval_metrics)
        self.tuning_metric = tuning_metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.input_preprocessing = input_preprocessing
        self.random_state = random_state

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        *,
        dataset_name: str,
        classifier_name: str,
        resample_id: str,
    ) -> ExperimentResult:
        """Run the configuration on a single train/test partition.

        Applies optional preprocessing, selects and fits the best estimator,
        predicts on train and (when present) test splits, computes all
        evaluation metrics and timing keys, and returns an
        :class:`ExperimentResult`. It does **not** persist anything to disk; the
        caller is responsible for saving the result.

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

        dataset_name : str
            Name of the dataset, forwarded to the returned
            :class:`ExperimentResult`.

        classifier_name : str
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
            classifier_name=self.configuration["classifier"],
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv_n_folds=self.cv,
            cv_metric=self.tuning_metric,
            param_grid=self.configuration["parameters"],
        )

        _fit_start = time()
        optimal_estimator.fit(train_inputs, y_train)
        _fit_elapsed = time() - _fit_start

        if not isinstance(optimal_estimator, GridSearchCV):
            optimal_estimator.refit_time_ = _fit_elapsed
            optimal_estimator.best_params_ = self.configuration["parameters"]
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
            optimal_estimator.best_params_ = self.configuration["parameters"]
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
            classifier_name=classifier_name,
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
