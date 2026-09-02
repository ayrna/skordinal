"""Experiment runner for a single classifier configuration."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from skordinal.metrics import get_ordinal_scorer

from ._base import (
    _check_input_preprocessing,
    _check_metric_names,
    _check_tuning_metric,
    _compute_metric,
    _set_nested_random_state,
)
from ._model_config import ModelConfig
from ._results import ExperimentResult


class Experiment:
    """Run a single classifier configuration on one train/test partition.

    Wraps one ``ModelConfig`` together with the cross-validation and
    preprocessing settings shared across partitions. Calling ``run`` applies
    optional preprocessing, selects and fits the best estimator, predicts on
    the train and (when present) test splits, computes all evaluation metrics
    and timing keys, and returns an ``ExperimentResult``. Nothing is written
    to disk.

    Parameters
    ----------
    model : ModelConfig
        Bound estimator and optional hyper-parameter grid describing what to
        run. When ``model.needs_search`` is ``True`` a ``GridSearchCV`` is
        constructed; otherwise the estimator is fitted directly using any
        fixed parameters from ``model.fixed_params()``.

    eval_metrics : list of str
        Metric names to compute for every partition (e.g.
        ``["mean_absolute_error", "average_mean_absolute_error"]``). Names
        must match a ``skordinal.metrics`` metric that scores predicted
        labels, which excludes ``ranked_probability_score``. Each name is
        stripped and resolved at construction, so a typo fails before any
        fitting starts.

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
        ``random_state`` for each partition. Fitted on the training split
        only, then applied to the train split and, when present, the test
        split. ``None`` means no preprocessing.

    random_state : int or None, default=None
        Seed forwarded to the base estimator, to the ``StratifiedKFold``
        splitter of a hyper-parameter search, and to the
        ``input_preprocessing`` clone. When ``None``, each keeps its own
        default random behaviour.

    Raises
    ------
    TypeError
        If ``model`` is not a ``ModelConfig``, if ``eval_metrics`` is a
        bare string, not iterable, or holds a non-string name, or if
        ``input_preprocessing`` is neither ``None`` nor a transformer
        instance.

    ValueError
        If ``eval_metrics`` is empty or holds an unregistered label-metric
        name, or if a string ``tuning_metric`` is not a registered scorer name.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from skordinal.experiments import Experiment, ModelConfig  # doctest: +SKIP
    >>> exp = Experiment(  # doctest: +SKIP
    ...     ModelConfig(SVC(), param_grid={"C": [0.1, 1.0]}),
    ...     eval_metrics=["mean_absolute_error"],
    ... )
    >>> result = exp.run(  # doctest: +SKIP
    ...     X_train, y_train, X_test, y_test,
    ...     dataset_name="balance-scale",
    ...     classifier_name="SVM",
    ...     resample_id=0,
    ... )
    """

    def __init__(
        self,
        model: ModelConfig,
        *,
        eval_metrics: list[str],
        tuning_metric: str | Callable[..., float] | None = "neg_mean_absolute_error",
        cv: int = 3,
        n_jobs: int = 1,
        input_preprocessing: BaseEstimator | None = None,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(model, ModelConfig):
            raise TypeError(
                f"'model' must be a ModelConfig instance; got {type(model).__name__!r}."
            )
        eval_metrics = _check_metric_names(eval_metrics, param="eval_metrics")
        _check_tuning_metric(tuning_metric)
        _check_input_preprocessing(input_preprocessing)

        self.model = model
        self.eval_metrics: list[str] = eval_metrics
        self.tuning_metric = tuning_metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.input_preprocessing = input_preprocessing
        self.random_state = random_state

    def _build_scaler(self) -> BaseEstimator | None:
        """Return a fresh, seeded scaler for one partition, or None."""
        if self.input_preprocessing is None:
            return None
        # Never fit the caller's instance, and seed it like the estimator
        scaler = clone(self.input_preprocessing)
        _set_nested_random_state(scaler, self.random_state)
        return scaler

    @staticmethod
    def _predict_proba_or_none(estimator: Any, inputs: np.ndarray) -> np.ndarray | None:
        """Return class probabilities, or ``None`` when the estimator cannot."""
        if not hasattr(estimator, "predict_proba"):
            return None
        try:
            return estimator.predict_proba(inputs)
        except AttributeError as exc:
            warnings.warn(
                f"predict_proba raised AttributeError; probabilities are omitted: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        *,
        dataset_name: str,
        classifier_name: str,
        resample_id: int,
        train_index: np.ndarray | None = None,
        test_index: np.ndarray | None = None,
    ) -> ExperimentResult:
        """Run the configuration on a single train/test partition.

        Applies optional preprocessing, selects and fits the best estimator,
        predicts on train and (when present) test splits, computes all
        evaluation metrics and timing keys, and returns an
        ``ExperimentResult``. It does not persist anything to disk; the
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
            ``ExperimentResult``.

        classifier_name : str
            Configuration label, used as ``classifier_name`` in the returned
            ``ExperimentResult``.

        resample_id : int
            Partition index, forwarded to the returned ``ExperimentResult``.

        train_index : ndarray of shape (n_train_samples,) or None, default=None
            Zero-based positions of the training samples in the original
            dataset array; forwarded to the returned ``ExperimentResult`` and
            used as the ``Pattern ID`` column.

        test_index : ndarray of shape (n_test_samples,) or None, default=None
            Zero-based positions of the test samples in the original dataset
            array; forwarded to the returned ``ExperimentResult`` and used as
            the ``Pattern ID`` column.

        Returns
        -------
        ExperimentResult
            Fully populated result for this partition. No side effects.

        Raises
        ------
        ValueError
            If ``y_test`` is given without ``X_test``.
        """
        if y_test is not None and X_test is None:
            raise ValueError("'y_test' was given without 'X_test'.")

        # Preprocess into locals so the caller's arrays are never mutated
        train_inputs: np.ndarray = X_train
        test_inputs: np.ndarray | None = X_test
        scaler = self._build_scaler()

        if scaler is not None:
            train_inputs = scaler.fit(train_inputs).transform(train_inputs)
            # A train-only run has no test split to transform
            if X_test is not None:
                test_inputs = scaler.transform(X_test)

        # Keep the refit metadata in locals, never on the estimator itself
        base = self.model.build(self.random_state)
        if self.model.needs_search:
            scorer = get_ordinal_scorer(self.tuning_metric)
            splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            search = GridSearchCV(
                base,
                param_grid=self.model.search_grid(),
                scoring=scorer,
                n_jobs=self.n_jobs,
                cv=splitter,
                error_score="raise",
            )
            search.fit(train_inputs, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            refit_time = search.refit_time_
            cv_time_train = search.cv_results_["mean_fit_time"].mean()
            cv_time_test = search.cv_results_["mean_score_time"].mean()
        else:
            if self.model.param_grid:
                base.set_params(**self.model.fixed_params())
            fit_start = perf_counter()
            base.fit(train_inputs, y_train)
            refit_time = perf_counter() - fit_start
            best_estimator = base
            best_params = self.model.fixed_params()
            cv_time_train = np.nan
            cv_time_test = np.nan

        train_predicted_y = best_estimator.predict(train_inputs)

        test_predicted_y = None
        elapsed = np.nan
        if y_test is not None:
            assert test_inputs is not None
            start = perf_counter()
            test_predicted_y = np.asarray(best_estimator.predict(test_inputs))
            elapsed = perf_counter() - start

        # The fitted scale, so a split missing a class keeps its real gaps
        classes = getattr(best_estimator, "classes_", None)
        train_metrics: OrderedDict[str, Any] = OrderedDict()
        test_metrics: OrderedDict[str, Any] = OrderedDict()
        for metric_name in self.eval_metrics:
            train_score = _compute_metric(
                metric_name, y_train, train_predicted_y, labels=classes
            )
            train_metrics[metric_name + "_train"] = train_score

            test_metrics[metric_name + "_test"] = np.nan
            if y_test is not None:
                assert test_predicted_y is not None
                test_score = _compute_metric(
                    metric_name, y_test, test_predicted_y, labels=classes
                )
                test_metrics[metric_name + "_test"] = test_score

        # Assemble timing keys, the cv_* pair stays NaN unless a search ran
        train_metrics["cv_time_train"] = cv_time_train
        test_metrics["cv_time_test"] = cv_time_test
        train_metrics["time_train"] = refit_time
        test_metrics["time_test"] = elapsed

        train_y_proba = self._predict_proba_or_none(best_estimator, train_inputs)
        y_proba = None
        if y_test is not None:
            assert test_inputs is not None
            y_proba = self._predict_proba_or_none(best_estimator, test_inputs)

        return ExperimentResult(
            dataset_name=dataset_name,
            classifier_name=classifier_name,
            resample_id=resample_id,
            train_predicted_y=train_predicted_y,
            test_predicted_y=test_predicted_y,
            y_proba=y_proba,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            best_params=best_params,
            best_model=best_estimator,
            train_true_y=y_train,
            test_true_y=y_test,
            train_index=train_index,
            test_index=test_index,
            train_y_proba=train_y_proba,
            scaler=scaler,
        )
