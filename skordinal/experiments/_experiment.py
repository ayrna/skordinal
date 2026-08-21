"""Experiment runner for a single classifier configuration."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from time import perf_counter
from typing import Any

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from skordinal.metrics import get_ordinal_scorer
from skordinal.metrics._metrics import _LABEL_METRICS

from ._model_config import ModelConfig
from ._results import ExperimentResult


def _compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute a single ordinal metric by name."""
    key = metric_name.strip()
    if key not in _LABEL_METRICS:
        raise ValueError(
            f"Unknown metric name: {metric_name!r}. Available: {sorted(_LABEL_METRICS)}."
        )
    return _LABEL_METRICS[key](y_true, y_pred)


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
        labels, which excludes ``ranked_probability_score``.

    tuning_metric : str, default="neg_mean_absolute_error"
        Metric used as the cross-validation scoring criterion when selecting
        the best hyper-parameter combination. Must be recognised by
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
        only, then applied to the train split and, when present, the test
        split. ``None`` means no preprocessing.

    random_state : int or None, default=None
        Seed used for two sources of randomness: the base estimator and the
        cross-validation splitter (``StratifiedKFold``) used during
        hyper-parameter search. When ``None``, both use their own default
        random behaviour.

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
        tuning_metric: str = "neg_mean_absolute_error",
        cv: int = 3,
        n_jobs: int = 1,
        input_preprocessing: str | None = None,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(model, ModelConfig):
            raise TypeError(
                f"'model' must be a ModelConfig instance; got {type(model).__name__!r}."
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

        self.model = model
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

        """
        # Apply preprocessing on local copies so the caller's arrays are not mutated.
        train_inputs: np.ndarray = X_train
        test_inputs: np.ndarray | None = X_test

        if self.input_preprocessing in {"norm", "std"}:
            scaler_cls = (
                preprocessing.MinMaxScaler
                if self.input_preprocessing == "norm"
                else preprocessing.StandardScaler
            )
            scaler = scaler_cls().fit(train_inputs)
            train_inputs = scaler.transform(train_inputs)
            # A train-only run has no test split to transform
            if X_test is not None:
                test_inputs = scaler.transform(X_test)

        # Select and fit the best estimator, keeping the refit metadata in
        # locals so nothing is ever injected onto the estimator itself
        base = self.model.build(self.random_state)
        if self.model.needs_search:
            scorer = (
                get_ordinal_scorer(self.tuning_metric)
                if isinstance(self.tuning_metric, str)
                else self.tuning_metric
            )
            splitter = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            search = GridSearchCV(
                base,
                param_grid=self.model.param_grid,
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

        # Predict on the training split.
        train_predicted_y = best_estimator.predict(train_inputs)

        # Predict on the test split when it is present.
        test_predicted_y = None
        elapsed = np.nan
        if y_test is not None:
            assert test_inputs is not None
            start = perf_counter()
            test_predicted_y = np.asarray(best_estimator.predict(test_inputs))
            elapsed = perf_counter() - start

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

        # Assemble timing keys, the cv_* pair stays NaN unless a search ran
        train_metrics["cv_time_train"] = cv_time_train
        test_metrics["cv_time_test"] = cv_time_test
        train_metrics["time_train"] = refit_time
        test_metrics["time_test"] = elapsed

        # Compute class probabilities on each split when supported
        train_y_proba = _predict_proba_or_none(best_estimator, train_inputs)
        y_proba = None
        if y_test is not None:
            assert test_inputs is not None
            y_proba = _predict_proba_or_none(best_estimator, test_inputs)

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
            best_params=best_params,
            best_model=best_estimator,
            train_true_y=y_train,
            test_true_y=y_test,
            train_index=train_index,
            test_index=test_index,
            train_y_proba=train_y_proba,
        )
