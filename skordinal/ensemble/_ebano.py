"""EBANO (Ensemble BAsed on uNimodal Ordinal classifiers)."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, check_cv, cross_val_predict
from sklearn.utils import Bunch, check_random_state, get_tags
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import _num_samples, check_is_fitted, column_or_1d

from skordinal.metrics import get_ordinal_scorer, get_ordinal_scorer_names
from skordinal.utils.validation import check_ordinal_targets


class _ProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Classifier adapter for scoring a precomputed probability matrix."""

    def __init__(self, classes_: NDArray, probabilities_: NDArray) -> None:
        self.classes_ = classes_
        self.probabilities_ = probabilities_

    def predict_proba(self, X: ArrayLike) -> NDArray:
        return self.probabilities_

    def predict(self, X: ArrayLike) -> NDArray:
        return self.classes_[np.argmax(self.probabilities_, axis=1)]


class EBANO(ClassifierMixin, _BaseComposition):
    """Ordinal probability ensemble with cross-validated weight selection.

    Each estimator's out-of-fold probabilities are combined using random
    non-negative weights that sum to one. The combination with the highest
    ``scoring`` value is retained, then the base estimators are fitted on all
    the training data.

    Already-fitted estimators can be wrapped in
    :class:`sklearn.frozen.FrozenEstimator`, which ``clone`` keeps fitted, or
    passed bare with ``cv="prefit"``, which uses them as fitted.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Named classifiers implementing ``fit`` and ``predict_proba``. An
        estimator can be set to ``"drop"`` to exclude it.
    n_iter : int, default=1000
        Number of random weight vectors to evaluate.
    cv : int, cross-validation generator, iterable, "prefit" or None, default=None
        Determines the cross-validation splitting strategy used in
        ``cross_val_predict`` to choose the weights. ``None`` uses 5-fold
        stratified cross-validation. ``"prefit"`` assumes that the
        ``estimators`` are already fitted and chooses the weights on ``X``.
    scoring : str or callable, default="neg_average_mean_absolute_error"
        Scorer used to select the weights. Scorers
        registered by :mod:`skordinal.metrics` are also supported.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel when computing cross-validated
        probabilities.
    random_state : int, RandomState instance or None, default=None
        Controls random weight generation.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The sorted class labels observed during fitting.
    estimators_ : list of fitted estimators
        The elements of the ``estimators`` parameter, having been fitted on the
        training data. If an estimator has been set to ``"drop"``, it will not
        appear in ``estimators_``. When ``cv="prefit"``, ``estimators_`` is set
        to ``estimators`` and is not fitted again.
    named_estimators_ : Bunch
        Fitted estimators accessed by their names. Dropped estimators appear
        as ``"drop"``.
    weights_ : ndarray of shape (n_estimators,)
        Weights selected by cross-validation.
    best_score_ : float
        Score achieved by ``weights_`` on the out-of-fold predictions.
    n_features_in_ : int
        Number of features seen during fit. Only defined if the underlying
        estimator exposes such an attribute when fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Only defined if the underlying
        estimators expose such an attribute when fit.

    Notes
    -----
    ``cv`` must be a partition of the data, because it is passed to
    :func:`sklearn.model_selection.cross_val_predict`. Splitters such as
    :class:`sklearn.model_selection.ShuffleSplit` are therefore not supported.

    With ``cv="prefit"`` the weights are selected on the probabilities the
    fitted estimators give for the ``X`` and ``y`` passed to :meth:`fit`. If
    the estimators were trained on those same samples, the probabilities are
    optimistic and the weights will be biased towards the estimators that
    overfit most. Pass data the estimators have not seen (e.g. a validation
    set) unless you intentionally want that behaviour.

    ``X`` is not validated or converted. The base estimators validate it, so
    text, DataFrames, sparse matrices or missing values are accepted whenever
    the base estimators accept them.

    Examples
    --------
    >>> from skordinal.ensemble import EBANO
    >>> from skordinal.linear_model import LogisticAT, LogisticIT
    >>> model = EBANO([("at", LogisticAT()), ("it", LogisticIT())], random_state=0)
    """

    _parameter_constraints: dict = {
        "estimators": [list],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "scoring": [str, callable],
        "n_jobs": [Integral, None],
        "random_state": ["random_state", None],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        *,
        n_iter: int = 1000,
        cv: int | str | BaseCrossValidator | Iterable | None = None,
        scoring: str = "neg_average_mean_absolute_error",
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
    ) -> None:
        self.estimators = estimators
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep: bool = True) -> dict:
        return self._get_params("estimators", deep=deep)

    def set_params(self, **params) -> EBANO:
        return self._set_params("estimators", **params)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        try:
            active = [
                e
                for _, e in self.estimators
                if not (isinstance(e, str) and e == "drop")
            ]
            base_tags = [get_tags(e) for e in active]
            tags.input_tags.sparse = all(t.input_tags.sparse for t in base_tags)
            tags.input_tags.allow_nan = all(t.input_tags.allow_nan for t in base_tags)
        except (TypeError, ValueError, AttributeError):
            # Malformed estimators: fit() reports it with a clear message.
            pass
        return tags

    def _validate_estimators(self) -> list[tuple[str, BaseEstimator]]:
        """Return the active (non-dropped) ``(name, estimator)`` pairs."""
        if not isinstance(self.estimators, list) or not self.estimators:
            raise ValueError(
                "'estimators' must be a non-empty list of (name, estimator) tuples."
            )
        if not all(
            isinstance(item, (tuple, list)) and len(item) == 2
            for item in self.estimators
        ):
            raise ValueError(
                "'estimators' must be a non-empty list of (name, estimator) tuples."
            )

        names, estimators = zip(*self.estimators)
        self._validate_names(names)

        active = [
            (name, estimator)
            for name, estimator in self.estimators
            if not (isinstance(estimator, str) and estimator == "drop")
        ]
        if not active:
            raise ValueError("At least one estimator must not be dropped.")
        for _, estimator in active:
            if (
                not isinstance(estimator, BaseEstimator)
                or not is_classifier(estimator)
                or not hasattr(estimator, "predict_proba")
            ):
                raise ValueError(
                    f"{type(estimator).__name__} must be a classifier that "
                    "implements predict_proba."
                )
        return active

    def _get_scorer(self):
        if isinstance(self.scoring, str) and self.scoring in get_ordinal_scorer_names():
            return get_ordinal_scorer(self.scoring)
        return check_scoring(self, scoring=self.scoring)

    @staticmethod
    def _check_classes(
        names: list[str], estimators: list, *expected_classes: NDArray
    ) -> None:
        """Raise if a fitted estimator predicts none of the expected class sets."""
        for name, estimator in zip(names, estimators):
            estimator_classes = np.asarray(estimator.classes_)
            if not any(np.array_equal(estimator_classes, c) for c in expected_classes):
                raise ValueError(
                    f"Estimator {name!r} has classes {estimator_classes.tolist()}, "
                    f"but EBANO expects {expected_classes[0].tolist()}."
                )

    @staticmethod
    def _validate_probabilities(
        probabilities: ArrayLike, n_samples: int, n_classes: int, name: str
    ) -> NDArray:
        probabilities = np.asarray(probabilities)
        if probabilities.shape != (n_samples, n_classes):
            raise ValueError(
                f"Estimator {name!r} returned probabilities with shape "
                f"{probabilities.shape}; expected {(n_samples, n_classes)}."
            )
        return probabilities

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: ArrayLike, y: ArrayLike) -> EBANO:
        """Fit the ensemble and select its probability-blending weights."""
        y = column_or_1d(y, warn=True)
        self.classes_, y_encoded = check_ordinal_targets(y)
        y_scoring = self.classes_[y_encoded]
        active = self._validate_estimators()
        n_samples, n_classes = _num_samples(X), len(self.classes_)

        names = [name for name, _ in active]
        prefit = isinstance(self.cv, str) and self.cv == "prefit"
        if prefit:
            fitted_estimators = [estimator for _, estimator in active]
            for estimator in fitted_estimators:
                check_is_fitted(estimator)
            # An EBANO's own estimators_ were fitted on the class positions
            self._check_classes(
                names, fitted_estimators, self.classes_, np.arange(n_classes)
            )
        else:
            cv = deepcopy(check_cv(self.cv, y_encoded, classifier=True))
            # Seed the copy so that every estimator gets the same folds
            if hasattr(cv, "random_state") and cv.random_state is None:
                cv.random_state = np.random.RandomState()
        oof_probabilities = []
        for name, estimator in active:
            if prefit:
                probabilities = estimator.predict_proba(X)
            else:
                probabilities = cross_val_predict(
                    estimator,
                    X,
                    y_encoded,
                    cv=deepcopy(cv),
                    method="predict_proba",
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                )
            oof_probabilities.append(
                self._validate_probabilities(probabilities, n_samples, n_classes, name)
            )
        stacked = np.stack(oof_probabilities)  # (n_estimators, n_samples, n_classes)

        rng = check_random_state(self.random_state)
        scorer = self._get_scorer()
        best_score = -np.inf
        best_weights: NDArray

        for _ in range(self.n_iter):
            weights = rng.uniform(size=len(active))
            weights /= weights.sum()
            combined = np.tensordot(weights, stacked, axes=1)
            score = float(
                scorer(_ProbabilityClassifier(self.classes_, combined), X, y_scoring)
            )
            if not np.isfinite(score):
                raise ValueError(
                    f"The scoring function returned a non-finite value: {score}."
                )
            if score > best_score:
                best_score = score
                best_weights = weights

        if not prefit:
            fitted_estimators = [
                clone(estimator).fit(X, y_encoded) for _, estimator in active
            ]
            # A frozen estimator keeps the labels it was fitted on
            self._check_classes(
                names, fitted_estimators, self.classes_, np.arange(n_classes)
            )

        self.weights_ = best_weights
        self.best_score_ = best_score
        self.estimators_ = fitted_estimators
        self._estimator_names_ = names
        # Delete the feature names of a previous fit, as validate_data does
        if hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")
        for estimator in fitted_estimators:
            if hasattr(estimator, "feature_names_in_"):
                self.feature_names_in_ = estimator.feature_names_in_
        fitted_by_name = dict(zip(self._estimator_names_, fitted_estimators))
        self.named_estimators_ = Bunch(
            **{name: fitted_by_name.get(name, "drop") for name, _ in self.estimators}
        )
        return self

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit, if the first estimator exposes it."""
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute n_features_in_"
            ) from nfe
        return self.estimators_[0].n_features_in_

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Return the weighted sum of the base-estimator probabilities."""
        check_is_fitted(self)
        n_samples, n_classes = _num_samples(X), len(self.classes_)
        probabilities = [
            self._validate_probabilities(
                estimator.predict_proba(X), n_samples, n_classes, name
            )
            for name, estimator in zip(self._estimator_names_, self.estimators_)
        ]
        return np.tensordot(self.weights_, np.stack(probabilities), axes=1)

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels from the weighted class probabilities."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
