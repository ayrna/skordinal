"""Cost-sensitive wrapper meta-estimator for ordinal classification."""

from __future__ import annotations

import inspect

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.extmath import normalize_proba_rows
from skordinal.utils.validation import check_ordinal_targets


def _ordinal_weights(p: int, y_enc: np.ndarray) -> np.ndarray:
    """Return per-sample weights for the one-vs-rest sub-problem of class p.

    Positive samples get weight 1; negative samples get a weight proportional
    to their ordinal distance to p, rescaled to sum to the negative count.

    """
    neg_mask = y_enc != p
    n_neg = neg_mask.sum()
    unnorm = np.abs(p - y_enc).astype(np.float64) + 1.0
    S = unnorm[neg_mask].sum()
    return np.where(neg_mask, unnorm * (n_neg / S), 1.0)


def _resolve_estimator(estimator):
    """Return the base estimator, substituting the default when None."""
    return estimator if estimator is not None else LogisticRegression()


def _estimators_has(attr):
    """Build an available_if predicate for a shared sub-estimator capability."""

    def check(self):
        target = (
            self.estimators_[0]
            if hasattr(self, "estimators_")
            else _resolve_estimator(self.estimator)
        )
        return hasattr(target, attr)

    return check


class CostSensitiveWrapper(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Cost-sensitive one-vs-rest meta-estimator for ordinal classification.

    Trains one binary one-vs-rest sub-classifier per ordinal class, reweighting
    negative samples by their ordinal distance to the focal class so that
    farther classes cost more. Predictions take the argmax over the per-class
    score matrix. The default base classifier is LogisticRegression.

    Parameters
    ----------
    estimator : classifier instance or None, default=None
        Base binary classifier. Must implement a ``fit`` method that accepts a
        ``sample_weight`` keyword argument. If ``None``, a LogisticRegression
        with default hyperparameters is used.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    estimators_ : list of length n_classes
        Fitted clones of the base estimator, one per ordinal class.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when X has feature
        names that are all strings.

    Notes
    -----
    ``predict`` and ``predict_proba`` score sub-classifiers differently
    (``decision_function`` vs. each one's calibrated probability) and are
    not guaranteed to agree, the same asymmetry as ``sklearn.svm.SVC``.

    References
    ----------
    .. [1] P. A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
       F. Fernández-Navarro, and C. Hervás-Martínez, "Ordinal Regression
       Methods: Survey and Experimental Study", IEEE Transactions on
       Knowledge and Data Engineering, vol. 28, no. 1, pp. 127-146, 2016,
       https://doi.org/10.1109/TKDE.2015.2457911

    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"]), None],
    }

    def __init__(self, estimator=None) -> None:
        self.estimator = estimator

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike) -> CostSensitiveWrapper:
        """Fit one cost-sensitive binary classifier per ordinal class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If the resolved base estimator is not a classifier, or if its
            ``fit`` method does not accept ``sample_weight``.

        """
        X, y = validate_data(
            self, X, y, accept_sparse=False, ensure_2d=True, dtype=None
        )

        self.classes_, y_enc = check_ordinal_targets(y)
        K = self.classes_.size

        base = _resolve_estimator(self.estimator)

        if not is_classifier(base):
            raise ValueError(
                f"estimator must be a classifier; got {type(base).__name__}"
            )

        if "sample_weight" not in inspect.signature(base.fit).parameters:
            raise ValueError(
                f"Estimator {type(base).__name__} does not accept sample_weight "
                f"in fit; CostSensitiveWrapper requires it."
            )

        self.estimators_ = []
        for p in range(K):
            y_bin = (y_enc == p).astype(np.intp)
            est_p = clone(base)
            est_p.fit(X, y_bin, sample_weight=_ordinal_weights(p, y_enc))
            self.estimators_.append(est_p)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict ordinal class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.

        AttributeError
            If no sub-estimator exposes ``decision_function`` or ``predict_proba``.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

        n = X.shape[0]
        K = len(self.estimators_)

        use_df = all(hasattr(e, "decision_function") for e in self.estimators_)
        use_proba = all(hasattr(e, "predict_proba") for e in self.estimators_)

        if not use_df and not use_proba:
            raise AttributeError(
                "Base estimators expose neither decision_function nor "
                "predict_proba; CostSensitiveWrapper.predict requires one of them."
            )

        S = np.empty((n, K), dtype=np.float64)
        if use_df:
            for p, est_p in enumerate(self.estimators_):
                S[:, p] = est_p.decision_function(X)
        else:
            for p, est_p in enumerate(self.estimators_):
                idx = int(np.searchsorted(est_p.classes_, 1))
                S[:, p] = est_p.predict_proba(X)[:, idx]

        return self.classes_[S.argmax(axis=1)]

    @available_if(_estimators_has("predict_proba"))
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return class-probability estimates for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. Each row is non-negative and sums to one.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.

        AttributeError
            If the base estimator does not expose ``predict_proba``, in which
            case this method is not present on the instance at all.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

        n = X.shape[0]
        K = len(self.estimators_)

        P = np.empty((n, K), dtype=np.float64)
        for p, est_p in enumerate(self.estimators_):
            idx = int(np.searchsorted(est_p.classes_, 1))
            P[:, p] = est_p.predict_proba(X)[:, idx]

        return normalize_proba_rows(P)
