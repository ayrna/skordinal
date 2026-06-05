"""Regressor wrapper meta-estimator for ordinal classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_regressor,
)
from sklearn.svm import SVR
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.validation import check_is_fitted

from skordinal.utils._sklearn_compat import validate_data
from skordinal.utils.validation import check_ordinal_targets


class RegressorWrapper(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Regressor-based meta-estimator for ordinal classification.

    Wraps any scikit-learn regressor and adapts it for ordinal classification
    by training on the zero-based rank encoding of the labels and rounding each
    continuous prediction to the nearest rank, breaking ties toward the lower
    class. The default base estimator is an SVR.

    Parameters
    ----------
    estimator : regressor instance or None, default=None
        Base regressor. Must implement ``fit`` and ``predict``. If ``None``,
        an SVR with default hyperparameters is used.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    estimator_ : regressor instance
        Fitted clone of the base estimator.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when X has feature
        names that are all strings.

    References
    ----------
    .. [1] P. A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
       F. Fernández-Navarro, and C. Hervás-Martínez, "Ordinal Regression
       Methods: Survey and Experimental Study", IEEE Transactions on
       Knowledge and Data Engineering, vol. 28, no. 1, pp. 127-146, 2016,
       https://doi.org/10.1109/TKDE.2015.2457911

    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
    }

    def __init__(self, estimator=None) -> None:
        self.estimator = estimator

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike) -> RegressorWrapper:
        """Fit the wrapped regressor to ``(X, y)``.

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
            If the resolved base estimator is not a regressor.

        """
        X, y = validate_data(
            self, X, y, accept_sparse=False, ensure_2d=True, dtype=None
        )

        self.classes_, y_enc = check_ordinal_targets(y)

        base = self.estimator if self.estimator is not None else SVR()

        if not is_regressor(base):
            raise ValueError(
                f"estimator must be a regressor; got {type(base).__name__}"
            )

        self.estimator_ = clone(base)
        self.estimator_.fit(X, y_enc)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict ordinal class labels by rounding regressor outputs.

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

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

        y_cont = self.estimator_.predict(X)
        ranks = np.arange(self.classes_.size, dtype=float)
        idx = np.abs(y_cont[:, None] - ranks[None, :]).argmin(axis=1)
        return self.classes_[idx]
