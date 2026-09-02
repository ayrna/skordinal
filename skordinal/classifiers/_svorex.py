"""Support Vector for Ordinal Regression (Explicit constraints) (SVOREX)."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.validation import check_ordinal_targets

from . import _libsvor as svorex  # type: ignore[attr-defined]


class SVOREX(ClassifierMixin, BaseEstimator):
    """Support Vector for Ordinal Regression (Explicit constraints).

    This class derives from the Algorithm Class and implements the SVOREX method.
    This class uses SVOREX implementation by W. Chu et al
    (http://www.gatsby.ucl.ac.uk/~chuwei/svor.htm).

    Parameters
    ----------
    C : float, default=1
        Set the parameter C.

    kernel : str, default="rbf"
        Set type of kernel function.
        - rbf: use Gaussian RBF kernel
        - linear: use imbalanced Linear kernel
        - poly: use Polynomial kernel with order p

    degree : int, default=2
        Degree of the polynomial kernel; must be at least 1. Ignored by
        the ``rbf`` and ``linear`` kernels.

    tol : float, default=0.001
        Set tolerance of termination criterion.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for the RBF kernel. Ignored by the linear
        and polynomial kernels.

        - ``'scale'``: ``1 / (n_features * X.var())``. Falls back to
          ``1.0`` when ``X.var() == 0``.
        - ``'auto'``: ``1 / n_features``.
        - float: used as-is. Must be strictly positive.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Array that contains all different class labels found in the original dataset.

    thresholds_ : ndarray of shape (n_classes - 1,)
        Fitted ordered thresholds partitioning the latent projection into
        class regions.  ``predict`` assigns
        ``classes_[(predict_projection(X)[:, None] > thresholds_).sum(axis=1)]``.

    model_ : object
        Fitted estimator.

    References
    ----------
    .. [1] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and
           experimental study", IEEE Transactions on Knowledge and Data Engineering,
           Vol. 28. Issue 1, 2016, https://doi.org/10.1109/TKDE.2015.2457911

    .. [2] W. Chu and S. S. Keerthi, "Support Vector Ordinal Regression", Neural
           Computation, vol. 19, no. 3, pp. 792-815, 2007,
           http://10.1162/neco.2007.19.3.792

    """

    _parameter_constraints: dict = {
        "C": [Interval(Real, 0.0, None, closed="neither")],
        "kernel": [StrOptions({"rbf", "linear", "poly"})],
        "degree": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "gamma": [
            StrOptions({"scale", "auto"}),
            Interval(Real, 0.0, None, closed="neither"),
        ],
    }

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 2,
        tol: float = 0.001,
        gamma: float | str = "scale",
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.tol = tol
        self.gamma = gamma

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike) -> SVOREX:
        """Fit the model with the training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training patterns array, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid or data has wrong format.

        """
        X, y = validate_data(self, X, y)
        self.classes_, y_encoded = check_ordinal_targets(y)

        arg = ""
        if self.kernel == "linear":
            arg = "-L"
        elif self.kernel == "poly":
            arg = "-P {}".format(self.degree)
        # kernel == "rbf" maps to the C core's default (gaussian); no flag emitted.

        # Resolve gamma to a scalar before passing it to the C backend
        n_features = X.shape[1]
        if self.gamma == "scale":
            x_var = X.var()
            gamma_value = 1.0 / (n_features * x_var) if x_var != 0.0 else 1.0
        elif self.gamma == "auto":
            gamma_value = 1.0 / n_features
        else:
            gamma_value = float(self.gamma)

        # -M 0 SVOREX
        options = "svor -M 0 {} -T {} -K {} -C {}".format(
            arg, str(self.tol), str(gamma_value), str(self.C)
        )
        self.model_ = svorex.fit((y_encoded + 1).tolist(), X.tolist(), options)
        # biasj are the backend's ordered cutpoints
        self.thresholds_ = np.asarray(self.model_["biasj"], dtype=np.float64)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test patterns array, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If the input is invalid.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred, _ = svorex.predict(X.tolist(), self.model_)
        return self.classes_[np.asarray(y_pred).astype(int) - 1]

    def predict_projection(self, X: ArrayLike) -> np.ndarray:
        """Return the raw latent projection for each sample.

        The kernel projection ``f(x)`` is the raw latent projection
        (ordinal-axis score) that ``thresholds_`` partitions into class
        regions, on the same scale.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test patterns array, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        projection : ndarray of shape (n_samples,)
            Raw kernel projection for each sample.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If the input is invalid.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        _, projection = svorex.predict(X.tolist(), self.model_)
        return np.asarray(projection, dtype=np.float64)
