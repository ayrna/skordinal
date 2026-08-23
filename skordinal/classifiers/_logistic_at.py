"""All-Threshold logistic ordinal classifier (LogisticAT)."""

import warnings
from numbers import Integral, Real

import numpy as np
import scipy.optimize
import scipy.special
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.extmath import (
    cumproba_to_proba,
    params_to_thresholds,
    repair_cumproba,
    thresholds_grad,
    thresholds_to_params,
)
from skordinal.utils.validation import check_ordinal_targets


class LogisticAT(ClassifierMixin, BaseEstimator):
    """All-Threshold logistic ordinal classifier.

    ``LogisticAT`` [1]_ fits a shared linear projection ``f(x) = w^T x`` and
    ``K-1`` ordered thresholds ``b_1 <= ... <= b_{K-1}`` by minimising a
    margin-based surrogate of the absolute error.  For every (sample,
    threshold) pair ``(i, k)`` the model incurs a logistic loss that
    penalises the projection ``f(x_i)`` for being on the wrong side of
    threshold ``b_k``.  Optimisation uses L-BFGS-B over an unconstrained
    reparametrisation that keeps the thresholds ordered.

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularisation strength applied to ``w`` only; thresholds are
        unpenalised.  Must be >= 0.

    max_iter : int, default=1000
        Maximum number of L-BFGS-B iterations.

    tol : float, default=1e-5
        Convergence tolerance forwarded to L-BFGS-B as both ``ftol`` and
        ``gtol``.  Must be strictly positive.

    class_weight : dict, "balanced", or None, default=None
        Per-class weights applied to each sample's loss contribution.
        Supply a dict ``{class_label: weight}`` to set weights manually,
        or ``"balanced"`` to let the estimator compute weights
        proportional to ``n_samples / (n_classes * np.bincount(y))``.
        ``None`` assigns weight 1 to every sample.  The L2 penalty on
        ``w`` and the threshold parameters are not affected.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during ``fit``, sorted in ascending order.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in_,), dtype object
        Feature names seen during ``fit``.  Defined only when ``X`` is a
        ``pandas.DataFrame``.

    coef_ : ndarray of shape (n_features_in_,)
        Linear projection weights ``w``.  No bias term — per-class biases
        are absorbed into ``thresholds_``.

    thresholds_ : ndarray of shape (n_classes - 1,)
        Fitted ordered thresholds ``b_1 <= ... <= b_{K-1}``.

    n_iter_ : int
        Number of L-BFGS-B iterations executed during ``fit``.

    loss_ : float
        Final objective value (All-Threshold loss + L2 penalty) at
        convergence.

    Notes
    -----
    The All-Threshold loss sums, over all ``(i, k)`` pairs, a logistic
    surrogate of the absolute error:

    .. math::

        J(w, t) = \\frac{1}{n} \\sum_{i,k}
                  \\log\\bigl(1 + \\exp(-s_{i,k}(b_k - w^T x_i))\\bigr)
                  + \\frac{\\alpha}{2n} \\|w\\|^2

    where ``s_{i,k} = +1`` if ``y_i <= k`` else ``-1``.  The formula is
    shown unweighted; ``class_weight`` multiplies each sample's term.
    L-BFGS-B optimises the packed vector ``[w; t]`` where ``t`` is the
    unconstrained representation of the thresholds; ``b[0] = t[0]``,
    ``b[k] = t[0] + t[1]^2 + ... + t[k]^2`` for ``k >= 1``.

    Standardising features (zero mean, unit variance) is recommended;
    extreme feature scales can degrade the fit under the fixed solver
    tolerance.

    ``predict`` uses the cumulative-median rule, which is not in general
    equal to the modal prediction ``classes_[argmax(predict_proba(X),
    axis=1)]``; see ``predict`` for the exact rule and its rationale.

    References
    ----------
    .. [1] F. Pedregosa, F. Bach, and A. Gramfort,
       "On the Consistency of Ordinal Regression Methods,"
       Journal of Machine Learning Research, vol. 18, no. 55,
       pp. 1-35, 2017.
       http://jmlr.org/papers/v18/15-495.html

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers import LogisticAT
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 4))
    >>> y = np.repeat([1, 2, 3, 4], 20)
    >>> clf = LogisticAT(alpha=1.0).fit(X, y)
    >>> clf.predict(X[:5])  # doctest: +SKIP
    array([1, 2, 3, 4, ...])
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "class_weight": [StrOptions({"balanced"}), dict, None],
    }

    def __init__(
        self,
        alpha=1.0,
        max_iter=1000,
        tol=1e-5,
        class_weight=None,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the All-Threshold logistic model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training patterns.

        y : array-like of shape (n_samples,)
            Ordinal target labels. Must contain at least 2 unique classes.

        Returns
        -------
        self : LogisticAT
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``y`` contains fewer than 2 unique classes.
        """
        X, y = validate_data(self, X, y, dtype=np.float64)
        self.classes_, y_enc = check_ordinal_targets(y)
        sample_weight = compute_sample_weight(self.class_weight, y)

        n_features = X.shape[1]
        n_classes = len(self.classes_)

        params0 = self._init_params(n_features, n_classes)

        result = scipy.optimize.minimize(
            fun=self._objective,
            x0=params0,
            args=(X, y_enc, sample_weight, n_features, n_classes),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "gtol": self.tol,
            },
        )

        if not result.success:
            warnings.warn(
                f"LogisticAT failed to converge after {self.max_iter} "
                "iterations. Increase max_iter or set alpha > 0.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.coef_ = result.x[:n_features]
        self.thresholds_ = params_to_thresholds(result.x[n_features:])
        self.n_iter_ = int(result.nit)
        self.loss_ = float(result.fun)

        return self

    def _project(self, X):
        """Compute the raw latent projection for pre-validated X."""
        return X @ self.coef_

    def _cumproba(self, X):
        """Compute raw cumulative probabilities on pre-validated X."""
        f = self._project(X)
        eta = self.thresholds_[None, :] - f[:, None]
        return scipy.special.expit(eta)

    def predict_cumproba(self, X):
        """Return cumulative class probabilities ``P(Y <= k | x)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns.

        Returns
        -------
        F : ndarray of shape (n_samples, n_classes - 1)
            ``F[:, k]`` equals ``P(Y <= k+1 | x)`` for ``k = 0, ..., K-2``.
            Columns are non-decreasing along the class axis; isotonic repair
            is applied to enforce strict validity.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return repair_cumproba(self._cumproba(X))

    def predict_proba(self, X):
        """Return per-class probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Non-negative class probabilities; each row sums to 1.0.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return cumproba_to_proba(self._cumproba(X), repair=True)

    def predict(self, X):
        """Predict ordinal class labels for patterns in X.

        The predicted class follows the cumulative-median rule: the number
        of fitted thresholds the projection ``f(x) = w^T x`` exceeds,
        equivalently the first class whose cumulative probability
        ``P(Y <= k | x)`` reaches 0.5.  This median is the risk-minimising
        decision for the absolute-error surrogate the model optimises, and
        unlike the mode (``argmax(predict_proba(X))``) it can differ when
        adjacent thresholds are close.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels drawn from ``self.classes_``.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        f = self._project(X)
        labels_enc = (f[:, np.newaxis] > self.thresholds_[np.newaxis, :]).sum(axis=1)
        return self.classes_[labels_enc]

    def predict_projection(self, X):
        """Return the raw latent projection for each sample.

        The linear projection ``f(x) = w^T x`` is the raw latent
        projection (ordinal-axis score) that ``thresholds_`` partitions
        into class regions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns.

        Returns
        -------
        projection : ndarray of shape (n_samples,)
            Raw linear projection for each sample.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._project(X)

    def _init_params(self, n_features, n_classes):
        """Build the initial packed parameter vector ``[w; t]``."""
        w = np.zeros(n_features)

        quantiles = np.arange(1, n_classes) / n_classes
        b_init = np.log(quantiles / (1.0 - quantiles))

        t = thresholds_to_params(b_init)

        return np.concatenate([w, t])

    def _objective(
        self,
        params,
        X,
        y_enc,
        sample_weight,
        n_features,
        n_classes,
    ):
        """Return the All-Threshold loss + L2 objective and gradient."""
        w = params[:n_features]
        t = params[n_features:]
        thresholds = params_to_thresholds(t)

        n = X.shape[0]
        f = X @ w
        k_idx = np.arange(n_classes - 1)
        # All (sample, threshold) pairs: s = +1 when y_i <= k else -1
        s = np.where(y_enc[:, None] <= k_idx[None, :], 1.0, -1.0)
        margin = s * (thresholds[None, :] - f[:, None])

        # Per-pair logistic loss: softplus(-margin)
        J_pairs = np.logaddexp(0.0, -margin)
        J = (J_pairs * sample_weight[:, None]).sum() / n + (self.alpha / (2 * n)) * (
            w @ w
        )

        sigma = scipy.special.expit(-margin)

        sw_col = sample_weight[:, None]
        grad_f = ((s * sigma) * sw_col).sum(axis=1) / n
        grad_w = X.T @ grad_f + (self.alpha / n) * w

        grad_b = -((s * sigma) * sw_col).sum(axis=0) / n
        grad_t = thresholds_grad(t, grad_b)

        return float(J), np.concatenate([grad_w, grad_t])
