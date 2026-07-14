"""Proportional Odds Model (POM) for ordinal classification."""

import math
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

# Saturation bounds for the complementary-log-log link: exp(-exp(30))
# underflows to 0 on float64, and exp(-exp(-500)) is indistinguishable from
# 1.  The bounds also guard the density exp(z - exp(z)) against
# inner-exp(z) overflow, not only the CDF.  Reused (negated) for loglog
# via the z -> -z tail symmetry between the two links
_CLOGLOG_Z_MIN = -500.0
_CLOGLOG_Z_MAX = 30.0

# Per-class probability floor used in the objective
_PROBA_FLOOR = 1e-12


class POM(ClassifierMixin, BaseEstimator):
    """Proportional Odds Model for ordinal classification.

    POM [1]_ postulates a shared linear projection ``f(x) = w^T x`` and
    ``K-1`` ordered thresholds ``b_1 <= ... <= b_{K-1}`` fitted jointly by
    maximum likelihood. The cumulative class probability is modelled as
    ``P(Y <= k | x) = G(b_k - w^T x)`` where ``G`` is the inverse-link
    function selected by the ``link`` parameter. See [2]_ for a survey of
    ordinal regression methods.

    Parameters
    ----------
    link : str, default="logit"
        Inverse-link function ``G``. One of ``{"logit", "probit",
        "cloglog", "loglog", "cauchit"}``.

        - ``"logit"``: standard logistic CDF, equivalent to the standard
          proportional-odds model.
        - ``"probit"``: standard normal CDF.
        - ``"cloglog"``: complementary log-log CDF
          ``G(z) = 1 - exp(-exp(z))``.
        - ``"loglog"``: log-log (Gumbel) CDF ``G(z) = exp(-exp(-z))``.
        - ``"cauchit"``: Cauchy CDF ``G(z) = 0.5 + arctan(z) / pi``.

    alpha : float, default=0.0
        L2 regularisation strength applied to ``w`` only; thresholds are
        unpenalised. Must be >= 0. When ``alpha=0`` the model maximises
        the pure log-likelihood.

    solver : {"lbfgs", "newton-cg", "bfgs"}, default="lbfgs"
        Optimisation algorithm forwarded to
        ``scipy.optimize.minimize``. All solvers consume the analytic
        objective gradient.

        - ``"lbfgs"``: limited-memory BFGS (``"L-BFGS-B"``). The
          default; low memory cost, suitable for many features.
        - ``"newton-cg"``: Newton conjugate gradient. Uses curvature
          information; converges in fewer iterations on
          well-conditioned problems.
        - ``"bfgs"``: full BFGS. Maintains a dense Hessian
          approximation; best when the number of features is small.

    max_iter : int, default=1000
        Maximum number of solver iterations.

    tol : float, default=1e-5
        Convergence tolerance forwarded to the solver (``ftol`` and ``gtol``
        for ``"lbfgs"``, ``gtol`` for ``"bfgs"``, ``xtol`` for
        ``"newton-cg"``). Must be strictly positive.

    class_weight : dict, "balanced", or None, default=None
        Per-class weights applied to each sample's loss contribution.
        Supply a dict ``{class_label: weight}`` to set weights manually,
        or ``"balanced"`` to let the estimator compute weights
        proportional to ``n_samples / (n_classes * np.bincount(y))``.
        ``None`` assigns weight 1 to every sample.  The L2 penalty on
        ``w`` is not affected.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique ordinal class labels seen during ``fit``, in ascending
        order.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in_,), dtype object
        Feature names seen during ``fit``.  Defined only when ``X`` is a
        ``pandas.DataFrame``.

    coef_ : ndarray of shape (n_features_in_,)
        Linear projection weights ``w``. No bias term — the per-class
        biases are absorbed into ``thresholds_``.

    thresholds_ : ndarray of shape (n_classes - 1,)
        Fitted ordered thresholds ``b_1 <= ... <= b_{K-1}``.

    n_iter_ : int
        Number of solver iterations executed during ``fit``.

    loss_ : float
        Final objective value (negative log-likelihood plus L2 penalty)
        at termination; set even when the solver does not converge.

    Notes
    -----
    The chosen ``solver`` optimises the packed parameter vector ``[w; t]``
    where ``w`` is the linear projection and ``t`` is the unconstrained
    reparametrisation of the thresholds (``b[0] = t[0]``,
    ``b[k] = t[0] + t[1]^2 + ... + t[k]^2`` for ``k >= 1``), which
    guarantees they remain non-decreasing throughout optimisation.

    The objective is the weighted negative log-likelihood plus an L2
    penalty on ``w``:

    .. math::

        J(w, t) = -\\frac{1}{n} \\sum_{i=1}^{n} \\log h_{y_i}(x_i)
                  + \\frac{\\alpha}{2n} \\|w\\|^2

    where ``h_k(x) = P(Y = k | x)`` is obtained from the cumulative-link
    differences.  The formula is shown unweighted; ``class_weight``
    multiplies each sample's term.

    On linearly separable data with ``alpha=0`` the estimate is not
    finite (coefficients may diverge); set ``alpha > 0`` to regularise.

    Standardising features (zero mean, unit variance) is recommended:
    extreme feature scales interact poorly with the fixed solver
    tolerance and can silently degrade the fit.

    References
    ----------
    .. [1] P. McCullagh, "Regression models for ordinal data",
       Journal of the Royal Statistical Society. Series B (Methodological),
       vol. 42, no. 2, pp. 109-142, 1980.

    .. [2] P. A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
       F. Fernández-Navarro, and C. Hervás-Martínez, "Ordinal regression
       methods: survey and experimental study," IEEE Transactions on
       Knowledge and Data Engineering, vol. 28, no. 1, 2016.
       https://doi.org/10.1109/TKDE.2015.2457911

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers import POM
    >>> rng = np.random.default_rng(0)
    >>> y = np.repeat([1, 2, 3, 4], 20)
    >>> X = y[:, None] + 0.3 * rng.standard_normal((80, 4))
    >>> clf = POM(link="logit", alpha=0.1).fit(X, y)
    >>> clf.predict(X[::16])  # doctest: +SKIP
    array([1, 1, 2, 3, 4])
    """

    _parameter_constraints: dict = {
        "link": [StrOptions({"logit", "probit", "cloglog", "loglog", "cauchit"})],
        "alpha": [Interval(Real, 0.0, None, closed="left")],
        "solver": [StrOptions({"lbfgs", "newton-cg", "bfgs"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "class_weight": [StrOptions({"balanced"}), dict, None],
    }

    def __init__(
        self,
        link="logit",
        alpha=0.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-5,
        class_weight=None,
    ):
        self.link = link
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the Proportional Odds Model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training patterns.

        y : array-like of shape (n_samples,)
            Ordinal target labels. Must contain at least 2 unique classes.

        Returns
        -------
        self : POM
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``y`` contains fewer than 2 unique classes.
        """
        X, y = validate_data(self, X, y, dtype=np.float64)
        self.classes_, y_enc = check_ordinal_targets(y)
        sample_weight = compute_sample_weight(self.class_weight, y)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        params0 = self._init_params(n_features, n_classes)

        # Select the scipy method and forward tol to the tolerance key(s)
        # that method understands (avoiding unknown-option warnings)
        options = {"maxiter": self.max_iter}
        if self.solver == "lbfgs":
            method = "L-BFGS-B"
            options["ftol"] = self.tol
            options["gtol"] = self.tol
        elif self.solver == "bfgs":
            method = "BFGS"
            options["gtol"] = self.tol
        else:
            method = "Newton-CG"
            options["xtol"] = self.tol

        result = scipy.optimize.minimize(
            fun=self._objective,
            x0=params0,
            args=(X, y_enc, sample_weight, n_features, n_classes),
            method=method,
            jac=True,
            options=options,
        )

        if not result.success:
            warnings.warn(
                f"POM did not converge (stopped after {result.nit} "
                f"iterations: {result.message}). Consider increasing "
                "max_iter, setting alpha > 0, or standardising X.",
                ConvergenceWarning,
                stacklevel=2,
            )

        self.coef_ = result.x[:n_features]
        self.thresholds_ = params_to_thresholds(result.x[n_features:])
        self.n_iter_ = int(result.nit)
        self.loss_ = float(result.fun)

        return self

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

        The predicted class is ``classes_[argmax(predict_proba(X),
        axis=1)]``, the class with the highest estimated probability.
        Exact probability ties resolve to the lower class, i.e. the
        first occurrence in ``classes_``.

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
        proba = cumproba_to_proba(self._cumproba(X), repair=True)
        return self.classes_[proba.argmax(axis=1)]

    def _cumproba(self, X):
        """Compute raw cumulative probabilities on pre-validated X."""
        f = X @ self.coef_  # (n,)
        eta = self.thresholds_[np.newaxis, :] - f[:, np.newaxis]  # (n, K-1)
        # Predict path only needs the CDF; skip the density computation
        return self._link_cdf_pdf(eta, need_pdf=False)[0]

    def _link_cdf_pdf(self, z, need_pdf=True):
        """Evaluate the inverse-link CDF and, optionally, its density."""
        if self.link == "logit":
            s = scipy.special.expit(z)
            if not need_pdf:
                return s, None
            return s, s * (1.0 - s)
        if self.link == "probit":
            cdf = scipy.special.ndtr(z)
            if not need_pdf:
                return cdf, None
            pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
            return cdf, pdf
        if self.link == "cloglog":
            z_clamp = np.clip(z, _CLOGLOG_Z_MIN, _CLOGLOG_Z_MAX)
            ez = np.exp(z_clamp)
            cdf = -np.expm1(-ez)
            if not need_pdf:
                return cdf, None
            pdf = np.exp(z_clamp - ez)
            return cdf, pdf
        if self.link == "loglog":
            z_clamp = np.clip(z, -_CLOGLOG_Z_MAX, -_CLOGLOG_Z_MIN)
            enz = np.exp(-z_clamp)
            cdf = np.exp(-enz)
            if not need_pdf:
                return cdf, None
            pdf = np.exp(-z_clamp - enz)
            return cdf, pdf
        # cauchit
        cdf = 0.5 + np.arctan(z) / math.pi
        if not need_pdf:
            return cdf, None
        pdf = 1.0 / (math.pi * (1.0 + z**2))
        return cdf, pdf

    def _objective(
        self,
        params,
        X,
        y_enc,
        sample_weight,
        n_features,
        n_classes,
    ):
        """Compute the NLL + L2 objective and its gradient."""
        w = params[:n_features]
        t = params[n_features:]
        thresholds = params_to_thresholds(t)  # (K-1,)
        n_samples = X.shape[0]

        f = X @ w  # (n,)

        # Gather-only evaluation: each sample's likelihood needs at most
        # two link values (the cumulative bounds around its own class),
        # so avoid building the full (n, K-1) cdf/pdf grid
        # The upper cdf is fixed at 1 for y == K-1 and the lower at 0
        # for y == 0; the masks keep those rows out of the link gathers
        hi_mask = y_enc < n_classes - 1
        lo_mask = y_enc > 0
        idx_hi = np.minimum(y_enc, n_classes - 2)
        idx_lo = np.maximum(y_enc - 1, 0)

        cdf_hi, pdf_hi = self._link_cdf_pdf(thresholds[idx_hi] - f)
        cdf_lo, pdf_lo = self._link_cdf_pdf(thresholds[idx_lo] - f)
        cdf_hi = np.where(hi_mask, cdf_hi, 1.0)
        cdf_lo = np.where(lo_mask, cdf_lo, 0.0)

        # True-class probability per sample (raw, before floor)
        h_true_raw = cdf_hi - cdf_lo
        clamped = h_true_raw <= _PROBA_FLOOR
        h_true = np.maximum(h_true_raw, _PROBA_FLOOR)

        # Objective: weighted NLL + L2
        J = -(sample_weight * np.log(h_true)).sum() / n_samples
        J += self.alpha / (2.0 * n_samples) * np.dot(w, w)

        # Error derivative -sample_weight/h_true where not clamped, 0
        # elsewhere.  Where h_true <= _PROBA_FLOOR the floor is active
        # (flat region), so the subgradient is zero — zeroing e_true
        # keeps the analytic gradient consistent with the clamped
        # objective
        e_true = np.where(clamped, 0.0, -sample_weight / h_true)

        g_hi = np.where(hi_mask, e_true * pdf_hi, 0.0)
        g_lo = np.where(lo_mask, -e_true * pdf_lo, 0.0)

        s = -(g_hi + g_lo)  # (n,)
        grad_w = X.T @ s / n_samples + (self.alpha / n_samples) * w

        # Gradient w.r.t. the ordered thresholds: scatter each sample's
        # contribution back onto the (at most two) thresholds it touched
        grad_b = (
            np.bincount(idx_hi, weights=g_hi, minlength=n_classes - 1)
            + np.bincount(idx_lo, weights=g_lo, minlength=n_classes - 1)
        )[: n_classes - 1] / n_samples

        # Chain rule through the cumsum-of-squares threshold reparametrisation
        grad_t = thresholds_grad(t, grad_b)

        return float(J), np.concatenate([grad_w, grad_t])

    def _init_params(self, n_features, n_classes):
        """Build the deterministic initial parameter vector."""
        w = np.zeros(n_features)

        # Equal-class-mass quantile spacing under w=0
        quantiles = np.arange(1, n_classes) / n_classes  # (K-1,)

        if self.link == "logit":
            b_init = np.log(quantiles / (1.0 - quantiles))
        elif self.link == "probit":
            b_init = scipy.special.ndtri(quantiles)
        elif self.link == "cloglog":
            b_init = np.log(-np.log1p(-quantiles))
        elif self.link == "loglog":
            b_init = -np.log(-np.log(quantiles))
        else:
            # cauchit
            b_init = np.tan(math.pi * (quantiles - 0.5))

        t = thresholds_to_params(b_init)

        return np.concatenate([w, t])
