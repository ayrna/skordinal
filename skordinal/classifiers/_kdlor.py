"""Kernel Discriminant Learning for Ordinal Regression (KDLOR)."""

import warnings
from numbers import Integral, Real

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.extmath import cumproba_to_proba, repair_cumproba
from skordinal.utils.validation import check_ordinal_targets

# Largest KKT residual tolerated before warning on non-convergence
_KDLOR_KKT_TOLERANCE = 1e-3


class KDLOR(ClassifierMixin, BaseEstimator):
    """Kernel Discriminant Learning for Ordinal Regression.

    KDLOR [1]_ projects training patterns into a one-dimensional kernel
    discriminant subspace that maximises the margin between adjacent
    ordinal class means relative to the within-class scatter. The dual
    of this optimisation is a small convex quadratic programme. Ordered
    thresholds are placed at the midpoints of adjacent projected class
    means. See [2]_ for a survey of ordinal regression methods.

    Parameters
    ----------
    C : float, default=0.1
        Equality-constraint right-hand side in the quadratic programme.
        Because the dual QP is positively homogeneous, ``C`` rescales
        the whole dual solution, including ``thresholds_``; the class
        predictions are invariant to it, though ``predict_proba`` is
        not. Must be strictly positive.

    u : float, default=0.001
        Ridge added to the within-class scatter matrix, as a fraction
        of its mean diagonal. Must be strictly positive.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'laplacian'}, default='rbf'
        Kernel function used to map inputs into feature space. Passed
        directly to ``sklearn.metrics.pairwise.pairwise_kernels``.

    degree : int, default=3
        Degree for the polynomial kernel. Ignored by other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient.

        - ``'scale'``: ``1 / (n_features * X.var())``. Falls back to
          ``1.0`` when ``X.var() == 0``.
        - ``'auto'``: ``1 / n_features``.
        - float: used as-is. Must be strictly positive.

    coef0 : float, default=0.0
        Independent term in the polynomial and sigmoid kernels.

    tol : float, default=1e-5
        Convergence tolerance passed as ``ftol`` to the SLSQP solver.

    max_iter : int, default=200
        Maximum number of SLSQP iterations passed as ``maxiter`` to
        the solver.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique ordinal class labels seen during ``fit``, in ascending
        order.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in_,), dtype object
        Feature names seen during ``fit``. Defined only when ``X`` is a
        ``pandas.DataFrame``.

    gamma_ : float
        Resolved kernel coefficient.

    X_fit_ : ndarray of shape (n_samples, n_features)
        Training patterns retained for kernel evaluation at predict
        time.

    dual_coef_ : ndarray of shape (n_samples,)
        Projection vector expressed in kernel space. The
        one-dimensional latent projection of a sample is the kernel
        matrix between that sample and ``X_fit_`` contracted with
        ``dual_coef_``.

    thresholds_ : ndarray of shape (n_classes - 1,)
        Ordered decision boundaries; midpoints of adjacent projected
        class means.

    n_iter_ : int
        Number of SLSQP iterations performed.

    Notes
    -----
    Because the ridge on the within-class scatter matrix scales with
    that matrix, and the dual QP is positively homogeneous, predictions
    are invariant to any positive rescaling of the kernel matrix (e.g.
    rescaling ``X`` under a linear kernel).

    If the SLSQP solver does not converge, a ``ConvergenceWarning`` is
    emitted only when the Karush-Kuhn-Tucker residual of the dual
    solution exceeds a small fixed tolerance.

    References
    ----------
    .. [1] B.-Y. Sun, J. Li, D. D. Wu, X.-M. Zhang, and W.-B. Li,
       "Kernel discriminant learning for ordinal regression", IEEE
       Transactions on Knowledge and Data Engineering, vol. 22, no. 6,
       pp. 906-910, 2010.
       https://doi.org/10.1109/TKDE.2009.170

    .. [2] P. A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
       F. Fernández-Navarro, and C. Hervás-Martínez, "Ordinal
       regression methods: survey and experimental study", IEEE
       Transactions on Knowledge and Data Engineering, vol. 28, no. 1,
       2016.
       https://doi.org/10.1109/TKDE.2015.2457911

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers import KDLOR
    >>> rng = np.random.default_rng(0)
    >>> y = np.repeat([1, 2, 3], 30)
    >>> X = y[:, None] + 0.3 * rng.standard_normal((90, 4))
    >>> clf = KDLOR(kernel="rbf").fit(X, y)
    >>> clf.predict(X[::18])  # doctest: +SKIP
    array([1, 1, 2, 2, 3])
    """

    _parameter_constraints: dict = {
        "C": [Interval(Real, 0.0, None, closed="neither")],
        "u": [Interval(Real, 0.0, None, closed="neither")],
        "kernel": [StrOptions({"linear", "poly", "rbf", "sigmoid", "laplacian"})],
        "degree": [Interval(Integral, 1, None, closed="left")],
        "gamma": [
            StrOptions({"scale", "auto"}),
            Interval(Real, 0.0, None, closed="neither"),
        ],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        *,
        C=0.1,
        u=0.001,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-5,
        max_iter=200,
    ):
        self.C = C
        self.u = u
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def _kernel_params(self, gamma):
        """Build kwargs for ``sklearn.metrics.pairwise.pairwise_kernels``."""
        if self.kernel in ("rbf", "laplacian", "sigmoid"):
            params = {"gamma": gamma}
            if self.kernel == "sigmoid":
                params["coef0"] = self.coef0
        elif self.kernel == "poly":
            params = {"gamma": gamma, "degree": self.degree, "coef0": self.coef0}
        else:
            # Linear kernel needs no extra parameters
            params = {}
        return params

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the KDLOR model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training patterns.

        y : array-like of shape (n_samples,)
            Ordinal target labels. Must contain at least 2 unique classes.

        Returns
        -------
        self : KDLOR
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``y`` contains fewer than 2 unique classes, or if the
            kernel and within-class scatter matrices overflow to
            non-finite values (rescale ``X`` or reduce ``gamma``).

        scipy.linalg.LinAlgError
            If the within-class scatter matrix remains numerically
            singular even after the ridge regularisation (in which case
            increasing ``u`` is recommended).
        """
        X, y = validate_data(self, X, y, dtype=np.float64)
        self.classes_, y_enc = check_ordinal_targets(y)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Resolve gamma to a scalar before passing to pairwise_kernels
        if self.gamma == "scale":
            x_var = X.var()
            gamma_val = 1.0 / (n_features * x_var) if x_var != 0.0 else 1.0
        elif self.gamma == "auto":
            gamma_val = 1.0 / n_features
        else:
            gamma_val = float(self.gamma)

        self.gamma_ = gamma_val
        self.X_fit_ = X

        K_train = pairwise_kernels(
            X, X, metric=self.kernel, **self._kernel_params(gamma_val)
        )

        # Indicator matmul: E[i, c] = 1 iff y_enc[i] == c
        # M_sums[:, c] sums K_train columns of class c; M_means divides
        # by n_c
        class_counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
        E = np.zeros((n_samples, n_classes), dtype=np.float64)
        E[np.arange(n_samples), y_enc] = 1.0
        M_sums = K_train @ E  # (n, K)
        M_means = M_sums / class_counts  # (n, K)
        D = M_means[:, 1:] - M_means[:, :-1]  # (n, K-1)

        # Within-class scatter: H = K K^T - sum_c (1/n_c) s_c s_c^T
        #                          = K K^T - (M_sums / counts) @ M_sums^T
        H = K_train @ K_train - (M_sums / class_counts) @ M_sums.T
        # Ridge relative to H's own scale; fall back to an absolute
        # ridge when H is exactly zero (e.g. constant training data)
        scale = H.diagonal().mean()
        if scale <= 0.0:
            scale = 1.0
        H[np.diag_indices_from(H)] += self.u * scale
        # Symmetrise exactly; cho_factor requires it
        H = (H + H.T) * 0.5

        if not np.isfinite(H).all():
            raise ValueError(
                "The within-class scatter matrix contains non-finite values, "
                "usually from kernel overflow on large inputs.  Rescale X or "
                "reduce gamma."
            )

        try:
            c_factor = scipy.linalg.cho_factor(H, lower=True, check_finite=False)
        except scipy.linalg.LinAlgError as exc:
            raise scipy.linalg.LinAlgError(
                "Within-class scatter matrix H is not positive definite even "
                "after ridge regularisation.  Try increasing the parameter u "
                f"(current value: {self.u})."
            ) from exc

        HinvD = scipy.linalg.cho_solve(c_factor, D, check_finite=False)  # (n, K-1)
        Q = D.T @ HinvD  # (K-1, K-1)
        # Symmetrise exactly; the analytic gradient assumes symmetry
        Q = (Q + Q.T) * 0.5

        alpha, result = self._solve_qp(Q)
        self.n_iter_ = int(result.nit)

        # dual_coef_ = 0.5 * H^{-1} D @ alpha = 0.5 * HinvD @ alpha
        self.dual_coef_ = 0.5 * (HinvD @ alpha)  # (n,)

        m_proj = self.dual_coef_ @ M_means  # (K,)
        self.thresholds_ = (m_proj[1:] + m_proj[:-1]) * 0.5  # (K-1,)

        if n_classes > 2 and not np.all(np.diff(self.thresholds_) > 0):
            warnings.warn(
                "KDLOR: thresholds_ are not strictly increasing.  The QP "
                "solution may be degenerate.  Consider adjusting C or u.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        return self

    def _solve_qp(self, Q):
        """Solve the KDLOR dual QP on the unit simplex and rescale by C."""
        # Solving on the unit simplex removes the C-dependence of the
        # objective; dividing by max|Q| brings it to order 1 so the
        # absolute ftol is meaningful regardless of kernel and data
        # scale
        k_minus_1 = Q.shape[0]
        q_scale = np.abs(Q).max()
        Q_hat = Q / q_scale if q_scale > 0.0 else Q
        beta0 = np.full(k_minus_1, 1.0 / k_minus_1, dtype=np.float64)
        constraints = {
            "type": "eq",
            "fun": lambda b: b.sum() - 1.0,
            "jac": lambda b: np.ones(k_minus_1, dtype=np.float64),
        }
        bounds = [(0.0, None)] * k_minus_1
        result = scipy.optimize.minimize(
            lambda b, Q: float(0.5 * b @ Q @ b),
            beta0,
            args=(Q_hat,),
            jac=lambda b, Q: Q @ b,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )
        beta = result.x

        if not result.success:
            # Complementary slackness: at the optimum Q_hat @ beta is
            # constant only on the active support (beta > 0); entries
            # where beta == 0 may sit above that constant, so excluding
            # them from the residual avoids penalising a boundary
            # optimum
            g = Q_hat @ beta
            support = beta > 1e-9
            mu = g[support].mean()
            residual = np.abs(g[support] - mu).max()
            if not support.all():
                residual = max(residual, max(0.0, (mu - g[~support]).max()))
            if residual > _KDLOR_KKT_TOLERANCE:
                warnings.warn(
                    f"KDLOR SLSQP did not converge: {result.message}. "
                    f"KKT residual = {residual:.3e}.",
                    category=ConvergenceWarning,
                    stacklevel=3,
                )

        return beta * self.C, result

    def _project(self, X):
        """Compute the raw latent projection for pre-validated X."""
        K_test = pairwise_kernels(
            X,
            self.X_fit_,
            metric=self.kernel,
            **self._kernel_params(self.gamma_),
        )  # (m, n)
        return K_test @ self.dual_coef_  # (m,)

    def predict(self, X):
        """Predict ordinal class labels via the threshold-counting rule.

        For each sample the predicted class is ``classes_[c]``, where
        ``c`` is the number of thresholds that the kernel projection
        ``f(x)`` strictly exceeds. Classes need not be contiguous, so
        ``c`` is an index into ``classes_``, not a class value. A
        projection exactly equal to a threshold is not counted (strict
        ``>``): ties go to the lower class. The result is not in
        general equal to ``classes_[argmax(predict_proba(X), axis=1)]``
        — see the Notes on ``predict_proba`` for why.

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
        projection = self._project(X)  # (m,)
        wx = projection[:, np.newaxis] - self.thresholds_[np.newaxis, :]  # (m, K-1)
        labels_enc = (wx > 0).sum(axis=1)  # (m,) in [0, K-1]
        return self.classes_[labels_enc]

    def _cumproba(self, X):
        """Compute raw cumulative probabilities on pre-validated X."""
        projection = self._project(X)
        return expit(self.thresholds_[np.newaxis, :] - projection[:, np.newaxis])

    def predict_cumproba(self, X):
        """Cumulative class probabilities for each sample.

        Computes ``P(y <= k | x) = sigmoid(threshold_k - f(x))`` for each
        ordinal threshold ``k``, where ``f(x)`` is the kernel projection
        score.

        The sigmoid is not a calibrated probability; see the Notes on
        ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        cumproba : ndarray of shape (n_samples, n_classes - 1)
            Entry ``[i, k]`` is the estimated probability that sample ``i``
            belongs to class ``k`` or lower. Isotonic repair enforces that
            each row is non-decreasing.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return repair_cumproba(self._cumproba(X))

    def predict_proba(self, X):
        """Class probability estimates for each sample.

        Derives class probabilities from the cumulative probability estimates
        via finite differencing.

        Notes
        -----
        Because KDLOR is a geometric (not probabilistic) model, the sigmoid
        used in ``predict_cumproba`` has no calibrated temperature. The
        derived class probabilities are therefore best treated as ordinal
        scores rather than as calibrated probabilities. In particular,
        ``classes_[argmax(predict_proba(X), axis=1)]`` is not in general
        equal to ``predict``: ``predict`` follows the canonical KDLOR rule
        of counting threshold crossings on the projection axis. Users who
        need calibrated probabilities should wrap KDLOR with
        ``sklearn.calibration.CalibratedClassifierCV``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Row-stochastic matrix of class probability estimates.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return cumproba_to_proba(self._cumproba(X), repair=True)
