"""Cumulative Link Model output layer implemented with NumPy."""

from numbers import Integral, Real

import numpy as np
from scipy.special import expit, ndtr

from skordinal.utils.extmath import params_to_thresholds, thresholds_grad

_LINK_FUNCTIONS = ("logit", "probit", "cloglog")

# Beyond these bounds the links saturate exactly in float64 (exp(50)
# already maps cloglog to 0 or 1 and the normal density underflows to 0
# past |eta| = 40), so clipping avoids overflow without changing any value
_CLOGLOG_MAX_ETA = 50.0
_PROBIT_MAX_ABS_ETA = 40.0


class CLM:
    """Cumulative Link Model output layer.

    NumPy port of the CLM layer of Vargas et al. [1]_. It maps a
    latent scalar projection ``z`` of each sample to the probabilities of
    ``n_classes`` ordered categories:

        P(Y <= k | z) = F(theta_k - z),    k = 1, ..., n_classes - 1

    where ``F`` is the inverse link function and ``theta`` are ordered
    thresholds.

    The layer is stateless: its trainable parameters are passed explicitly
    to every method, so they can be packed with the rest of a network into
    the single 1D vector optimised by :class:`MLPBaseClassifier`.

    The ``n_classes - 1`` unconstrained parameters ``params = [b, a_1, ...,
    a_{K-2}]`` are mapped to thresholds as

        theta_0 = b
        theta_k = b + sum_{j=1..k} (a_j^2 + min_distance)

    which keeps ``theta`` non-decreasing, with consecutive gaps of at least
    ``min_distance``, for any value of ``params``.

    Parameters
    ----------
    n_classes : int
        Number of ordinal classes. Must be at least 2.

    link_function : {'logit', 'probit', 'cloglog'}, default='logit'
        Link function relating the cumulative probabilities to the latent
        projection.

    min_distance : float, default=0.0
        Minimum distance between consecutive thresholds.

    Attributes
    ----------
    n_params : int
        Number of trainable parameters, ``n_classes - 1``.

    References
    ----------
    .. [1] V. M. Vargas, P. A. Gutiérrez, and C. Hervás-Martínez,
       "Cumulative link models for deep ordinal classification,"
       Neurocomputing, vol. 401, pp. 48-58, 2020.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers._clm import CLM
    >>> clm = CLM(n_classes=3, link_function="logit")
    >>> params = clm.init_params()
    >>> clm.thresholds(params)
    array([-2.,  2.])
    >>> proba = clm.forward(np.array([-2.0, 0.0, 2.0]), params)
    >>> proba.shape
    (3, 3)
    >>> np.allclose(proba.sum(axis=1), 1.0)
    True
    """

    def __init__(self, n_classes, link_function="logit", min_distance=0.0):
        if (
            not isinstance(n_classes, Integral)
            or isinstance(n_classes, bool)
            or n_classes < 2
        ):
            raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}.")
        if link_function not in _LINK_FUNCTIONS:
            raise ValueError(
                f"link_function must be one of {_LINK_FUNCTIONS}, "
                f"got {link_function!r}."
            )
        if (
            not isinstance(min_distance, Real)
            or isinstance(min_distance, bool)
            or not np.isfinite(min_distance)
            or min_distance < 0
        ):
            raise ValueError(
                f"min_distance must be a finite float >= 0, got {min_distance!r}."
            )

        self.n_classes = int(n_classes)
        self.link_function = link_function
        self.min_distance = float(min_distance)

    @property
    def n_params(self):
        """Number of trainable parameters of the layer."""
        return self.n_classes - 1

    def init_params(self):
        """Return parameters whose thresholds are spread evenly over [-2, 2].

        Every ``a_j`` is non-zero, so no threshold gap starts at the
        stationary point of the cumsum-of-squares map.

        Returns
        -------
        params : ndarray of shape (n_classes - 1,)
            Initial unconstrained parameters.
        """
        params = np.zeros(self.n_params)
        if self.n_params > 1:
            gap = 4.0 / (self.n_params - 1)
            params[1:] = np.sqrt(gap)
            # Centre the thresholds on zero
            params[0] = -0.5 * (self.n_params - 1) * (gap + self.min_distance)
        return params

    def thresholds(self, params):
        """Map unconstrained parameters to ordered thresholds.

        Parameters
        ----------
        params : array-like of shape (n_classes - 1,)
            Unconstrained parameters.

        Returns
        -------
        thresholds : ndarray of shape (n_classes - 1,)
            Non-decreasing thresholds.
        """
        params = self._check_params(params)
        return params_to_thresholds(params) + self.min_distance * np.arange(
            self.n_params
        )

    def cumproba(self, z, params):
        """Compute the cumulative probabilities ``P(Y <= k | z)``.

        Parameters
        ----------
        z : array-like of shape (n_samples,) or (n_samples, 1)
            Latent projection of each sample.

        params : array-like of shape (n_classes - 1,)
            Unconstrained parameters.

        Returns
        -------
        cumproba : ndarray of shape (n_samples, n_classes - 1)
            Cumulative probabilities, non-decreasing along each row.
        """
        cdf, _, _ = self._link(self._eta(z, params))
        return cdf

    def forward(self, z, params):
        """Compute the class probabilities ``P(Y = k | z)``.

        Parameters
        ----------
        z : array-like of shape (n_samples,) or (n_samples, 1)
            Latent projection of each sample.

        params : array-like of shape (n_classes - 1,)
            Unconstrained parameters.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. Each row is non-negative and sums to 1.
        """
        eta = self._eta(z, params)
        cdf, sf, _ = self._link(eta)

        proba = np.empty((eta.shape[0], self.n_classes))
        proba[:, 0] = cdf[:, 0]
        # Differencing the survival function is more accurate when both
        # cumulative probabilities are close to one
        proba[:, 1:-1] = np.where(
            eta[:, :-1] > 0.0, sf[:, :-1] - sf[:, 1:], np.diff(cdf, axis=1)
        )
        proba[:, -1] = sf[:, -1]
        np.maximum(proba, 0.0, out=proba)
        return proba

    def backward(self, z, params, grad_proba):
        """Back-propagate a gradient through the layer.

        Parameters
        ----------
        z : array-like of shape (n_samples,) or (n_samples, 1)
            Latent projection of each sample, as passed to ``forward``.

        params : array-like of shape (n_classes - 1,)
            Unconstrained parameters, as passed to ``forward``.

        grad_proba : array-like of shape (n_samples, n_classes)
            Gradient of a scalar loss with respect to the output of
            ``forward``.

        Returns
        -------
        grad_z : ndarray of the same shape as ``z``
            Gradient of the loss with respect to ``z``.

        grad_params : ndarray of shape (n_classes - 1,)
            Gradient of the loss with respect to ``params``.
        """
        z_shape = np.shape(z)
        eta = self._eta(z, params)
        grad_proba = np.asarray(grad_proba, dtype=float)
        if grad_proba.shape != (eta.shape[0], self.n_classes):
            raise ValueError(
                f"grad_proba must have shape {(eta.shape[0], self.n_classes)}, "
                f"got {grad_proba.shape}."
            )

        _, _, pdf = self._link(eta)

        # P_k = F_k - F_{k-1}, so dL/dF_k = dL/dP_k - dL/dP_{k+1}
        grad_cdf = grad_proba[:, :-1] - grad_proba[:, 1:]
        grad_eta = grad_cdf * pdf

        # eta = theta - z
        grad_z = -np.sum(grad_eta, axis=1).reshape(z_shape)
        grad_params = thresholds_grad(
            self._check_params(params), np.sum(grad_eta, axis=0)
        )
        return grad_z, grad_params

    def _check_params(self, params):
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_params,):
            raise ValueError(
                f"params must have shape ({self.n_params},), got {params.shape}."
            )
        return params

    def _eta(self, z, params):
        z = np.asarray(z, dtype=float)
        if z.ndim == 2 and z.shape[1] == 1:
            z = z[:, 0]
        if z.ndim != 1:
            raise ValueError(
                f"z must have shape (n_samples,) or (n_samples, 1), got {z.shape}."
            )
        return self.thresholds(params)[np.newaxis, :] - z[:, np.newaxis]

    def _link(self, eta):
        """Return the inverse link F(eta), its complement and its derivative."""
        if self.link_function == "logit":
            cdf = expit(eta)
            sf = expit(-eta)
            pdf = cdf * sf
        elif self.link_function == "probit":
            cdf = ndtr(eta)
            sf = ndtr(-eta)
            eta_clipped = np.clip(eta, -_PROBIT_MAX_ABS_ETA, _PROBIT_MAX_ABS_ETA)
            pdf = np.exp(-0.5 * eta_clipped**2) / np.sqrt(2.0 * np.pi)
        else:  # 'cloglog'
            exp_eta = np.exp(np.minimum(eta, _CLOGLOG_MAX_ETA))
            cdf = -np.expm1(-exp_eta)
            sf = np.exp(-exp_eta)
            pdf = exp_eta * sf
        return cdf, sf, pdf
