"""Margin functions of the threshold losses."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import expit


class Margin(ABC):
    """Penalty ``phi`` of a signed margin ``z = s * eta``.

    ``s = +1`` when the true class lies at or below the threshold and ``-1``
    otherwise, so a margin on the right side of its threshold is positive and
    ``phi`` decreases with ``z``. A margin gives ``phi``, its two derivatives,
    the estimate of ``P(Y <= k)`` from the unsigned margin ``eta`` and
    ``log_odds_scale``, the factor ``c`` such that ``c * log(num / den)``
    minimises ``num * phi(t) + den * phi(-t)``. Margins are stateless, and
    :func:`get_margin` returns shared instances.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name that :func:`get_margin` resolves."""

    @property
    @abstractmethod
    def log_odds_scale(self) -> float:
        """Factor ``c`` of the block minimiser ``c * log(num / den)``."""

    @abstractmethod
    def phi(self, z: np.ndarray) -> np.ndarray:
        """Return the penalty of each signed margin.

        Parameters
        ----------
        z : ndarray
            Signed margins.

        Returns
        -------
        phi : ndarray of the same shape as ``z``
            ``phi(z)``.
        """

    @abstractmethod
    def derivatives(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the first and second derivatives of ``phi``.

        Parameters
        ----------
        z : ndarray
            Signed margins.

        Returns
        -------
        dphi : ndarray of the same shape as ``z``
            ``phi'(z)``.

        d2phi : ndarray of the same shape as ``z``
            ``phi''(z)``.
        """

    @abstractmethod
    def cumproba(self, eta: np.ndarray) -> np.ndarray:
        """Return the estimate of ``P(Y <= k)`` from the unsigned margins.

        Parameters
        ----------
        eta : ndarray
            Unsigned margins ``theta_k - f``.

        Returns
        -------
        cumproba : ndarray of the same shape as ``eta``
            Estimate of ``P(Y <= k)``.
        """

    def gradient_hessian(
        self, eta: np.ndarray, above: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the two derivatives of ``phi(s * eta)`` with respect to ``eta``.

        The default signs the margins and calls :meth:`derivatives`, and a
        margin overrides it when it has a cheaper closed form.

        Parameters
        ----------
        eta : ndarray
            Unsigned margins ``theta_k - f``.

        above : ndarray of bool of the same shape as ``eta``
            Whether the sample lies above the threshold, where ``s = -1``.

        Returns
        -------
        gradient : ndarray of the same shape as ``eta``
            First derivative with respect to ``eta``.

        hessian : ndarray of the same shape as ``eta``
            Second derivative with respect to ``eta``.
        """
        s = np.where(above, -1.0, 1.0)
        dphi, d2phi = self.derivatives(s * eta)
        return s * dphi, d2phi


class LogisticMargin(Margin):
    """Logistic margin ``phi(z) = log(1 + exp(-z))``.

    Each threshold term is the negative log-likelihood of a logistic model of
    ``Y <= k``, so ``sigma(eta)`` estimates ``P(Y <= k)`` (a composite
    likelihood) and the intercept-only minimiser is the log-odds.
    """

    name = "logistic"
    log_odds_scale = 1.0

    def phi(self, z: np.ndarray) -> np.ndarray:
        """Return ``log(1 + exp(-z))`` without overflow."""
        return np.logaddexp(0.0, -z)

    def derivatives(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``-sigma(-z)`` and ``sigma(z) * sigma(-z)``."""
        # Each sigmoid is evaluated directly, so neither loses precision near 1
        sigma_minus = expit(-z)
        return -sigma_minus, sigma_minus * expit(z)

    def cumproba(self, eta: np.ndarray) -> np.ndarray:
        """Return ``sigma(eta)``."""
        return expit(eta)

    def gradient_hessian(
        self, eta: np.ndarray, above: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``above - sigma(-eta)`` and ``sigma(-eta) * sigma(eta)``."""
        sigma, sigma_minus = expit(eta), expit(-eta)
        return np.where(above, sigma, -sigma_minus), sigma * sigma_minus


class ExponentialMargin(Margin):
    """Exponential margin ``phi(z) = exp(-z)``, the loss of ORBoost.

    The population minimiser of the exponential loss is half the log-odds, so
    ``sigma(2 * eta)`` estimates ``P(Y <= k)`` [1]_.

    References
    ----------
    .. [1] J. Friedman, T. Hastie and R. Tibshirani, "Additive logistic
       regression: a statistical view of boosting", The Annals of Statistics,
       vol. 28, no. 2, pp. 337-407, 2000.
    """

    name = "exponential"
    log_odds_scale = 0.5

    def phi(self, z: np.ndarray) -> np.ndarray:
        """Return ``exp(-z)``."""
        return np.exp(-z)

    def derivatives(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``-exp(-z)`` and ``exp(-z)``."""
        e = np.exp(-z)
        return -e, e

    def cumproba(self, eta: np.ndarray) -> np.ndarray:
        """Return ``sigma(2 * eta)``."""
        return expit(2.0 * eta)


_MARGINS = {margin.name: margin for margin in (LogisticMargin(), ExponentialMargin())}


def get_margin(margin: str | Margin) -> Margin:
    """Return the margin given by name, or the margin instance given.

    Names resolve to shared instances, since margins are stateless.

    Parameters
    ----------
    margin : {"logistic", "exponential"} or Margin
        Name of the margin, or a margin instance.

    Returns
    -------
    margin : Margin
        The shared instance of that name, or ``margin`` itself.

    Raises
    ------
    ValueError
        If ``margin`` is an unknown name.
    """
    if isinstance(margin, Margin):
        return margin
    if margin not in _MARGINS:
        raise ValueError(
            f"Unknown margin {margin!r}; valid margins are "
            f"{', '.join(repr(name) for name in _MARGINS)}."
        )
    return _MARGINS[margin]
