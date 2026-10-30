"""Losses of the threshold models and their abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from skordinal._loss.margin import BaseMargin, get_margin
from skordinal.utils.extmath import cumproba_to_proba


class BaseLoss(ABC):
    """Loss of a threshold model, written on the margins ``eta_ik = theta_k - f_i``.

    A threshold model scores sample ``i`` with a latent projection ``f_i``, the
    ``raw_prediction`` of scikit-learn's losses, and places ``K - 1`` ordered
    thresholds ``theta`` between its ``K`` classes. ``F(eta_ik)`` estimates the
    probability of the low event ``Y <= k``, so a larger ``f`` always means a
    larger class. Each loss is a function of the margins, and what it returns
    is already differentiated with respect to ``f`` and ``theta``, so that the
    estimators only chain through their own parameters (``X @ w``, a booster, a
    network).

    Conventions shared by every loss.

    - ``y_true`` holds class indices ``0, ..., K - 1`` (``intp``),
      ``raw_prediction`` has shape ``(n_samples,)`` and ``thresholds`` shape
      ``(n_classes - 1,)``, all ``float64`` ndarrays, with ``n_samples >= 1``,
      some positive weight and ``n_classes >= 2``. Nothing is validated or
      converted, except by ``cumproba_to_proba`` in :meth:`predict_proba`.
    - Per-sample outputs are already multiplied by ``sample_weight``, and only
      the gradient in the thresholds is summed over samples, so the estimator
      forms ``J = loss.sum() / n + penalty``.
    - A loss holds no state. It is built from the estimator's parameters.

    Subclasses implement the abstract methods. This class derives :meth:`loss`
    and :meth:`predict_proba` from them.
    """

    def loss(
        self,
        y_true: np.ndarray,
        raw_prediction: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the weighted loss of each sample.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Class indices in ``0, ..., n_classes - 1``.

        raw_prediction : ndarray of shape (n_samples,)
            Latent projection ``f`` of each sample.

        thresholds : ndarray of shape (n_classes - 1,)
            Ordered thresholds.

        sample_weight : ndarray of shape (n_samples,), default=None
            Weight of each sample, ``None`` for unit weights.

        Returns
        -------
        loss : ndarray of shape (n_samples,)
            ``sample_weight[i] * l_i``, not summed over samples.
        """
        return self.loss_gradient(y_true, raw_prediction, thresholds, sample_weight)[0]

    @abstractmethod
    def loss_gradient(
        self,
        y_true: np.ndarray,
        raw_prediction: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the weighted loss and its gradients in one pass.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Class indices in ``0, ..., n_classes - 1``.

        raw_prediction : ndarray of shape (n_samples,)
            Latent projection ``f`` of each sample.

        thresholds : ndarray of shape (n_classes - 1,)
            Ordered thresholds.

        sample_weight : ndarray of shape (n_samples,), default=None
            Weight of each sample, ``None`` for unit weights.

        Returns
        -------
        loss : ndarray of shape (n_samples,)
            ``sample_weight[i] * l_i``, not summed over samples.

        gradient : ndarray of shape (n_samples,)
            Derivative of ``loss`` with respect to ``raw_prediction``.

        threshold_gradient : ndarray of shape (n_classes - 1,)
            Derivative of ``loss.sum()`` with respect to ``thresholds``.
        """

    @abstractmethod
    def gradient_hessian(
        self,
        y_true: np.ndarray,
        raw_prediction: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the first and second derivatives of each sample's loss in ``f``.

        Samples do not interact, so the second derivatives are the diagonal of
        the Hessian. Both are what XGBoost and LightGBM take as the gradient and
        Hessian of a custom objective.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Class indices in ``0, ..., n_classes - 1``.

        raw_prediction : ndarray of shape (n_samples,)
            Latent projection ``f`` of each sample.

        thresholds : ndarray of shape (n_classes - 1,)
            Ordered thresholds.

        sample_weight : ndarray of shape (n_samples,), default=None
            Weight of each sample, ``None`` for unit weights.

        Returns
        -------
        gradient : ndarray of shape (n_samples,)
            Derivative of ``loss`` with respect to ``raw_prediction``.

        hessian : ndarray of shape (n_samples,)
            Second derivative of ``loss`` with respect to ``raw_prediction``.
        """

    @abstractmethod
    def predict_cumproba(
        self, raw_prediction: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Return the estimates of ``P(Y <= k)`` at every threshold.

        Parameters
        ----------
        raw_prediction : ndarray of shape (n_samples,)
            Latent projection ``f`` of each sample.

        thresholds : ndarray of shape (n_classes - 1,)
            Ordered thresholds.

        Returns
        -------
        cumproba : ndarray of shape (n_samples, n_classes - 1)
            Estimate of ``P(Y <= k)`` for each sample and threshold ``k``.
        """

    def predict_proba(
        self, raw_prediction: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Return the probability of each class.

        The cumulative probabilities of :meth:`predict_cumproba` are made
        non-decreasing by isotonic regression where needed and differenced.

        Parameters
        ----------
        raw_prediction : ndarray of shape (n_samples,)
            Latent projection ``f`` of each sample.

        thresholds : ndarray of shape (n_classes - 1,)
            Ordered thresholds.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of each class for each sample, with rows summing to one.
        """
        return cumproba_to_proba(self.predict_cumproba(raw_prediction, thresholds))

    @abstractmethod
    def fit_intercept_only(
        self,
        y_true: np.ndarray,
        n_classes: int,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the optimal thresholds without covariates, at ``f = 0``.

        It deviates from scikit-learn's ``fit_intercept_only`` in two ways.
        ``n_classes`` is explicit, because a loss holds no number of classes, and
        the result is the ``n_classes - 1`` thresholds instead of one intercept.
        A class without mass counts as half a sample of the mean positive weight,
        so the result is finite and depends on the weights only through their
        ratios, and with every class present it is the exact optimum.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Class indices in ``0, ..., n_classes - 1``.

        n_classes : int
            Number of classes, explicit because a class may have no sample.

        sample_weight : ndarray of shape (n_samples,), default=None
            Weight of each sample, ``None`` for unit weights.

        Returns
        -------
        thresholds : ndarray of shape (n_classes - 1,)
            Ordered optimal thresholds.
        """


def _pav(log_num, log_den):
    """Return the log-ratios ``log(num[k] / den[k])`` pooled until non-decreasing.

    The masses are given in log scale. Pool adjacent violators merges a block
    with the one before it while that one has a larger log-ratio, adding their
    masses. A margin's optimal threshold is non-decreasing in the log-ratio, so
    it maps the pooled log-ratios to the ordered minimisers of a separable
    threshold problem.
    """
    blocks = []
    for k, (num, den) in enumerate(zip(log_num, log_den)):
        start, value = k, num - den
        while blocks and blocks[-1][4] > value:
            start, _, num_before, den_before, _ = blocks.pop()
            num, den = np.logaddexp(num_before, num), np.logaddexp(den_before, den)
            value = num - den
        blocks.append([start, k, num, den, value])
    log_ratios = np.empty(len(log_num))
    for start, end, _, _, value in blocks:
        log_ratios[start : end + 1] = value
    return log_ratios


class _BaseThresholdLoss(BaseLoss):
    """The margin, probabilities and initial thresholds both threshold losses share."""

    def __init__(self, margin: str | BaseMargin = "logistic"):
        self.margin = get_margin(margin)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(margin={self.margin.name!r})"

    def predict_cumproba(
        self, raw_prediction: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Return the margin's estimate of ``P(Y <= k)`` at every threshold."""
        return self.margin.cumproba(thresholds - raw_prediction[:, None])

    @abstractmethod
    def _pairs(self, log_masses):
        """Return the log-masses on each side of every threshold."""

    def fit_intercept_only(
        self,
        y_true: np.ndarray,
        n_classes: int,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the ordered optimal thresholds at ``f = 0``, in closed form.

        Each threshold is ``margin.optimal_threshold`` of the log-ratio of the
        class masses on its two sides, ordered with pool adjacent violators.
        Pooled thresholds are tied, and a tie is a zero parameter with zero
        gradient in ``params_to_thresholds``, so a solver started there
        separates the ties first.
        """
        counts = np.bincount(y_true, sample_weight, n_classes).astype(np.float64)
        n_weighted = (
            len(y_true) if sample_weight is None else np.count_nonzero(sample_weight)
        )
        # An empty class weighs half a mean sample, so no frequency is 0 or 1
        counts[counts == 0.0] = 0.5 * counts.sum() / n_weighted
        log_num, log_den = self._pairs(np.log(counts))
        return self.margin.optimal_threshold(_pav(log_num, log_den))


class AllThresholdLoss(_BaseThresholdLoss):
    """All-threshold loss ``sum_k phi(s_ik * eta_ik)`` over every threshold.

    ``s_ik = +1`` if ``y_i <= k`` and ``-1`` otherwise, so every margin of a
    sample is penalised. With the logistic margin it is the objective of
    ``LogisticAT`` and with the exponential margin that of ORBoost-All. Threshold
    ``k`` sees the mass of the classes up to ``k`` against the mass of the
    classes above it, so its optima at ``f = 0`` are ordered by construction.

    Parameters
    ----------
    margin : {"logistic", "exponential"} or BaseMargin, default="logistic"
        Margin function ``phi``.

    References
    ----------
    .. [1] J. D. M. Rennie and N. Srebro, "Loss functions for preference
       levels: regression with discrete ordered labels", IJCAI Multidisciplinary
       Workshop on Advances in Preference Handling, 2005.

    .. [2] H.-T. Lin and L. Li, "Large-margin thresholded ensembles for ordinal
       regression: theory and practice", Algorithmic Learning Theory, LNAI
       4264, 319-333, 2006.
    """

    def loss_gradient(
        self,
        y_true: np.ndarray,
        raw_prediction: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the weighted loss and its gradients in ``f`` and ``thresholds``."""
        s = np.where(y_true[:, None] <= np.arange(len(thresholds)), 1.0, -1.0)
        z = s * (thresholds - raw_prediction[:, None])
        loss = self.margin.phi(z).sum(axis=1)
        # Derivative of each term in eta, where deta/df = -1 and deta/dtheta = 1
        g = s * self.margin.derivatives(z)[0]
        if sample_weight is not None:
            loss *= sample_weight
            g *= sample_weight[:, None]
        return loss, -g.sum(axis=1), g.sum(axis=0)

    def gradient_hessian(
        self,
        y_true: np.ndarray,
        raw_prediction: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``dl_i/df_i`` and ``d2l_i/df_i2`` of each sample."""
        above = y_true[:, None] > np.arange(len(thresholds))
        g, h = self.margin.gradient_hessian(thresholds - raw_prediction[:, None], above)
        grad, hess = -g.sum(axis=1), h.sum(axis=1)
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess

    def _pairs(self, log_masses):
        """Return the log-masses of the classes up to and above each threshold."""
        below = np.logaddexp.accumulate(log_masses)[:-1]
        above = np.logaddexp.accumulate(log_masses[::-1])[::-1][1:]
        return below, above


class ImmediateThresholdLoss(_BaseThresholdLoss):
    """Immediate-threshold loss ``phi(eta_{i, y_i}) + phi(-eta_{i, y_i - 1})``.

    Each sample only sees the threshold above its class (``k = y_i``, when
    ``y_i < K - 1``) and the one below (``k = y_i - 1``, when ``y_i > 0``).
    With the logistic margin it is the objective of ``LogisticIT`` and with the
    exponential margin that of ORBoost-LR. Threshold ``k`` only depends on
    the classes ``k`` and ``k + 1``, so its optima need not be ordered and are
    ordered with pool adjacent violators. The probabilities apply the margin's
    estimate to every threshold, as the all-threshold loss does, and are a
    heuristic, because each sample only informs its two adjacent thresholds.

    Parameters
    ----------
    margin : {"logistic", "exponential"} or BaseMargin, default="logistic"
        Margin function ``phi``.

    References
    ----------
    .. [1] J. D. M. Rennie and N. Srebro, "Loss functions for preference
       levels: regression with discrete ordered labels", IJCAI Multidisciplinary
       Workshop on Advances in Preference Handling, 2005.

    .. [2] H.-T. Lin and L. Li, "Large-margin thresholded ensembles for ordinal
       regression: theory and practice", Algorithmic Learning Theory, LNAI
       4264, 319-333, 2006.
    """

    def _adjacent(self, y_true, raw_prediction, thresholds):
        """Return rows, thresholds and signed margins of the upper and lower terms."""
        i_up = np.flatnonzero(y_true < len(thresholds))
        i_lo = np.flatnonzero(y_true > 0)
        k_up, k_lo = y_true[i_up], y_true[i_lo] - 1
        upper = (i_up, k_up, thresholds[k_up] - raw_prediction[i_up])
        lower = (i_lo, k_lo, raw_prediction[i_lo] - thresholds[k_lo])
        return upper, lower

    def loss_gradient(
        self,
        y_true: np.ndarray,
        raw_prediction: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the weighted loss and its gradients in ``f`` and ``thresholds``."""
        upper, lower = self._adjacent(y_true, raw_prediction, thresholds)
        (i_up, k_up, z_up), (i_lo, k_lo, z_lo) = upper, lower
        d_up, d_lo = self.margin.derivatives(z_up)[0], self.margin.derivatives(z_lo)[0]
        loss, gradient = np.zeros(len(y_true)), np.zeros(len(y_true))
        loss[i_up] = self.margin.phi(z_up)
        loss[i_lo] += self.margin.phi(z_lo)
        if sample_weight is not None:
            loss *= sample_weight
            d_up, d_lo = d_up * sample_weight[i_up], d_lo * sample_weight[i_lo]
        # dz_up = dtheta_y - df and dz_lo = df - dtheta_{y - 1}
        gradient[i_up] = -d_up
        gradient[i_lo] += d_lo
        m = len(thresholds)
        threshold_gradient = np.bincount(k_up, d_up, m) - np.bincount(k_lo, d_lo, m)
        return loss, gradient, threshold_gradient

    def gradient_hessian(
        self,
        y_true: np.ndarray,
        raw_prediction: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``dl_i/df_i`` and ``d2l_i/df_i2`` of each sample."""
        upper, lower = self._adjacent(y_true, raw_prediction, thresholds)
        (i_up, _, z_up), (i_lo, _, z_lo) = upper, lower
        d_up, h_up = self.margin.derivatives(z_up)
        d_lo, h_lo = self.margin.derivatives(z_lo)
        grad, hess = np.zeros(len(y_true)), np.zeros(len(y_true))
        grad[i_up] = -d_up
        grad[i_lo] += d_lo
        hess[i_up] = h_up
        hess[i_lo] += h_lo
        if sample_weight is not None:
            grad *= sample_weight
            hess *= sample_weight
        return grad, hess

    def _pairs(self, log_masses):
        """Return the log-masses of the two classes adjacent to each threshold."""
        return log_masses[:-1], log_masses[1:]
