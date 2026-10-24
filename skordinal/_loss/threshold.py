"""Threshold losses of Rennie and Srebro (2005)."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from skordinal._loss.base import BaseLoss
from skordinal._loss.margin import Margin, get_margin


def _pav(log_num, log_den, scale):
    """Return the ordered minimisers of a separable threshold problem.

    Threshold ``k`` contributes a convex term whose minimiser is
    ``scale * log(num[k] / den[k])``, with the masses given in log scale. Pool
    adjacent violators merges a block with the one before it while that one has
    a larger value, adding their masses.
    """
    blocks = []
    for k, (num, den) in enumerate(zip(log_num, log_den)):
        start, value = k, scale * (num - den)
        while blocks and blocks[-1][4] > value:
            start, _, num_before, den_before, _ = blocks.pop()
            num, den = np.logaddexp(num_before, num), np.logaddexp(den_before, den)
            value = scale * (num - den)
        blocks.append([start, k, num, den, value])
    thresholds = np.empty(len(log_num))
    for start, end, _, _, value in blocks:
        thresholds[start : end + 1] = value
    return thresholds


class _ThresholdLoss(BaseLoss):
    """The margin, probabilities and initial thresholds both threshold losses share."""

    def __init__(self, margin: str | Margin = "logistic"):
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

    def init_thresholds(
        self,
        y_true: np.ndarray,
        n_classes: int,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the ordered optimal thresholds at ``f = 0``, in closed form.

        Each threshold is ``margin.log_odds_scale`` times the log-odds of the
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
        return _pav(log_num, log_den, self.margin.log_odds_scale)


class AllThresholdLoss(_ThresholdLoss):
    """All-threshold loss ``sum_k phi(s_ik * eta_ik)`` over every threshold.

    ``s_ik = +1`` if ``y_i <= k`` and ``-1`` otherwise, so every margin of a
    sample is penalised. With the logistic margin it is the objective of
    ``LogisticAT`` and with the exponential margin that of ORBoost-All. Threshold
    ``k`` sees the mass of the classes up to ``k`` against the mass of the
    classes above it, so its optima at ``f = 0`` are ordered by construction.

    Parameters
    ----------
    margin : {"logistic", "exponential"} or Margin, default="logistic"
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


class ImmediateThresholdLoss(_ThresholdLoss):
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
    margin : {"logistic", "exponential"} or Margin, default="logistic"
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
