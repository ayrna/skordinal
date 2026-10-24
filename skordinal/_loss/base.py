"""Base class of the losses of the threshold models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

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
    def init_thresholds(
        self,
        y_true: np.ndarray,
        n_classes: int,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the optimal thresholds without covariates, at ``f = 0``.

        The analogue of ``fit_intercept_only`` in scikit-learn's losses. A class
        without mass counts as half a sample of the mean positive weight, so the
        result is finite and depends on the weights only through their ratios,
        and with every class present it is the exact optimum.

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
