"""Cross-entropy loss regularised with beta soft labels."""

from numbers import Real

import numpy as np
from scipy.special import logsumexp

from ._beta_distribution import get_beta_soft_labels


class BetaLoss:
    """Cross-entropy against beta soft labels.

    NumPy port of the beta regularised loss of Vargas et al. [1]_. Each
    one-hot target is blended with the beta soft label of its class,

        T = (1 - eta) * Y + eta * Y @ cls_probs

    and the loss is the weighted mean cross-entropy between ``T`` and the
    softmax of the logits. Only cross-entropy is supported as the base
    loss.

    Parameters
    ----------
    n_classes : int
        Number of classes. The named parameter sets cover
        ``3 <= n_classes <= 14``.

    params_set : {"standard", "gen"} or dict, default="standard"
        Parameters of the beta distribution of each class. See
        :func:`get_beta_soft_labels`.

    eta : float, default=1.0
        Weight of the soft labels in the targets, in ``[0, 1]``. ``1.0``
        uses the soft labels alone and ``0.0`` recovers the standard
        cross-entropy on one-hot targets.

    Attributes
    ----------
    cls_probs : ndarray of shape (n_classes, n_classes)
        Beta soft label of each class, one per row.

    References
    ----------
    .. [1] V. M. Vargas, P. A. Gutiérrez, and C. Hervás-Martínez,
       "Unimodal regularisation based on beta distribution for deep
       ordinal regression," Pattern Recognition, vol. 122, 108310, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers._beta_loss import BetaLoss
    >>> loss_fn = BetaLoss(n_classes=3, eta=1.0)
    >>> Y = np.eye(3)[[0, 2]]
    >>> logits = np.array([[2.0, 0.0, -2.0], [-1.0, 0.0, 3.0]])
    >>> loss, grad = loss_fn.loss_and_grad(logits, Y)
    >>> grad.shape
    (2, 3)
    """

    def __init__(self, n_classes, params_set="standard", eta=1.0):
        if (
            not isinstance(eta, Real)
            or isinstance(eta, bool)
            or not np.isfinite(eta)
            or not 0.0 <= eta <= 1.0
        ):
            raise ValueError(f"eta must be a float in [0, 1], got {eta!r}.")

        self.cls_probs = get_beta_soft_labels(n_classes, params_set)
        self.n_classes = int(n_classes)
        self.params_set = params_set
        self.eta = float(eta)

    def soft_targets(self, Y):
        """Blend one-hot targets with the beta soft labels.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_classes)
            One-hot encoded targets.

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            Soft targets ``(1 - eta) * Y + eta * Y @ cls_probs``.
        """
        Y = self._check_2d(Y, "Y")
        return (1.0 - self.eta) * Y + self.eta * (Y @ self.cls_probs)

    def loss_and_grad(self, logits, Y, sample_weight=None):
        """Compute the loss and its gradient with respect to the logits.

        Parameters
        ----------
        logits : array-like of shape (n_samples, n_classes)
            Unnormalised class scores.

        Y : array-like of shape (n_samples, n_classes)
            One-hot encoded targets.

        sample_weight : array-like of shape (n_samples,) or None, default=None
            Per-sample weights. ``None`` weights every sample by one.

        Returns
        -------
        loss : float
            ``-sum_i w_i sum_k T_ik log softmax(logits)_ik / sum_i w_i``.

        grad : ndarray of shape (n_samples, n_classes)
            Gradient of ``loss`` with respect to ``logits``.
        """
        logits = self._check_2d(logits, "logits")
        T = self.soft_targets(Y)
        if T.shape[0] != logits.shape[0]:
            raise ValueError(
                f"logits and Y must have the same number of samples, got "
                f"{logits.shape[0]} and {T.shape[0]}."
            )

        if sample_weight is None:
            sample_weight = np.ones(logits.shape[0])
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != (logits.shape[0],):
            raise ValueError(
                f"sample_weight must have shape ({logits.shape[0]},), got "
                f"{sample_weight.shape}."
            )
        weight_sum = np.sum(sample_weight)

        log_proba = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.sum(sample_weight[:, np.newaxis] * T * log_proba) / weight_sum

        # The soft labels lose ~1e-9 of mass at the interval ends, so keep
        # the row sums of T instead of assuming they are one
        grad = np.exp(log_proba) * T.sum(axis=1, keepdims=True) - T
        grad *= sample_weight[:, np.newaxis] / weight_sum

        return float(loss), grad

    def _check_2d(self, array, name):
        array = np.asarray(array, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_classes:
            raise ValueError(
                f"{name} must have shape (n_samples, {self.n_classes}), got "
                f"{array.shape}."
            )
        return array
