"""Synthetic ordinal classification dataset generator."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, validate_params


def _resolve_weights(weights, n_classes):
    """Return class proportions summing to one (mirrors make_classification)."""
    if weights is None:
        return np.full(n_classes, 1.0 / n_classes)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) not in (n_classes, n_classes - 1):
        raise ValueError(
            f"weights must have length n_classes ({n_classes}) or "
            f"n_classes - 1 ({n_classes - 1}); got {len(weights)}."
        )
    if len(weights) == n_classes - 1:
        weights = np.append(weights, 1.0 - weights.sum())
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and sum to a positive value.")
    return weights / weights.sum()


@validate_params(
    {
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "n_features": [Interval(Integral, 1, None, closed="left")],
        "n_classes": [Interval(Integral, 2, None, closed="left")],
        "n_informative": [Interval(Integral, 1, None, closed="left")],
        "noise": [Interval(Real, 0, None, closed="left")],
        "weights": ["array-like", None],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def make_ordinal_classification(
    n_samples=100,
    n_features=10,
    n_classes=5,
    *,
    n_informative=5,
    noise=0.1,
    weights=None,
    random_state=None,
):
    """Generate a synthetic ordinal classification dataset.

    Uses a latent-variable data-generating process: a continuous latent score
    is computed from a linear combination of ``n_informative`` features, scaled
    to unit variance, perturbed with Gaussian noise, and then thresholded into
    ``n_classes`` ordinal bins. Because the cut points are quantiles of the
    latent score, the realised class frequencies follow ``weights`` (uniform by
    default). The noise pushes samples across *adjacent* boundaries, so it
    introduces ordinally-coherent label uncertainty rather than arbitrary
    class flips.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.

    n_features : int, default=10
        Total number of features.

    n_classes : int, default=5
        Number of ordinal classes (bins). Must be at least 2.

    n_informative : int, default=5
        Number of features that drive the latent score. The remaining
        ``n_features - n_informative`` features are irrelevant noise. Must
        satisfy ``n_informative <= n_features``.

    noise : float, default=0.1
        Standard deviation of the Gaussian noise added to the latent score.
        The latent score is standardised to unit variance first, so ``noise``
        is a dimensionless difficulty knob whose effect is consistent across
        ``n_informative`` and ``n_classes``. ``0`` yields a noise-free
        (deterministic) labelling.

    weights : array-like of shape (n_classes,) or (n_classes - 1,), default=None
        Proportion of samples assigned to each class. If ``None``, classes are
        balanced. If ``n_classes - 1`` values are given, the last proportion is
        inferred as ``1 - sum(weights)``. Values are normalised to sum to one.

    random_state : int, RandomState instance, or None, default=None
        Seed or random number generator for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix, dtype ``float64``.

    y : ndarray of shape (n_samples,)
        Ordinal integer class labels in ``{0, 1, ..., n_classes-1}``,
        dtype ``intp``.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.datasets import make_ordinal_classification
    >>> X, y = make_ordinal_classification(n_samples=200, n_classes=4, random_state=0)
    >>> X.shape, y.shape
    ((200, 10), (200,))
    >>> _, y = make_ordinal_classification(
    ...     n_samples=1000, n_classes=3, weights=[0.6, 0.3, 0.1], random_state=0
    ... )
    >>> np.round(np.bincount(y) / y.size, 1).tolist()
    [0.6, 0.3, 0.1]
    """
    if n_informative > n_features:
        raise ValueError(
            f"n_informative ({n_informative}) must not exceed "
            f"n_features ({n_features})."
        )
    proportions = _resolve_weights(weights, n_classes)

    rng = check_random_state(random_state)

    X = rng.randn(n_samples, n_features)
    w = rng.randn(n_informative)
    z = X[:, :n_informative] @ w
    # Standardise the latent score so `noise` acts on a unit-variance scale
    std = z.std()
    if std > 0:
        z = z / std
    if noise > 0:
        z = z + noise * rng.randn(n_samples)

    # Quantile cut points make the class frequencies follow `proportions`
    cut_percentiles = np.cumsum(proportions)[:-1] * 100.0
    thresholds = np.percentile(z, cut_percentiles)
    y = np.searchsorted(thresholds, z).astype(np.intp)

    return X.astype(np.float64), y
