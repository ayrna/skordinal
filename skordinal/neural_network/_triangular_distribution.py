"""Soft labels drawn from triangular distributions."""

from numbers import Integral, Real

import numpy as np

# Largest alpha2 for which every middle-class triangle stays inside its own
# interval and its two neighbours; beyond it rows leak mass to farther
# classes and below 0, so they no longer sum to one
_MAX_ALPHA2 = 2.0 / 9.0


def _get_intervals(n_classes):
    """Split [0, 1] into ``n_classes`` consecutive intervals.

    The end points are pulled in by ``1e-9``, as in dlordinal.

    Returns
    -------
    intervals : ndarray of shape (n_classes, 2)
        Lower and upper bound of each interval.
    """
    points = np.linspace(1e-9, 1 - 1e-9, n_classes + 1)
    return np.column_stack([points[:-1], points[1:]])


def _triangular_cdf(x, a, b, c):
    """Evaluate the CDF of the triangular distribution.

    Broadcasts over all arguments. The degenerate triangles ``c == a``
    and ``c == b`` are supported.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the CDF.

    a, b, c : array-like
        Lower bound, upper bound and mode, with ``a <= c <= b`` and
        ``a < b``.

    Returns
    -------
    cdf : ndarray
        ``P(X <= x)``.
    """
    x, a, b, c = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, a, b, c))
    )

    # Guard the denominators so the branch that np.select discards for a
    # degenerate triangle does not divide by zero
    left_den = (b - a) * (c - a)
    right_den = (b - a) * (b - c)
    left = (x - a) ** 2 / np.where(left_den > 0, left_den, 1.0)
    right = 1.0 - (b - x) ** 2 / np.where(right_den > 0, right_den, 1.0)

    return np.select([x <= a, x <= c, x < b], [0.0, left, right], default=1.0)


def _triangular_params(n_classes, alpha2):
    """Return the lower bound, upper bound and mode of each class triangle."""
    sqrt_2alpha2 = np.sqrt(2 * alpha2)

    c_minus = (1 - 2 * alpha2) * (2 * alpha2 - sqrt_2alpha2)
    alpha1 = ((1 - np.sqrt(1 - 4 * c_minus)) / 2) ** 2
    c1 = (
        ((n_classes - 1) / n_classes) ** 2
        * (1 - 2 * alpha2)
        * (sqrt_2alpha2 * (sqrt_2alpha2 - 1))
    )
    alpha3 = ((1 - np.sqrt(1 - 4 * c1)) / 2) ** 2

    a = np.empty(n_classes)
    b = np.empty(n_classes)
    c = np.empty(n_classes)

    a[0], b[0], c[0] = 0.0, 1.0 / ((1.0 - np.sqrt(alpha1)) * n_classes), 0.0

    # The middle classes solve a quadratic; dlordinal picks the root that
    # puts a below (j - 1) / J and b above j / J
    j = np.arange(2, n_classes, dtype=float)
    den = 2.0 * n_classes * (1 - 2 * alpha2)
    a_num = 2.0 * j - 2 - 4 * j * alpha2 + 2 * alpha2
    b_num = 2.0 * j - 4 * j * alpha2 + 2 * alpha2
    a_plus = (a_num + sqrt_2alpha2) / den
    b_plus = (b_num + sqrt_2alpha2) / den
    a[1:-1] = np.where(
        a_plus < (j - 1.0) / n_classes, a_plus, (a_num - sqrt_2alpha2) / den
    )
    b[1:-1] = np.where(b_plus > j / n_classes, b_plus, (b_num - sqrt_2alpha2) / den)
    c[1:-1] = (a[1:-1] + b[1:-1]) / 2.0

    a_last = 1.0 + 1.0 / (n_classes * (np.sqrt(alpha3) - 1.0))
    a[-1] = (
        a_last if a_last > 0.0 else 1.0 - 1.0 / (n_classes * (np.sqrt(alpha3) - 1.0))
    )
    b[-1], c[-1] = 1.0, 1.0

    return a, b, c


def get_triangular_soft_labels(n_classes, alpha2=0.01):
    r"""Build soft labels from triangular distributions.

    NumPy port of the triangular soft labelling of Vargas et al. [1]_.
    The interval :math:`[0, 1]` is split into :math:`J` equal intervals,
    one per class. The soft label of class :math:`j` is a triangular
    distribution :math:`p(x; a_j, b_j, c_j)` with lower bound
    :math:`a_j`, upper bound :math:`b_j` and mode :math:`c_j`, and its
    probability for class :math:`k` is the mass that distribution puts on
    interval :math:`k`.

    The triangles are chosen so that every middle class leaves
    probability :math:`\alpha_2` on each of its two neighbours and
    :math:`1 - 2\alpha_2` on itself:

    .. math::
        a_j = \begin{cases}
            0 & j = 1 \\
            \frac{2j - 2 - 4j\alpha_2 + 2\alpha_2 \pm
              \sqrt{2\alpha_2}}{2J(1 - 2\alpha_2)} & 1 < j < J \\
            1 + \frac{1}{\pm J (\sqrt{\alpha_3} - 1)} & j = J
        \end{cases}

    .. math::
        b_j = \begin{cases}
            \frac{1}{(1 - \sqrt{\alpha_1})J} & j = 1 \\
            \frac{2j - 4j\alpha_2 + 2\alpha_2 \pm
              \sqrt{2\alpha_2}}{2J(1 - 2\alpha_2)} & 1 < j < J \\
            1 & j = J
        \end{cases}

    .. math::
        c_j = \begin{cases}
            0 & j = 1 \\
            \frac{a_j + b_j}{2} & 1 < j < J \\
            1 & j = J
        \end{cases}

    where the signs are chosen so that :math:`a_j < (j - 1) / J` and
    :math:`b_j > j / J`. The first and last classes are half triangles
    whose errors :math:`\alpha_1` and :math:`\alpha_3` follow from
    :math:`\alpha_2`:

    .. math::
        \alpha_1 = \left(\frac{1 - \sqrt{1 - 4(1 - 2\alpha_2)
            (2\alpha_2 - \sqrt{2\alpha_2})}}{2}\right)^2

    .. math::
        \alpha_3 = \left(\frac{1 - \sqrt{1 - 4\left(\frac{J - 1}{J}\right)^2
            (1 - 2\alpha_2)\sqrt{2\alpha_2}(\sqrt{2\alpha_2} - 1)}}{2}\right)^2

    Parameters
    ----------
    n_classes : int
        Number of classes :math:`J`. Must be at least 2.

    alpha2 : float, default=0.01
        Probability :math:`\alpha_2` given to each neighbour of a middle
        class. Must lie in ``[0, 2/9]`` when ``n_classes >= 3``: beyond
        ``2/9`` a middle triangle spills past its two neighbours and the
        rows stop being distributions. With ``n_classes == 2`` there is no
        middle class and any value in ``[0, 1]`` is accepted.

    Returns
    -------
    probs : ndarray of shape (n_classes, n_classes)
        Soft labels. Row ``j`` is the target distribution of class ``j``.
        As in dlordinal, the intervals exclude ``1e-9`` at both ends, so
        rows sum to one up to about ``1e-8``.

    Raises
    ------
    ValueError
        If ``n_classes`` is not an integer of at least 2, or ``alpha2`` is
        not a finite number in its valid range.

    References
    ----------
    .. [1] V. M. Vargas, P. A. Gutiérrez, J. Barbero-Gómez, and
       C. Hervás-Martínez, "Soft labelling based on triangular distributions
       for ordinal classification," Information Fusion, vol. 93,
       pp. 258-267, 2023.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.neural_network._triangular_distribution import (
    ...     get_triangular_soft_labels,
    ... )
    >>> np.round(get_triangular_soft_labels(5), 4)
    array([[0.9885, 0.0115, 0.    , 0.    , 0.    ],
           [0.01  , 0.98  , 0.01  , 0.    , 0.    ],
           [0.    , 0.01  , 0.98  , 0.01  , 0.    ],
           [0.    , 0.    , 0.01  , 0.98  , 0.01  ],
           [0.    , 0.    , 0.    , 0.0051, 0.9949]])
    """
    if (
        not isinstance(n_classes, Integral)
        or isinstance(n_classes, bool)
        or n_classes < 2
    ):
        raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}.")

    max_alpha2 = 1.0 if n_classes == 2 else _MAX_ALPHA2
    if (
        not isinstance(alpha2, Real)
        or isinstance(alpha2, bool)
        or not np.isfinite(alpha2)
        or not 0.0 <= alpha2 <= max_alpha2
    ):
        raise ValueError(
            f"alpha2 must be a float in [0, {max_alpha2:.4g}] for "
            f"n_classes={n_classes}, got {alpha2!r}."
        )

    n_classes = int(n_classes)
    a, b, c = _triangular_params(n_classes, float(alpha2))
    intervals = _get_intervals(n_classes)

    cdf_upper = _triangular_cdf(
        intervals[np.newaxis, :, 1], a[:, None], b[:, None], c[:, None]
    )
    cdf_lower = _triangular_cdf(
        intervals[np.newaxis, :, 0], a[:, None], b[:, None], c[:, None]
    )
    return cdf_upper - cdf_lower
