"""Soft labels drawn from beta distributions."""

from numbers import Integral, Real

import numpy as np
from scipy.special import betainc

# Parameters [p, q, a] of the beta distribution of each class, per number of
# classes, copied from dlordinal
_BETA_PARAMS_SETS = {
    "standard": {
        3: [[1, 4, 1], [4, 4, 1], [4, 1, 1]],
        4: [[1, 6, 1], [6, 10, 1], [10, 6, 1], [6, 1, 1]],
        5: [[1, 8, 1], [6, 14, 1], [12, 12, 1], [14, 6, 1], [8, 1, 1]],
        6: [[1, 10, 1], [7, 20, 1], [15, 20, 1], [20, 15, 1], [20, 7, 1], [10, 1, 1]],
        7: [
            [1, 12, 1],
            [7, 26, 1],
            [16, 28, 1],
            [24, 24, 1],
            [28, 16, 1],
            [26, 7, 1],
            [12, 1, 1],
        ],
        8: [
            [1, 14, 1],
            [7, 31, 1],
            [17, 37, 1],
            [27, 35, 1],
            [35, 27, 1],
            [37, 17, 1],
            [31, 7, 1],
            [14, 1, 1],
        ],
        9: [
            [1, 16, 1],
            [8, 40, 1],
            [18, 47, 1],
            [30, 47, 1],
            [40, 40, 1],
            [47, 30, 1],
            [47, 18, 1],
            [40, 8, 1],
            [16, 1, 1],
        ],
        10: [
            [1, 18, 1],
            [8, 45, 1],
            [19, 57, 1],
            [32, 59, 1],
            [45, 55, 1],
            [55, 45, 1],
            [59, 32, 1],
            [57, 19, 1],
            [45, 8, 1],
            [18, 1, 1],
        ],
        11: [
            [1, 21, 1],
            [8, 51, 1],
            [20, 68, 1],
            [34, 73, 1],
            [48, 69, 1],
            [60, 60, 1],
            [69, 48, 1],
            [73, 34, 1],
            [68, 20, 1],
            [51, 8, 1],
            [21, 1, 1],
        ],
        12: [
            [1, 23, 1],
            [8, 56, 1],
            [20, 76, 1],
            [35, 85, 1],
            [51, 85, 1],
            [65, 77, 1],
            [77, 65, 1],
            [85, 51, 1],
            [85, 35, 1],
            [76, 20, 1],
            [56, 8, 1],
            [23, 1, 1],
        ],
        13: [
            [1, 25, 1],
            [8, 61, 1],
            [20, 84, 1],
            [36, 98, 1],
            [53, 100, 1],
            [70, 95, 1],
            [84, 84, 1],
            [95, 70, 1],
            [100, 53, 1],
            [98, 36, 1],
            [84, 20, 1],
            [61, 8, 1],
            [25, 1, 1],
        ],
        14: [
            [1, 27, 1],
            [2, 17, 1],
            [5, 23, 1],
            [9, 27, 1],
            [13, 28, 1],
            [18, 28, 1],
            [23, 27, 1],
            [27, 23, 1],
            [28, 18, 1],
            [28, 13, 1],
            [27, 9, 1],
            [23, 5, 1],
            [17, 2, 1],
            [27, 1, 1],
        ],
    },
    "gen": {
        3: [[1, 2, 2], [4, 4, 1], [6, 1, 0.5]],
        4: [[1, 3, 2], [6, 10, 1], [10, 6, 1], [8, 1, 0.5]],
        5: [[1, 4, 2], [6, 14, 1], [12, 12, 1], [14, 6, 1], [10, 1, 0.5]],
        6: [[1, 5, 2], [7, 20, 1], [15, 20, 1], [20, 15, 1], [20, 7, 1], [12, 1, 0.5]],
        7: [
            [1, 6, 2],
            [7, 26, 1],
            [16, 28, 1],
            [24, 24, 1],
            [28, 16, 1],
            [26, 7, 1],
            [14, 1, 0.5],
        ],
        8: [
            [1, 7, 2],
            [7, 31, 1],
            [17, 37, 1],
            [27, 35, 1],
            [35, 27, 1],
            [37, 17, 1],
            [31, 7, 1],
            [16, 1, 0.5],
        ],
        9: [
            [1, 8, 2],
            [8, 40, 1],
            [18, 47, 1],
            [30, 47, 1],
            [40, 40, 1],
            [47, 30, 1],
            [47, 18, 1],
            [40, 8, 1],
            [20, 1, 0.5],
        ],
        10: [
            [1, 9, 2],
            [8, 45, 1],
            [19, 57, 1],
            [32, 59, 1],
            [45, 55, 1],
            [55, 45, 1],
            [59, 32, 1],
            [57, 19, 1],
            [45, 8, 1],
            [24, 1, 0.5],
        ],
        11: [
            [1, 10, 2],
            [8, 51, 1],
            [20, 68, 1],
            [34, 73, 1],
            [48, 69, 1],
            [60, 60, 1],
            [69, 48, 1],
            [73, 34, 1],
            [68, 20, 1],
            [51, 8, 1],
            [28, 1, 0.5],
        ],
        12: [
            [1, 11, 2],
            [8, 56, 1],
            [20, 76, 1],
            [35, 85, 1],
            [51, 85, 1],
            [65, 77, 1],
            [77, 65, 1],
            [85, 51, 1],
            [85, 35, 1],
            [76, 20, 1],
            [56, 8, 1],
            [32, 1, 0.5],
        ],
        13: [
            [1, 12, 2],
            [8, 61, 1],
            [20, 84, 1],
            [36, 98, 1],
            [53, 100, 1],
            [70, 95, 1],
            [84, 84, 1],
            [95, 70, 1],
            [100, 53, 1],
            [98, 36, 1],
            [84, 20, 1],
            [61, 8, 1],
            [36, 1, 0.5],
        ],
        14: [
            [1, 13, 2],
            [2, 17, 1],
            [5, 23, 1],
            [9, 27, 1],
            [13, 28, 1],
            [18, 28, 1],
            [23, 27, 1],
            [27, 23, 1],
            [28, 18, 1],
            [28, 13, 1],
            [27, 9, 1],
            [23, 5, 1],
            [17, 2, 1],
            [40, 1, 0.5],
        ],
    },
}


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


def _beta_cdf(x, p, q, a=1.0):
    """CDF of the generalised beta distribution, ``I_{x^a}(p, q)``.

    dlordinal evaluates the same function as
    ``x^(a p) / (p B(p, q)) 2F1(p, 1 - q; p + 1; x^a)``, which overflows
    ``gamma(p + q)`` for large shapes and cancels into small negative
    differences; the regularised incomplete beta function avoids both.

    Parameters
    ----------
    x : array-like
        Points in ``[0, 1]``.

    p, q : float
        Shape parameters, strictly positive.

    a : float, default=1.0
        Scaling exponent, strictly positive.

    Returns
    -------
    cdf : ndarray
        ``P(X <= x)``.
    """
    return betainc(p, q, np.asarray(x, dtype=float) ** a)


def _check_params(n_classes, params_set):
    """Resolve ``params_set`` to the validated ``(p, q, a)`` triplets."""
    if isinstance(params_set, str):
        if params_set not in _BETA_PARAMS_SETS:
            raise ValueError(
                f"params_set must be one of {sorted(_BETA_PARAMS_SETS)} or a dict, "
                f"got {params_set!r}."
            )
        table = _BETA_PARAMS_SETS[params_set]
    elif isinstance(params_set, dict):
        table = params_set
    else:
        raise ValueError(
            f"params_set must be a str or a dict, got {type(params_set).__name__}."
        )

    if n_classes not in table:
        raise ValueError(
            f"params_set {params_set!r} defines no parameters for "
            f"n_classes={n_classes}; available: {sorted(table)}."
        )

    triplets = table[n_classes]
    if len(triplets) != n_classes:
        raise ValueError(
            f"params_set must give {n_classes} [p, q, a] triplets for "
            f"n_classes={n_classes}, got {len(triplets)}."
        )

    params = []
    for triplet in triplets:
        if len(triplet) != 3 or not all(
            isinstance(v, Real) and not isinstance(v, bool) and np.isfinite(v) and v > 0
            for v in triplet
        ):
            raise ValueError(
                f"Each beta parameter triplet must be three finite positive "
                f"numbers [p, q, a], got {triplet!r}."
            )
        params.append(tuple(float(v) for v in triplet))
    return params


def get_beta_soft_labels(n_classes, params_set="standard"):
    r"""Build soft labels from beta distributions.

    NumPy port of the unimodal beta soft labelling of Vargas et al. [1]_.
    The interval :math:`[0, 1]` is split into :math:`J` equal intervals,
    one per class. The soft label of class :math:`j` is a generalised beta
    distribution with parameters :math:`(p_j, q_j, a_j)`, whose CDF is the
    regularised incomplete beta function evaluated at :math:`x^{a_j}`:

    .. math::
        F_j(x) = I_{x^{a_j}}(p_j, q_j)
            = \frac{x^{a_j p_j}}{p_j B(p_j, q_j)}
              \, {}_2F_1(p_j, 1 - q_j; p_j + 1; x^{a_j})

    Its probability for class :math:`k` is the mass that distribution
    puts on interval :math:`k`:

    .. math::
        P_{jk} = F_j\left(\frac{k}{J}\right) - F_j\left(\frac{k - 1}{J}\right)

    Parameters
    ----------
    n_classes : int
        Number of classes :math:`J`.

    params_set : {"standard", "gen"} or dict, default="standard"
        Parameters of the beta distribution of each class. The named sets
        are the ones published with dlordinal and define
        ``3 <= n_classes <= 14``. A dict maps the number of classes to a
        list of ``n_classes`` triplets ``[p, q, a]`` of finite positive
        numbers, e.g. ``{3: [[1, 4, 1], [4, 4, 1], [4, 1, 1]]}``.

    Returns
    -------
    probs : ndarray of shape (n_classes, n_classes)
        Soft labels. Row ``j`` is the target distribution of class ``j``.
        As in dlordinal, the intervals exclude ``1e-9`` at both ends, so
        rows sum to one up to about ``1e-8``.

    Raises
    ------
    ValueError
        If ``n_classes`` is not an integer of at least 2, ``params_set`` is
        neither a known name nor a dict, it defines no parameters for
        ``n_classes``, or its triplets are malformed.

    References
    ----------
    .. [1] V. M. Vargas, P. A. Gutiérrez, and C. Hervás-Martínez,
       "Unimodal regularisation based on beta distribution for deep
       ordinal regression," Pattern Recognition, vol. 122, 108310, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers._beta_distribution import get_beta_soft_labels
    >>> np.round(get_beta_soft_labels(3), 4)
    array([[0.8025, 0.1852, 0.0123],
           [0.1733, 0.6534, 0.1733],
           [0.0123, 0.1852, 0.8025]])
    """
    if (
        not isinstance(n_classes, Integral)
        or isinstance(n_classes, bool)
        or n_classes < 2
    ):
        raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}.")
    n_classes = int(n_classes)

    params = np.array(_check_params(n_classes, params_set))
    p, q, a = (params[:, i, np.newaxis] for i in range(3))
    intervals = _get_intervals(n_classes)

    return _beta_cdf(intervals[:, 1], p, q, a) - _beta_cdf(intervals[:, 0], p, q, a)
