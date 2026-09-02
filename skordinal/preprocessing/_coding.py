"""Ordinal coding utilities."""

from numbers import Integral

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions, validate_params

_VALID_DECOMPOSITIONS = (
    "ordered_partitions",
    "one_vs_next",
    "one_vs_followers",
    "one_vs_previous",
)


@validate_params(
    {
        "n_classes": [Interval(Integral, 2, None, closed="left")],
        "decomposition": [StrOptions(set(_VALID_DECOMPOSITIONS))],
    },
    prefer_skip_nested_validation=True,
)
def build_coding_matrix(n_classes, decomposition):
    """Return the coding matrix for an ordinal decomposition strategy.

    The resulting matrix has one row per class and one column per
    binary subproblem. Each entry is in ``{-1, 0, +1}``: ``+1`` marks
    the positive group of the subproblem, ``-1`` the negative group,
    ``0`` excludes the class from that subproblem.

    Parameters
    ----------
    n_classes : int
        Number of ordinal classes (must be ``>= 2``).

    decomposition : {'ordered_partitions', 'one_vs_next', 'one_vs_followers', 'one_vs_previous'}
        Decomposition strategy.

        - ``'ordered_partitions'``: subproblem ``k`` is
          ``{k+1, ..., K-1} vs {0, ..., k}``. See [1]_.
        - ``'one_vs_next'``: subproblem ``k`` is class ``k`` vs class
          ``k+1`` (other classes excluded).
        - ``'one_vs_followers'``: subproblem ``k`` is class ``k`` vs
          ``{k+1, ..., K-1}`` (preceding classes excluded).
        - ``'one_vs_previous'``: subproblem ``k`` is class ``k+1`` vs
          ``{0, ..., k}`` (following classes excluded).

    Returns
    -------
    coding : ndarray of shape (n_classes, n_classes - 1), dtype np.intp
        Coding matrix with values in ``{-1, 0, +1}``.

    Raises
    ------
    ValueError
        If ``n_classes < 2`` or ``decomposition`` is not one of the
        recognised strategies.

    Examples
    --------
    >>> from skordinal.preprocessing import build_coding_matrix
    >>> build_coding_matrix(2, "ordered_partitions")
    array([[-1],
           [ 1]])
    >>> build_coding_matrix(4, "ordered_partitions")
    array([[-1, -1, -1],
           [ 1, -1, -1],
           [ 1,  1, -1],
           [ 1,  1,  1]])

    References
    ----------
    .. [1] E. Frank and M. Hall, "A Simple Approach to Ordinal
       Classification", in Proc. 12th European Conference on Machine Learning
       (ECML 2001), pp. 145-156, 2001.
    """
    K = int(n_classes)
    coding = np.zeros((K, K - 1), dtype=np.intp)

    if decomposition == "ordered_partitions":
        for k in range(K - 1):
            coding[: k + 1, k] = -1
            coding[k + 1 :, k] = 1
    elif decomposition == "one_vs_next":
        for k in range(K - 1):
            coding[k, k] = 1
            coding[k + 1, k] = -1
    elif decomposition == "one_vs_followers":
        for k in range(K - 1):
            coding[k, k] = 1
            coding[k + 1 :, k] = -1
    else:  # one_vs_previous
        for k in range(K - 1):
            coding[: k + 1, k] = -1
            coding[k + 1, k] = 1

    return coding
