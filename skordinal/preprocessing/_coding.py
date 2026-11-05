"""Ordinal coding utilities."""

from numbers import Integral

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions, validate_params

_VALID_DECOMPOSITIONS = (
    "cumulative",
    "adjacent",
    "sequential",
    "sequential_reverse",
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

    decomposition : {'cumulative', 'adjacent', 'sequential', 'sequential_reverse'}
        Decomposition, named after the cumulative, sequential and adjacent
        categories models. Binary subproblem ``k`` compares the classes
        ``{k+1, ..., K-1}`` with ``{0, ..., k}`` in ``'cumulative'`` [1]_, class
        ``k`` with class ``k+1`` in ``'adjacent'``, class ``k`` with the classes
        ``{k+1, ..., K-1}`` in ``'sequential'``, and class ``k+1`` with
        ``{0, ..., k}`` in ``'sequential_reverse'``, the sequential model with
        the class order reversed. The other classes are left out of the
        subproblem.

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
    >>> build_coding_matrix(2, "cumulative")
    array([[-1],
           [ 1]])
    >>> build_coding_matrix(4, "cumulative")
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

    if decomposition == "cumulative":
        for k in range(K - 1):
            coding[: k + 1, k] = -1
            coding[k + 1 :, k] = 1
    elif decomposition == "adjacent":
        for k in range(K - 1):
            coding[k, k] = 1
            coding[k + 1, k] = -1
    elif decomposition == "sequential":
        for k in range(K - 1):
            coding[k, k] = 1
            coding[k + 1 :, k] = -1
    else:  # sequential_reverse
        for k in range(K - 1):
            coding[: k + 1, k] = -1
            coding[k + 1, k] = 1

    return coding
