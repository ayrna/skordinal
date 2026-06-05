"""Ordinal coding utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

_VALID_DECOMPOSITIONS = (
    "ordered_partitions",
    "one_vs_next",
    "one_vs_followers",
    "one_vs_previous",
)


def ordinal_to_binary_cumulative(
    y: ArrayLike,
    classes: ArrayLike,
) -> np.ndarray:
    """Encode ordinal targets into K-1 binary cumulative problems.

    Column ``k`` represents the binary problem
    "class > ``classes[k]``": entries are ``1`` when the original label
    exceeds the ``k``-th class threshold and ``0`` otherwise. Each row
    is therefore non-increasing in ``k``.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Ordinal targets.

    classes : array-like of shape (n_classes,)
        Sorted unique class labels.

    Returns
    -------
    Y_binary : ndarray of shape (n_samples, n_classes - 1), dtype np.intp
        Cumulative binary encoding.

    Raises
    ------
    ValueError
        If ``len(classes) < 2``, or if any element of ``y`` is not present
        in ``classes``.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.preprocessing import ordinal_to_binary_cumulative
    >>> classes = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 2, 1])
    >>> ordinal_to_binary_cumulative(y, classes)
    array([[0, 0],
           [1, 0],
           [1, 1],
           [1, 0]])
    """
    y = np.asarray(y)
    classes = np.asarray(classes)
    if len(classes) < 2:
        raise ValueError(
            f"classes must contain at least 2 unique labels; got {len(classes)}."
        )
    if not np.isin(y, classes).all():
        unknown = np.setdiff1d(np.unique(y), classes)
        raise ValueError(f"y contains values not in classes; got {unknown.tolist()}.")
    return (y[:, None] > classes[:-1]).astype(np.intp)


def binary_cumulative_to_ordinal(
    binary_preds: ArrayLike,
    classes: ArrayLike,
) -> np.ndarray:
    """Decode K-1 binary predictions back to ordinal class labels.

    Accepts either hard ``{0, 1}`` predictions or continuous
    probabilities in ``[0, 1]``; both are reduced via ``> 0.5`` to a
    per-row count of crossed thresholds, which indexes into
    ``classes``.

    This routine does NOT correct monotonicity violations — call
    :func:`skordinal.utils.validation.check_monotonic_probabilities`
    beforehand if the binary predictions may be non-monotone.

    Parameters
    ----------
    binary_preds : array-like of shape (n_samples, n_classes - 1)
        Hard or soft cumulative binary predictions.

    classes : array-like of shape (n_classes,)
        Sorted unique class labels.

    Returns
    -------
    y : ndarray of shape (n_samples,)
        Decoded ordinal labels, drawn from ``classes``.

    Raises
    ------
    ValueError
        If ``len(classes) < 2`` or if
        ``binary_preds.shape[1] != len(classes) - 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.preprocessing import binary_cumulative_to_ordinal
    >>> classes = np.array([0, 1, 2])
    >>> B = np.array([[0, 0], [1, 0], [1, 1]])
    >>> binary_cumulative_to_ordinal(B, classes)
    array([0, 1, 2])
    """
    binary_preds = np.asarray(binary_preds)
    classes = np.asarray(classes)
    if len(classes) < 2:
        raise ValueError(
            f"classes must contain at least 2 unique labels; got {len(classes)}."
        )
    if binary_preds.ndim != 2 or binary_preds.shape[1] != len(classes) - 1:
        raise ValueError(
            f"binary_preds must have shape (n_samples, {len(classes) - 1}); "
            f"got {binary_preds.shape}."
        )
    indices = (binary_preds > 0.5).sum(axis=1)
    return classes[indices]


def build_coding_matrix(
    n_classes: int,
    decomposition: str,
) -> np.ndarray:
    """Return the coding matrix for an ordinal decomposition strategy.

    The resulting matrix has one row per class and one column per
    binary subproblem. Each entry is in ``{-1, 0, +1}``: ``+1`` marks
    the positive group of the subproblem, ``-1`` the negative group,
    ``0`` excludes the class from that subproblem.

    Parameters
    ----------
    n_classes : int
        Number of ordinal classes (must be ``>= 3``).

    decomposition : {'ordered_partitions', 'one_vs_next', 'one_vs_followers', 'one_vs_previous'}
        Decomposition strategy.

        - ``'ordered_partitions'``: subproblem ``k`` is
          ``{0, ..., k} vs {k+1, ..., K-1}``. See [1]_.
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
        If ``n_classes < 3`` or ``decomposition`` is not one of the
        recognised strategies.

    Examples
    --------
    >>> from skordinal.preprocessing import build_coding_matrix
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
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 3:
        raise ValueError(f"n_classes must be an integer >= 3; got {n_classes!r}.")
    if decomposition not in _VALID_DECOMPOSITIONS:
        raise ValueError(
            f"decomposition must be one of {list(_VALID_DECOMPOSITIONS)}; "
            f"got {decomposition!r}."
        )

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
