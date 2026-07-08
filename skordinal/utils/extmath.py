"""Mathematical utilities shared across ordinal classifiers."""

import numpy as np
from sklearn.isotonic import isotonic_regression
from sklearn.utils import check_array


def params_to_thresholds(params):
    """Map unconstrained threshold parameters to ordered thresholds.

    Applies the cumsum-of-squares transform:

        b[0] = params[0]
        b[k] = params[0] + sum_{j=1..k} params[j]^2   for k >= 1

    This guarantees ``b[0] <= b[1] <= ... <= b[K-2]`` for any unconstrained
    ``params``, enabling gradient-based optimisation without explicit
    ordering constraints.

    Parameters
    ----------
    params : ndarray of shape (K-1,)
        Unconstrained parameter vector. The first element is free; the
        remaining elements are squared to form non-negative increments.

    Returns
    -------
    b : ndarray of shape (K-1,)
        Non-decreasing threshold vector.

    Raises
    ------
    ValueError
        If ``params`` is empty.

    Notes
    -----
    A zero entry in ``params[1:]`` ties two thresholds and is also a
    stationary point of any pulled-back gradient, since the
    ``2 * params[j]`` chain-rule factor in ``thresholds_grad`` vanishes
    there. Gradient-based optimisers therefore cannot separate
    thresholds initialised exactly tied; initialise with strictly
    increasing thresholds instead.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import params_to_thresholds
    >>> params_to_thresholds(np.array([1.0, 2.0, -3.0]))
    array([ 1.,  5., 14.])
    """
    params = np.asarray(params, dtype=float).ravel()
    if params.size == 0:
        raise ValueError("params must contain at least one element")

    t_sq = np.empty_like(params)
    t_sq[0] = params[0]
    t_sq[1:] = params[1:] ** 2
    return np.cumsum(t_sq)


def thresholds_to_params(thresholds):
    """Invert ``params_to_thresholds`` to recover unconstrained parameters.

    Returns ``params`` such that
    ``params_to_thresholds(params) == thresholds`` (up to floating-point
    error) for an ordered ``thresholds``.

    Parameters
    ----------
    thresholds : ndarray of shape (K-1,)
        Ordered threshold values.

    Returns
    -------
    params : ndarray of shape (K-1,)
        Unconstrained parameter vector. ``params[0] = thresholds[0]`` and
        ``params[j] = sqrt(max(thresholds[j] - thresholds[j-1], 0))`` for
        ``j >= 1``.

    Raises
    ------
    ValueError
        If ``thresholds`` is empty.

    Notes
    -----
    Tied input thresholds produce an exactly-zero ``params`` entry,
    which is a stationary point of the pulled-back gradient; see the
    ``Notes`` section of ``params_to_thresholds``.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import (
    ...     thresholds_to_params, params_to_thresholds
    ... )
    >>> b = np.array([1., 5., 14.])
    >>> t = thresholds_to_params(b)
    >>> np.allclose(params_to_thresholds(t), b)
    True
    """
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    if thresholds.size == 0:
        raise ValueError("thresholds must contain at least one element")

    params = np.empty_like(thresholds)
    params[0] = thresholds[0]
    if params.size > 1:
        params[1:] = np.sqrt(np.maximum(np.diff(thresholds), 0.0))
    return params


def thresholds_grad(params, grad_thresholds):
    """Push a threshold-space gradient back through the parameter map.

    Applies the chain rule through the cumsum-of-squares transform used
    by ``params_to_thresholds``: given the gradient of a loss with
    respect to the ordered thresholds, returns the gradient with respect
    to the unconstrained parameters.

    Parameters
    ----------
    params : ndarray of shape (K-1,)
        Unconstrained parameter vector, as passed to
        ``params_to_thresholds``.

    grad_thresholds : ndarray of shape (K-1,)
        Gradient of the loss with respect to the ordered thresholds
        ``b = params_to_thresholds(params)``.

    Returns
    -------
    grad_params : ndarray of shape (K-1,)
        Gradient of the loss with respect to ``params``.

    Raises
    ------
    ValueError
        If ``params`` is empty, or if ``grad_thresholds`` does not have
        the same number of elements as ``params``.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import (
    ...     thresholds_grad, params_to_thresholds
    ... )
    >>> t = np.array([1.0, 2.0, -3.0])
    >>> grad_b = np.array([1.0, 0.5, -2.0])
    >>> analytic = thresholds_grad(t, grad_b)
    >>> eps = 1e-6
    >>> numeric = np.array([
    ...     (grad_b @ params_to_thresholds(t + eps * e)
    ...      - grad_b @ params_to_thresholds(t - eps * e)) / (2 * eps)
    ...     for e in np.eye(t.size)
    ... ])
    >>> np.allclose(analytic, numeric, atol=1e-4)
    True
    """
    params = np.asarray(params, dtype=float).ravel()
    if params.size == 0:
        raise ValueError("params must contain at least one element")

    grad_thresholds = np.asarray(grad_thresholds, dtype=float).ravel()
    if grad_thresholds.size != params.size:
        raise ValueError(
            "params and grad_thresholds must have the same shape "
            "(number of elements), got sizes "
            f"{params.size} and {grad_thresholds.size}"
        )

    n_params = params.size
    grad_params = np.empty(n_params)
    # Take suffix sums via a reversed cumsum, then restore the order
    cumsum_grad = np.cumsum(grad_thresholds[::-1])[::-1]
    grad_params[0] = cumsum_grad[0]
    if n_params > 1:
        grad_params[1:] = 2.0 * params[1:] * cumsum_grad[1:]
    return grad_params


def _check_cumproba(cumproba):
    """Validate a cumulative-probability matrix and cast it to float64."""
    cumproba = check_array(
        cumproba, ensure_2d=True, dtype=np.float64, input_name="cumproba"
    )

    if cumproba.min() < 0.0 or cumproba.max() > 1.0:
        raise ValueError(
            f"cumproba entries must lie in [0, 1], got range "
            f"[{cumproba.min():.4g}, {cumproba.max():.4g}]"
        )

    return cumproba


def _isotonic_repair(cumproba):
    """Repair non-monotonic rows of an already validated cumproba matrix."""
    repaired = cumproba.copy()
    # Refit only violating rows; isotonic regression leaves monotone
    # in-range rows unchanged
    violating = np.flatnonzero((np.diff(cumproba, axis=1) < 0.0).any(axis=1))
    for i in violating:
        repaired[i] = isotonic_regression(
            cumproba[i], y_min=0.0, y_max=1.0, increasing=True
        )
    return repaired


def repair_cumproba(cumproba):
    """Apply row-wise isotonic regression to enforce monotonicity in [0, 1].

    Each row of ``cumproba`` is independently fitted with isotonic regression
    bounded to ``[0, 1]`` and required to be non-decreasing. The returned
    array has the same shape as the input.

    This is the canonical primitive for repairing cumulative probability
    outputs produced by ordinal classifiers whose backends (e.g. raw sigmoid,
    independent binary boosters) do not guarantee monotonicity.

    Parameters
    ----------
    cumproba : array-like of shape (n_samples, n_thresholds)
        Cumulative probabilities. Each entry should lie in ``[0, 1]``.

    Returns
    -------
    repaired : ndarray of shape (n_samples, n_thresholds), dtype np.float64
        Each row is non-decreasing with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``cumproba`` is not 2-D, has zero columns, contains NaN /
        inf values, or has any entry outside ``[0, 1]``. Upstream
        ``ValueError`` from ``check_array`` (e.g. NaN inputs) is
        propagated unchanged.

    TypeError
        If ``cumproba`` is a sparse matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import repair_cumproba
    >>> raw = np.array([[0.4, 0.2, 0.7]])  # column 1 violates monotonicity
    >>> repair_cumproba(raw)
    array([[0.3, 0.3, 0.7]])
    """
    cumproba = _check_cumproba(cumproba)
    return _isotonic_repair(cumproba)


def proba_to_cumproba(proba):
    """Convert class-wise ``P(Y = k)`` to cumulative ``P(Y <= k)``.

    Takes a matrix of class-wise probabilities ``P(Y = k | x)`` for
    ``k = 1, ..., n_classes`` and returns the cumulative probabilities
    ``P(Y <= k | x)`` for ``k = 1, ..., n_classes - 1``, i.e. all
    partial row sums except the last (which is always 1).

    This is the inverse of ``cumproba_to_proba``: for valid ``proba``
    (non-negative rows summing to ~1),
    ``cumproba_to_proba(proba_to_cumproba(proba))`` recovers ``proba``
    up to floating-point error. The result is clipped to ``[0, 1]`` (a
    cumulative sum can drift a few ulps above 1) so it stays valid
    input for ``repair_cumproba`` and ``cumproba_to_proba``.

    It does not validate its input; callers must pass valid
    probability rows.

    Parameters
    ----------
    proba : array-like of shape (n_samples, n_classes), dtype float
        Class-wise probabilities. Each row should be non-negative and
        sum to ~1.

    Returns
    -------
    cumproba : ndarray of shape (n_samples, n_classes - 1), dtype np.float64
        Cumulative probabilities. Each row is non-decreasing with
        values in ``[0, 1]`` for valid input.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import (
    ...     proba_to_cumproba, cumproba_to_proba
    ... )
    >>> P = np.array([[0.2, 0.3, 0.4, 0.1]])
    >>> proba_to_cumproba(P)
    array([[0.2, 0.5, 0.9]])
    >>> np.allclose(cumproba_to_proba(proba_to_cumproba(P)), P)
    True
    """
    proba = np.asarray(proba, dtype=np.float64)
    cumproba = np.cumsum(proba, axis=1)[:, :-1]
    np.clip(cumproba, 0.0, 1.0, out=cumproba)
    return cumproba


def cumproba_to_proba(cumproba, repair=True):
    """Convert cumulative ``P(Y <= k)`` to class-wise ``P(Y = k)``.

    Takes a matrix of cumulative probabilities ``P(Y <= k | x)`` for
    ``k = 1, ..., n_classes - 1`` and returns a matrix of class-wise
    probabilities ``P(Y = k | x)`` for ``k = 1, ..., n_classes``.

    When ``repair=True``, monotonicity violations in each row are
    silently fixed via isotonic regression before differencing. When
    ``repair=False``, any non-monotonic row triggers a ``ValueError``.

    Special row behaviours:

    - An all-zero row becomes ``[0, 0, ..., 1]``; the final class absorbs
      all probability mass.
    - An all-one row becomes ``[1, 0, ..., 0]``; the first class absorbs
      all probability mass.

    Parameters
    ----------
    cumproba : array-like of shape (n_samples, n_classes - 1)
        Cumulative probabilities. Each row should be a non-decreasing
        sequence of values in ``[0, 1]``.

    repair : bool, default=True
        If ``True``, apply isotonic regression row-wise to enforce
        monotonicity before differencing, then clip and renormalise.
        If ``False``, raise ``ValueError`` when any row is non-monotonic.

    Returns
    -------
    class_proba : ndarray of shape (n_samples, n_classes), dtype np.float64
        Class-wise probabilities. Each row is non-negative and sums to
        ``1.0`` up to floating-point rounding, in both the
        ``repair=True`` and ``repair=False`` branches.

    Raises
    ------
    ValueError
        If ``cumproba`` is not 2-D, has zero columns, contains NaN / inf
        values, contains values outside ``[0, 1]``, or — when
        ``repair=False`` — has any row whose entries are not
        non-decreasing. Upstream ``ValueError`` from ``check_array``
        (e.g. NaN inputs) is propagated unchanged.

    TypeError
        If ``cumproba`` is a sparse matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import cumproba_to_proba
    >>> cumproba = np.array([[0.2, 0.5, 0.9]])
    >>> cumproba_to_proba(cumproba)
    array([[0.2, 0.3, 0.4, 0.1]])
    """
    cumproba = _check_cumproba(cumproba)

    n_samples, n_thresholds = cumproba.shape
    class_proba = np.empty((n_samples, n_thresholds + 1), dtype=np.float64)

    if not repair:
        diffs = np.diff(cumproba, axis=1)
        if (diffs < 0.0).any():
            raise ValueError(
                f"cumproba rows must be non-decreasing, got minimum diff "
                f"{diffs.min():.4g}"
            )
        class_proba[:, 0] = cumproba[:, 0]
        class_proba[:, 1:-1] = diffs
        class_proba[:, -1] = 1.0 - cumproba[:, -1]
        return class_proba

    repaired = _isotonic_repair(cumproba)
    class_proba[:, 0] = repaired[:, 0]
    class_proba[:, 1:-1] = np.diff(repaired, axis=1)
    class_proba[:, -1] = 1.0 - repaired[:, -1]

    np.clip(class_proba, 0.0, None, out=class_proba)
    row_sums = class_proba.sum(axis=1, keepdims=True)
    # Guard against division by zero on an all-zero row, though y_min=0
    # already rules this out
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    class_proba /= row_sums
    return class_proba


def normalize_proba_rows(scores, *, floor=np.finfo(np.float64).tiny):
    """Clip a score matrix to a positive floor and row-normalise to sum to 1.

    Each entry is clipped to ``>= floor`` and each row is divided by its
    sum; a second clip-and-renormalise keeps every output entry strictly
    positive even when the first division underflows a tiny entry to
    zero. Callers must pass finite ``scores`` — non-finite input or an
    overflowing row sum raises.

    This is the shared primitive used by ensemble meta-estimators that fuse
    sub-classifier scores via a product or sum combiner, where underflow to
    zero would otherwise produce degenerate rows.

    Parameters
    ----------
    scores : ndarray of shape (n_samples, n_classes), dtype float
        Raw (non-negative) score matrix. Negative values are clipped up to
        ``floor``.

    floor : float, default=numpy.finfo(numpy.float64).tiny
        Minimum value every entry is clipped to before normalisation. Must
        be strictly positive and finite.

    Returns
    -------
    proba : ndarray of shape (n_samples, n_classes), dtype float64
        Row-normalised probability matrix. Every entry is strictly
        positive, though an entry may fall below ``floor`` after
        normalisation. Each row sums to 1.0 up to floating-point
        rounding.

    Raises
    ------
    ValueError
        If ``floor`` is not strictly positive and finite (this also
        rejects ``NaN``), or if ``scores`` contains NaN or positive
        infinity, or if any row sum overflows to infinity.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import normalize_proba_rows
    >>> S = np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
    >>> P = normalize_proba_rows(S)
    >>> np.allclose(P.sum(axis=1), 1.0)
    True
    >>> bool((P > 0).all())
    True
    """
    if not 0.0 < floor < np.inf:
        raise ValueError(f"floor must be strictly positive and finite, got {floor!r}")

    scores = np.array(scores, dtype=np.float64)
    np.clip(scores, floor, None, out=scores)
    row_sums = scores.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all():
        raise ValueError("scores must be finite and row sums must not overflow")
    scores /= row_sums

    # Clip to the smallest positive float64 and re-normalise; the first
    # division can underflow an entry to exactly zero
    tiny = np.finfo(np.float64).tiny
    np.clip(scores, tiny, None, out=scores)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores


def losses_to_proba(losses):
    """Convert a per-class loss matrix to row-normalised probabilities.

    The conversion is ``softmax(1 / (losses + tiny))``: each row is first
    inverted (so that smaller losses become larger scores), then shifted by
    the row max for numerical stability before exponentiation, then divided
    by its row sum. Infinite losses are accepted, mapping to a zero
    score for that class. The mapping is a heuristic kept for backward
    compatibility: its argmax always matches the loss argmin, but the
    probabilities are scale-sensitive — saturating toward the uniform
    distribution for losses much larger than 1 — and should not be read
    as calibrated.

    Parameters
    ----------
    losses : ndarray of shape (n_samples, n_classes), dtype float
        Per-class non-negative loss values.

    Returns
    -------
    proba : ndarray of shape (n_samples, n_classes), dtype float64
        Row-normalised probability matrix. Each row is non-negative and
        sums to 1.0 up to floating-point rounding.

    Raises
    ------
    ValueError
        If ``losses`` contains any negative value or NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.extmath import losses_to_proba
    >>> L = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    >>> np.round(losses_to_proba(L), 4)
    array([[0.9921, 0.0067, 0.0013],
           [0.0013, 0.0067, 0.9921]])
    """
    losses = np.asarray(losses, dtype=np.float64)
    if not (losses >= 0.0).all():
        raise ValueError("losses must be non-negative and not NaN")

    tiny = np.finfo(np.float64).tiny
    scores = 1.0 / (losses + tiny)
    scores -= scores.max(axis=1, keepdims=True)
    proba = np.exp(scores)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba
