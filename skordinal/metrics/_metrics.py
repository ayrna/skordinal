"""Metrics for ordinal classification."""

from __future__ import annotations

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.utils import check_array, check_consistent_length


def _check_metric_inputs(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Coerce metric inputs to 1-D arrays and validate length consistency.

    Two-dimensional inputs are interpreted as one-hot encoded labels and
    collapsed via ``argmax`` along the last axis. Centralises the input
    coercion that every public ordinal metric needs.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted labels or class scores.

    Returns
    -------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim > 1:
        y_true_arr = y_true_arr.argmax(axis=-1)
    if y_pred_arr.ndim > 1:
        y_pred_arr = y_pred_arr.argmax(axis=-1)
    check_consistent_length(y_true_arr, y_pred_arr)
    return y_true_arr, y_pred_arr


def _check_proba_inputs(
    y_true: ArrayLike, y_proba: ArrayLike, *, sum_atol: float = 1e-6
) -> tuple[NDArray, NDArray[np.float64]]:
    """Validate inputs for probabilistic ordinal metrics.

    ``y_true`` may be 1-D class labels or a 2-D one-hot matrix.
    ``y_proba`` must be a 2-D matrix coercible to ``float64`` whose rows
    sum to approximately one.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        Ground truth labels.

    y_proba : array-like of shape (n_samples, n_classes)
        Predicted class probability matrix.

    sum_atol : float, default=1e-6
        Absolute tolerance for the row-sum check.

    Returns
    -------
    y_true : ndarray of shape (n_samples,)
    y_proba : ndarray of shape (n_samples, n_classes), dtype float64

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_proba`` have inconsistent length, or if any
        row of ``y_proba`` does not sum to 1 within ``sum_atol``.
    """
    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim > 1:
        y_true_arr = y_true_arr.argmax(axis=-1)
    y_proba_arr = check_array(
        y_proba, ensure_2d=True, dtype="float64", input_name="y_proba"
    )
    check_consistent_length(y_true_arr, y_proba_arr)
    row_sums = y_proba_arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=sum_atol):
        raise ValueError(
            f"y_proba rows must sum to 1 (atol={sum_atol}); got row-sum "
            f"range [{row_sums.min():.6g}, {row_sums.max():.6g}]"
        )
    return y_true_arr, y_proba_arr


def _recall_per_class(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    labels: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return per-class recall as a 1-D float64 ndarray.

    Thin wrapper around :func:`sklearn.metrics.recall_score` with
    ``average=None`` and ``zero_division=0``. Centralises the call so
    public sensitivity-based metrics share one implementation.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.

    y_pred : ndarray of shape (n_samples,)
        Predicted labels.

    labels : array-like of shape (n_classes,), default=None
        Labels in the order to score. If ``None``, all unique labels are
        used.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    sensitivities : ndarray of shape (n_classes,), dtype float64
    """
    return np.asarray(
        recall_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        dtype=np.float64,
    )


def _per_class_mae(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    labels: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return per-class mean absolute error as a 1-D float64 ndarray.

    Drops rows of the confusion matrix with no support (zero true
    samples for that class) so divisions remain finite. Shared by
    :func:`average_mean_absolute_error` and
    :func:`maximum_mean_absolute_error`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.

    y_pred : ndarray of shape (n_samples,)
        Predicted labels.

    labels : array-like of shape (n_classes,), default=None
        Labels to index the confusion matrix.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    per_class_mae : ndarray of shape (n_classes_with_support,), dtype float64
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    n_class = cm.shape[0]
    costs = np.abs(np.arange(n_class)[:, None] - np.arange(n_class)[None, :])
    errors = costs * cm
    support = cm.sum(axis=1).astype(np.float64)
    non_zero = support > 0
    return errors[non_zero].sum(axis=1) / support[non_zero]


def average_mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute the average per-class mean absolute error.

    For each class with at least one ground-truth sample, the mean
    absolute error is computed using the class label as the numerical
    score. The per-class errors are then averaged with equal weight,
    which makes the metric robust to class imbalance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to the confusion matrix.

    Returns
    -------
    score : float
        Average per-class mean absolute error.

    Notes
    -----
    Classes with no ground-truth samples are excluded from the average.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import average_mean_absolute_error
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> average_mean_absolute_error(y_true, y_pred)
    0.125

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    return float(_per_class_mae(y_true, y_pred, sample_weight=sample_weight).mean())


def geometric_mean(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute the geometric mean of per-class sensitivities.

    Sensitivity (recall) is computed for every class from the confusion
    matrix and the result is the geometric mean across classes. The
    metric penalises poor performance on minority classes more strongly
    than a simple average.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to the confusion matrix.

    Returns
    -------
    score : float
        Geometric mean of the per-class sensitivities.

    Notes
    -----
    Classes with no ground-truth samples are treated as sensitivity 1,
    leaving them out of the geometric mean. This avoids collapsing the
    score to zero when a class has no support.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import geometric_mean
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> geometric_mean(y_true, y_pred)
    0.8408964152537145

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    sum_by_class = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivities = np.diag(cm) / sum_by_class.astype("double")
    sensitivities[sum_by_class == 0] = 1
    return float(pow(np.prod(sensitivities), 1.0 / cm.shape[0]))


def gmsec(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Geometric mean of the sensitivities of the extreme ordinal classes.

    Proposed in :footcite:t:`vargas2024improving` to assess the
    classification performance on the first and the last classes of an
    ordinal scale. Returns the geometric mean of the recall of the
    lowest and the highest classes that appear in ``y_true``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to ``recall_score``.

    Returns
    -------
    score : float
        Geometric mean of the sensitivities of the extreme classes.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import gmsec
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> gmsec(y_true, y_pred)
    0.7071067811865476

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    sensitivities = _recall_per_class(y_true, y_pred, sample_weight=sample_weight)
    return float(np.sqrt(sensitivities[0] * sensitivities[-1]))


def maximum_mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute the maximum per-class mean absolute error.

    Returns the largest per-class MAE across classes with at least one
    ground-truth sample. Useful for monitoring the worst-performing
    class under imbalance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to the confusion matrix.

    Returns
    -------
    score : float
        Maximum per-class mean absolute error.

    Notes
    -----
    Classes with no ground-truth samples are excluded from the maximum.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import maximum_mean_absolute_error
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> maximum_mean_absolute_error(y_true, y_pred)
    0.5

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    return float(_per_class_mae(y_true, y_pred, sample_weight=sample_weight).max())


def minimum_sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Lowest per-class sensitivity.

    Returns the minimum recall across all classes present in
    ``y_true``. The metric flags the class on which the classifier
    performs worst regardless of class prevalence.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to ``recall_score``.

    Returns
    -------
    score : float
        Minimum per-class sensitivity.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import minimum_sensitivity
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> minimum_sensitivity(y_true, y_pred)
    0.5

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    sensitivities = _recall_per_class(y_true, y_pred, sample_weight=sample_weight)
    return float(np.min(sensitivities))


def mean_zero_one_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Fraction of misclassified samples (error rate).

    Equivalent to ``1 - accuracy``; the complementary measure of the
    Correct Classification Rate.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to the confusion matrix.

    Returns
    -------
    score : float
        Mean zero-one error.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import mean_zero_one_error
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mean_zero_one_error(y_true, y_pred)
    0.2857142857142857

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    return float(1 - np.diagonal(cm).sum() / cm.sum())


def kendalls_tau(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Kendall's tau rank correlation coefficient.

    Measures the ordinal association between ``y_true`` and ``y_pred``
    using the number of concordant minus discordant pairs. Computed via
    :func:`scipy.stats.kendalltau`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    score : float
        Kendall's tau in the range [-1, 1].

    Notes
    -----
    Does not accept ``sample_weight`` because the underlying scipy
    backend does not support it.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import kendalls_tau
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> kendalls_tau(y_true, y_pred)
    0.8140915784106943

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    corr, _ = scipy.stats.kendalltau(y_true, y_pred)
    return float(corr)


def weighted_kappa(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Weighted Cohen's kappa with linear ordinal weights.

    A version of the kappa statistic that assigns different weights to
    different levels of disagreement, so off-by-one errors penalise the
    score less than far-off ones. The current implementation uses
    linear weights ``w_{ij} = |i - j|``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to the confusion matrix.

    Returns
    -------
    score : float
        Weighted kappa in the range [-1, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import weighted_kappa
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> weighted_kappa(y_true, y_pred)
    0.7586206896551724

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    n_class = cm.shape[0]
    costs = np.abs(np.arange(n_class)[:, None] - np.arange(n_class)[None, :])
    f = 1 - costs

    n = cm.sum()
    x = cm / n

    r = x.sum(axis=1)
    s = x.sum(axis=0)
    Ex = r.reshape(-1, 1) * s
    po = (x * f).sum()
    pe = (Ex * f).sum()
    return float((po - pe) / (1 - pe))


def spearmans_rho(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Spearman's rank correlation coefficient between two ordinal vectors.

    A non-parametric measure of monotonic association computed from
    label values directly (treated as numerical scores). Returns ``0.0`` when one
    of the inputs is constant.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    score : float
        Spearman's rho in the range [-1, 1].

    Notes
    -----
    Does not accept ``sample_weight``: weighted Spearman is not
    implemented in scipy and the rank correlation has no canonical
    weighted definition.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import spearmans_rho
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> spearmans_rho(y_true, y_pred)
    0.9165444688834581

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    num = (y_true_centered * y_pred_centered).sum()
    div = np.sqrt((y_true_centered**2).sum() * (y_pred_centered**2).sum())

    if num == 0:
        return 0.0
    return float(num / div)


def ranked_probability_score(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Ranked probability score for ordinal class probabilities.

    Quadratic distance between the cumulative ground-truth indicator
    function and the cumulative predicted distribution, averaged over
    samples. The lower the value, the better. Defined for ordinal
    targets in :footcite:t:`janitza2016random`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels encoded as 0-based integer indices.

    y_proba : array-like of shape (n_samples, n_classes)
        Predicted class probability distribution. Each row must sum to
        approximately ``1`` (``atol=1e-6``); otherwise ``ValueError`` is
        raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to :func:`numpy.average`.

    Returns
    -------
    score : float
        Ranked probability score; lower is better.

    Notes
    -----
    Samples whose ``y_true`` falls outside ``[0, n_classes)`` are
    counted with a per-sample contribution of ``1.0``.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import ranked_probability_score
    >>> y_true = np.array([0, 0, 3, 2])
    >>> y_pred = np.array(
    ...     [[0.2, 0.4, 0.2, 0.2],
    ...      [0.7, 0.1, 0.1, 0.1],
    ...      [0.5, 0.05, 0.1, 0.35],
    ...      [0.1, 0.05, 0.65, 0.2]])
    >>> ranked_probability_score(y_true, y_pred)
    0.5068750000000001

    """
    y_true, y_proba = _check_proba_inputs(y_true, y_proba)
    y_true = y_true.astype(np.intp)
    n_samples, n_classes = y_proba.shape

    in_range = (y_true >= 0) & (y_true < n_classes)
    y_oh = np.zeros_like(y_proba)
    rows = np.arange(n_samples)[in_range]
    y_oh[rows, y_true[in_range]] = 1.0

    y_oh_cum = y_oh.cumsum(axis=1)
    y_proba_cum = y_proba.cumsum(axis=1)

    per_sample = np.power(y_proba_cum - y_oh_cum, 2).sum(axis=1)
    per_sample[~in_range] = 1.0

    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return float(np.average(per_sample, weights=weights))


def accuracy_off1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
) -> float:
    """1-off accuracy: predictions in adjacent classes count as correct.

    A prediction is counted as correct when it lies within one class
    of the ground truth, on either side of the ordinal scale.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    labels : array-like of shape (n_classes,), default=None
        Labels of the classes used to index the confusion matrix. If
        ``None``, the labels are inferred from ``y_true``.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights forwarded to the confusion matrix.

    Returns
    -------
    score : float
        1-off accuracy in the range [0, 1].

    Notes
    -----
    Both adjacent diagonals are counted: a prediction one class above
    or one class below the truth contributes to the score.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import accuracy_off1_score
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 1])
    >>> accuracy_off1_score(y_true, y_pred)
    0.8571428571428571

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    if labels is None:
        labels = np.unique(y_true)

    conf_mat = confusion_matrix(
        y_true, y_pred, labels=labels, sample_weight=sample_weight
    )
    n = conf_mat.shape[0]
    mask = np.eye(n, n) + np.eye(n, n, k=1) + np.eye(n, n, k=-1)
    correct = mask * conf_mat

    return float(np.sum(correct) / np.sum(conf_mat))
