"""Metrics for ordinal classification."""

import numpy as np
import scipy.stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    recall_score,
)
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import _check_sample_weight


def _check_labels(arr, name):
    """Validate a label array and collapse a 2-D one to 1-D hard labels.

    ``dtype=None`` keeps string label sets usable. Validation runs before
    the ``argmax``, which would otherwise mask a NaN behind its own index
    and quietly turn a 3-D input into a 2-D one.
    """
    arr = check_array(arr, ensure_2d=False, dtype=None, input_name=name)
    if arr.ndim > 1:
        return arr.argmax(axis=-1) if arr.shape[1] > 1 else arr.ravel()
    return arr


def _check_metric_weight(y_true, sample_weight):
    """Validate, reshape, and cast a metric's sample weights to float64.

    Every check is scikit-learn's own, so the messages match any other
    estimator; the all-zero rejection backports, with its exact message,
    the one ``_check_sample_weight`` itself performs from scikit-learn
    1.9. A ``(n, 1)`` weight is raveled first, which scikit-learn refuses.
    """
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight)
    if weights.ndim == 2 and weights.shape[1] == 1:
        weights = weights.ravel()
    weights = _check_sample_weight(
        weights, y_true, dtype=np.float64, ensure_non_negative=True
    )
    # scikit-learn < 1.9 accepts an all-zero vector and lets the 0/0
    # surface as nan downstream
    if not weights.any():
        raise ValueError("Sample weights must contain at least one non-zero number.")
    return weights


def _check_metric_inputs(y_true, y_pred):
    """Coerce a metric's targets to validated 1-D label arrays.

    Either input may be 1-D labels or 2-D: a one-hot matrix is collapsed
    via ``argmax`` along the last axis, a single column is raveled to its
    original values. Raises ``ValueError`` if either is empty, more than
    2-D, complex, holds a non-finite value, or if the two differ in length.
    """
    y_true_arr = _check_labels(y_true, "y_true")
    y_pred_arr = _check_labels(y_pred, "y_pred")
    check_consistent_length(y_true_arr, y_pred_arr)
    return y_true_arr, y_pred_arr


def _check_proba_inputs(y_true, y_proba):
    """Coerce and validate the inputs of a probabilistic ordinal metric.

    ``y_true`` is collapsed as in ``_check_metric_inputs``. ``y_proba``
    must be coercible to ``float64`` with every entry in ``[0, 1]``; a 1-D
    or single-column input is the positive-class probability of a binary
    problem and is expanded to ``[1 - p, p]``, and rows must then sum to 1
    within ``atol=1e-6, rtol=0``. Raises ``ValueError`` on a malformed
    ``y_true``, a length mismatch, an out-of-range entry, or a row that
    does not sum to 1.
    """
    y_true_arr = _check_labels(y_true, "y_true")
    y_proba_arr = check_array(
        y_proba, ensure_2d=False, dtype="float64", input_name="y_proba"
    )
    if y_proba_arr.ndim == 1:
        y_proba_arr = y_proba_arr.reshape(-1, 1)
    check_consistent_length(y_true_arr, y_proba_arr)

    lo, hi = y_proba_arr.min(), y_proba_arr.max()
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"y_proba entries must lie in [0, 1]; got range [{lo:.6g}, {hi:.6g}]"
        )

    if y_proba_arr.shape[1] == 1:
        y_proba_arr = np.hstack([1.0 - y_proba_arr, y_proba_arr])

    row_sums = y_proba_arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0):
        raise ValueError(
            "y_proba rows must sum to 1 (atol=1e-6, rtol=0); got row-sum "
            f"range [{row_sums.min():.6g}, {row_sums.max():.6g}]"
        )
    return y_true_arr, y_proba_arr


def _recall_per_class(y_true, y_pred, *, labels=None, sample_weight=None):
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


def _per_class_mae(y_true, y_pred, *, labels=None, sample_weight=None):
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


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def average_mean_absolute_error(y_true, y_pred, *, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
    return float(_per_class_mae(y_true, y_pred, sample_weight=sample_weight).mean())


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def geometric_mean(y_true, y_pred, *, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    sum_by_class = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivities = np.diag(cm) / sum_by_class.astype("double")
    sensitivities[sum_by_class == 0] = 1
    return float(pow(np.prod(sensitivities), 1.0 / cm.shape[0]))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def gmsec(y_true, y_pred, *, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
    sensitivities = _recall_per_class(y_true, y_pred, sample_weight=sample_weight)
    return float(np.sqrt(sensitivities[0] * sensitivities[-1]))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def mean_extreme_sensitivity(y_true, y_pred, *, sample_weight=None):
    """Arithmetic mean of the sensitivities of the extreme ordinal classes.

    Assesses the balanced performance between the extreme classes of an
    ordinal scale using the arithmetic mean. Returns the mean of the recall
    of the lowest and the highest classes that appear in ``y_true``.

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
        Arithmetic mean of the sensitivities of the extreme classes.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import mean_extreme_sensitivity
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mean_extreme_sensitivity(y_true, y_pred)
    0.75

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    sample_weight = _check_metric_weight(y_true, sample_weight)
    sensitivities = _recall_per_class(y_true, y_pred, sample_weight=sample_weight)
    return float((sensitivities[0] + sensitivities[-1]) / 2.0)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def maximum_mean_absolute_error(y_true, y_pred, *, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
    return float(_per_class_mae(y_true, y_pred, sample_weight=sample_weight).max())


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def minimum_sensitivity(y_true, y_pred, *, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
    sensitivities = _recall_per_class(y_true, y_pred, sample_weight=sample_weight)
    return float(np.min(sensitivities))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def mean_zero_one_error(y_true, y_pred, *, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    return float(1 - np.diagonal(cm).sum() / cm.sum())


@validate_params(
    {"y_true": ["array-like"], "y_pred": ["array-like"]},
    prefer_skip_nested_validation=True,
)
def kendalls_tau(y_true, y_pred):
    """Kendall's tau rank correlation coefficient.

    Measures the ordinal association between ``y_true`` and ``y_pred``
    using the number of concordant minus discordant pairs. Computed via
    :func:`scipy.stats.kendalltau`. Returns ``0.0`` when one of the
    inputs is constant.

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
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    corr, _ = scipy.stats.kendalltau(y_true, y_pred)
    return float(corr)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def weighted_kappa(y_true, y_pred, *, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
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


@validate_params(
    {"y_true": ["array-like"], "y_pred": ["array-like"]},
    prefer_skip_nested_validation=True,
)
def spearmans_rho(y_true, y_pred):
    """Spearman's rank correlation coefficient between two ordinal vectors.

    A non-parametric measure of monotonic association between the ranks of
    ``y_true`` and ``y_pred``, with average ranks assigned to ties. Computed
    via :func:`scipy.stats.spearmanr`. Returns ``0.0`` when one of the inputs
    is constant.

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
    0.8464861424907173

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    corr, _ = scipy.stats.spearmanr(y_true, y_pred)
    return float(corr)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_proba": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def ranked_probability_score(y_true, y_proba, *, sample_weight=None):
    """Ranked probability score for ordinal class probabilities.

    Quadratic distance between the cumulative ground-truth indicator
    function and the cumulative predicted distribution, averaged over
    samples. The lower the value, the better. Defined for ordinal
    targets in :footcite:t:`janitza2016random`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels encoded as 0-based integer indices.

    y_proba : array-like of shape (n_samples, n_classes), (n_samples, 1), or (n_samples,)
        Predicted class probability distribution. A 1-D input, or a
        single-column 2-D input, is treated as the positive-class
        probability of a binary problem and expanded to two columns,
        ``[1 - p, p]``. Every entry must lie in ``[0, 1]`` and each row
        must sum to approximately ``1`` (``atol=1e-6, rtol=0``).

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
    sample_weight = _check_metric_weight(y_true, sample_weight)

    in_range = (y_true >= 0) & (y_true < n_classes)
    y_oh = np.zeros_like(y_proba)
    rows = np.arange(n_samples)[in_range]
    y_oh[rows, y_true[in_range]] = 1.0

    y_oh_cum = y_oh.cumsum(axis=1)
    y_proba_cum = y_proba.cumsum(axis=1)

    per_sample = np.power(y_proba_cum - y_oh_cum, 2).sum(axis=1)
    per_sample[~in_range] = 1.0

    return float(np.average(per_sample, weights=sample_weight))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "labels": ["array-like", None],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def accuracy_off1_score(y_true, y_pred, *, labels=None, sample_weight=None):
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
    sample_weight = _check_metric_weight(y_true, sample_weight)
    if labels is None:
        labels = np.unique(y_true)

    conf_mat = confusion_matrix(
        y_true, y_pred, labels=labels, sample_weight=sample_weight
    )
    n = conf_mat.shape[0]
    mask = np.eye(n, n) + np.eye(n, n, k=1) + np.eye(n, n, k=-1)
    correct = mask * conf_mat

    return float(np.sum(correct) / np.sum(conf_mat))


_LABEL_METRICS = {
    "accuracy_off1_score": accuracy_off1_score,
    "accuracy_score": accuracy_score,
    "average_mean_absolute_error": average_mean_absolute_error,
    "geometric_mean": geometric_mean,
    "gmsec": gmsec,
    "kendalls_tau": kendalls_tau,
    "maximum_mean_absolute_error": maximum_mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "mean_extreme_sensitivity": mean_extreme_sensitivity,
    "mean_zero_one_error": mean_zero_one_error,
    "minimum_sensitivity": minimum_sensitivity,
    "spearmans_rho": spearmans_rho,
    "weighted_kappa": weighted_kappa,
}


def _resolve_label_metric(name):
    """Return the label metric registered under name, or raise ValueError."""
    if name not in _LABEL_METRICS:
        raise ValueError(
            f"Unknown metric name: {name!r}. Available: {sorted(_LABEL_METRICS)}."
        )
    return _LABEL_METRICS[name]
