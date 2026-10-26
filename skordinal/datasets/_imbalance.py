"""Class-imbalance metrics for ordinal classification targets."""

from __future__ import annotations

from collections import Counter

from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import check_array


def _class_counts(y) -> dict[object, int]:
    """Return the count of each class in a one-dimensional target array."""
    targets = check_array(y, ensure_2d=False, dtype=None, input_name="y")
    if targets.ndim != 1:
        raise ValueError(f"y must be one-dimensional; got shape {targets.shape}.")

    counts = dict(Counter(targets.tolist()))
    if len(counts) < 2:
        raise ValueError("y must contain at least two distinct classes.")
    return counts


def _class_imbalance_ratios(class_counts: dict[object, int]) -> list[float]:
    """Return the per-class imbalance ratios for validated class counts."""
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)
    return [
        (total_samples - count) / ((n_classes - 1) * count)
        for count in class_counts.values()
    ]


@validate_params({"y": ["array-like"]}, prefer_skip_nested_validation=True)
def compute_dataset_air(y) -> float:
    """Compute the Average Imbalance Ratio (AIR) from class labels.

    For each class :math:`j`, its imbalance ratio is
    ``(N - n_j) / ((K - 1) * n_j)``, where ``N`` is the number of target
    values, ``n_j`` is the count for class ``j``, and ``K`` is the number of
    classes. AIR is the mean of these per-class ratios, so a balanced dataset
    has AIR equal to 1.

    The unnormalized per-class ratio was introduced for ordinal class
    imbalance by Pérez-Ortiz et al. [1]_. The normalization by ``K - 1``,
    which makes the ratio equal to 1 for a balanced dataset, follows the
    later formulation in Vargas et al. [2]_.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels. ``y`` can be taken from a dataset returned by
        :func:`~skordinal.datasets.load_dataset` or supplied directly from
        any other source.

    Returns
    -------
    air : float
        Mean of the per-class imbalance ratios.

    Raises
    ------
    ValueError
        If ``y`` is not one-dimensional or contains fewer than two distinct
        classes.

    References
    ----------
    .. [1] M. Pérez-Ortiz, A. Sáez, J. Sánchez-Monedero, P. A. Gutiérrez,
           and C. Hervás-Martínez, "Tackling the ordinal and imbalance nature
           of a melanoma image classification problem," 2016 International
           Joint Conference on Neural Networks (IJCNN), pp. 2156-2163, 2016.
           https://doi.org/10.1109/IJCNN.2016.7727466
    .. [2] V. M. Vargas, D. Guijo-Rubio, R. Ayllón-Gavilán, A. M.
           Gómez-Orellana, P. A. Gutiérrez, and C. Hervás-Martínez,
           "Soft labelling for deep ordinal classification: an experimental
           review," IEEE Transactions on Knowledge and Data Engineering,
           2026. https://doi.org/10.1109/TKDE.2026.3681678

    Examples
    --------
    >>> from skordinal.datasets import compute_dataset_air, load_dataset
    >>> dataset = load_dataset("balance_scale")
    >>> air = compute_dataset_air(dataset.target)
    """
    class_counts = _class_counts(y)
    class_irs = _class_imbalance_ratios(class_counts)
    return sum(class_irs) / len(class_irs)


@validate_params({"y": ["array-like"]}, prefer_skip_nested_validation=True)
def compute_dataset_mir(y) -> float:
    """Compute the Maximum Imbalance Ratio (MIR) from class labels.

    MIR is the number of samples in the majority class divided by the number
    of samples in the minority class. A perfectly balanced dataset has MIR
    equal to 1.

    The majority-to-minority count ratio is a standard class-imbalance measure.
    It has been used in the context of multi-class problems, where
    Ortigosa-Hernández et al. [1]_ discuss its use and limitations.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels. ``y`` can be taken from a dataset returned by
        :func:`~skordinal.datasets.load_dataset` or supplied directly from
        any other source.

    Returns
    -------
    mir : float
        Ratio of the majority-class count to the minority-class count.

    Raises
    ------
    ValueError
        If ``y`` is not one-dimensional or contains fewer than two distinct
        classes.

    References
    ----------
    .. [1] J. Ortigosa-Hernández, I. Inza, and J. A. Lozano, "Measuring the
           class-imbalance extent of multi-class problems," Pattern
           Recognition Letters, vol. 98, pp. 32-38, 2017.
           https://doi.org/10.1016/j.patrec.2017.08.002

    Examples
    --------
    >>> from skordinal.datasets import compute_dataset_mir, load_dataset
    >>> dataset = load_dataset("balance_scale")
    >>> mir = compute_dataset_mir(dataset.target)
    """
    class_counts = _class_counts(y)
    return max(class_counts.values()) / min(class_counts.values())
