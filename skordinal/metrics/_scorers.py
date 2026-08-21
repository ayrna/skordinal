"""Scorer registry for ordinal classification metrics."""

from sklearn.metrics import accuracy_score, make_scorer, mean_absolute_error
from sklearn.utils._param_validation import validate_params

from ._metrics import (
    accuracy_off1_score,
    average_mean_absolute_error,
    geometric_mean,
    gmsec,
    kendalls_tau,
    maximum_mean_absolute_error,
    mean_extreme_sensitivity,
    mean_zero_one_error,
    minimum_sensitivity,
    spearmans_rho,
    weighted_kappa,
)

_SCORERS = {
    "neg_average_mean_absolute_error": make_scorer(
        average_mean_absolute_error, greater_is_better=False
    ),
    "neg_mean_absolute_error": make_scorer(
        mean_absolute_error, greater_is_better=False
    ),
    "neg_maximum_mean_absolute_error": make_scorer(
        maximum_mean_absolute_error, greater_is_better=False
    ),
    "neg_mean_zero_one_error": make_scorer(
        mean_zero_one_error, greater_is_better=False
    ),
    "accuracy_score": make_scorer(accuracy_score),
    "accuracy_off1_score": make_scorer(accuracy_off1_score),
    "geometric_mean": make_scorer(geometric_mean),
    "gmsec": make_scorer(gmsec),
    "mean_extreme_sensitivity": make_scorer(mean_extreme_sensitivity),
    "kendalls_tau": make_scorer(kendalls_tau),
    "minimum_sensitivity": make_scorer(minimum_sensitivity),
    "spearmans_rho": make_scorer(spearmans_rho),
    "weighted_kappa": make_scorer(weighted_kappa),
}

__all__ = ["get_ordinal_scorer", "list_ordinal_scorers"]


@validate_params({"name": [str]}, prefer_skip_nested_validation=True)
def get_ordinal_scorer(name):
    """Return a scikit-learn-compatible scorer by name.

    Parameters
    ----------
    name : str
        Scorer name. Use :func:`list_ordinal_scorers` for the full list.
        Leading and trailing whitespace is stripped before lookup.

    Returns
    -------
    scorer : callable
        A scorer compatible with :class:`~sklearn.model_selection.GridSearchCV`
        and :func:`~sklearn.model_selection.cross_val_score`.

    Raises
    ------
    ValueError
        If ``name`` is not a registered scorer name.

    Examples
    --------
    >>> from skordinal.metrics import get_ordinal_scorer
    >>> scorer = get_ordinal_scorer("neg_mean_absolute_error")
    >>> callable(scorer)
    True

    """
    key = name.strip()
    if key in _SCORERS:
        return _SCORERS[key]
    if f"neg_{key}" in _SCORERS:
        raise ValueError(
            f"Unknown scorer name: {name!r}. A scorer must be "
            f"greater-is-better, so a loss is only registered as 'neg_{key}'."
        )
    raise ValueError(
        f"Unknown scorer name: {name!r}. Available: {list_ordinal_scorers()}."
    )


def list_ordinal_scorers():
    """Return the sorted list of registered ordinal scorer names.

    Returns
    -------
    names : list of str
        Sorted list of all scorer names accepted by :func:`get_ordinal_scorer`.

    Examples
    --------
    >>> from skordinal.metrics import list_ordinal_scorers
    >>> all_scorers = list_ordinal_scorers()
    >>> type(all_scorers)
    <class 'list'>
    >>> "neg_mean_absolute_error" in all_scorers
    True
    >>> "accuracy_score" in all_scorers
    True

    """
    return sorted(_SCORERS)
