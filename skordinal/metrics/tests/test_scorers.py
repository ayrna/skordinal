"""Tests for the public scorer API."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils._param_validation import InvalidParameterError

from skordinal.metrics import (
    accuracy_off1_score,
    accuracy_score,
    average_mean_absolute_error,
    geometric_mean,
    get_ordinal_scorer,
    gmsec,
    kendalls_tau,
    list_ordinal_scorers,
    maximum_mean_absolute_error,
    mean_absolute_error,
    mean_extreme_sensitivity,
    mean_zero_one_error,
    minimum_sensitivity,
    ranked_probability_score,
    spearmans_rho,
    weighted_kappa,
)


def test_list_ordinal_scorers_is_sorted():
    """list_ordinal_scorers returns names in sorted order."""
    names = list_ordinal_scorers()
    assert names == sorted(names)


def test_list_ordinal_scorers_returns_new_list():
    """Two calls return equal but non-identical list objects."""
    assert list_ordinal_scorers() == list_ordinal_scorers()
    assert list_ordinal_scorers() is not list_ordinal_scorers()


def test_scorers_not_in_public_all():
    """Private symbol _SCORERS is not exported from skordinal.metrics."""
    import skordinal.metrics as m

    assert "_SCORERS" not in m.__all__


@pytest.mark.parametrize(
    "name, metric_fn",
    [
        ("accuracy_off1_score", accuracy_off1_score),
        ("accuracy_score", accuracy_score),
        ("geometric_mean", geometric_mean),
        ("gmsec", gmsec),
        ("mean_extreme_sensitivity", mean_extreme_sensitivity),
        ("minimum_sensitivity", minimum_sensitivity),
        ("spearmans_rho", spearmans_rho),
        ("kendalls_tau", kendalls_tau),
        ("weighted_kappa", weighted_kappa),
    ],
)
def test_utility_scorer_sign(name, metric_fn):
    """Utility scorers have sign +1 and wrap the expected metric function."""
    scorer = get_ordinal_scorer(name)
    assert scorer._score_func is metric_fn
    assert scorer._sign == 1


@pytest.mark.parametrize(
    "name, metric_fn",
    [
        ("neg_average_mean_absolute_error", average_mean_absolute_error),
        ("neg_mean_absolute_error", mean_absolute_error),
        ("neg_maximum_mean_absolute_error", maximum_mean_absolute_error),
        ("neg_mean_zero_one_error", mean_zero_one_error),
        ("neg_ranked_probability_score", ranked_probability_score),
        ("average_mean_absolute_error", average_mean_absolute_error),
        ("mean_absolute_error", mean_absolute_error),
        ("maximum_mean_absolute_error", maximum_mean_absolute_error),
        ("mean_zero_one_error", mean_zero_one_error),
    ],
)
def test_loss_scorer_sign(name, metric_fn):
    """Loss scorers have sign -1 and wrap the expected metric function."""
    scorer = get_ordinal_scorer(name)
    assert scorer._score_func is metric_fn
    assert scorer._sign == -1


def test_whitespace_stripped():
    """Leading and trailing whitespace in the name is ignored."""
    assert get_ordinal_scorer("  neg_mean_absolute_error  ")._sign == -1


def test_gridcv_with_loss_scorer():
    """Loss scorer integrates with GridSearchCV: best_score_ is non-positive."""
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = np.repeat([0, 1, 2], 10)

    gs = GridSearchCV(
        LogisticRegression(max_iter=200),
        param_grid={"C": [0.1, 1.0]},
        scoring=get_ordinal_scorer("neg_mean_absolute_error"),
        cv=StratifiedKFold(n_splits=2),
    )
    gs.fit(X, y)
    assert gs.best_score_ <= 0


def test_get_ordinal_scorer_rejects_non_string():
    """A non-string name is rejected at the parameter boundary."""
    with pytest.raises(InvalidParameterError):
        get_ordinal_scorer(123)


def test_get_ordinal_scorer_value_error():
    """Unknown name raises ValueError mentioning the requested name."""
    with pytest.raises(ValueError, match="roc_auc"):
        get_ordinal_scorer("roc_auc")


def test_scorer_names_present():
    """Every expected scorer name appears in list_ordinal_scorers()."""
    names = list_ordinal_scorers()
    expected = [
        "neg_average_mean_absolute_error",
        "neg_mean_absolute_error",
        "neg_maximum_mean_absolute_error",
        "neg_mean_zero_one_error",
        "neg_ranked_probability_score",
        "accuracy_score",
        "accuracy_off1_score",
        "geometric_mean",
        "gmsec",
        "kendalls_tau",
        "mean_extreme_sensitivity",
        "minimum_sensitivity",
        "spearmans_rho",
        "weighted_kappa",
    ]
    for name in expected:
        assert name in names, f"{name!r} missing from list_ordinal_scorers()"
