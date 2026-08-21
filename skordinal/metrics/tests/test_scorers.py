"""Tests for the public scorer API."""

import numpy as np
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
    spearmans_rho,
    weighted_kappa,
)

_EXPECTED_SCORERS = [
    ("neg_average_mean_absolute_error", average_mean_absolute_error, -1),
    ("neg_maximum_mean_absolute_error", maximum_mean_absolute_error, -1),
    ("neg_mean_absolute_error", mean_absolute_error, -1),
    ("neg_mean_zero_one_error", mean_zero_one_error, -1),
    ("accuracy_off1_score", accuracy_off1_score, 1),
    ("accuracy_score", accuracy_score, 1),
    ("geometric_mean", geometric_mean, 1),
    ("gmsec", gmsec, 1),
    ("kendalls_tau", kendalls_tau, 1),
    ("mean_extreme_sensitivity", mean_extreme_sensitivity, 1),
    ("minimum_sensitivity", minimum_sensitivity, 1),
    ("spearmans_rho", spearmans_rho, 1),
    ("weighted_kappa", weighted_kappa, 1),
]

_LOSS_METRIC_FNS = {
    average_mean_absolute_error,
    maximum_mean_absolute_error,
    mean_absolute_error,
    mean_zero_one_error,
}


@pytest.mark.parametrize(
    "name, metric_fn, sign",
    _EXPECTED_SCORERS,
    ids=[entry[0] for entry in _EXPECTED_SCORERS],
)
def test_registered_scorer_contract(name, metric_fn, sign):
    """Each scorer wraps its metric, and only a loss is negated."""
    scorer = get_ordinal_scorer(name)
    assert scorer._score_func is metric_fn
    assert scorer._sign == sign
    assert (metric_fn in _LOSS_METRIC_FNS) == name.startswith("neg_")


def test_scorer_names_match_expected():
    """Names are exactly the pinned contract, sorted, in a fresh list."""
    names = list_ordinal_scorers()
    assert names == sorted(entry[0] for entry in _EXPECTED_SCORERS)
    assert list_ordinal_scorers() is not names


def test_scorers_not_in_public_all():
    """Private symbol _SCORERS is not exported from skordinal.metrics."""
    import skordinal.metrics as m

    assert "_SCORERS" not in m.__all__


def test_get_ordinal_scorer_returns_fresh_copy():
    """Mutating a returned scorer does not affect subsequent lookups."""
    first = get_ordinal_scorer("neg_mean_absolute_error")
    assert get_ordinal_scorer("neg_mean_absolute_error") is not first
    first._kwargs["sample_weight"] = None
    assert "sample_weight" not in get_ordinal_scorer("neg_mean_absolute_error")._kwargs


@pytest.mark.parametrize(
    "name, labels",
    [
        ("neg_mean_absolute_error", [0, 1, 2]),
        ("accuracy_score", [0, 1, 2]),
    ],
    ids=["loss", "utility"],
)
def test_scorer_drives_gridsearchcv(name, labels):
    """Every scorer shape works in GridSearchCV with the sign its name implies."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    y = np.repeat(np.array(labels), 60 // len(labels))

    gs = GridSearchCV(
        LogisticRegression(max_iter=500),
        param_grid={"C": [0.1, 1.0]},
        scoring=get_ordinal_scorer(name),
        cv=StratifiedKFold(n_splits=3),
    ).fit(X, y)

    if name.startswith("neg_"):
        assert gs.best_score_ <= 0
    else:
        assert gs.best_score_ >= 0


def test_get_ordinal_scorer_rejects_non_string():
    """A non-string name is rejected at the parameter boundary."""
    with pytest.raises(InvalidParameterError):
        get_ordinal_scorer(123)


def test_get_ordinal_scorer_value_error():
    """Unknown name raises ValueError mentioning the requested name."""
    with pytest.raises(ValueError, match="roc_auc"):
        get_ordinal_scorer("roc_auc")


def test_whitespace_stripped():
    """Leading and trailing whitespace in the name is ignored."""
    assert get_ordinal_scorer("  neg_mean_absolute_error  ")._sign == -1
