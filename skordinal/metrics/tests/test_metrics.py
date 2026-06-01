"""Tests for the metrics module."""

import inspect

import numpy as np
import numpy.testing as npt
import pytest

from skordinal.metrics import (
    accuracy_off1_score,
    accuracy_score,
    average_mean_absolute_error,
    geometric_mean,
    gmsec,
    kendalls_tau,
    maximum_mean_absolute_error,
    mean_absolute_error,
    mean_extreme_sensitivity,
    mean_zero_one_error,
    minimum_sensitivity,
    ranked_probability_score,
    spearmans_rho,
    weighted_kappa,
)
from skordinal.metrics._metrics import _check_metric_inputs, _check_proba_inputs

_WEIGHTED_METRICS = [
    accuracy_off1_score,
    average_mean_absolute_error,
    geometric_mean,
    gmsec,
    maximum_mean_absolute_error,
    mean_extreme_sensitivity,
    mean_zero_one_error,
    minimum_sensitivity,
    weighted_kappa,
    ranked_probability_score,
]

_WEIGHTED_METRIC_IDS = [fn.__name__ for fn in _WEIGHTED_METRICS]

_Y_PROBA_6 = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.3, 0.5],
        [0.3, 0.5, 0.2],
        [0.6, 0.3, 0.1],
        [0.1, 0.2, 0.7],
    ]
)


def test_check_metric_inputs_1d_passthrough():
    """1-D arrays are returned unchanged by _check_metric_inputs."""
    y_t = np.array([0, 1, 2, 1])
    y_p = np.array([0, 2, 2, 1])
    out_t, out_p = _check_metric_inputs(y_t, y_p)
    assert np.array_equal(out_t, y_t)
    assert np.array_equal(out_p, y_p)


def test_check_metric_inputs_one_hot_argmax():
    """2-D one-hot inputs are collapsed to 1-D label vectors via argmax."""
    y_t_oh = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_p_oh = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    out_t, out_p = _check_metric_inputs(y_t_oh, y_p_oh)
    assert np.array_equal(out_t, np.array([0, 1, 2]))
    assert np.array_equal(out_p, np.array([1, 0, 2]))


def test_check_metric_inputs_length_mismatch_raises():
    """Mismatched lengths raise ValueError."""
    with pytest.raises(ValueError):
        _check_metric_inputs([0, 1, 2], [0, 1])


def test_check_proba_inputs_requires_2d_proba():
    """1-D y_proba raises; valid 2-D passes; one-hot y_true is collapsed."""
    with pytest.raises(ValueError):
        _check_proba_inputs(np.array([0, 1]), np.array([0.6, 0.4]))

    y_t = np.array([0, 1, 2])
    y_p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    out_t, out_p = _check_proba_inputs(y_t, y_p)
    assert np.array_equal(out_t, y_t)
    npt.assert_allclose(out_p, y_p)

    out_t_oh, _ = _check_proba_inputs(np.eye(3), y_p)
    assert np.array_equal(out_t_oh, y_t)


def test_check_proba_inputs_rejects_unnormalised_rows():
    """Rows not summing to 1 raise ValueError mentioning row-sum."""
    y_t = np.array([0, 1, 2])
    y_p_bad = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]) * 2
    with pytest.raises(ValueError, match="row"):
        _check_proba_inputs(y_t, y_p_bad)


def test_check_proba_inputs_accepts_within_atol():
    """Rows summing to 1 within atol are accepted without error."""
    y_t = np.array([0, 1, 2])
    y_p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    y_p[0, 0] += 5e-7
    _check_proba_inputs(y_t, y_p)


def test_accuracy_score():
    """accuracy_score correctly classifies a known label sequence."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(accuracy_score(y_true, y_pred), 0.8000, decimal=4)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], 5 / 6),
        ([0, 1, 2, 3, 4], [0, 2, 1, 4, 3], 1.0),
    ],
)
def test_accuracy_off1_score(y_true, y_pred, expected):
    """accuracy_off1_score counts predictions within one ordinal class of truth."""
    npt.assert_almost_equal(accuracy_off1_score(y_true, y_pred), expected, decimal=6)


def test_accuracy_off1_score_lower_diagonal():
    """Predictions one class below truth all count as correct (off1 = 1.0)."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([0, 1, 2])
    npt.assert_allclose(accuracy_off1_score(y_true, y_pred), 1.0)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 0, 1, 1], [0, 1, 0, 1], 0.5),
        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], 0.0),
        ([0, 0, 2, 1], [0, 2, 0, 1], 1.0),
        ([0, 0, 2, 1, 3], [2, 2, 0, 3, 1], 2.0),
    ],
)
def test_average_mean_absolute_error(y_true, y_pred, expected):
    """average_mean_absolute_error equals the mean of per-class MAEs."""
    npt.assert_almost_equal(
        average_mean_absolute_error(y_true, y_pred), expected, decimal=6
    )


def test_average_mean_absolute_error_pred_only_class_excluded():
    """Classes that appear only in predictions are excluded from AMAE."""
    npt.assert_almost_equal(
        average_mean_absolute_error([0, 1, 2, 3, 3], [0, 1, 2, 3, 4]), 0.125, decimal=6
    )


def test_geometric_mean():
    """geometric_mean returns the geometric mean of per-class sensitivities."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(geometric_mean(y_true, y_pred), 0.7991, decimal=4)


def test_geometric_mean_empty_class_treated_as_one():
    """A class with zero total weight is treated as sensitivity 1, not 0."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    w = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    npt.assert_almost_equal(geometric_mean(y_true, y_pred, sample_weight=w), 1.0)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 0, 1, 1], [0, 1, 0, 1], 0.5),
        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], 1.0),
    ],
)
def test_gmsec(y_true, y_pred, expected):
    """gmsec equals the geometric mean of the two extreme class sensitivities."""
    npt.assert_almost_equal(gmsec(y_true, y_pred), expected, decimal=6)


def test_mean_absolute_error():
    """mean_absolute_error computes the correct global MAE for a known sequence."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(mean_absolute_error(y_true, y_pred), 0.3000, decimal=4)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 0, 1, 1], [0, 1, 0, 1], 0.5),
        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], 0.0),
        ([0, 0, 2, 1], [0, 2, 0, 1], 2.0),
        ([0, 0, 2, 1, 3], [2, 2, 0, 3, 1], 2.0),
    ],
)
def test_maximum_mean_absolute_error(y_true, y_pred, expected):
    """maximum_mean_absolute_error equals the worst per-class MAE."""
    npt.assert_almost_equal(
        maximum_mean_absolute_error(y_true, y_pred), expected, decimal=6
    )


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 0, 1, 2, 3, 0, 0], [0, 1, 1, 2, 3, 0, 1], 0.75),
        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], 1.0),
        ([0, 0, 1, 1], [1, 1, 1, 1], 0.5),
    ],
)
def test_mean_extreme_sensitivity(y_true, y_pred, expected):
    """mean_extreme_sensitivity returns the arithmetic mean of the two extreme class recalls."""
    npt.assert_almost_equal(
        mean_extreme_sensitivity(y_true, y_pred), expected, decimal=6
    )


def test_maximum_mean_absolute_error_pred_only_class_excluded():
    """Classes that appear only in predictions are excluded from MMAE."""
    npt.assert_almost_equal(
        maximum_mean_absolute_error([0, 1, 2, 3, 3], [0, 1, 2, 3, 4]), 0.5, decimal=6
    )


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 0, 1, 1], [0, 1, 0, 1], 0.5),
        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], 1.0),
    ],
)
def test_minimum_sensitivity(y_true, y_pred, expected):
    """minimum_sensitivity returns the lowest per-class recall."""
    npt.assert_almost_equal(minimum_sensitivity(y_true, y_pred), expected, decimal=6)


def test_mean_zero_one_error():
    """mean_zero_one_error returns the fraction of misclassified samples."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(mean_zero_one_error(y_true, y_pred), 0.2000, decimal=4)


def test_ranked_probability_score():
    """ranked_probability_score returns the correct RPS for a known prediction."""
    y_true = np.array([0, 0, 3, 2])
    y_pred = np.array(
        [
            [0.2, 0.4, 0.2, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.5, 0.05, 0.1, 0.35],
            [0.1, 0.05, 0.65, 0.2],
        ]
    )
    npt.assert_almost_equal(
        ranked_probability_score(y_true, y_pred), 0.506875, decimal=6
    )


def test_ranked_probability_score_out_of_range():
    """y_true values outside [0, n_classes) contribute 1.0 per-sample RPS."""
    y_true = np.array([0, 5, 1])
    y_proba = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0]])
    npt.assert_almost_equal(
        ranked_probability_score(y_true, y_proba), 1.0 / 3, decimal=6
    )


def test_kendalls_tau():
    """kendalls_tau returns the correct rank correlation for a known sequence."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(kendalls_tau(y_true, y_pred), 0.6240, decimal=4)


def test_weighted_kappa():
    """weighted_kappa returns the correct kappa for a known sequence."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(weighted_kappa(y_true, y_pred), 0.6703, decimal=4)


def test_spearmans_rho():
    """spearmans_rho returns the correct rank correlation for a known sequence."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(spearmans_rho(y_true, y_pred), 0.6429, decimal=4)


def test_spearmans_rho_constant_input():
    """spearmans_rho returns 0.0 when y_true is constant (zero numerator)."""
    npt.assert_equal(spearmans_rho(np.array([1, 1, 1, 1]), np.array([0, 1, 2, 3])), 0.0)


def test_metric_names_in_all():
    """All public metric names are present in skordinal.metrics.__all__."""
    import skordinal.metrics as m

    expected = [
        "accuracy_score",
        "average_mean_absolute_error",
        "geometric_mean",
        "mean_absolute_error",
        "maximum_mean_absolute_error",
        "mean_extreme_sensitivity",
        "minimum_sensitivity",
        "mean_zero_one_error",
        "kendalls_tau",
        "weighted_kappa",
        "spearmans_rho",
        "ranked_probability_score",
        "gmsec",
        "accuracy_off1_score",
    ]
    for name in expected:
        assert name in m.__all__, f"{name!r} missing from __all__"


@pytest.mark.parametrize("fn", _WEIGHTED_METRICS, ids=_WEIGHTED_METRIC_IDS)
def test_sample_weight_is_keyword_only(fn):
    """sample_weight is a keyword-only parameter in every weighted metric."""
    sig = inspect.signature(fn)
    assert sig.parameters["sample_weight"].kind == inspect.Parameter.KEYWORD_ONLY


def test_accuracy_off1_score_labels_is_keyword_only():
    """labels is a keyword-only parameter of accuracy_off1_score."""
    sig = inspect.signature(accuracy_off1_score)
    assert sig.parameters["labels"].kind == inspect.Parameter.KEYWORD_ONLY


def test_correlation_metrics_reject_sample_weight():
    """kendalls_tau and spearmans_rho raise TypeError when sample_weight is passed."""
    y_t = np.array([0, 1, 2, 1, 0])
    y_p = np.array([0, 1, 2, 0, 1])
    w = np.ones(5)
    for fn in (kendalls_tau, spearmans_rho):
        with pytest.raises(TypeError):
            fn(y_t, y_p, sample_weight=w)


@pytest.mark.parametrize("fn", _WEIGHTED_METRICS, ids=_WEIGHTED_METRIC_IDS)
def test_metric_unit_sample_weight_matches_unweighted(fn):
    """All-ones sample_weight produces the same result as no weight."""
    y_t = np.array([0, 1, 2, 1, 0, 2])
    y_p = np.array([0, 1, 1, 2, 0, 2])
    n = len(y_t)
    w = np.ones(n)

    if fn is ranked_probability_score:
        unweighted = fn(y_t, _Y_PROBA_6)
        weighted = fn(y_t, _Y_PROBA_6, sample_weight=w)
    else:
        unweighted = fn(y_t, y_p)
        weighted = fn(y_t, y_p, sample_weight=w)

    npt.assert_allclose(weighted, unweighted)


def test_metric_zero_weight_excludes_sample():
    """Setting a sample's weight to 0 removes its contribution to accuracy_off1_score."""
    y_t = np.array([0, 1, 2, 1, 0, 3])
    y_p = np.array([3, 1, 2, 1, 0, 3])
    n = len(y_t)

    unit_w = np.ones(n)
    zero_w = np.ones(n)
    zero_w[0] = 0.0

    score_unit = accuracy_off1_score(y_t, y_p, sample_weight=unit_w)
    score_zero = accuracy_off1_score(y_t, y_p, sample_weight=zero_w)
    assert score_zero > score_unit


@pytest.mark.parametrize(
    "fn",
    _WEIGHTED_METRICS + [kendalls_tau, spearmans_rho],
    ids=_WEIGHTED_METRIC_IDS + ["kendalls_tau", "spearmans_rho"],
)
def test_metric_returns_python_float(fn):
    """Every public metric returns a Python float, not a numpy scalar."""
    y_t = np.array([0, 1, 2, 1, 0, 2])
    y_p = np.array([0, 1, 1, 2, 0, 2])

    if fn is ranked_probability_score:
        result = fn(y_t, _Y_PROBA_6)
    else:
        result = fn(y_t, y_p)

    assert type(result) is float, (
        f"{fn.__name__} returned {type(result).__name__}, expected float"
    )
