"""Tests for the metrics module."""

import inspect

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.utils._param_validation import InvalidParameterError

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
from skordinal.metrics._metrics import (
    _check_metric_inputs,
    _check_metric_weight,
    _check_proba_inputs,
)

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


@pytest.fixture
def y_proba_6():
    """Six rows of three-class probabilities."""
    return np.array(
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


def test_check_metric_inputs_column_vector_ravel():
    """(n, 1) column vectors are raveled, not argmaxed to all-zeros."""
    y_t_col = np.array([[0], [1], [2]])
    y_p_col = np.array([[0], [2], [2]])
    out_t, out_p = _check_metric_inputs(y_t_col, y_p_col)
    assert np.array_equal(out_t, np.array([0, 1, 2]))
    assert np.array_equal(out_p, np.array([0, 2, 2]))


def test_check_metric_inputs_length_mismatch_raises():
    """Mismatched lengths raise ValueError."""
    with pytest.raises(ValueError):
        _check_metric_inputs([0, 1, 2], [0, 1])


@pytest.mark.parametrize(
    "value, match", [(np.nan, "NaN"), (np.inf, "infinity")], ids=["nan", "inf"]
)
@pytest.mark.parametrize("which", ["y_true", "y_pred"])
@pytest.mark.parametrize("shape", ["1d", "one_hot"])
def test_check_metric_inputs_rejects_non_finite(value, match, which, shape):
    """A non-finite value in either input raises before the argmax collapse hides it."""
    y_t = np.array([0.0, 1.0, 2.0])
    y_p = np.array([0.0, 1.0, 1.0])
    if shape == "one_hot":
        y_t = np.eye(3)
        y_p = np.eye(3)[[0, 1, 1]]
    arr = y_t if which == "y_true" else y_p
    arr = arr.copy()
    if arr.ndim == 1:
        arr[0] = value
    else:
        arr[0, 0] = value
    if which == "y_true":
        y_t = arr
    else:
        y_p = arr
    with pytest.raises(ValueError, match=match):
        _check_metric_inputs(y_t, y_p)


@pytest.mark.parametrize(
    "y_t, match",
    [
        (np.array([0, 1, 2], dtype=np.int64), None),
        (np.array([0, 1, 10**400], dtype=object), None),
        (np.array(["low", "mid", "high"]), None),
        (np.array([0, 1, np.nan], dtype=object), "NaN"),
        (np.array([0.0, np.nan, "x"], dtype=object), "NaN"),
        (np.array([0 + 0j, 1 + 0j, 2 + 0j]), "Complex"),
    ],
    ids=[
        "int64",
        "object_huge_int",
        "strings",
        "object_nan",
        "object_mixed",
        "complex",
    ],
)
def test_check_metric_inputs_dtype_handling(y_t, match):
    """Integer, huge-int and string labels pass; NaN and complex raise."""
    y_p = np.array([0, 1, 1])
    if match is None:
        _check_metric_inputs(y_t, y_p)
    else:
        with pytest.raises(ValueError, match=match):
            _check_metric_inputs(y_t, y_p)


@pytest.mark.parametrize(
    "y_t, y_p, match",
    [
        (np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), "dim 3"),
        (np.array([]), np.array([]), "0 sample"),
    ],
    ids=["3d", "empty"],
)
def test_check_metric_inputs_rejects_malformed_shape(y_t, y_p, match):
    """A 3-D or empty input raises instead of collapsing to a plausible score."""
    with pytest.raises(ValueError, match=match):
        _check_metric_inputs(y_t, y_p)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([-0.5]),
        np.array([0.5]),
        np.array([1.7]),
        np.array([0.5], dtype=object),
    ],
    ids=["negative_frac", "half", "above_one", "object_frac"],
)
def test_check_proba_inputs_rejects_non_integer_y_true(bad):
    """A non-integer y_true raises instead of being truncated to a valid index."""
    with pytest.raises(ValueError, match="integer-valued"):
        _check_proba_inputs(bad, np.array([[0.1, 0.2, 0.7]]))


def test_check_proba_inputs_2d_and_one_hot():
    """A valid 2-D y_proba passes through; one-hot y_true is collapsed."""
    y_t = np.array([0, 1, 2])
    y_p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    out_t, out_p = _check_proba_inputs(y_t, y_p)
    assert np.array_equal(out_t, y_t)
    npt.assert_allclose(out_p, y_p)

    out_t_oh, _ = _check_proba_inputs(np.eye(3), y_p)
    assert np.array_equal(out_t_oh, y_t)


@pytest.mark.parametrize("as_column", [False, True], ids=["1d", "column"])
def test_check_proba_inputs_expands_binary_proba(as_column):
    """A 1-D or (n, 1) y_proba is expanded to the [1 - p, p] positive-class columns."""
    y_t = np.array([0, 1])
    p = np.array([0.7, 0.3])
    proba_input = p.reshape(-1, 1) if as_column else p
    out_t, out_p = _check_proba_inputs(y_t, proba_input)
    assert np.array_equal(out_t, y_t)
    npt.assert_allclose(out_p, np.column_stack([1.0 - p, p]))


def test_check_proba_inputs_rejects_out_of_range_entries():
    """A y_proba entry outside [0, 1] raises, even if rows still sum to 1."""
    y_t = np.array([0, 1, 2])
    y_p = np.array([[1.4, -0.4, 0.0], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _check_proba_inputs(y_t, y_p)


def test_check_proba_inputs_column_vector_ravel():
    """(n, 1) column vector y_true is raveled, not argmaxed to all-zeros."""
    y_t_col = np.array([[0], [1], [2]])
    y_p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    out_t, out_p = _check_proba_inputs(y_t_col, y_p)
    assert np.array_equal(out_t, np.array([0, 1, 2]))
    npt.assert_allclose(out_p, y_p)


@pytest.mark.parametrize(
    "y_t, y_p, match",
    [
        (
            np.array([0.0, np.nan, 2.0]),
            np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]),
            "NaN",
        ),
        (np.zeros((2, 2, 3)), np.full((2, 2), 0.5), "dim 3"),
    ],
    ids=["nan", "3d"],
)
def test_check_proba_inputs_routes_y_true_through_check_labels(y_t, y_p, match):
    """The proba gateway rejects a malformed y_true before the argmax collapse."""
    with pytest.raises(ValueError, match=match):
        _check_proba_inputs(y_t, y_p)


@pytest.mark.parametrize(
    "fn", [kendalls_tau, spearmans_rho], ids=["kendalls_tau", "spearmans_rho"]
)
@pytest.mark.parametrize(
    "y_t, y_p, match",
    [
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, np.inf, 2.0]), "infinity"),
        (np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), "dim 3"),
        (np.array([]), np.array([]), "0 sample"),
    ],
    ids=["inf", "3d", "empty"],
)
def test_correlation_metrics_reject_invalid_input(fn, y_t, y_p, match):
    """kendalls_tau and spearmans_rho reject inf, 3-D and empty input."""
    with pytest.raises(ValueError, match=match):
        fn(y_t, y_p)


def test_check_proba_inputs_rejects_unnormalised_rows():
    """Rows not summing to 1 raise ValueError mentioning row-sum."""
    y_t = np.array([0, 1, 2])
    y_p_bad = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]) * 0.5
    with pytest.raises(ValueError, match="row"):
        _check_proba_inputs(y_t, y_p_bad)


@pytest.mark.parametrize(
    "delta, raises", [(5e-7, False), (5e-6, True)], ids=["within_atol", "rtol_zero"]
)
def test_check_proba_inputs_row_sum_atol_boundary(delta, raises):
    """A row-sum delta is accepted within atol and rejected past it, rtol being 0."""
    y_t = np.array([0, 1, 2])
    y_p = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    y_p[0, 0] += delta
    if raises:
        with pytest.raises(ValueError, match="row"):
            _check_proba_inputs(y_t, y_p)
    else:
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


def test_ranked_probability_score_expands_binary_proba():
    """1-D and (n, 1) y_proba give the same RPS as the explicit two-column form."""
    y_true = np.array([1, 0])
    p = np.array([0.8, 0.3])
    expected = ranked_probability_score(y_true, np.column_stack([1.0 - p, p]))
    npt.assert_allclose(ranked_probability_score(y_true, p), expected)
    npt.assert_allclose(ranked_probability_score(y_true, p.reshape(-1, 1)), expected)


@pytest.mark.parametrize(
    "y_true, y_proba, expected",
    [
        (
            np.array([0, 5, 1]),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0]]),
            2.0 / 3,
        ),
        (np.array([5]), np.array([[0.2, 0.2, 0.2, 0.2, 0.2]]), 4.0),
    ],
    ids=["3_classes_mixed", "5_classes_single"],
)
def test_ranked_probability_score_out_of_range(y_true, y_proba, expected):
    """An out-of-range y_true is penalised by n_classes - 1, not a fixed constant."""
    npt.assert_almost_equal(
        ranked_probability_score(y_true, y_proba), expected, decimal=6
    )


def test_ranked_probability_score_out_of_range_no_better_than_worst_in_range():
    """An out-of-range label scores no better than the worst in-range one."""
    y_proba = np.array([[0.0, 0.0, 1.0]])
    out_of_range = ranked_probability_score(np.array([9]), y_proba)
    worst_in_range = max(
        ranked_probability_score(np.array([label]), y_proba) for label in range(3)
    )
    assert out_of_range >= worst_in_range


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3],
            0.6240,
        ),
        ([1, 1, 1, 1], [0, 1, 2, 3], 0.0),
    ],
)
def test_kendalls_tau(y_true, y_pred, expected):
    """kendalls_tau returns the correct rank correlation for known sequences."""
    npt.assert_almost_equal(kendalls_tau(y_true, y_pred), expected, decimal=4)


def test_weighted_kappa():
    """weighted_kappa returns the correct kappa for a known sequence."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    npt.assert_almost_equal(weighted_kappa(y_true, y_pred), 0.6703, decimal=4)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3],
            0.6429,
        ),
        ([0, 0, 1, 2, 3, 0, 0], [0, 1, 1, 2, 3, 0, 1], 0.8465),
    ],
)
def test_spearmans_rho(y_true, y_pred, expected):
    """spearmans_rho returns the correct rank correlation for known sequences."""
    npt.assert_almost_equal(spearmans_rho(y_true, y_pred), expected, decimal=4)


def test_spearmans_rho_constant_input():
    """spearmans_rho returns 0.0 when one of the inputs is constant."""
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
def test_metric_unit_sample_weight_matches_unweighted(fn, y_proba_6):
    """All-ones sample_weight produces the same result as no weight."""
    y_t = np.array([0, 1, 2, 1, 0, 2])
    y_p = np.array([0, 1, 1, 2, 0, 2])
    n = len(y_t)
    w = np.ones(n)

    if fn is ranked_probability_score:
        unweighted = fn(y_t, y_proba_6)
        weighted = fn(y_t, y_proba_6, sample_weight=w)
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
    "weight, match",
    [
        (np.array([1.0, -1.0, 1.0]), "Negative"),
        (np.array([1.0, np.inf, 1.0]), "infinity"),
        (np.array([1.0, np.nan, 1.0]), "NaN"),
        (np.zeros(3), "non-zero"),
        (np.ones((3, 2)), "1D array"),
        (np.ones(2), "expected"),
    ],
    ids=["negative", "inf", "nan", "all_zero", "2d", "length_mismatch"],
)
def test_check_metric_weight_rejects_invalid(weight, match):
    """Negative, non-finite, all-zero, 2-D, or mis-length weights all raise."""
    with pytest.raises(ValueError, match=match):
        _check_metric_weight(np.array([0, 1, 2]), weight)


def test_check_metric_weight_ravels_column_vector():
    """A (n, 1) weight is raveled to the same result as its 1-D form."""
    y_t = np.array([0, 1, 2])
    w_1d = np.array([1.0, 2.0, 3.0])
    w_col = w_1d.reshape(-1, 1)
    out_1d = _check_metric_weight(y_t, w_1d)
    out_col = _check_metric_weight(y_t, w_col)
    npt.assert_array_equal(out_1d, out_col)


@pytest.mark.parametrize("fn", _WEIGHTED_METRICS, ids=_WEIGHTED_METRIC_IDS)
def test_metric_routes_sample_weight_through_the_check(fn, y_proba_6):
    """Every weighted metric rejects an all-zero sample_weight."""
    y_t = np.array([0, 1, 2, 1, 0, 2])
    y_p = np.array([0, 1, 1, 2, 0, 2])
    x = y_proba_6 if fn is ranked_probability_score else y_p
    with pytest.raises(ValueError, match="non-zero"):
        fn(y_t, x, sample_weight=np.zeros(len(y_t)))


@pytest.mark.parametrize(
    "fn",
    _WEIGHTED_METRICS + [kendalls_tau, spearmans_rho],
    ids=_WEIGHTED_METRIC_IDS + ["kendalls_tau", "spearmans_rho"],
)
def test_metric_returns_python_float(fn, y_proba_6):
    """Every public metric returns a Python float, not a numpy scalar."""
    y_t = np.array([0, 1, 2, 1, 0, 2])
    y_p = np.array([0, 1, 1, 2, 0, 2])

    if fn is ranked_probability_score:
        result = fn(y_t, y_proba_6)
    else:
        result = fn(y_t, y_p)

    assert type(result) is float, (
        f"{fn.__name__} returned {type(result).__name__}, expected float"
    )


def test_metric_rejects_non_array_like_y_true():
    """A scalar y_true is rejected at the parameter boundary."""
    # Decorator fires before _check_metric_inputs
    with pytest.raises(InvalidParameterError):
        average_mean_absolute_error(1.0, [1, 2])
