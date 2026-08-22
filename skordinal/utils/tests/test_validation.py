"""Tests for the validation utilities."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skordinal.utils.validation import (
    _rank_encode_labels,
    check_ordinal_targets,
    check_thresholds,
)

_ALL_HELPERS = [
    check_ordinal_targets,
    check_thresholds,
]
_HELPER_NAMES = [f.__name__ for f in _ALL_HELPERS]


@pytest.mark.parametrize("func", _ALL_HELPERS, ids=_HELPER_NAMES)
def test_api_no_mutable_defaults(func):
    """No helper has a mutable default argument."""
    for default in func.__defaults__ or ():
        assert not isinstance(default, (list, dict, np.ndarray)), (
            f"{func.__name__} has a mutable default: {default!r}"
        )


def test_cot_known_encoding():
    """Exact encoding, class order, dtype, round-trip, and contiguity."""
    y = [3, 1, 2, 1, 3]
    classes, y_encoded = check_ordinal_targets(y)
    assert_array_equal(classes, [1, 2, 3])
    assert_array_equal(y_encoded, [2, 0, 1, 0, 2])
    assert y_encoded.dtype == np.intp
    assert_array_equal(classes[y_encoded], y)
    assert set(y_encoded.tolist()) == set(range(len(classes)))
    assert np.all(np.diff(classes) > 0)


def test_cot_integer_valued_floats_accepted():
    """Float labels with integer values are accepted; classes dtype stays float."""
    classes, y_encoded = check_ordinal_targets(np.array([1.0, 2.0, 3.0]))
    assert classes.dtype.kind == "f"
    assert y_encoded.dtype == np.intp


def test_cot_non_contiguous_labels():
    """Gap labels are allowed; encoding is still 0-based."""
    classes, y_encoded = check_ordinal_targets([10, 20, 30])
    assert_array_equal(y_encoded, [0, 1, 2])
    assert_array_equal(classes, [10, 20, 30])


@pytest.mark.parametrize(
    "y, match",
    [
        (None, r"requires y to be passed"),
        ([], None),
        (np.array([[1, 2], [3, 4]]), r"y must be a 1D array"),
        # "1 class" singular, not "1 classes"
        ([1, 1, 1], r"y must contain at least 2 unique classes, got 1 class\.$"),
        (np.array(["a", "b"], dtype=object), None),
        ([np.nan, 1.0, 2.0], None),
    ],
    ids=["none", "empty", "2d", "single-class", "object-dtype", "nan"],
)
def test_cot_invalid_input_raises(y, match):
    """Each invalid-input shape / dtype / cardinality raises ValueError."""
    expected = ValueError if match is not None else (ValueError, TypeError)
    with pytest.raises(expected, match=match):
        check_ordinal_targets(y)


@pytest.mark.parametrize(
    "thresholds",
    [
        [-1.0, 0.0, 1.0, 2.0],
        [0.0],
        [0.0, np.nextafter(0.0, 1.0)],
    ],
    ids=["generic", "binary-edge", "smallest-gap"],
)
def test_ct_valid_returns_none(thresholds):
    """Valid threshold vectors return None."""
    assert check_thresholds(thresholds) is None


@pytest.mark.parametrize(
    "thresholds, match",
    [
        ([0.0, 0.0, 1.0], r"strictly increasing"),
        ([1.0, 0.0], r"strictly increasing"),
        ([0.0, np.inf], r"finite"),
        ([np.nan, 1.0], r"finite"),
        ([], r"length >= 1"),
        ([[0.0, 1.0]], r"1D array"),
    ],
    ids=["equal", "decreasing", "inf", "nan", "empty", "2d"],
)
def test_ct_invalid_raises(thresholds, match):
    """Each invalid threshold vector raises ValueError with a specific message."""
    with pytest.raises(ValueError, match=match):
        check_thresholds(thresholds)


@pytest.mark.parametrize(
    "y, classes, expected",
    [
        (np.array([1, 3, 5, 3, 1]), np.array([1, 3, 5]), [0, 1, 2, 1, 0]),
        ([1, 3, 5, 3, 1], [1, 3, 5], [0, 1, 2, 1, 0]),
        (np.array([1.0, 5.0, 3.0]), np.array([1.0, 3.0, 5.0]), [0, 2, 1]),
    ],
    ids=["ndarray", "list", "float"],
)
def test_rel_known_encoding(y, classes, expected):
    """Exact ranks, np.intp dtype, and a clean round-trip through classes."""
    encoded = _rank_encode_labels(y, classes)
    assert_array_equal(encoded, expected)
    assert encoded.dtype == np.intp
    assert_array_equal(np.asarray(classes)[encoded], y)


def test_rel_above_max_returns_length_without_raising():
    """A label above the highest known class encodes to len(classes)."""
    encoded = _rank_encode_labels(np.array([1, 3, 6]), np.array([1, 3, 5]))
    assert_array_equal(encoded, [0, 1, 3])


@pytest.mark.parametrize(
    "y, match",
    [
        (np.array([1, 2, 3]), r"\[2\]"),
        (np.array([0, 3, 5]), r"\[0\]"),
        (np.array([-7, 1]), r"\[-7\]"),
    ],
    ids=["interior-gap", "below-min", "below-min-negative"],
)
def test_rel_unknown_label_raises(y, match):
    """Only labels above the last class escape. Every other unknown raises."""
    with pytest.raises(
        ValueError,
        match=r"y contains labels not present in the given label set: " + match,
    ):
        _rank_encode_labels(y, np.array([1, 3, 5]))


def test_rel_unknown_labels_reported_sorted_and_deduped():
    """Offending labels are listed ascending and deduplicated, with the label set."""
    with pytest.raises(ValueError, match=r"label set: \[2, 4\]; labels=\[1, 3, 5\]\."):
        _rank_encode_labels(np.array([4, 2, 4, 2]), np.array([1, 3, 5]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_rel_non_finite_float_raises(bad):
    """NaN and both infinities are rejected before any encoding happens."""
    with pytest.raises(ValueError, match=r"^y contains non-finite values\.$"):
        _rank_encode_labels(np.array([1.0, bad]), np.array([1.0, 3.0, 5.0]))


def test_rel_empty_y_returns_empty():
    """An empty y encodes to an empty np.intp array without raising."""
    encoded = _rank_encode_labels(np.array([], dtype=int), np.array([1, 3, 5]))
    assert encoded.shape == (0,)
    assert encoded.dtype == np.intp


@pytest.mark.parametrize(
    "y, expected, match",
    [
        (np.array([7, 7]), [0, 0], None),
        (np.array([9]), [1], None),
        (np.array([1]), None, r"labels not present.*\[1\].*labels=\[7\]"),
    ],
    ids=["exact", "above", "below"],
)
def test_rel_single_element_classes(y, expected, match):
    """A one-class label set exercises the degenerate clip bound correctly."""
    classes = np.array([7])
    if match is None:
        assert_array_equal(_rank_encode_labels(y, classes), expected)
    else:
        with pytest.raises(ValueError, match=match):
            _rank_encode_labels(y, classes)


def test_rel_input_name_swaps_both_messages():
    """input_name= replaces the leading 'y' in the non-finite and unknown errors."""
    classes = np.array([1, 3, 5])
    with pytest.raises(ValueError, match=r"y_true contains non-finite values"):
        _rank_encode_labels(np.array([1.0, np.inf]), classes, input_name="y_true")
    with pytest.raises(ValueError, match=r"y_true contains labels not present"):
        _rank_encode_labels(np.array([1, 2, 3]), classes, input_name="y_true")


def test_rel_string_labels_encode():
    """Non-numeric labels encode by rank and still reject unknown values."""
    classes = np.array(["a", "b", "c"])
    assert_array_equal(_rank_encode_labels(np.array(["a", "c"]), classes), [0, 2])
    with pytest.raises(ValueError, match=r"label set: \['bb'\]"):
        _rank_encode_labels(np.array(["bb"]), classes)
