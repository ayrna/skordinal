"""Tests for the ordinal coding utilities."""

import numpy as np
import numpy.testing as npt
import pytest

from skordinal.preprocessing import (
    binary_cumulative_to_ordinal,
    build_coding_matrix,
    ordinal_to_binary_cumulative,
)


@pytest.mark.parametrize(
    "n, K",
    [(5, 4), (1, 2)],
)
def test_encode_output_shape_and_dtype(n, K):
    """Output shape is (n, K-1) with integer dtype."""
    classes = np.arange(K)
    rng = np.random.default_rng(0)
    y = rng.integers(0, K, size=n)
    result = ordinal_to_binary_cumulative(y, classes)
    assert result.shape == (n, K - 1)
    assert result.dtype == np.intp


@pytest.mark.parametrize(
    "y, classes, expected",
    [
        # K=2: single threshold
        (
            np.array([0, 1, 0]),
            np.array([0, 1]),
            np.array([[0], [1], [0]]),
        ),
        # K=3: standard triangular pattern
        (
            np.array([0, 1, 2, 1, 0]),
            np.array([0, 1, 2]),
            np.array([[0, 0], [1, 0], [1, 1], [1, 0], [0, 0]]),
        ),
        # K=4: non-zero-indexed labels
        (
            np.array([10, 20, 30, 40]),
            np.array([10, 20, 30, 40]),
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]),
        ),
    ],
)
def test_encode_correctness(y, classes, expected):
    """Hand-traced known cases encode correctly for K=2, K=3, and K=4 non-zero-indexed labels."""
    npt.assert_array_equal(ordinal_to_binary_cumulative(y, classes), expected)


def test_encode_non_contiguous_labels():
    """Non-contiguous class values produce the same triangular pattern."""
    classes = np.array([3, 5, 9])
    y = np.array([3, 5, 9])
    expected = np.array([[0, 0], [1, 0], [1, 1]])
    npt.assert_array_equal(ordinal_to_binary_cumulative(y, classes), expected)


def test_encode_empty_input_returns_zero_rows():
    """n_samples == 0 returns shape (0, K-1) without raising."""
    classes = np.array([0, 1, 2, 3])
    result = ordinal_to_binary_cumulative(np.array([], dtype=int), classes)
    assert result.shape == (0, 3)


def test_encode_raises_on_unknown_label():
    """A label absent from classes triggers ValueError."""
    classes = np.array([0, 1, 2])
    with pytest.raises(ValueError, match=r"not in classes"):
        ordinal_to_binary_cumulative(np.array([0, 5, 1]), classes)


@pytest.mark.parametrize("classes", [np.array([]), np.array([0])])
def test_encode_raises_on_too_few_classes(classes):
    """len(classes) < 2 triggers ValueError."""
    with pytest.raises(ValueError, match=r"at least 2 unique labels"):
        ordinal_to_binary_cumulative(np.array([0]), classes)


def test_decode_output_shape_and_dtype_matches_classes():
    """Output shape is (n,) and dtype matches classes.dtype."""
    classes = np.array(["low", "mid", "high"])
    B = np.array([[0, 0], [1, 0], [1, 1]])
    result = binary_cumulative_to_ordinal(B, classes)
    assert result.shape == (3,)
    assert result.dtype == classes.dtype


@pytest.mark.parametrize(
    "B, classes, expected",
    [
        # Hard {0,1} predictions with K=4
        (
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]),
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3]),
        ),
        # Non-zero-indexed hard predictions
        (
            np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]]),
            np.array([10, 20, 30, 40]),
            np.array([10, 30, 40]),
        ),
        # Soft (probability) predictions including boundary cases
        (
            np.array(
                [
                    [0.10, 0.05, 0.01],  # all <= 0.5 -> class 0
                    [0.90, 0.40, 0.10],  # one > 0.5 -> class 1
                    [0.95, 0.80, 0.30],  # two > 0.5 -> class 2
                    [0.99, 0.95, 0.60],  # three > 0.5 -> class 3
                    [0.51, 0.49, 0.51],  # non-monotone: 2 out of 3 > 0.5 -> class 2
                    [0.50, 0.50, 0.50],  # exactly 0.5 is NOT > 0.5 -> class 0
                ]
            ),
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 3, 2, 0]),
        ),
    ],
)
def test_decode_correctness(B, classes, expected):
    """Hard and soft predictions decode correctly; p==0.5 maps to the lower class."""
    npt.assert_array_equal(binary_cumulative_to_ordinal(B, classes), expected)


def test_decode_empty_input_returns_zero_rows():
    """n_samples == 0 returns shape (0,) without raising."""
    classes = np.array([0, 1, 2, 3])
    result = binary_cumulative_to_ordinal(np.zeros((0, 3), dtype=int), classes)
    assert result.shape == (0,)


def test_decode_raises_on_shape_mismatch():
    """Wrong number of columns in binary_preds raises ValueError."""
    classes = np.array([0, 1, 2])  # expects K-1 = 2 columns
    with pytest.raises(ValueError, match=r"shape"):
        binary_cumulative_to_ordinal(np.zeros((4, 5), dtype=int), classes)


@pytest.mark.parametrize("classes", [np.array([]), np.array([0])])
def test_decode_raises_on_too_few_classes(classes):
    """len(classes) < 2 triggers ValueError."""
    with pytest.raises(ValueError, match=r"at least 2 unique labels"):
        binary_cumulative_to_ordinal(np.array([[0]]), classes)


@pytest.mark.parametrize("K", [3, 10])
@pytest.mark.parametrize(
    "strategy",
    ["ordered_partitions", "one_vs_next", "one_vs_followers", "one_vs_previous"],
)
def test_coding_matrix_shape_dtype_and_values(K, strategy):
    """Shape is (K, K-1), dtype is np.intp, and values are a subset of {-1, 0, +1}."""
    coding = build_coding_matrix(K, strategy)
    assert coding.shape == (K, K - 1)
    assert coding.dtype == np.intp
    assert set(np.unique(coding).tolist()).issubset({-1, 0, 1})


@pytest.mark.parametrize(
    "strategy, expected",
    [
        (
            "ordered_partitions",
            np.array([[-1, -1], [1, -1], [1, 1]]),
        ),
        (
            "one_vs_next",
            np.array([[1, 0], [-1, 1], [0, -1]]),
        ),
        (
            "one_vs_followers",
            np.array([[1, 0], [-1, 1], [-1, -1]]),
        ),
        (
            "one_vs_previous",
            np.array([[-1, -1], [1, -1], [0, 1]]),
        ),
    ],
)
def test_coding_matrix_k3_reference(strategy, expected):
    """K=3 hand-traced reference matrices match expected values for all four strategies."""
    npt.assert_array_equal(build_coding_matrix(3, strategy), expected)


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_ordered_partitions_structural_property(K):
    """ordered_partitions has no zeros; column k is -1 for rows 0..k and +1 for rows k+1..K-1."""
    coding = build_coding_matrix(K, "ordered_partitions")
    assert (coding != 0).all(), "ordered_partitions must contain no zeros"
    for k in range(K - 1):
        expected_col = np.where(np.arange(K) <= k, -1, 1)
        npt.assert_array_equal(coding[:, k], expected_col)


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_one_vs_next_structural_property(K):
    """one_vs_next: each column has exactly one +1, one -1, and all other entries are 0."""
    coding = build_coding_matrix(K, "one_vs_next")
    for k in range(K - 1):
        col = coding[:, k]
        assert (col == 1).sum() == 1, f"column {k} must have exactly one +1"
        assert (col == -1).sum() == 1, f"column {k} must have exactly one -1"
        assert (col == 0).sum() == K - 2, f"column {k} must have K-2 zeros"


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_one_vs_followers_structural_property(K):
    """one_vs_followers: column k is +1 at row k, -1 at rows k+1..K-1, and 0 elsewhere."""
    coding = build_coding_matrix(K, "one_vs_followers")
    for k in range(K - 1):
        col = coding[:, k]
        assert col[k] == 1
        npt.assert_array_equal(col[k + 1 :], -1)
        npt.assert_array_equal(col[:k], 0)


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_one_vs_previous_structural_property(K):
    """one_vs_previous: column k is -1 at rows 0..k, +1 at row k+1, and 0 elsewhere."""
    coding = build_coding_matrix(K, "one_vs_previous")
    for k in range(K - 1):
        col = coding[:, k]
        npt.assert_array_equal(col[: k + 1], -1)
        assert col[k + 1] == 1
        npt.assert_array_equal(col[k + 2 :], 0)


@pytest.mark.parametrize("invalid", [-1, 0, 2])
def test_coding_matrix_raises_on_n_classes_below_three(invalid):
    """n_classes < 3 raises ValueError."""
    with pytest.raises(ValueError, match=r"n_classes"):
        build_coding_matrix(invalid, "ordered_partitions")


def test_coding_matrix_raises_on_unknown_decomposition():
    """Unrecognised decomposition string raises ValueError."""
    with pytest.raises(ValueError, match=r"decomposition"):
        build_coding_matrix(4, "not-a-strategy")


@pytest.mark.parametrize("K", [2, 3, 5])
def test_round_trip_encode_decode(K):
    """decode(encode(y, classes), classes) == y for hard predictions."""
    classes = np.arange(K)
    rng = np.random.default_rng(0)
    y = rng.integers(0, K, size=50)
    npt.assert_array_equal(
        binary_cumulative_to_ordinal(ordinal_to_binary_cumulative(y, classes), classes),
        y,
    )
