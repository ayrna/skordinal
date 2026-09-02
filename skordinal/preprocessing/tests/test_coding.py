"""Tests for the ordinal coding utilities."""

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skordinal.preprocessing import build_coding_matrix


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


@pytest.mark.parametrize("invalid", [-1, 0, 1])
def test_coding_matrix_raises_on_n_classes_below_two(invalid):
    """n_classes < 2 raises ValueError."""
    with pytest.raises(ValueError, match=r"n_classes"):
        build_coding_matrix(invalid, "ordered_partitions")


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("ordered_partitions", np.array([[-1], [1]])),
        ("one_vs_next", np.array([[1], [-1]])),
        ("one_vs_followers", np.array([[1], [-1]])),
        ("one_vs_previous", np.array([[-1], [1]])),
    ],
)
def test_coding_matrix_k2_reference(strategy, expected):
    """K=2 yields a single well-formed binary subproblem for every strategy."""
    npt.assert_array_equal(build_coding_matrix(2, strategy), expected)


def test_coding_matrix_raises_on_unknown_decomposition():
    """Unrecognised decomposition string raises ValueError."""
    with pytest.raises(ValueError, match=r"decomposition"):
        build_coding_matrix(4, "not-a-strategy")


def test_build_coding_matrix_rejects_non_integer_n_classes():
    """A non-integer n_classes is rejected at the parameter boundary."""
    with pytest.raises(InvalidParameterError):
        build_coding_matrix(3.5, "ordered_partitions")
