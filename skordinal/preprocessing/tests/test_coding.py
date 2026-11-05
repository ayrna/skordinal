"""Tests for the ordinal coding utilities."""

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skordinal.preprocessing import build_coding_matrix


@pytest.mark.parametrize("K", [3, 10])
@pytest.mark.parametrize(
    "decomposition",
    ["cumulative", "adjacent", "sequential", "sequential_reverse"],
)
def test_coding_matrix_shape_dtype_and_values(K, decomposition):
    """Shape is (K, K-1), dtype is np.intp, and values are a subset of {-1, 0, +1}."""
    coding = build_coding_matrix(K, decomposition)
    assert coding.shape == (K, K - 1)
    assert coding.dtype == np.intp
    assert set(np.unique(coding).tolist()).issubset({-1, 0, 1})


@pytest.mark.parametrize(
    "decomposition, expected",
    [
        (
            "cumulative",
            np.array([[-1, -1], [1, -1], [1, 1]]),
        ),
        (
            "adjacent",
            np.array([[1, 0], [-1, 1], [0, -1]]),
        ),
        (
            "sequential",
            np.array([[1, 0], [-1, 1], [-1, -1]]),
        ),
        (
            "sequential_reverse",
            np.array([[-1, -1], [1, -1], [0, 1]]),
        ),
    ],
)
def test_coding_matrix_k3_reference(decomposition, expected):
    """Hand-traced K=3 matrices match the expected ones for every decomposition."""
    npt.assert_array_equal(build_coding_matrix(3, decomposition), expected)


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_cumulative_structural_property(K):
    """In cumulative, column k is -1 for rows 0..k and +1 for rows k+1..K-1."""
    coding = build_coding_matrix(K, "cumulative")
    assert (coding != 0).all(), "cumulative must contain no zeros"
    for k in range(K - 1):
        expected_col = np.where(np.arange(K) <= k, -1, 1)
        npt.assert_array_equal(coding[:, k], expected_col)


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_adjacent_structural_property(K):
    """In adjacent, each column has one +1, one -1 and zeros elsewhere."""
    coding = build_coding_matrix(K, "adjacent")
    for k in range(K - 1):
        col = coding[:, k]
        assert (col == 1).sum() == 1, f"column {k} must have exactly one +1"
        assert (col == -1).sum() == 1, f"column {k} must have exactly one -1"
        assert (col == 0).sum() == K - 2, f"column {k} must have K-2 zeros"


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_sequential_structural_property(K):
    """In sequential, column k is +1 at row k, -1 at rows k+1..K-1 and 0 elsewhere."""
    coding = build_coding_matrix(K, "sequential")
    for k in range(K - 1):
        col = coding[:, k]
        assert col[k] == 1
        npt.assert_array_equal(col[k + 1 :], -1)
        npt.assert_array_equal(col[:k], 0)


@pytest.mark.parametrize("K", [3, 5, 10])
def test_coding_matrix_sequential_reverse_structural_property(K):
    """In sequential_reverse, column k is -1 at rows 0..k, +1 at row k+1, else 0."""
    coding = build_coding_matrix(K, "sequential_reverse")
    for k in range(K - 1):
        col = coding[:, k]
        npt.assert_array_equal(col[: k + 1], -1)
        assert col[k + 1] == 1
        npt.assert_array_equal(col[k + 2 :], 0)


@pytest.mark.parametrize("invalid", [-1, 0, 1])
def test_coding_matrix_raises_on_n_classes_below_two(invalid):
    """n_classes < 2 raises ValueError."""
    with pytest.raises(ValueError, match=r"n_classes"):
        build_coding_matrix(invalid, "cumulative")


@pytest.mark.parametrize(
    "decomposition, expected",
    [
        ("cumulative", np.array([[-1], [1]])),
        ("adjacent", np.array([[1], [-1]])),
        ("sequential", np.array([[1], [-1]])),
        ("sequential_reverse", np.array([[-1], [1]])),
    ],
)
def test_coding_matrix_k2_reference(decomposition, expected):
    """K=2 yields a single well-formed binary subproblem for every decomposition."""
    npt.assert_array_equal(build_coding_matrix(2, decomposition), expected)


def test_coding_matrix_raises_on_unknown_decomposition():
    """Unrecognised decomposition string raises ValueError."""
    with pytest.raises(ValueError, match=r"decomposition"):
        build_coding_matrix(4, "not-a-strategy")


def test_build_coding_matrix_rejects_non_integer_n_classes():
    """A non-integer n_classes is rejected at the parameter boundary."""
    with pytest.raises(InvalidParameterError):
        build_coding_matrix(3.5, "cumulative")
