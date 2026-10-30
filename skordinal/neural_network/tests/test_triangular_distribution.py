"""Tests for the triangular soft label distribution."""

import warnings

import numpy as np
import pytest

from skordinal.neural_network._triangular_distribution import (
    _MAX_ALPHA2,
    _get_intervals,
    _triangular_cdf,
    _triangular_params,
    get_triangular_soft_labels,
)

# Output of dlordinal.soft_labelling.get_triangular_soft_labels (dlordinal
# 2.2.1), printed with 12 significant digits
_DLORDINAL_REFERENCE = {
    (2, 0.01): np.array(
        [
            [0.988454949125, 0.0115450473049],
            [0.000835921702805, 0.999164074413],
        ]
    ),
    (2, 0.05): np.array(
        [
            [0.972182071775, 0.0278179248922],
            [0.00216131391147, 0.997838682274],
        ]
    ),
    (2, 0.1): np.array(
        [
            [0.971385960188, 0.0286140364887],
            [0.00222912360003, 0.997770872589],
        ]
    ),
    (3, 0.01): np.array(
        [
            [0.988454947532, 0.0115450471131, 0.0],
            [0.0100000002428, 0.979999999514, 0.0100000002428],
            [0.0, 0.00253517947186, 0.99746481483],
        ]
    ),
    (3, 0.05): np.array(
        [
            [0.972182070386, 0.0278179246142, 0.0],
            [0.0500000004325, 0.899999999135, 0.0500000004325],
            [0.0, 0.00641257157058, 0.99358742291],
        ]
    ),
    (3, 0.1): np.array(
        [
            [0.971385958807, 0.0286140362077, 0.0],
            [0.100000000494, 0.799999999011, 0.100000000494],
            [0.0, 0.006608078381, 0.993391916107],
        ]
    ),
    (5, 0.01): np.array(
        [
            [0.988454944345, 0.0115450467295, 0.0, 0.0, 0.0],
            [0.0100000007285, 0.979999999514, 0.00999999975716, 0.0, 0.0],
            [0.0, 0.0100000002428, 0.979999999514, 0.0100000002428, 0.0],
            [0.0, 0.0, 0.00999999975716, 0.979999999514, 0.0100000007285],
            [0.0, 0.0, 0.0, 0.00505524254956, 0.994944748161],
        ]
    ),
    (5, 0.05): np.array(
        [
            [0.972182067609, 0.0278179240584, 0.0, 0.0, 0.0],
            [0.0500000012974, 0.899999999135, 0.0499999995675, 0.0, 0.0],
            [0.0, 0.0500000004325, 0.899999999135, 0.0500000004325, 0.0],
            [0.0, 0.0, 0.0499999995675, 0.899999999135, 0.0500000012974],
            [0.0, 0.0, 0.0, 0.01254458379, 0.98745540733],
        ]
    ),
    (5, 0.1): np.array(
        [
            [0.971385956046, 0.0286140356455, 0.0, 0.0, 0.0],
            [0.100000001483, 0.799999999011, 0.0999999995056, 0.0, 0.0],
            [0.0, 0.100000000494, 0.799999999011, 0.100000000494, 0.0],
            [0.0, 0.0, 0.0999999995056, 0.799999999011, 0.100000001483],
            [0.0, 0.0, 0.0, 0.0129176254565, 0.98708236568],
        ]
    ),
    (10, 0.01): np.array(
        [
            [0.988454936378, 0.0115450457705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0100000019427,
                0.979999999514,
                0.00999999854294,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0100000014571,
                0.979999999514,
                0.00999999902863,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0100000009714,
                0.979999999514,
                0.00999999951431,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0100000004857, 0.979999999514, 0.01, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.01, 0.979999999514, 0.0100000004857, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.00999999951431,
                0.979999999514,
                0.0100000009714,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.00999999902863,
                0.979999999514,
                0.0100000014571,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.00999999854294,
                0.979999999514,
                0.0100000019427,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00784007364527, 0.992159908126],
        ]
    ),
    (10, 0.05): np.array(
        [
            [0.972182060667, 0.0278179226687, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0500000034596,
                0.899999999135,
                0.0499999974053,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0500000025947,
                0.899999999135,
                0.0499999982702,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0500000017298,
                0.899999999135,
                0.0499999991351,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0500000008649, 0.899999999135, 0.05, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.05, 0.899999999135, 0.0500000008649, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0499999991351,
                0.899999999135,
                0.0500000017298,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0499999982702,
                0.899999999135,
                0.0500000025947,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0499999974053,
                0.899999999135,
                0.0500000034596,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0191709245962, 0.980829058173],
        ]
    ),
    (10, 0.1): np.array(
        [
            [0.971385949143, 0.0286140342401, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.100000003955,
                0.799999999011,
                0.0999999970334,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.100000002967,
                0.799999999011,
                0.0999999980223,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.100000001978,
                0.799999999011,
                0.0999999990111,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.100000000989, 0.799999999011, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.799999999011, 0.100000000989, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0999999990111,
                0.799999999011,
                0.100000001978,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0999999980223,
                0.799999999011,
                0.100000002967,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0999999970334,
                0.799999999011,
                0.100000003955,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0197301935858, 0.980269789223],
        ]
    ),
    (2, 0.5): np.array(
        [
            [0.999999996, 0.0],
            [0.0, 0.999999996],
        ]
    ),
    (2, 0.9): np.array(
        [
            [0.918607074813, 0.0813929223286],
            [0.00714481791843, 0.99285517842],
        ]
    ),
    (5, 2 / 9): np.array(
        [
            [0.987654312691, 0.0123456784198, 0.0, 0.0, 0.0],
            [0.222222223556, 0.555555554667, 0.222222221778, 0.0, 0.0],
            [1.99999996089e-18, 0.222222222667, 0.555555554667, 0.222222222667, 0.0],
            [0.0, 2.2222201228e-19, 0.222222221778, 0.555555554667, 0.222222223556],
            [0.0, 0.0, 0.0, 0.00541636753037, 0.994583623206],
        ]
    ),
}

# Example from the dlordinal docstring
_DLORDINAL_DOCSTRING_EXAMPLE = np.array(
    [
        [0.98845494, 0.01154505, 0.0, 0.0, 0.0],
        [0.01, 0.98, 0.01, 0.0, 0.0],
        [0.0, 0.01, 0.98, 0.01, 0.0],
        [0.0, 0.0, 0.01, 0.98, 0.01],
        [0.0, 0.0, 0.0, 0.00505524, 0.99494475],
    ]
)

N_CLASSES = [2, 3, 4, 5, 7, 10, 20]


def _valid_alpha2(n_classes):
    return [0.0, 0.01, 0.05, 0.1, 0.2, _MAX_ALPHA2] + (
        [0.5, 0.9, 1.0] if n_classes == 2 else []
    )


# -----------------
# Reference values
# -----------------


@pytest.mark.parametrize("key", list(_DLORDINAL_REFERENCE))
def test_triangular_soft_labels_match_dlordinal(key):
    """The port reproduces dlordinal's soft labels."""
    n_classes, alpha2 = key

    np.testing.assert_allclose(
        get_triangular_soft_labels(n_classes, alpha2),
        _DLORDINAL_REFERENCE[key],
        atol=1e-8,
    )


def test_triangular_soft_labels_match_dlordinal_docstring_example():
    """The port reproduces the example of the dlordinal docstring."""
    np.testing.assert_allclose(
        get_triangular_soft_labels(5), _DLORDINAL_DOCSTRING_EXAMPLE, atol=1e-8
    )


def test_triangular_soft_labels_default_alpha2():
    """The default alpha2 is 0.01."""
    np.testing.assert_array_equal(
        get_triangular_soft_labels(4), get_triangular_soft_labels(4, 0.01)
    )


# -----------------
# Validation
# -----------------


@pytest.mark.parametrize("n_classes", [1, 0, -3, 2.0, 3.5, "3", None, True])
def test_triangular_soft_labels_rejects_invalid_n_classes(n_classes):
    """``n_classes`` must be an integer of at least 2."""
    with pytest.raises(ValueError, match="n_classes must be an integer >= 2"):
        get_triangular_soft_labels(n_classes)


@pytest.mark.parametrize(
    "n_classes, alpha2",
    [
        (3, -1e-6),
        (3, 0.2223),
        (3, 0.3),
        (5, 0.5),
        (10, 1.0),
        (2, -0.1),
        (2, 1.0001),
        (3, np.nan),
        (3, np.inf),
        (3, "0.1"),
        (3, None),
        (3, True),
    ],
)
def test_triangular_soft_labels_rejects_invalid_alpha2(n_classes, alpha2):
    """``alpha2`` must be a finite number in its valid range."""
    with pytest.raises(ValueError, match="alpha2 must be a float in"):
        get_triangular_soft_labels(n_classes, alpha2)


def test_triangular_soft_labels_error_reports_range_and_value():
    """The error message states the allowed range and the value received."""
    with pytest.raises(ValueError, match=r"\[0, 0\.2222\].*n_classes=4.*0\.3"):
        get_triangular_soft_labels(4, 0.3)


@pytest.mark.parametrize("n_classes", [np.int64(4), np.int32(3)])
def test_triangular_soft_labels_accepts_numpy_integers(n_classes):
    """NumPy integers are valid class counts."""
    assert get_triangular_soft_labels(n_classes).shape == (n_classes, n_classes)


def test_triangular_soft_labels_accepts_numpy_floats():
    """NumPy floats are valid values of alpha2."""
    np.testing.assert_array_equal(
        get_triangular_soft_labels(4, np.float32(0.05)),
        get_triangular_soft_labels(4, float(np.float32(0.05))),
    )


@pytest.mark.parametrize("n_classes", [3, 5, 10])
def test_triangular_soft_labels_accept_the_boundary_alpha2(n_classes):
    """``alpha2 = 2/9`` is still a valid distribution."""
    probs = get_triangular_soft_labels(n_classes, _MAX_ALPHA2)

    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)


# -----------------
# Structure of the soft labels
# -----------------


@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_triangular_soft_labels_are_distributions(n_classes):
    """Rows are non-negative and sum to one up to the trimmed ends."""
    for alpha2 in _valid_alpha2(n_classes):
        probs = get_triangular_soft_labels(n_classes, alpha2)

        assert probs.shape == (n_classes, n_classes)
        assert probs.dtype == np.float64
        assert np.isfinite(probs).all()
        assert np.all(probs >= 0.0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_triangular_soft_labels_are_unimodal_on_the_diagonal(n_classes):
    """Each row peaks on its own class and decreases away from it."""
    for alpha2 in _valid_alpha2(n_classes):
        if n_classes == 2 and alpha2 > 0.5:
            continue
        probs = get_triangular_soft_labels(n_classes, alpha2)

        np.testing.assert_array_equal(probs.argmax(axis=1), np.arange(n_classes))
        for j, row in enumerate(probs):
            assert np.all(np.diff(row[: j + 1]) >= -1e-12)
            assert np.all(np.diff(row[j:]) <= 1e-12)


@pytest.mark.parametrize("n_classes", [3, 4, 5, 10])
@pytest.mark.parametrize("alpha2", [0.01, 0.05, 0.1, 0.2, _MAX_ALPHA2])
def test_triangular_soft_labels_middle_rows(n_classes, alpha2):
    """Middle classes give alpha2 to each neighbour and nothing beyond."""
    probs = get_triangular_soft_labels(n_classes, alpha2)

    for j in range(1, n_classes - 1):
        np.testing.assert_allclose(probs[j, j - 1], alpha2, atol=1e-8)
        np.testing.assert_allclose(probs[j, j + 1], alpha2, atol=1e-8)
        np.testing.assert_allclose(probs[j, j], 1.0 - 2.0 * alpha2, atol=1e-8)
        far = np.abs(np.arange(n_classes) - j) > 1
        np.testing.assert_allclose(probs[j, far], 0.0, atol=1e-8)


@pytest.mark.parametrize("n_classes", [2, 3, 5, 10])
@pytest.mark.parametrize("alpha2", [0.01, 0.1, 0.2])
def test_triangular_soft_labels_extreme_rows_only_reach_one_neighbour(
    n_classes, alpha2
):
    """The first and last classes only leak into their single neighbour."""
    probs = get_triangular_soft_labels(n_classes, alpha2)

    np.testing.assert_allclose(probs[0, 2:], 0.0, atol=1e-8)
    np.testing.assert_allclose(probs[-1, :-2], 0.0, atol=1e-8)
    assert probs[0, 1] > 0.0
    assert probs[-1, -2] > 0.0


@pytest.mark.parametrize("n_classes", [2, 3, 6])
def test_triangular_soft_labels_zero_alpha2_is_identity(n_classes):
    """Without smoothing every class keeps all of its probability."""
    np.testing.assert_allclose(
        get_triangular_soft_labels(n_classes, 0.0), np.eye(n_classes), atol=1e-8
    )


@pytest.mark.parametrize("n_classes", [3, 5, 10])
def test_triangular_soft_labels_smooth_more_with_larger_alpha2(n_classes):
    """A larger alpha2 moves probability off the diagonal for middle classes."""
    alphas = np.linspace(0.0, _MAX_ALPHA2, 12)
    diagonals = np.array(
        [np.diag(get_triangular_soft_labels(n_classes, a))[1:-1] for a in alphas]
    )
    neighbours = np.array(
        [get_triangular_soft_labels(n_classes, a)[1, 0] for a in alphas]
    )

    assert np.all(np.diff(diagonals, axis=0) < 0.0)
    assert np.all(np.diff(neighbours) > 0.0)


@pytest.mark.parametrize("n_classes", [3, 5, 8])
def test_triangular_soft_labels_middle_rows_are_symmetric(n_classes):
    """Middle rows put the same mass on both neighbours."""
    probs = get_triangular_soft_labels(n_classes, 0.1)

    for j in range(1, n_classes - 1):
        assert probs[j, j - 1] == pytest.approx(probs[j, j + 1], abs=1e-8)


def test_triangular_soft_labels_do_not_warn():
    """No floating-point warnings across the whole valid domain."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for n_classes in N_CLASSES:
            for alpha2 in _valid_alpha2(n_classes):
                get_triangular_soft_labels(n_classes, alpha2)


def test_triangular_soft_labels_return_fresh_arrays():
    """Mutating a result does not affect later calls."""
    probs = get_triangular_soft_labels(4, 0.05)
    probs[:] = 0.0

    assert get_triangular_soft_labels(4, 0.05).sum() > 0.0


# -----------------
# Triangle parameters
# -----------------


@pytest.mark.parametrize("n_classes", N_CLASSES)
def test_triangular_params_define_valid_triangles(n_classes):
    """Every class triangle satisfies a <= c <= b with a < b."""
    for alpha2 in _valid_alpha2(n_classes):
        a, b, c = _triangular_params(n_classes, alpha2)

        assert np.all(a <= c)
        assert np.all(c <= b)
        assert np.all(a < b)
        assert a[0] == 0.0 and c[0] == 0.0
        assert b[-1] == 1.0 and c[-1] == 1.0


@pytest.mark.parametrize("n_classes", [3, 5, 10])
@pytest.mark.parametrize("alpha2", [0.01, 0.1, _MAX_ALPHA2])
def test_triangular_params_middle_triangles_stay_within_neighbours(n_classes, alpha2):
    """Middle triangles cover their class and stop inside the neighbours."""
    a, b, _ = _triangular_params(n_classes, alpha2)
    j = np.arange(2, n_classes)

    assert np.all(a[1:-1] <= (j - 1) / n_classes)
    assert np.all(b[1:-1] >= j / n_classes)
    assert np.all(a[1:-1] >= (j - 2) / n_classes - 1e-12)
    assert np.all(b[1:-1] <= (j + 1) / n_classes + 1e-12)


def test_triangular_params_middle_triangles_are_symmetric():
    """The mode of a middle class is the centre of its class interval."""
    n_classes = 6
    _, _, c = _triangular_params(n_classes, 0.1)
    centres = (np.arange(1, n_classes - 1) + 0.5) / n_classes

    np.testing.assert_allclose(c[1:-1], centres, atol=1e-12)


# -----------------
# Intervals
# -----------------


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_get_intervals_split_unit_interval(n_classes):
    """Intervals are consecutive, equal, and trimmed by 1e-9 at the ends."""
    intervals = _get_intervals(n_classes)

    assert intervals.shape == (n_classes, 2)
    assert intervals[0, 0] == 1e-9
    assert intervals[-1, 1] == 1 - 1e-9
    np.testing.assert_array_equal(intervals[1:, 0], intervals[:-1, 1])
    widths = np.diff(intervals, axis=1).ravel()
    np.testing.assert_allclose(widths, widths[0])


# -----------------
# Triangular CDF
# -----------------


def _cdf_reference(x, a, b, c):
    """Closed-form triangular CDF for a single point."""
    if x <= a:
        return 0.0
    if x <= c:
        return (x - a) ** 2 / ((b - a) * (c - a))
    if x < b:
        return 1.0 - (b - x) ** 2 / ((b - a) * (b - c))
    return 1.0


@pytest.mark.parametrize(
    "a, b, c", [(0.0, 1.0, 0.5), (-1.0, 2.0, 0.3), (0.2, 0.4, 0.25)]
)
def test_triangular_cdf_matches_closed_form(a, b, c):
    """The vectorised CDF agrees with the closed form at many points."""
    x = np.linspace(a - 0.5, b + 0.5, 101)

    np.testing.assert_allclose(
        _triangular_cdf(x, a, b, c), [_cdf_reference(v, a, b, c) for v in x]
    )


def test_triangular_cdf_at_the_mode():
    """At the mode the CDF is (c - a) / (b - a), not one as in dlordinal."""
    assert _triangular_cdf(0.3, 0.0, 1.0, 0.3) == pytest.approx(0.3)
    assert _triangular_cdf(0.5, 0.0, 2.0, 0.5) == pytest.approx(0.25)


def test_triangular_cdf_at_the_bounds():
    """The CDF is 0 at and below a, and 1 at and above b."""
    np.testing.assert_array_equal(
        _triangular_cdf(np.array([-5.0, 0.0, 1.0, 7.0]), 0.0, 1.0, 0.4),
        [0.0, 0.0, 1.0, 1.0],
    )


def test_triangular_cdf_is_continuous_and_monotone():
    """The CDF is non-decreasing and continuous across the mode."""
    a, b, c = 0.1, 0.9, 0.6
    x = np.linspace(0.0, 1.0, 2001)
    cdf = _triangular_cdf(x, a, b, c)

    assert np.all(np.diff(cdf) >= 0.0)
    assert np.max(np.abs(np.diff(cdf))) < 5e-3
    eps = 1e-10
    assert _triangular_cdf(c - eps, a, b, c) == pytest.approx(
        _triangular_cdf(c + eps, a, b, c), abs=1e-8
    )


@pytest.mark.parametrize(
    "a, b, c, x, expected",
    [
        # Left-degenerate triangle (mode at the lower bound): 1 - (b - x)^2 / b^2
        (0.0, 2.0, 0.0, 1.0, 0.75),
        (0.0, 2.0, 0.0, 0.0, 0.0),
        # Right-degenerate triangle (mode at the upper bound): x^2 / b^2
        (0.0, 2.0, 2.0, 1.0, 0.25),
        (0.0, 2.0, 2.0, 2.0, 1.0),
    ],
)
def test_triangular_cdf_degenerate_triangles(a, b, c, x, expected):
    """Half triangles with the mode at either bound do not divide by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _triangular_cdf(x, a, b, c) == pytest.approx(expected)


def test_triangular_cdf_broadcasts():
    """Points and parameters broadcast against each other."""
    x = np.array([[0.25, 0.5, 0.75]])
    a = np.zeros((2, 1))
    b = np.ones((2, 1))
    c = np.array([[0.5], [1.0]])

    cdf = _triangular_cdf(x, a, b, c)

    assert cdf.shape == (2, 3)
    np.testing.assert_allclose(cdf[0], [0.125, 0.5, 0.875])
    np.testing.assert_allclose(cdf[1], [0.0625, 0.25, 0.5625])
