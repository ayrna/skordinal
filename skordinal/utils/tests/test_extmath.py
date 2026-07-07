"""Tests for skordinal.utils.extmath mathematical helpers."""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skordinal.utils.extmath import (
    cumproba_to_proba,
    losses_to_proba,
    normalize_proba_rows,
    params_to_thresholds,
    proba_to_cumproba,
    repair_cumproba,
    thresholds_grad,
    thresholds_to_params,
)

_EXTMATH_HELPERS = [
    params_to_thresholds,
    thresholds_to_params,
    thresholds_grad,
    proba_to_cumproba,
    cumproba_to_proba,
    repair_cumproba,
    losses_to_proba,
    normalize_proba_rows,
]
_EXTMATH_NAMES = [f.__name__ for f in _EXTMATH_HELPERS]


@pytest.mark.parametrize("func", _EXTMATH_HELPERS, ids=_EXTMATH_NAMES)
def test_api_no_mutable_defaults(func):
    """No extmath helper has a mutable default argument."""
    for default in func.__defaults__ or ():
        assert not isinstance(default, (list, dict, np.ndarray)), (
            f"{func.__name__} has a mutable default: {default!r}"
        )


def test_api_cumproba_to_proba_default_repair():
    """cumproba_to_proba defaults repair=True."""
    sig = inspect.signature(cumproba_to_proba)
    assert sig.parameters["repair"].default is True


def test_p2t_known_values():
    """params_to_thresholds produces the expected non-decreasing vector."""
    t = np.array([1.0, 2.0, -3.0])
    result = params_to_thresholds(t)
    # b[0] = 1.0; b[1] = 1.0 + 4.0 = 5.0; b[2] = 5.0 + 9.0 = 14.0
    assert_allclose(result, [1.0, 5.0, 14.0], atol=1e-12)


def test_p2t_output_is_non_decreasing():
    """Output of params_to_thresholds is always non-decreasing."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = rng.standard_normal(5)
        b = params_to_thresholds(t)
        assert np.all(np.diff(b) >= 0.0)


def test_p2t_single_element():
    """Single-element parameter passes through unchanged."""
    result = params_to_thresholds(np.array([3.0]))
    assert_allclose(result, [3.0], atol=1e-12)


def test_t2p_round_trip():
    """params_to_thresholds(thresholds_to_params(b)) == b for ordered b."""
    b = np.array([1.0, 5.0, 14.0])
    t = thresholds_to_params(b)
    assert_allclose(params_to_thresholds(t), b, atol=1e-10)


@pytest.mark.parametrize(
    "b",
    [
        np.array([-2.0, 0.0, 3.0]),
        np.array([0.5]),
        np.array([1.0, 1.0, 2.0]),  # non-strictly-increasing (flat segment OK)
    ],
    ids=["generic", "single", "flat-segment"],
)
def test_t2p_round_trip_parametrised(b):
    """Round-trip holds for various ordered threshold vectors."""
    t = thresholds_to_params(b)
    assert_allclose(params_to_thresholds(t), b, atol=1e-10)


@pytest.mark.parametrize(
    "call",
    [
        lambda: params_to_thresholds(np.array([])),
        lambda: thresholds_to_params(np.array([])),
        lambda: thresholds_grad(np.array([]), np.array([])),
    ],
    ids=["params_to_thresholds", "thresholds_to_params", "thresholds_grad"],
)
def test_threshold_functions_empty_input_raises(call):
    """Each threshold helper rejects an empty input vector."""
    with pytest.raises(ValueError, match=r"must contain at least one element"):
        call()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_thresholds_grad_matches_finite_difference(seed):
    """thresholds_grad matches a central-difference gradient check."""
    rng = np.random.default_rng(seed)
    t = rng.standard_normal(5)
    grad_b = rng.standard_normal(5)
    eps = 1e-6
    grad_t = thresholds_grad(t, grad_b)
    grad_t_fd = np.empty_like(t)
    for i in range(len(t)):
        t_plus, t_minus = t.copy(), t.copy()
        t_plus[i] += eps
        t_minus[i] -= eps
        loss_plus = grad_b @ params_to_thresholds(t_plus)
        loss_minus = grad_b @ params_to_thresholds(t_minus)
        grad_t_fd[i] = (loss_plus - loss_minus) / (2 * eps)
    assert_allclose(grad_t, grad_t_fd, atol=1e-5)


def test_thresholds_grad_single_element():
    """A single-parameter gradient passes through unchanged."""
    grad_params = thresholds_grad(np.array([3.0]), np.array([5.0]))
    assert_allclose(grad_params, [5.0], atol=1e-12)


def test_thresholds_grad_shape_mismatch_raises():
    """A grad_thresholds size differing from params size raises ValueError."""
    with pytest.raises(ValueError, match=r"must have the same shape"):
        thresholds_grad(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_thresholds_grad_ravels_2d_input():
    """A 2-D grad_thresholds is ravelled to match its 1-D counterpart."""
    t = np.array([1.0, 2.0, -3.0])
    grad_b_1d = np.array([1.0, 0.5, -2.0])
    grad_b_2d = grad_b_1d.reshape(3, 1)
    assert_allclose(thresholds_grad(t, grad_b_2d), thresholds_grad(t, grad_b_1d))


def test_repair_cumproba_already_monotone_unchanged():
    """A monotone input row is returned bit-exact."""
    cumproba = np.array([[0.1, 0.4, 0.8]])
    result = repair_cumproba(cumproba)
    assert_array_equal(result, cumproba)


def test_repair_cumproba_fixes_violation():
    """A non-monotone row is repaired to be non-decreasing."""
    cumproba = np.array([[0.4, 0.2, 0.7]])
    result = repair_cumproba(cumproba)
    assert result.shape == cumproba.shape
    assert np.all(np.diff(result, axis=1) >= 0.0)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_repair_cumproba_mixed_batch_preserves_monotone_row():
    """Batching with a violating row does not affect a monotone row."""
    cumproba = np.array([[0.1, 0.4, 0.8], [0.4, 0.2, 0.7]])
    result = repair_cumproba(cumproba)
    assert_array_equal(result[0], cumproba[0])
    assert np.all(np.diff(result[1]) >= 0.0)


def test_repair_cumproba_idempotent():
    """Applying repair_cumproba twice equals applying it once."""
    rng = np.random.default_rng(42)
    cumproba = rng.uniform(0, 1, size=(10, 4))
    once = repair_cumproba(cumproba)
    twice = repair_cumproba(once)
    assert_allclose(twice, once, atol=1e-12)


def test_repair_cumproba_values_in_unit_interval():
    """Repaired output entries stay in [0, 1]."""
    rng = np.random.default_rng(7)
    cumproba = rng.uniform(0, 1, size=(20, 5))
    result = repair_cumproba(cumproba)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


@pytest.mark.parametrize(
    "cumproba",
    [np.array([[-0.1, 0.5]]), np.array([[0.5, 1.2]])],
    ids=["negative-entry", "above-one-entry"],
)
def test_repair_cumproba_out_of_range_raises(cumproba):
    """Entries outside [0, 1] raise ValueError."""
    with pytest.raises(ValueError, match=r"cumproba entries must lie in \[0, 1\]"):
        repair_cumproba(cumproba)


def test_c2p_valid_input_output_values():
    """Output values, row sums, and non-negativity for a monotonic input."""
    cumproba = np.array([[0.2, 0.5, 0.9]])
    class_proba = cumproba_to_proba(cumproba)
    assert_allclose(class_proba, [[0.2, 0.3, 0.4, 0.1]], atol=1e-12)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)


def test_c2p_repair_true_repairs_violation():
    """Monotonicity violations are repaired silently when repair=True."""
    cumproba = np.array([[0.5, 0.3, 0.9]])
    class_proba = cumproba_to_proba(cumproba, repair=True)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)


def test_c2p_repair_true_matches_hand_computed_repair():
    """repair=True gives the exact PAVA-repaired class probabilities."""
    cumproba = np.array([[0.5, 0.3, 0.9]])
    class_proba = cumproba_to_proba(cumproba, repair=True)
    # Pool the violating pair to their average via PAVA: [0.4, 0.4, 0.9]
    assert_allclose(class_proba, [[0.4, 0.0, 0.5, 0.1]], atol=1e-12)


def test_c2p_repair_false_valid_input():
    """repair=False happy path with multiple rows."""
    cumproba = np.array([[0.2, 0.5, 0.9], [0.1, 0.4, 0.8]])
    class_proba = cumproba_to_proba(cumproba, repair=False)
    assert class_proba.shape == (2, 4)
    assert_allclose(class_proba[0], [0.2, 0.3, 0.4, 0.1], atol=1e-12)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)


def test_c2p_repair_false_raises_on_violation():
    """A non-monotonic row raises ValueError when repair=False."""
    with pytest.raises(ValueError, match=r"cumproba rows must be non-decreasing"):
        cumproba_to_proba(np.array([[0.5, 0.3]]), repair=False)


@pytest.mark.parametrize("repair", [True, False])
def test_c2p_k2_single_column(repair):
    """K=2 single-column input produces 2-column output for both branches."""
    cumproba = np.array([[0.3], [0.7]])
    class_proba = cumproba_to_proba(cumproba, repair=repair)
    assert class_proba.shape == (2, 2)
    assert_allclose(class_proba[0], [0.3, 0.7], atol=1e-12)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "cumproba",
    [np.array([[-0.1, 0.5]]), np.array([[0.5, 1.2]])],
    ids=["negative-entry", "above-one-entry"],
)
def test_c2p_out_of_range_raises(cumproba):
    """Entries outside [0, 1] raise ValueError."""
    with pytest.raises(ValueError, match=r"cumproba entries must lie in \[0, 1\]"):
        cumproba_to_proba(cumproba)


def test_c2p_zero_columns_raises():
    """A cumproba matrix with zero columns raises ValueError."""
    with pytest.raises(ValueError, match=r"minimum of 1 is required"):
        cumproba_to_proba(np.empty((3, 0)))


@pytest.mark.parametrize(
    "cumproba, mass_index",
    [
        (np.array([[0.0, 0.0, 0.0]]), -1),
        (np.array([[1.0, 1.0, 1.0]]), 0),
    ],
    ids=["all-zero-row", "all-one-row"],
)
def test_c2p_special_rows_concentrate_mass(cumproba, mass_index):
    """All-zero and all-one rows place all mass on a single class."""
    class_proba = cumproba_to_proba(cumproba)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)
    assert class_proba[0, mass_index] == pytest.approx(1.0)


def test_c2p_output_shape():
    """Output has n_classes = n_thresholds + 1 columns."""
    rng = np.random.default_rng(11)
    for n_thresholds in [1, 2, 5, 9]:
        cumproba = np.sort(rng.uniform(0, 1, size=(8, n_thresholds)), axis=1)
        result = cumproba_to_proba(cumproba)
        assert result.shape == (8, n_thresholds + 1)


def test_c2p_rows_sum_to_one_multirow():
    """All rows of the output sum to 1.0 within floating-point tolerance."""
    rng = np.random.default_rng(99)
    cumproba = np.sort(rng.uniform(0, 1, size=(50, 6)), axis=1)
    result = cumproba_to_proba(cumproba)
    assert_allclose(result.sum(axis=1), np.ones(50), atol=1e-12)


def test_p2c_known_values():
    """proba_to_cumproba produces the expected cumulative sums."""
    proba = np.array([[0.2, 0.3, 0.4, 0.1]])
    result = proba_to_cumproba(proba)
    assert_allclose(result, [[0.2, 0.5, 0.9]], atol=1e-12)


def test_p2c_round_trip_random_dirichlet():
    """cumproba_to_proba(proba_to_cumproba(P), repair=False) recovers P."""
    rng = np.random.default_rng(21)
    proba = rng.dirichlet(np.ones(5), size=30)
    recovered = cumproba_to_proba(proba_to_cumproba(proba), repair=False)
    assert_allclose(recovered, proba, atol=1e-10)


def test_p2c_round_trip_matches_doctest_example():
    """The doctest's round-trip direction holds under a tight tolerance."""
    proba = np.array([[0.2, 0.3, 0.4, 0.1]])
    recovered = cumproba_to_proba(proba_to_cumproba(proba))
    assert_allclose(recovered, proba, atol=1e-12)


def test_p2c_output_shape():
    """Output has n_classes - 1 columns for K classes."""
    rng = np.random.default_rng(11)
    for n_classes in [2, 3, 5, 9]:
        proba = rng.dirichlet(np.ones(n_classes), size=8)
        result = proba_to_cumproba(proba)
        assert result.shape == (8, n_classes - 1)


def test_p2c_output_dtype_is_float64():
    """Output dtype is float64 regardless of the input dtype."""
    proba = np.array([[0.5, 0.5]], dtype=np.float32)
    result = proba_to_cumproba(proba)
    assert result.dtype == np.float64


def test_p2c_k2_single_column():
    """K=2 input produces a single-column cumulative output."""
    proba = np.array([[0.3, 0.7], [0.6, 0.4]])
    result = proba_to_cumproba(proba)
    assert result.shape == (2, 1)
    assert_allclose(result, [[0.3], [0.6]], atol=1e-12)


def test_p2c_fp_overshoot_regression_clips_to_one():
    """A row whose raw cumsum overshoots 1.0 by a few ulp is clipped."""
    proba = np.append(np.full(100, 1.0 / 100), 0.0).reshape(1, -1)
    # Confirm the row genuinely overshoots before proba_to_cumproba clips it
    raw_partial = np.cumsum(proba, axis=1)[:, :-1]
    assert raw_partial.max() > 1.0
    cumproba = proba_to_cumproba(proba)
    assert cumproba.max() <= 1.0
    # The clipped output must remain valid input for downstream consumers
    repair_cumproba(cumproba)
    cumproba_to_proba(cumproba, repair=False)


def test_p2c_output_is_non_decreasing():
    """Output rows are non-decreasing for non-negative input."""
    rng = np.random.default_rng(3)
    proba = rng.uniform(0, 1, size=(20, 6))
    result = proba_to_cumproba(proba)
    assert np.all(np.diff(result, axis=1) >= 0.0)


def test_p2c_does_not_mutate_input_array():
    """proba_to_cumproba leaves its ndarray argument unchanged."""
    rng = np.random.default_rng(13)
    proba = rng.dirichlet(np.ones(4), size=10).astype(np.float64)
    saved = proba.copy()
    proba_to_cumproba(proba)
    np.testing.assert_array_equal(proba, saved)


def test_l2p_known_values():
    """losses_to_proba produces expected probabilities for a known input."""
    L = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    result = losses_to_proba(L)
    assert_allclose(
        result, [[0.9921, 0.0067, 0.0013], [0.0013, 0.0067, 0.9921]], atol=1e-4
    )


def test_l2p_rows_sum_to_one():
    """Every row of losses_to_proba output sums to 1.0."""
    rng = np.random.default_rng(5)
    L = rng.uniform(0, 2, size=(30, 5))
    result = losses_to_proba(L)
    assert_allclose(result.sum(axis=1), np.ones(30), atol=1e-12)


def test_l2p_all_entries_non_negative():
    """All entries of losses_to_proba output are >= 0."""
    rng = np.random.default_rng(6)
    L = rng.uniform(0, 5, size=(20, 4))
    result = losses_to_proba(L)
    assert np.all(result >= 0.0)


def test_l2p_minimum_loss_gets_maximum_probability():
    """The class with the minimum loss receives the highest probability."""
    L = np.array([[0.01, 1.0, 2.0]])
    result = losses_to_proba(L)
    assert result[0, 0] == result[0].max()


def test_l2p_symmetric_input_gives_symmetric_output():
    """Equal losses produce equal probabilities across classes."""
    L = np.array([[1.0, 1.0, 1.0]])
    result = losses_to_proba(L)
    assert_allclose(result[0], [1 / 3, 1 / 3, 1 / 3], atol=1e-12)


@pytest.mark.parametrize(
    "L",
    [np.array([[0.1, -0.2, 0.3]]), np.array([[0.1, np.nan, 0.3]])],
    ids=["negative", "nan"],
)
def test_l2p_negative_or_nan_loss_raises(L):
    """A negative or NaN loss entry raises ValueError."""
    with pytest.raises(ValueError, match=r"losses must be non-negative"):
        losses_to_proba(L)


def test_npr_rows_sum_to_one():
    """normalize_proba_rows output rows sum to 1.0 within atol=1e-12."""
    rng = np.random.default_rng(42)
    S = rng.uniform(0.0, 2.0, size=(30, 5))
    P = normalize_proba_rows(S)
    assert_allclose(P.sum(axis=1), np.ones(30), atol=1e-12)


def test_npr_all_entries_positive():
    """normalize_proba_rows guarantees all entries > 0 (clipped to floor)."""
    S = np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
    P = normalize_proba_rows(S)
    assert (P > 0.0).all(), "normalize_proba_rows should clip zeros to a positive floor"


def test_npr_underflow_regression_all_entries_strictly_positive():
    """An extreme-scale row no longer underflows to an exact zero entry."""
    S = np.array([[0.0, 1e20]])
    P = normalize_proba_rows(S)
    assert (P > 0.0).all()
    assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_npr_zero_row_becomes_valid():
    """An all-zero row is clipped to floor and normalised to uniform."""
    K = 4
    S = np.zeros((1, K), dtype=float)
    P = normalize_proba_rows(S)
    assert P.shape == (1, K)
    assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    # Clip every zero to the same tiny value, so the row becomes uniform
    assert_allclose(P[0], np.full(K, 1.0 / K), atol=1e-12)


def test_npr_shape_preserved():
    """normalize_proba_rows preserves the input array shape."""
    rng = np.random.default_rng(7)
    for shape in [(1, 2), (10, 3), (50, 7)]:
        S = rng.uniform(0.1, 1.0, size=shape)
        P = normalize_proba_rows(S)
        assert P.shape == shape, f"Shape changed: {shape} -> {P.shape}"


def test_npr_custom_floor_respected():
    """A custom floor clips input entries before normalisation."""
    S = np.array([[0.0, 1.0, 0.0]])
    floor = 0.1
    P = normalize_proba_rows(S, floor=floor)
    assert_allclose(P[0], [0.1 / 1.2, 1.0 / 1.2, 0.1 / 1.2], atol=1e-12)
    assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "floor",
    [0.0, -5.0, float("nan"), np.inf],
    ids=["zero", "negative", "nan", "inf"],
)
def test_npr_non_positive_floor_raises(floor):
    """A non-positive, non-finite, or NaN floor raises ValueError."""
    S = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match=r"floor must be strictly positive"):
        normalize_proba_rows(S, floor=floor)


def test_npr_nan_scores_raises():
    """A NaN entry in scores raises ValueError."""
    S = np.array([[np.nan, 1.0, 2.0]])
    with pytest.raises(ValueError, match=r"scores must be finite"):
        normalize_proba_rows(S)


def test_npr_inf_scores_raises():
    """An infinite entry in scores raises ValueError."""
    S = np.array([[np.inf, 1.0, 2.0]])
    with pytest.raises(ValueError, match=r"scores must be finite"):
        normalize_proba_rows(S)


def test_npr_finite_overflow_row_sum_raises():
    """A finite row whose sum overflows to infinity raises ValueError."""
    S = np.array([[1e308, 1e308]])
    with pytest.raises(ValueError, match=r"scores must be finite"):
        normalize_proba_rows(S)


def test_npr_already_normalised_input_stays_close():
    """A row summing to 1 with positive entries is approximately unchanged."""
    rng = np.random.default_rng(11)
    raw = rng.dirichlet(np.ones(5), size=20)
    P = normalize_proba_rows(raw.copy())
    assert_allclose(P, raw, atol=1e-12)


def test_npr_does_not_mutate_input_array():
    """normalize_proba_rows leaves its ndarray argument unchanged."""
    rng = np.random.default_rng(11)
    S = rng.uniform(0.1, 1.0, size=(20, 5)).astype(np.float64)
    saved = S.copy()
    normalize_proba_rows(S)
    np.testing.assert_array_equal(S, saved)


def test_npr_negative_scores_clipped_to_floor():
    """Negative input entries are clipped up to the tiny floor, not to zero."""
    S = np.array([[-1.0, 2.0, 3.0]])
    P = normalize_proba_rows(S)
    # -1.0 clips to tiny, so the row becomes [tiny, 2.0, 3.0] before dividing
    # by its sum of approximately 5.0
    assert 0.0 < P[0, 0] < 1e-300
    assert_allclose(P[0, 1:], [0.4, 0.6], atol=1e-12)
    assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
