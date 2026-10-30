"""Tests of the losses, their intercept-only thresholds and the PAV."""

import warnings

import numpy as np
import pytest
import scipy.optimize
from numpy.testing import assert_allclose, assert_array_equal

from skordinal._loss import AllThresholdLoss, ImmediateThresholdLoss
from skordinal._loss.tests.test_margin import FORMULAS
from skordinal.linear_model import LogisticAT, LogisticIT
from skordinal.utils.extmath import (
    params_to_thresholds,
    thresholds_grad,
    thresholds_to_params,
)

LOSSES = [
    AllThresholdLoss("logistic"),
    AllThresholdLoss("exponential"),
    ImmediateThresholdLoss("logistic"),
    ImmediateThresholdLoss("exponential"),
]


def make_problem(n_samples, n_classes, *, weighted=False, all_classes=False):
    """Return labels, increasing projections, ordered thresholds and sample weights."""
    rng = np.random.RandomState(0)
    y = rng.randint(n_classes, size=n_samples)
    if all_classes:
        y[:n_classes] = np.arange(n_classes)
    # Projections grow with the label, as those of a fitted model do
    f = 0.5 * (y - (n_classes - 1) / 2) + rng.standard_normal(n_samples)
    thresholds = np.sort(rng.standard_normal(n_classes - 1))
    sample_weight = None
    if weighted:
        sample_weight = rng.uniform(0.5, 2.0, size=n_samples)
        # Zero a third of the weights, sparing the first n_classes samples
        sample_weight[n_classes::3] = 0.0
    return y.astype(np.intp), f, thresholds, sample_weight


def optimal_thresholds(loss, y, sample_weight, n_classes):
    """Return the minimiser of the loss at ``f = 0`` by L-BFGS-B from three starts."""
    f = np.zeros(len(y))

    def objective(params):
        theta = params_to_thresholds(params)
        value, _, grad = loss.loss_gradient(y, f, theta, sample_weight)
        return value.sum(), thresholds_grad(params, grad)

    results = [
        scipy.optimize.minimize(
            objective,
            thresholds_to_params(np.linspace(low, high, n_classes - 1)),
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 0.0, "gtol": 1e-12, "maxiter": 10_000},
        )
        for low, high in [(-2.0, 2.0), (0.5, 1.5), (-1.0, 0.0)]
    ]
    return params_to_thresholds(min(results, key=lambda result: result.fun).x)


def naive_pav(num, den, scale):
    """Return the ordered optima ``scale * log(num / den)`` by naive merging."""
    blocks = [[k] for k in range(len(num))]
    while True:
        values = [scale * np.log(num[b].sum() / den[b].sum()) for b in blocks]
        unordered = [j for j in range(len(blocks) - 1) if values[j] > values[j + 1]]
        if not unordered:
            break
        # Merge the first pair out of order, adding its masses
        j = unordered[0]
        blocks[j : j + 2] = [blocks[j] + blocks[j + 1]]
    theta = np.empty(len(num))
    for block, value in zip(blocks, values):
        theta[block] = value
    return theta


def threshold_masses(loss, counts):
    """Return the masses on each side of every threshold from the per-class ones."""
    if isinstance(loss, AllThresholdLoss):
        return np.cumsum(counts)[:-1], np.cumsum(counts[::-1])[::-1][1:]
    return counts[:-1], counts[1:]


@pytest.fixture(params=LOSSES, ids=repr)
def loss(request):
    """One registered loss."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param((1, 2, False), id="n1-K2"),
        pytest.param((1, 5, True), id="n1-K5-weighted"),
        pytest.param((30, 2, True), id="n30-K2-weighted"),
        pytest.param((30, 5, False), id="n30-K5"),
        pytest.param((30, 5, True), id="n30-K5-weighted"),
    ]
)
def problem(request):
    """Random problem whose labels need not cover every class."""
    n_samples, n_classes, weighted = request.param
    return make_problem(n_samples, n_classes, weighted=weighted)


@pytest.fixture(
    params=[
        pytest.param((2, False), id="K2"),
        pytest.param((3, True), id="K3-weighted"),
        pytest.param((5, False), id="K5"),
        pytest.param((5, True), id="K5-weighted"),
        pytest.param(None, id="K5-alternating"),
    ]
)
def init_problem(request):
    """Labels, weights and classes where every class has mass."""
    if request.param is not None:
        n_classes, weighted = request.param
        y, _, _, w = make_problem(30, n_classes, weighted=weighted, all_classes=True)
        return y, w, n_classes
    # Alternating class sizes force the immediate-threshold PAV to merge blocks
    y = np.repeat(np.arange(5), [10, 2, 10, 2, 10])
    return y, np.random.RandomState(1).uniform(0.5, 2.0, size=len(y)), 5


def test_shapes_and_weights(loss, problem):
    """Outputs are well formed, consistent and weighted, and leave inputs intact."""
    y, f, thresholds, sample_weight = problem
    n, m = len(y), len(thresholds)
    weight = np.full(n, 2.0) if sample_weight is None else sample_weight
    inputs = [a.copy() for a in (y, f, thresholds, weight)]

    unweighted = loss.loss_gradient(y, f, thresholds)
    unit = loss.loss_gradient(y, f, thresholds, np.ones(n))
    weighted = loss.loss_gradient(y, f, thresholds, weight)
    doubled = loss.loss_gradient(y, f, thresholds, 2.0 * weight)
    grad, hess = loss.gradient_hessian(y, f, thresholds)
    grad_weighted, hess_weighted = loss.gradient_hessian(y, f, thresholds, weight)

    assert [a.shape for a in unweighted] == [(n,), (n,), (m,)]
    assert grad.shape == hess.shape == (n,)
    assert all(a.dtype == np.float64 for a in (*unweighted, grad, hess))
    # Non-negative convex margins give non-negative losses and second derivatives
    assert np.all(unweighted[0] >= 0.0) and np.all(hess >= 0.0)
    assert_allclose(grad, unweighted[1], rtol=1e-12)
    # None means unit weights
    for a, b in zip(unweighted, unit):
        assert_array_equal(a, b)
    assert_allclose(weighted[0], weight * unweighted[0], rtol=1e-12)
    assert_allclose(weighted[1], weight * unweighted[1], rtol=1e-12)
    assert_allclose(doubled[2], 2.0 * weighted[2], rtol=1e-12)
    assert_allclose(grad_weighted, weight * grad, rtol=1e-12)
    assert_allclose(hess_weighted, weight * hess, rtol=1e-12)
    for before, after in zip(inputs, (y, f, thresholds, weight)):
        assert_array_equal(before, after)


def test_finite_differences(loss, problem):
    """The gradients and the second derivative match central differences."""
    y, f, thresholds, sample_weight = problem
    eps = 1e-5

    _, grad_f, grad_thresholds = loss.loss_gradient(y, f, thresholds, sample_weight)
    _, hess = loss.gradient_hessian(y, f, thresholds, sample_weight)
    # Each f_i only enters its own sample, so all projections are perturbed at once
    fd_f = loss.loss(y, f + eps, thresholds, sample_weight)
    fd_f -= loss.loss(y, f - eps, thresholds, sample_weight)
    fd_thresholds = [
        loss.loss(y, f, thresholds + step, sample_weight).sum()
        - loss.loss(y, f, thresholds - step, sample_weight).sum()
        for step in eps * np.eye(len(thresholds))
    ]
    fd_hess = loss.loss_gradient(y, f + eps, thresholds, sample_weight)[1]
    fd_hess -= loss.loss_gradient(y, f - eps, thresholds, sample_weight)[1]

    assert_allclose(grad_f, fd_f / (2 * eps), rtol=1e-5, atol=1e-6)
    assert_allclose(grad_thresholds, np.divide(fd_thresholds, 2 * eps), atol=1e-6)
    assert_allclose(hess, fd_hess / (2 * eps), rtol=1e-5, atol=1e-6)


def test_init_thresholds(loss, init_problem):
    """init_thresholds is the ordered minimiser of the loss at ``f = 0``."""
    y, sample_weight, n_classes = init_problem

    theta = loss.init_thresholds(y, n_classes, sample_weight)

    assert theta.shape == (n_classes - 1,)
    assert np.all(np.diff(theta) >= 0.0)
    expected = optimal_thresholds(loss, y, sample_weight, n_classes)
    assert_allclose(theta, expected, atol=1e-6)


def test_init_thresholds_weight_scale(loss):
    """init_thresholds depends on the weights only through their ratios."""
    y, _, _, sample_weight = make_problem(30, 5, weighted=True, all_classes=True)
    # An empty class brings in the pseudo-count, which must scale with the weights
    y[y == 2] = 3

    theta = loss.init_thresholds(y, 5, sample_weight)
    scaled = loss.init_thresholds(y, 5, 0.01 * sample_weight)

    assert_allclose(scaled, theta, rtol=1e-12)


def test_probabilities(loss, problem):
    """predict_cumproba is non-decreasing in [0, 1] and predict_proba differences it."""
    y, f, thresholds, _ = problem

    cumproba = loss.predict_cumproba(f, thresholds)
    proba = loss.predict_proba(f, thresholds)

    assert cumproba.shape == (len(y), len(thresholds))
    assert np.all((cumproba >= 0.0) & (cumproba <= 1.0))
    assert np.all(np.diff(cumproba, axis=1) >= 0.0)
    expected = np.diff(cumproba, axis=1, prepend=0.0, append=1.0)
    assert_allclose(proba, expected, atol=1e-12)


@pytest.mark.parametrize(
    "estimator_cls, loss_cls",
    [(LogisticAT, AllThresholdLoss), (LogisticIT, ImmediateThresholdLoss)],
    ids=["LogisticAT", "LogisticIT"],
)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("weighted", [False, True])
def test_logistic_estimators(estimator_cls, loss_cls, n_classes, weighted):
    """The logistic losses rebuild the objective and gradient of LogisticAT/IT."""
    rng = np.random.RandomState(0)
    n, p, alpha = 40, 3, 0.7
    X = rng.standard_normal((n, p))
    y = rng.randint(n_classes, size=n)
    coef, t = rng.standard_normal(p), rng.standard_normal(n_classes - 1)
    weight = rng.uniform(0.5, 2.0, size=n) if weighted else None

    J_ref, grad_ref = estimator_cls(alpha=alpha)._objective(
        np.concatenate([coef, t]),
        X,
        y,
        np.ones(n) if weight is None else weight,
        p,
        n_classes,
    )
    value, grad_f, grad_t = loss_cls("logistic").loss_gradient(
        y, X @ coef, params_to_thresholds(t), weight
    )
    J = value.sum() / n + alpha / (2 * n) * (coef @ coef)
    grad_coef = X.T @ grad_f / n + alpha / n * coef
    grad = np.concatenate([grad_coef, thresholds_grad(t, grad_t / n)])

    assert_allclose(J, J_ref, rtol=1e-12)
    assert_allclose(grad, grad_ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "loss_cls, signs",
    [
        (AllThresholdLoss, [[1, 1], [-1, 1], [-1, -1], [-1, 1], [1, 1], [-1, -1]]),
        (ImmediateThresholdLoss, [[1, 0], [-1, 1], [0, -1], [-1, 1], [1, 0], [0, -1]]),
    ],
    ids=["AllThresholdLoss", "ImmediateThresholdLoss"],
)
@pytest.mark.parametrize("margin", list(FORMULAS))
def test_hand_formulas(loss_cls, signs, margin):
    """Loss, derivatives and probabilities follow the formulas term by term."""
    # The last two samples lie far above both thresholds, misclassified and not
    y, f = np.array([0, 1, 2, 1, 0, 2]), np.array([0.3, -0.2, 1.1, 0.4, 40.0, 40.0])
    thresholds = np.array([-0.5, 0.8])
    w = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 0.5])
    formulas = FORMULAS[margin]
    loss_ref, grad_f_ref, hess_ref = np.zeros(6), np.zeros(6), np.zeros(6)
    grad_t_ref = np.zeros(2)
    # signs[i][k] is the sign s of the term of sample i at threshold k, 0 if absent
    for i, k in np.ndindex(6, 2):
        s = signs[i][k]
        if s:
            z = s * (thresholds[k] - f[i])
            loss_ref[i] += w[i] * formulas["phi"](z)
            grad_f_ref[i] -= w[i] * s * formulas["dphi"](z)
            grad_t_ref[k] += w[i] * s * formulas["dphi"](z)
            hess_ref[i] += w[i] * formulas["d2phi"](z)
    cumproba_ref = np.vectorize(formulas["cumproba"])(thresholds - f[:, None])
    loss = loss_cls(margin)

    value, grad_f, grad_t = loss.loss_gradient(y, f, thresholds, w)
    _, hess = loss.gradient_hessian(y, f, thresholds, w)

    assert_allclose(value, loss_ref, rtol=1e-12)
    assert_allclose(grad_f, grad_f_ref, rtol=1e-12)
    assert_allclose(grad_t, grad_t_ref, rtol=1e-12)
    assert_allclose(hess, hess_ref, rtol=1e-12)
    assert_allclose(loss.predict_cumproba(f, thresholds), cumproba_ref, rtol=1e-12)


@pytest.mark.parametrize("n_classes", [3, 6])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("margin", list(FORMULAS))
def test_pav(n_classes, seed, margin):
    """Immediate-threshold initial thresholds match a naive isotonic merge."""
    rng = np.random.RandomState(seed)
    proportions = rng.dirichlet(np.full(n_classes, 0.7))
    y = np.concatenate([np.arange(n_classes), rng.choice(n_classes, 60, p=proportions)])
    w = rng.uniform(0.2, 3.0, size=len(y))

    theta = ImmediateThresholdLoss(margin).init_thresholds(y, n_classes, w)

    counts = np.bincount(y, w, n_classes)
    scale = FORMULAS[margin]["log_odds_scale"]
    assert_allclose(theta, naive_pav(counts[:-1], counts[1:], scale), rtol=1e-10)


@pytest.mark.parametrize("loss", LOSSES, ids=repr)
@pytest.mark.parametrize(
    "missing", [[2], [0, 1], [4]], ids=["middle", "first_two", "last"]
)
@pytest.mark.parametrize("how", ["absent", "absent_unweighted", "zero_weight"])
def test_empty_classes(loss, missing, how):
    """Empty classes count as half a mean sample and give finite thresholds silently."""
    y, _, _, w = make_problem(30, 5, weighted=True, all_classes=True)
    empty = np.isin(y, missing)
    if how == "zero_weight":
        w[empty] = 0.0
    else:
        y[empty] = 3
    if how == "absent_unweighted":
        w = None
    counts = np.bincount(y, w, 5).astype(np.float64)
    n_weighted = len(y) if w is None else np.count_nonzero(w)
    counts[counts == 0.0] = 0.5 * counts.sum() / n_weighted
    num, den = threshold_masses(loss, counts)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        theta = loss.init_thresholds(y, 5, w)

    assert np.all(np.isfinite(theta)) and np.all(np.diff(theta) >= 0.0)
    expected = naive_pav(num, den, FORMULAS[loss.margin.name]["log_odds_scale"])
    assert_allclose(theta, expected, rtol=1e-10)


@pytest.mark.parametrize("margin", list(FORMULAS))
def test_two_classes(margin):
    """With two classes the all-threshold and immediate-threshold losses coincide."""
    y, f, thresholds, w = make_problem(30, 2, weighted=True, all_classes=True)
    at, it = AllThresholdLoss(margin), ImmediateThresholdLoss(margin)
    args = (y, f, thresholds, w)

    for a, b in zip(at.loss_gradient(*args), it.loss_gradient(*args)):
        assert_allclose(a, b, rtol=1e-12)
    assert_allclose(at.gradient_hessian(*args), it.gradient_hessian(*args), rtol=1e-12)
    assert_allclose(
        at.predict_proba(f, thresholds), it.predict_proba(f, thresholds), rtol=1e-12
    )
    assert_allclose(at.init_thresholds(y, 2, w), it.init_thresholds(y, 2, w))
