"""Properties shared by every loss, run on each one registered in ``LOSSES``."""

import numpy as np
import pytest
import scipy.optimize
from numpy.testing import assert_allclose, assert_array_equal

from skordinal._loss import AllThresholdLoss, ImmediateThresholdLoss
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
