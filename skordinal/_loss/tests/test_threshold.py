"""Tests of the threshold losses, their initial thresholds and their PAV."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skordinal._loss import AllThresholdLoss, ImmediateThresholdLoss
from skordinal._loss.tests.test_common import LOSSES, make_problem
from skordinal._loss.tests.test_margin import FORMULAS
from skordinal.linear_model import LogisticAT, LogisticIT
from skordinal.utils.extmath import params_to_thresholds, thresholds_grad


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
