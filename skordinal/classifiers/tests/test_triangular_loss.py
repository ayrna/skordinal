"""Tests for the triangular soft label loss."""

import warnings

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import log_softmax, softmax

from skordinal.classifiers._triangular_distribution import get_triangular_soft_labels
from skordinal.classifiers._triangular_loss import TriangularLoss


def _dlordinal_reference(logits, y, cls_probs, eta):
    """Literal NumPy translation of dlordinal's CustomTargetsLoss.forward.

    ``y_true = (1 - eta) * one_hot(y) + eta * cls_probs[y]`` followed by
    ``CrossEntropyLoss(reduction="mean")`` with probability targets.
    """
    n_classes = cls_probs.shape[0]
    y_prob = cls_probs[y]
    target_oh = np.eye(n_classes)[y]
    y_true = (1.0 - eta) * target_oh + eta * y_prob
    return np.mean(-np.sum(y_true * log_softmax(logits, axis=1), axis=1))


@pytest.fixture
def rng():
    """Return a seeded random number generator."""
    return np.random.default_rng(0)


def _problem(rng, n_samples=12, n_classes=4):
    y = rng.integers(0, n_classes, size=n_samples)
    return rng.normal(scale=2.0, size=(n_samples, n_classes)), np.eye(n_classes)[y], y


# -----------------
# Construction
# -----------------


def test_triangular_loss_stores_configuration():
    """The constructor stores its arguments and the soft label matrix."""
    loss_fn = TriangularLoss(5, alpha2=0.1, eta=0.3)

    assert loss_fn.n_classes == 5
    assert loss_fn.alpha2 == 0.1
    assert loss_fn.eta == 0.3
    np.testing.assert_array_equal(loss_fn.cls_probs, get_triangular_soft_labels(5, 0.1))


def test_triangular_loss_defaults():
    """Defaults match dlordinal: alpha2=0.05 and eta=1.0."""
    loss_fn = TriangularLoss(3)

    assert loss_fn.alpha2 == 0.05
    assert loss_fn.eta == 1.0


@pytest.mark.parametrize("eta", [-0.1, 1.01, np.nan, np.inf, "1", None, True])
def test_triangular_loss_rejects_invalid_eta(eta):
    """``eta`` must be a finite number in [0, 1]."""
    with pytest.raises(ValueError, match="eta must be a float in"):
        TriangularLoss(3, eta=eta)


@pytest.mark.parametrize(
    "n_classes, alpha2", [(1, 0.05), (3, 0.3), (3, -0.1), (2.5, 0.05)]
)
def test_triangular_loss_validates_soft_label_arguments(n_classes, alpha2):
    """Invalid ``n_classes`` or ``alpha2`` are rejected at construction."""
    with pytest.raises(ValueError, match="n_classes|alpha2"):
        TriangularLoss(n_classes, alpha2=alpha2)


# -----------------
# Soft targets
# -----------------


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_triangular_loss_soft_targets_blend(rng, eta):
    """Soft targets interpolate between one-hot and the soft label rows."""
    loss_fn = TriangularLoss(5, alpha2=0.1, eta=eta)
    _, Y, y = _problem(rng, n_classes=5)

    expected = (1.0 - eta) * Y + eta * loss_fn.cls_probs[y]

    np.testing.assert_allclose(loss_fn.soft_targets(Y), expected)


def test_triangular_loss_soft_targets_eta_zero_is_one_hot(rng):
    """With eta=0 the targets are the one-hot labels."""
    _, Y, _ = _problem(rng)

    np.testing.assert_array_equal(TriangularLoss(4, eta=0.0).soft_targets(Y), Y)


def test_triangular_loss_soft_targets_eta_one_are_soft_labels(rng):
    """With eta=1 the targets are exactly the soft label rows."""
    loss_fn = TriangularLoss(4, alpha2=0.1, eta=1.0)
    _, Y, y = _problem(rng)

    np.testing.assert_allclose(loss_fn.soft_targets(Y), loss_fn.cls_probs[y])


@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
def test_triangular_loss_soft_targets_are_distributions(rng, eta):
    """Soft targets are non-negative and sum to one."""
    _, Y, _ = _problem(rng)
    T = TriangularLoss(4, alpha2=0.2, eta=eta).soft_targets(Y)

    assert np.all(T >= 0.0)
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("shape", [(5,), (5, 3), (5, 5), (2, 5, 4)])
def test_triangular_loss_rejects_wrong_target_shape(shape):
    """Targets must have one column per class."""
    with pytest.raises(ValueError, match="Y must have shape"):
        TriangularLoss(4).soft_targets(np.zeros(shape))


# -----------------
# Loss value
# -----------------


@pytest.mark.parametrize("n_classes", [2, 3, 5, 8])
@pytest.mark.parametrize("alpha2", [0.0, 0.01, 0.05, 0.2])
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_triangular_loss_matches_dlordinal(rng, n_classes, alpha2, eta):
    """The loss equals dlordinal's CustomTargetsLoss with cross-entropy."""
    loss_fn = TriangularLoss(n_classes, alpha2=alpha2, eta=eta)
    logits, Y, y = _problem(rng, n_samples=20, n_classes=n_classes)

    loss, _ = loss_fn.loss_and_grad(logits, Y)

    assert loss == pytest.approx(
        _dlordinal_reference(logits, y, loss_fn.cls_probs, eta), rel=1e-12
    )


def test_triangular_loss_eta_zero_is_standard_cross_entropy(rng):
    """With eta=0 the loss is the usual multinomial cross-entropy."""
    logits, Y, y = _problem(rng)
    loss, _ = TriangularLoss(4, alpha2=0.1, eta=0.0).loss_and_grad(logits, Y)

    expected = -np.mean(log_softmax(logits, axis=1)[np.arange(len(y)), y])

    assert loss == pytest.approx(expected, rel=1e-12)


def test_triangular_loss_returns_python_float(rng):
    """The loss is a plain float, as scipy.optimize expects."""
    logits, Y, _ = _problem(rng)
    loss, _ = TriangularLoss(4).loss_and_grad(logits, Y)

    assert type(loss) is float


def test_triangular_loss_is_minimised_by_the_soft_targets(rng):
    """Logits that reproduce the targets give a lower loss than any other."""
    loss_fn = TriangularLoss(4, alpha2=0.1, eta=1.0)
    _, Y, _ = _problem(rng)
    T = loss_fn.soft_targets(Y)
    best, _ = loss_fn.loss_and_grad(np.log(T + 1e-300), Y)

    for _ in range(10):
        other, _ = loss_fn.loss_and_grad(rng.normal(size=T.shape), Y)
        assert best < other


def test_triangular_loss_penalises_distant_errors_more():
    """Predicting a distant class costs more than predicting a neighbour."""
    loss_fn = TriangularLoss(5, alpha2=0.1, eta=1.0)
    Y = np.eye(5)[[2]]
    near = np.full((1, 5), -10.0)
    near[0, 3] = 10.0
    far = np.full((1, 5), -10.0)
    far[0, 0] = 10.0

    assert loss_fn.loss_and_grad(near, Y)[0] < loss_fn.loss_and_grad(far, Y)[0]


def test_triangular_loss_is_invariant_to_logit_shifts(rng):
    """Adding a constant to every logit of a sample changes nothing."""
    loss_fn = TriangularLoss(4, alpha2=0.1, eta=0.7)
    logits, Y, _ = _problem(rng)
    shift = rng.normal(scale=50.0, size=(logits.shape[0], 1))

    loss, grad = loss_fn.loss_and_grad(logits, Y)
    loss_shifted, grad_shifted = loss_fn.loss_and_grad(logits + shift, Y)

    assert loss_shifted == pytest.approx(loss, rel=1e-10)
    np.testing.assert_allclose(grad_shifted, grad, atol=1e-12)


# -----------------
# Sample weights
# -----------------


def test_triangular_loss_unit_weights_match_no_weights(rng):
    """Unit sample weights are the same as no sample weights."""
    loss_fn = TriangularLoss(4, alpha2=0.05)
    logits, Y, _ = _problem(rng)

    loss, grad = loss_fn.loss_and_grad(logits, Y)
    loss_w, grad_w = loss_fn.loss_and_grad(logits, Y, np.ones(len(Y)))

    assert loss_w == loss
    np.testing.assert_array_equal(grad_w, grad)


def test_triangular_loss_weights_are_normalised(rng):
    """Scaling every weight by a constant does not change the loss."""
    loss_fn = TriangularLoss(4, alpha2=0.05)
    logits, Y, _ = _problem(rng)
    w = rng.uniform(0.5, 2.0, size=len(Y))

    loss, grad = loss_fn.loss_and_grad(logits, Y, w)
    loss_scaled, grad_scaled = loss_fn.loss_and_grad(logits, Y, 7.0 * w)

    assert loss_scaled == pytest.approx(loss, rel=1e-12)
    np.testing.assert_allclose(grad_scaled, grad, atol=1e-15)


def test_triangular_loss_integer_weights_match_repeated_samples(rng):
    """A weight of k is the same as repeating the sample k times."""
    loss_fn = TriangularLoss(4, alpha2=0.1, eta=0.5)
    logits, Y, _ = _problem(rng)
    w = rng.integers(1, 4, size=len(Y))

    loss, _ = loss_fn.loss_and_grad(logits, Y, w.astype(float))
    loss_repeated, _ = loss_fn.loss_and_grad(
        np.repeat(logits, w, axis=0), np.repeat(Y, w, axis=0)
    )

    assert loss == pytest.approx(loss_repeated, rel=1e-12)


def test_triangular_loss_zero_weight_ignores_sample(rng):
    """A sample with zero weight contributes nothing."""
    loss_fn = TriangularLoss(4, alpha2=0.1)
    logits, Y, _ = _problem(rng)
    w = np.ones(len(Y))
    w[0] = 0.0

    loss, grad = loss_fn.loss_and_grad(logits, Y, w)
    loss_dropped, _ = loss_fn.loss_and_grad(logits[1:], Y[1:])

    assert loss == pytest.approx(loss_dropped, rel=1e-12)
    np.testing.assert_array_equal(grad[0], 0.0)


@pytest.mark.parametrize("shape", [(11,), (13,), (12, 1)])
def test_triangular_loss_rejects_wrong_weight_shape(rng, shape):
    """Sample weights must have one entry per sample."""
    logits, Y, _ = _problem(rng)

    with pytest.raises(ValueError, match="sample_weight must have shape"):
        TriangularLoss(4).loss_and_grad(logits, Y, np.ones(shape))


def test_triangular_loss_rejects_mismatched_samples(rng):
    """Logits and targets must describe the same samples."""
    logits, Y, _ = _problem(rng)

    with pytest.raises(ValueError, match="same number of samples"):
        TriangularLoss(4).loss_and_grad(logits, Y[:-1])


def test_triangular_loss_rejects_wrong_logit_shape(rng):
    """Logits must have one column per class."""
    _, Y, _ = _problem(rng)

    with pytest.raises(ValueError, match="logits must have shape"):
        TriangularLoss(4).loss_and_grad(np.zeros((len(Y), 3)), Y)


# -----------------
# Gradient
# -----------------


@pytest.mark.parametrize("n_classes", [2, 3, 6])
@pytest.mark.parametrize("eta", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_triangular_loss_gradient_matches_finite_difference(
    rng, n_classes, eta, weighted
):
    """The analytic gradient matches finite differences."""
    loss_fn = TriangularLoss(n_classes, alpha2=0.1, eta=eta)
    logits, Y, _ = _problem(rng, n_samples=8, n_classes=n_classes)
    w = rng.uniform(0.2, 3.0, size=len(Y)) if weighted else None
    shape = logits.shape

    error = check_grad(
        lambda z: loss_fn.loss_and_grad(z.reshape(shape), Y, w)[0],
        lambda z: loss_fn.loss_and_grad(z.reshape(shape), Y, w)[1].ravel(),
        logits.ravel(),
    )

    assert error < 1e-6


def test_triangular_loss_gradient_closed_form(rng):
    """The gradient is w_i (softmax_i * sum(T_i) - T_i) / sum(w)."""
    loss_fn = TriangularLoss(4, alpha2=0.1, eta=0.8)
    logits, Y, _ = _problem(rng)
    w = rng.uniform(0.5, 2.0, size=len(Y))
    T = loss_fn.soft_targets(Y)

    _, grad = loss_fn.loss_and_grad(logits, Y, w)
    expected = (
        w[:, None] * (softmax(logits, axis=1) * T.sum(1, keepdims=True) - T) / w.sum()
    )

    np.testing.assert_allclose(grad, expected, atol=1e-15)


def test_triangular_loss_gradient_rows_sum_to_zero_up_to_trimmed_mass(rng):
    """Shifting all logits of a sample is a flat direction of the loss."""
    logits, Y, _ = _problem(rng)
    _, grad = TriangularLoss(4, alpha2=0.1).loss_and_grad(logits, Y)

    np.testing.assert_allclose(grad.sum(axis=1), 0.0, atol=1e-12)


def test_triangular_loss_gradient_vanishes_at_the_targets(rng):
    """The gradient is zero when the softmax equals the soft targets."""
    loss_fn = TriangularLoss(4, alpha2=0.1, eta=1.0)
    _, Y, _ = _problem(rng)
    T = loss_fn.soft_targets(Y)

    _, grad = loss_fn.loss_and_grad(np.log(T + 1e-300), Y)

    np.testing.assert_allclose(grad, 0.0, atol=1e-8)


# -----------------
# Numerical stability
# -----------------


@pytest.mark.parametrize("scale", [1e3, 1e6])
def test_triangular_loss_is_stable_for_large_logits(rng, scale):
    """Huge logits give finite loss and gradient without warnings."""
    loss_fn = TriangularLoss(4, alpha2=0.1, eta=0.5)
    _, Y, _ = _problem(rng)
    logits = scale * np.sign(rng.normal(size=Y.shape))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss, grad = loss_fn.loss_and_grad(logits, Y)

    assert np.isfinite(loss)
    assert np.isfinite(grad).all()


def test_triangular_loss_confident_wrong_prediction_is_large_but_finite():
    """A confident mistake costs about the logit gap times the target mass."""
    loss_fn = TriangularLoss(3, alpha2=0.0, eta=0.0)
    Y = np.eye(3)[[0]]
    logits = np.array([[-1e3, 0.0, 1e3]])

    loss, _ = loss_fn.loss_and_grad(logits, Y)

    assert loss == pytest.approx(2e3, rel=1e-12)
