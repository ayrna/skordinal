"""Tests for the NumPy Cumulative Link Model output layer."""

import warnings

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import expit
from scipy.stats import norm

from skordinal.classifiers._clm import CLM

LINKS = ["logit", "probit", "cloglog"]


def _dlordinal_reference(projected, b, a, n_classes, link_function, min_distance):
    """Literal NumPy translation of ``dlordinal.output_layers.CLM.forward``."""
    a = a**2
    a = a + min_distance
    thresholds_param = np.concatenate((b, a))
    th = np.sum(
        np.tril(np.ones((n_classes - 1, n_classes - 1)))
        * np.reshape(
            np.tile(thresholds_param, (n_classes - 1,)),
            (n_classes - 1, n_classes - 1),
        ),
        axis=1,
    )

    projected = np.reshape(projected, (-1,))
    m = projected.shape[0]
    a_mat = np.reshape(np.tile(th, (m,)), (m, -1))
    b_mat = np.reshape(np.tile(projected, (n_classes - 1,)), (-1, m)).T
    z3 = a_mat - b_mat

    if link_function == "probit":
        a3T = norm.cdf(z3)
    elif link_function == "cloglog":
        a3T = 1 - np.exp(-np.exp(z3))
    else:
        a3T = expit(z3)

    a3 = np.concatenate((a3T, np.ones((m, 1))), axis=1)
    a3[:, 1:] = a3[:, 1:] - a3[:, 0:-1]
    return a3


@pytest.fixture
def rng():
    """Return a seeded random number generator."""
    return np.random.default_rng(0)


# -----------------
# Construction
# -----------------


@pytest.mark.parametrize("n_classes", [2, 3, 10, np.int64(4)])
@pytest.mark.parametrize("link_function", LINKS)
def test_clm_init_stores_configuration(n_classes, link_function):
    """The constructor stores its arguments and derives ``n_params``."""
    clm = CLM(n_classes, link_function=link_function, min_distance=0.5)

    assert clm.n_classes == n_classes
    assert isinstance(clm.n_classes, int)
    assert clm.link_function == link_function
    assert clm.min_distance == 0.5
    assert clm.n_params == n_classes - 1


def test_clm_default_arguments():
    """The default link is logit and the default minimum distance is zero."""
    clm = CLM(3)

    assert clm.link_function == "logit"
    assert clm.min_distance == 0.0


@pytest.mark.parametrize("n_classes", [1, 0, -3, 2.0, 2.5, "3", None, True])
def test_clm_rejects_invalid_n_classes(n_classes):
    """``n_classes`` must be an integer of at least 2."""
    with pytest.raises(ValueError, match="n_classes"):
        CLM(n_classes)


@pytest.mark.parametrize("link_function", ["identity", "Logit", "", None, 1])
def test_clm_rejects_invalid_link_function(link_function):
    """Only the logit, probit and cloglog links are supported."""
    with pytest.raises(ValueError, match="link_function"):
        CLM(3, link_function=link_function)


@pytest.mark.parametrize("min_distance", [-1e-3, -1, np.nan, np.inf, "0.1", None, True])
def test_clm_rejects_invalid_min_distance(min_distance):
    """``min_distance`` must be a finite non-negative number."""
    with pytest.raises(ValueError, match="min_distance"):
        CLM(3, min_distance=min_distance)


@pytest.mark.parametrize("min_distance", [0, 0.0, 1, 2.5, np.float32(0.1)])
def test_clm_accepts_valid_min_distance(min_distance):
    """Integer and floating minimum distances are accepted as floats."""
    clm = CLM(3, min_distance=min_distance)

    assert isinstance(clm.min_distance, float)
    assert clm.min_distance == pytest.approx(float(min_distance))


# -----------------
# Parameters and thresholds
# -----------------


@pytest.mark.parametrize("n_classes", [3, 4, 5, 8])
def test_clm_init_params_spread_thresholds_over_unit_interval(n_classes):
    """Initial thresholds are evenly spaced over [-2, 2]."""
    clm = CLM(n_classes)
    params = clm.init_params()

    assert params.shape == (n_classes - 1,)
    np.testing.assert_allclose(
        clm.thresholds(params), np.linspace(-2.0, 2.0, n_classes - 1), atol=1e-12
    )


@pytest.mark.parametrize("n_classes", [3, 4, 6])
@pytest.mark.parametrize("min_distance", [0.25, 1.0])
def test_clm_init_params_are_centred_with_min_distance(n_classes, min_distance):
    """With a minimum distance, initial thresholds stay centred and evenly spaced."""
    clm = CLM(n_classes, min_distance=min_distance)
    thresholds = clm.thresholds(clm.init_params())
    gaps = np.diff(thresholds)

    assert thresholds.mean() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(gaps, gaps[0])
    assert np.all(gaps >= min_distance)


@pytest.mark.parametrize("n_classes", [3, 5, 9])
def test_clm_init_params_avoid_stationary_gaps(n_classes):
    """No gap parameter starts at zero, the stationary point of the square map."""
    params = CLM(n_classes).init_params()

    assert np.all(params[1:] != 0.0)


def test_clm_init_params_binary():
    """A binary layer starts with one threshold at zero."""
    np.testing.assert_array_equal(CLM(2).init_params(), [0.0])


def test_clm_init_params_returns_fresh_array():
    """Mutating the initial parameters does not affect later calls."""
    clm = CLM(4)
    params = clm.init_params()
    params[:] = 100.0

    assert not np.array_equal(clm.init_params(), params)


@pytest.mark.parametrize("n_classes", [2, 3, 5, 10])
@pytest.mark.parametrize("min_distance", [0.0, 0.3, 2.0])
def test_clm_thresholds_are_ordered_with_min_distance(rng, n_classes, min_distance):
    """Any parameter vector maps to thresholds at least ``min_distance`` apart."""
    clm = CLM(n_classes, min_distance=min_distance)

    for _ in range(20):
        params = rng.normal(scale=3.0, size=n_classes - 1)
        thresholds = clm.thresholds(params)

        assert thresholds.shape == (n_classes - 1,)
        assert thresholds[0] == params[0]
        assert np.all(np.diff(thresholds) >= min_distance - 1e-12)


def test_clm_thresholds_closed_form():
    """Thresholds follow ``b + sum_j (a_j^2 + min_distance)``."""
    clm = CLM(4, min_distance=0.5)
    params = np.array([-1.0, 2.0, -3.0])

    np.testing.assert_allclose(clm.thresholds(params), [-1.0, 3.5, 13.0])


@pytest.mark.parametrize("method", ["thresholds", "cumproba", "forward"])
@pytest.mark.parametrize("params", [np.zeros(2), np.zeros(4), np.zeros((3, 1)), []])
def test_clm_rejects_wrong_params_shape(method, params):
    """Every method checks that ``params`` has shape ``(n_classes - 1,)``."""
    clm = CLM(4)
    args = (params,) if method == "thresholds" else (np.zeros(5), params)

    with pytest.raises(ValueError, match="params must have shape"):
        getattr(clm, method)(*args)


def test_clm_methods_do_not_mutate_inputs(rng):
    """The layer is stateless and leaves its inputs untouched."""
    clm = CLM(4, link_function="probit", min_distance=0.2)
    z = rng.normal(size=6)
    params = rng.normal(size=3)
    grad_proba = rng.normal(size=(6, 4))
    z_copy, params_copy, grad_copy = z.copy(), params.copy(), grad_proba.copy()

    clm.forward(z, params)
    clm.cumproba(z, params)
    clm.backward(z, params, grad_proba)

    np.testing.assert_array_equal(z, z_copy)
    np.testing.assert_array_equal(params, params_copy)
    np.testing.assert_array_equal(grad_proba, grad_copy)


# -----------------
# Forward pass
# -----------------


@pytest.mark.parametrize("link_function", LINKS)
@pytest.mark.parametrize("n_classes", [2, 3, 5, 7])
@pytest.mark.parametrize("min_distance", [0.0, 0.1, 1.0])
def test_clm_forward_matches_dlordinal(rng, link_function, n_classes, min_distance):
    """The NumPy layer reproduces the dlordinal torch implementation."""
    clm = CLM(n_classes, link_function=link_function, min_distance=min_distance)
    z = rng.normal(scale=2.0, size=25)
    params = rng.normal(size=n_classes - 1)

    expected = _dlordinal_reference(
        z, params[:1], params[1:], n_classes, link_function, min_distance
    )

    np.testing.assert_allclose(clm.forward(z, params), expected, atol=1e-12)


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_forward_matches_dlordinal_initial_state(rng, link_function):
    """dlordinal's initial state (b=0, a=1) gives thresholds 0, 1, 2, ..."""
    n_classes = 5
    clm = CLM(n_classes, link_function=link_function)
    params = np.array([0.0, 1.0, 1.0, 1.0])
    z = rng.normal(size=10)

    np.testing.assert_allclose(clm.thresholds(params), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        clm.forward(z, params),
        _dlordinal_reference(z, params[:1], params[1:], n_classes, link_function, 0.0),
        atol=1e-12,
    )


@pytest.mark.parametrize("link_function", LINKS)
@pytest.mark.parametrize("n_classes", [2, 4, 6])
def test_clm_forward_returns_valid_distributions(rng, link_function, n_classes):
    """Probabilities have the right shape, lie in [0, 1] and sum to one."""
    clm = CLM(n_classes, link_function=link_function)
    z = rng.normal(scale=5.0, size=50)
    proba = clm.forward(z, rng.normal(size=n_classes - 1))

    assert proba.shape == (50, n_classes)
    assert proba.dtype == np.float64
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_cumproba_closed_form(rng, link_function):
    """Cumulative probabilities equal ``F(theta_k - z)`` for each link."""
    clm = CLM(4, link_function=link_function, min_distance=0.1)
    z = rng.normal(size=12)
    params = rng.normal(size=3)
    eta = clm.thresholds(params)[np.newaxis, :] - z[:, np.newaxis]

    expected = {
        "logit": expit(eta),
        "probit": norm.cdf(eta),
        "cloglog": 1.0 - np.exp(-np.exp(eta)),
    }[link_function]

    np.testing.assert_allclose(clm.cumproba(z, params), expected, atol=1e-14)


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_cumproba_agrees_with_forward(rng, link_function):
    """Cumulative sums of ``forward`` recover ``cumproba``."""
    clm = CLM(5, link_function=link_function)
    z = rng.normal(scale=2.0, size=30)
    params = rng.normal(size=4)

    cumproba = clm.cumproba(z, params)

    assert cumproba.shape == (30, 4)
    assert np.all(np.diff(cumproba, axis=1) >= 0.0)
    np.testing.assert_allclose(
        np.cumsum(clm.forward(z, params), axis=1)[:, :-1], cumproba, atol=1e-12
    )


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_accepts_column_vector_projection(rng, link_function):
    """Projections of shape (n, 1) and (n,) give the same probabilities."""
    clm = CLM(4, link_function=link_function)
    z = rng.normal(size=8)
    params = rng.normal(size=3)

    np.testing.assert_array_equal(
        clm.forward(z[:, np.newaxis], params), clm.forward(z, params)
    )
    np.testing.assert_array_equal(
        clm.cumproba(z[:, np.newaxis], params), clm.cumproba(z, params)
    )


def test_clm_accepts_lists():
    """Array-likes are converted to arrays."""
    clm = CLM(3)

    np.testing.assert_array_equal(
        clm.forward([0.0, 1.0], [-1.0, 1.0]),
        clm.forward(np.array([0.0, 1.0]), np.array([-1.0, 1.0])),
    )


@pytest.mark.parametrize("z", [np.zeros((4, 2)), np.zeros((2, 2, 1)), np.float64(1.0)])
@pytest.mark.parametrize("method", ["cumproba", "forward"])
def test_clm_rejects_invalid_projection_shape(z, method):
    """The projection must be a vector or a single column."""
    clm = CLM(3)

    with pytest.raises(ValueError, match="z must have shape"):
        getattr(clm, method)(z, clm.init_params())


def test_clm_forward_handles_empty_batch():
    """An empty batch returns an empty probability matrix."""
    proba = CLM(4).forward(np.empty(0), np.zeros(3))

    assert proba.shape == (0, 4)


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_higher_projection_shifts_mass_to_higher_classes(link_function):
    """Increasing ``z`` lowers every ``P(Y <= k)`` and raises the expected class."""
    clm = CLM(5, link_function=link_function)
    params = clm.init_params()
    z = np.linspace(-6.0, 6.0, 50)

    cumproba = clm.cumproba(z, params)
    expected_class = clm.forward(z, params) @ np.arange(5)

    assert np.all(np.diff(cumproba, axis=0) <= 0.0)
    assert np.all(np.diff(expected_class) > 0.0)


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_projection_between_thresholds_predicts_that_class(link_function):
    """A projection far inside an interval puts most mass on its class."""
    clm = CLM(4, link_function=link_function, min_distance=20.0)
    params = np.array([-30.0, 0.0, 0.0])
    thresholds = clm.thresholds(params)
    centres = np.concatenate(
        [
            [thresholds[0] - 10.0],
            (thresholds[:-1] + thresholds[1:]) / 2,
            [thresholds[-1] + 10.0],
        ]
    )

    proba = clm.forward(centres, params)

    np.testing.assert_array_equal(proba.argmax(axis=1), np.arange(4))
    assert np.all(proba.max(axis=1) > 0.99)


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_extreme_projections_are_finite_and_one_hot(link_function):
    """Huge projections saturate to one-hot rows without warnings."""
    clm = CLM(4, link_function=link_function)
    z = np.array([-1e300, -1e6, -1e3, 1e3, 1e6, 1e300])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        proba = clm.forward(z, clm.init_params())
        grad_z, grad_params = clm.backward(z, clm.init_params(), np.ones((6, 4)))

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba[:3], np.tile([1.0, 0, 0, 0], (3, 1)))
    np.testing.assert_allclose(proba[3:], np.tile([0, 0, 0, 1.0], (3, 1)))
    assert np.isfinite(grad_z).all()
    assert np.isfinite(grad_params).all()


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_tied_thresholds_give_zero_middle_probability(link_function):
    """Equal thresholds leave no mass to the class between them."""
    clm = CLM(4, link_function=link_function)
    params = np.array([0.5, 0.0, 1.0])
    proba = clm.forward(np.linspace(-3.0, 3.0, 7), params)

    np.testing.assert_array_equal(proba[:, 1], 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_clm_logit_tail_probabilities_keep_relative_precision():
    """Tail probabilities are not lost to cancellation near one.

    A naive ``1 - F(eta)`` rounds to zero once ``F(eta)`` is within machine
    epsilon of one; the survival function keeps full relative precision.
    """
    clm = CLM(3, link_function="logit")
    params = np.array([0.0, 1.0])
    z = np.array([-40.0])

    proba = clm.forward(z, params)

    np.testing.assert_allclose(proba[0, 1], expit(-40.0) - expit(-41.0), rtol=1e-10)
    np.testing.assert_allclose(proba[0, 2], expit(-41.0), rtol=1e-10)
    assert proba[0, 2] > 0.0


def test_clm_probit_tail_probabilities_keep_relative_precision():
    """The probit link keeps relative precision in the upper tail."""
    clm = CLM(2, link_function="probit")
    proba = clm.forward(np.array([-10.0]), np.array([0.0]))

    np.testing.assert_allclose(proba[0, 1], norm.sf(10.0), rtol=1e-10)


def test_clm_cloglog_tail_probabilities_keep_relative_precision():
    """The cloglog link keeps relative precision in the lower tail."""
    clm = CLM(2, link_function="cloglog")
    proba = clm.forward(np.array([40.0]), np.array([0.0]))

    np.testing.assert_allclose(proba[0, 0], -np.expm1(-np.exp(-40.0)), rtol=1e-10)
    assert proba[0, 0] > 0.0


def test_clm_cloglog_is_asymmetric():
    """Unlike logit and probit, cloglog is not symmetric around zero."""
    clm = CLM(2, link_function="cloglog")
    proba = clm.forward(np.array([-1.0, 1.0]), np.array([0.0]))

    assert not np.isclose(proba[0, 0], proba[1, 1])


@pytest.mark.parametrize("link_function", ["logit", "probit"])
def test_clm_symmetric_links_are_symmetric(link_function):
    """For symmetric links, P(Y=1 | z) equals P(Y=2 | -z) with a zero threshold."""
    clm = CLM(2, link_function=link_function)
    z = np.linspace(-4.0, 4.0, 9)
    proba = clm.forward(z, np.array([0.0]))

    np.testing.assert_allclose(proba[:, 0], proba[::-1, 1], atol=1e-15)


def test_clm_links_give_different_probabilities(rng):
    """Each link function defines a different model."""
    z = rng.normal(size=10)
    params = np.array([-1.0, 1.0])
    probas = [CLM(3, link_function=link).forward(z, params) for link in LINKS]

    for i in range(len(probas)):
        for j in range(i + 1, len(probas)):
            assert not np.allclose(probas[i], probas[j])


# -----------------
# Backward pass
# -----------------


@pytest.mark.parametrize("link_function", LINKS)
@pytest.mark.parametrize("n_classes", [2, 3, 6])
@pytest.mark.parametrize("min_distance", [0.0, 0.5])
def test_clm_backward_matches_finite_difference(
    rng, link_function, n_classes, min_distance
):
    """Analytic gradients of ``<G, forward(z, params)>`` match finite differences."""
    clm = CLM(n_classes, link_function=link_function, min_distance=min_distance)
    n_samples = 9
    z = rng.normal(size=n_samples)
    params = rng.normal(size=n_classes - 1)
    grad_proba = rng.normal(size=(n_samples, n_classes))

    def loss(theta):
        return np.sum(grad_proba * clm.forward(theta[:n_samples], theta[n_samples:]))

    def grad(theta):
        return np.concatenate(
            clm.backward(theta[:n_samples], theta[n_samples:], grad_proba)
        )

    assert check_grad(loss, grad, np.concatenate([z, params])) < 1e-6


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_backward_of_log_likelihood_matches_finite_difference(rng, link_function):
    """Gradients of the negative log-likelihood match finite differences."""
    clm = CLM(4, link_function=link_function, min_distance=0.1)
    n_samples = 15
    Y = np.eye(4)[rng.integers(0, 4, size=n_samples)]
    z0 = rng.normal(size=n_samples)
    params0 = clm.init_params()

    def nll(theta):
        proba = clm.forward(theta[:n_samples], theta[n_samples:])
        return -np.sum(Y * np.log(proba))

    def grad(theta):
        z, params = theta[:n_samples], theta[n_samples:]
        return np.concatenate(clm.backward(z, params, -Y / clm.forward(z, params)))

    assert check_grad(nll, grad, np.concatenate([z0, params0])) < 1e-5


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_backward_preserves_projection_shape(rng, link_function):
    """``grad_z`` has the same shape as the projection passed in."""
    clm = CLM(3, link_function=link_function)
    params = rng.normal(size=2)
    grad_proba = rng.normal(size=(5, 3))
    z = rng.normal(size=5)

    grad_vec, grad_params_vec = clm.backward(z, params, grad_proba)
    grad_col, grad_params_col = clm.backward(z[:, np.newaxis], params, grad_proba)

    assert grad_vec.shape == (5,)
    assert grad_col.shape == (5, 1)
    assert grad_params_vec.shape == (2,)
    np.testing.assert_array_equal(grad_col[:, 0], grad_vec)
    np.testing.assert_array_equal(grad_params_col, grad_params_vec)


@pytest.mark.parametrize("link_function", LINKS)
def test_clm_backward_of_constant_is_zero(rng, link_function):
    """The rows always sum to one, so a uniform upstream gradient vanishes."""
    clm = CLM(5, link_function=link_function)
    z = rng.normal(size=10)
    params = rng.normal(size=4)

    grad_z, grad_params = clm.backward(z, params, np.full((10, 5), 3.0))

    np.testing.assert_allclose(grad_z, 0.0, atol=1e-15)
    np.testing.assert_allclose(grad_params, 0.0, atol=1e-15)


def test_clm_backward_zero_upstream_gives_zero(rng):
    """A zero upstream gradient produces zero gradients."""
    clm = CLM(4)
    grad_z, grad_params = clm.backward(
        rng.normal(size=6), rng.normal(size=3), np.zeros((6, 4))
    )

    np.testing.assert_array_equal(grad_z, 0.0)
    np.testing.assert_array_equal(grad_params, 0.0)


def test_clm_backward_is_linear_in_upstream_gradient(rng):
    """The backward pass is linear in the upstream gradient."""
    clm = CLM(4, link_function="probit")
    z = rng.normal(size=6)
    params = rng.normal(size=3)
    g1 = rng.normal(size=(6, 4))
    g2 = rng.normal(size=(6, 4))

    grad_sum = clm.backward(z, params, 2.0 * g1 - g2)
    grad_1 = clm.backward(z, params, g1)
    grad_2 = clm.backward(z, params, g2)

    for combined, part_1, part_2 in zip(grad_sum, grad_1, grad_2):
        np.testing.assert_allclose(combined, 2.0 * part_1 - part_2, atol=1e-14)


def test_clm_backward_projection_and_first_threshold_are_opposite(rng):
    """Shifting ``z`` and ``b`` together leaves the output unchanged.

    Since the output depends on ``theta - z`` and ``b`` shifts every
    threshold, ``sum(grad_z) = -grad_b``.
    """
    clm = CLM(4, link_function="cloglog")
    grad_z, grad_params = clm.backward(
        rng.normal(size=8), rng.normal(size=3), rng.normal(size=(8, 4))
    )

    assert np.sum(grad_z) == pytest.approx(-grad_params[0])


@pytest.mark.parametrize("grad_proba_shape", [(6, 3), (6, 5), (5, 4), (6,), (6, 4, 1)])
def test_clm_backward_rejects_wrong_upstream_shape(grad_proba_shape):
    """The upstream gradient must match the output of ``forward``."""
    clm = CLM(4)

    with pytest.raises(ValueError, match="grad_proba must have shape"):
        clm.backward(np.zeros(6), np.zeros(3), np.zeros(grad_proba_shape))


def test_clm_backward_rejects_wrong_params_shape():
    """``backward`` validates ``params`` like the other methods."""
    with pytest.raises(ValueError, match="params must have shape"):
        CLM(4).backward(np.zeros(6), np.zeros(2), np.zeros((6, 4)))


def test_clm_gradient_descent_fits_cumulative_model(rng):
    """Plain gradient descent through the layer recovers the true thresholds."""
    true_clm = CLM(3)
    true_params = np.array([-1.0, np.sqrt(2.0)])
    z = rng.normal(scale=2.0, size=4000)
    cumproba = true_clm.cumproba(z, true_params)
    u = rng.uniform(size=(z.size, 1))
    y = np.sum(u > cumproba, axis=1)
    Y = np.eye(3)[y]

    params = true_clm.init_params()
    for _ in range(3000):
        proba = true_clm.forward(z, params)
        _, grad_params = true_clm.backward(z, params, -Y / proba / z.size)
        params -= 0.5 * grad_params

    np.testing.assert_allclose(
        true_clm.thresholds(params), true_clm.thresholds(true_params), atol=0.15
    )
