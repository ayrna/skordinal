"""Tests of the margin functions and ``get_margin``."""

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skordinal._loss import ExponentialMargin, LogisticMargin, Margin, get_margin

FORMULAS = {
    "logistic": {
        "phi": lambda z: math.log1p(math.exp(-z)),
        "dphi": lambda z: -1.0 / (1.0 + math.exp(z)),
        "d2phi": lambda z: 1.0 / ((1.0 + math.exp(-z)) * (1.0 + math.exp(z))),
        "cumproba": lambda eta: 1.0 / (1.0 + math.exp(-eta)),
        "log_odds_scale": 1.0,
    },
    "exponential": {
        "phi": lambda z: math.exp(-z),
        "dphi": lambda z: -math.exp(-z),
        "d2phi": lambda z: math.exp(-z),
        "cumproba": lambda eta: 1.0 / (1.0 + math.exp(-2.0 * eta)),
        "log_odds_scale": 0.5,
    },
}


@pytest.mark.parametrize("name", list(FORMULAS))
def test_margin(name):
    """Margins and their gradient_hessian follow the formulas and finite differences."""
    margin, formulas = get_margin(name), FORMULAS[name]
    z = np.linspace(-6.0, 6.0, 25)
    # Alternate sides so that margins of both signs lie above their threshold
    above = np.arange(z.size) % 2 == 1
    eps = 1e-5

    dphi, d2phi = margin.derivatives(z)
    grad, hess = margin.gradient_hessian(z, above)

    assert margin.log_odds_scale == formulas["log_odds_scale"]
    for method, value in [
        ("phi", margin.phi(z)),
        ("dphi", dphi),
        ("d2phi", d2phi),
        ("cumproba", margin.cumproba(z)),
    ]:
        assert_allclose(value, np.vectorize(formulas[method])(z), rtol=1e-12)
    fd_dphi = (margin.phi(z + eps) - margin.phi(z - eps)) / (2 * eps)
    fd_d2phi = (margin.derivatives(z + eps)[0] - margin.derivatives(z - eps)[0]) / (
        2 * eps
    )
    assert_allclose(dphi, fd_dphi, rtol=1e-6)
    assert_allclose(d2phi, fd_d2phi, rtol=1e-6)
    # gradient_hessian differentiates phi(s * eta) with s = -1 above the threshold
    s = np.where(above, -1.0, 1.0)
    assert_allclose(grad, s * np.vectorize(formulas["dphi"])(s * z), rtol=1e-12)
    assert_allclose(hess, np.vectorize(formulas["d2phi"])(s * z), rtol=1e-12)


def test_logistic_extremes():
    """The logistic margin keeps its relative precision and is silent far from zero."""
    z = np.array([-800.0, -40.0, 40.0, 800.0])
    tail = np.exp(-40.0)
    margin = LogisticMargin()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = [margin.phi(z), *margin.derivatives(z), margin.cumproba(z)]
        values += list(margin.gradient_hessian(z, z < 0.0))

    expected = [
        [800.0, 40.0, tail, 0.0],
        [-1.0, -1.0, -tail, 0.0],
        [0.0, tail, tail, 0.0],
        [0.0, tail, 1.0, 1.0],
        [0.0, tail, -tail, 0.0],
        [0.0, tail, tail, 0.0],
    ]
    for value, reference in zip(values, expected):
        assert_allclose(value, reference, rtol=1e-12, atol=0.0)


def test_get_margin():
    """get_margin shares instances by name, passes instances through, lists names."""
    margin = LogisticMargin()

    assert get_margin("logistic") is get_margin("logistic")
    assert isinstance(get_margin("exponential"), ExponentialMargin)
    assert get_margin(margin) is margin
    with pytest.raises(ValueError, match="'logistic', 'exponential'"):
        get_margin("hinge")


def test_margin_contract():
    """A margin cannot be instantiated without name and log_odds_scale."""
    with pytest.raises(TypeError, match=r"log_odds_scale'?, '?name"):
        type("Hinge", (Margin,), {})()
