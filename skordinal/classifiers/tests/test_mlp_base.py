"""Tests for the shared MLP classifier base class."""

import numpy as np
import pytest
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from skordinal.classifiers._mlp_base import MLPBaseClassifier


class DummyMLP(MLPBaseClassifier):
    """Small implementation used to test the base class."""

    last_sample_weight = None

    def _initialize_parameters(self, rng):
        return rng.normal(size=self.n_features_in_ * self.n_classes_)

    def _unpack_parameters(self, params):
        self.weights_ = params.reshape(self.n_features_in_, self.n_classes_)

    def _cost_and_grad(self, params, X, Y, sample_weight):
        type(self).last_sample_weight = sample_weight.copy()
        return 0.5 * np.dot(params, params), params.copy()

    def _predict_proba(self, X):
        return softmax(X @ self.weights_, axis=1)


@pytest.fixture
def data():
    """Return a small ordinal classification dataset."""
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = np.array([10, 20, 30, 20])
    return X, y


def test_fit_returns_self_and_sets_fitted_attributes(data):
    """Fitting returns the estimator and exposes the shared fitted state."""
    X, y = data

    estimator = DummyMLP(max_iter=10, random_state=0)
    clf = estimator.fit(X, y)

    assert clf is estimator
    assert clf.classes_.tolist() == [10, 20, 30]
    assert clf.n_classes_ == 3
    assert clf.n_features_in_ == X.shape[1]
    assert np.isfinite(clf.loss_)
    assert clf.n_iter_ >= 0
    assert clf.weights_.shape == (X.shape[1], len(clf.classes_))


def test_fit_preserves_original_labels_in_predict(data):
    """Predictions use the original ordinal labels rather than encoded indices."""
    X, y = data

    clf = DummyMLP(max_iter=1, random_state=0).fit(X, y)

    assert set(clf.predict(X)).issubset(set(y))


def test_fit_warns_when_optimizer_does_not_converge(data, monkeypatch):
    """Fitting warns when L-BFGS-B does not converge."""
    X, y = data

    def fake_minimize(fun, x0, args, method, jac, options):
        cost, _ = fun(x0, *args)

        return type(
            "Result",
            (),
            {
                "fun": cost,
                "nit": 17,
                "x": np.ones_like(x0),
                "success": False,
                "message": "Maximum iterations reached",
            },
        )()

    monkeypatch.setattr(
        "skordinal.classifiers._mlp_base.scipy.optimize.minimize",
        fake_minimize,
    )

    with pytest.warns(ConvergenceWarning, match="failed to converge"):
        DummyMLP(max_iter=17).fit(X, y)


def test_predict_proba_has_valid_shape_and_values(data):
    """Predicted probabilities have one valid distribution per sample."""
    X, y = data

    proba = DummyMLP(max_iter=1, random_state=0).fit(X, y).predict_proba(X)

    assert proba.shape == (len(X), len(np.unique(y)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_predict_before_fit_raises(data):
    """Prediction methods reject an unfitted estimator."""
    X, _ = data
    clf = DummyMLP()

    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X)


def test_predict_rejects_wrong_number_of_features(data):
    """Prediction validates the feature count seen during fitting."""
    X, y = data
    clf = DummyMLP(max_iter=1).fit(X, y)

    with pytest.raises(ValueError):
        clf.predict(X[:, :1])


def test_random_state_reproduces_initialization(data):
    """The same random state produces the same fitted parameters."""
    X, y = data

    first = DummyMLP(max_iter=1, random_state=42).fit(X, y)
    second = DummyMLP(max_iter=1, random_state=42).fit(X, y)

    np.testing.assert_array_equal(first.weights_, second.weights_)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("n_hidden_layers", 0),
        ("n_hidden_units", 0),
        ("alpha", -1.0),
        ("max_iter", 0),
    ],
)
def test_invalid_hyperparameters_are_rejected(data, parameter, value):
    """Parameter constraints are enforced when fitting."""
    X, y = data

    with pytest.raises(ValueError, match=rf"The '{parameter}' parameter"):
        DummyMLP(**{parameter: value}).fit(X, y)


def test_class_weight_is_converted_to_sample_weight(data):
    """Class weights reach the objective as per-sample weights."""
    X, y = data

    DummyMLP(class_weight={10: 2.0, 20: 1.0, 30: 3.0}, max_iter=1).fit(X, y)

    np.testing.assert_array_equal(
        DummyMLP.last_sample_weight, np.array([2.0, 1.0, 3.0, 1.0])
    )


def test_optimizer_uses_lbfgs_and_unpacks_result(data, monkeypatch):
    """The base class configures L-BFGS-B and stores its optimized parameters."""
    X, y = data
    calls = {}

    def fake_minimize(fun, x0, args, method, jac, options):
        calls.update(method=method, jac=jac, options=options)
        cost, gradient = fun(x0, *args)
        assert gradient.shape == x0.shape

        return type(
            "Result",
            (),
            {
                "fun": cost,
                "nit": 4,
                "x": np.ones_like(x0),
                "success": True,
                "message": "CONVERGENCE: dummy result",
            },
        )()

    monkeypatch.setattr(
        "skordinal.classifiers._mlp_base.scipy.optimize.minimize", fake_minimize
    )

    clf = DummyMLP(max_iter=17).fit(X, y)

    assert calls == {
        "method": "L-BFGS-B",
        "jac": True,
        "options": {
            "maxiter": 17,
            "ftol": 1e-5,
            "gtol": 1e-5,
        },
    }
    assert clf.n_iter_ == 4
    np.testing.assert_array_equal(clf.weights_, np.ones_like(clf.weights_))
