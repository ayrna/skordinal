"""Tests for the MLP classifier trained with triangular soft labels."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.special import log_softmax
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.estimator_checks import parametrize_with_checks

from skordinal.datasets import make_ordinal_classification
from skordinal.neural_network import MLPTriangularClassifier
from skordinal.neural_network._base import MLPBaseClassifier
from skordinal.neural_network._triangular_distribution import get_triangular_soft_labels
from skordinal.neural_network._triangular_loss import TriangularLoss

ACTIVATIONS = ["sigmoid", "tanh", "relu"]


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.2],
            [2.0, 2.0],
            [2.1, 1.9],
            [1.9, 2.1],
            [4.0, 4.0],
            [4.1, 3.9],
            [3.9, 4.1],
        ]
    )


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


@pytest.fixture
def ordinal_data():
    """Create a synthetic ordinal dataset for behavioural tests."""
    return make_ordinal_classification(
        n_samples=120,
        n_features=4,
        n_classes=4,
        n_informative=4,
        noise=0.1,
        random_state=0,
    )


@pytest.fixture(autouse=True)
def _ignore_convergence_warnings():
    """Most tests stop L-BFGS-B early on purpose."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        yield


def _prepare(clf, X, Y):
    """Set the state ``fit`` builds before calling ``_cost_and_grad``."""
    clf.n_features_in_ = X.shape[1]
    clf.n_classes_ = Y.shape[1]
    return clf._initialize_parameters(np.random.RandomState(0))


def _n_params(n_features, n_classes, n_hidden_layers, n_hidden_units):
    n_params = n_features * n_hidden_units + n_hidden_units
    n_params += (n_hidden_layers - 1) * (n_hidden_units**2 + n_hidden_units)
    return n_params + n_hidden_units * n_classes + n_classes


# -----------------
# scikit-learn API
# -----------------


@parametrize_with_checks(
    [MLPTriangularClassifier(max_iter=50, random_state=0)],
    expected_failed_checks=lambda est: {
        "check_classifiers_classes": "Ordinal targets must be numeric."
    },
)
def test_mlp_triangular_sklearn_compatible(estimator, check):
    """The estimator passes scikit-learn's estimator checks."""
    check(estimator)


def test_mlp_triangular_is_mlp_base_classifier():
    """The classifier is built on the shared MLP base class."""
    assert issubclass(MLPTriangularClassifier, MLPBaseClassifier)


def test_mlp_triangular_default_parameters():
    """Constructor defaults are stored unchanged."""
    assert MLPTriangularClassifier().get_params() == {
        "n_hidden_layers": 1,
        "n_hidden_units": 4,
        "alpha": 0.01,
        "activation": "sigmoid",
        "alpha2": 0.05,
        "eta": 1.0,
        "class_weight": None,
        "max_iter": 500,
        "random_state": None,
        "tol": 1e-5,
    }


def test_mlp_triangular_parameter_constraints_match_init_params():
    """Constraint keys match the public constructor parameters."""
    init_params = set(
        inspect.signature(MLPTriangularClassifier.__init__).parameters
    ) - {"self"}

    assert set(MLPTriangularClassifier._parameter_constraints) == init_params


def test_mlp_triangular_clone_and_set_params():
    """New parameters round-trip through clone and set_params."""
    cloned = clone(MLPTriangularClassifier(alpha2=0.1, eta=0.5))

    assert cloned.alpha2 == 0.1
    assert cloned.eta == 0.5

    cloned.set_params(alpha2=0.2, eta=0.0)
    assert cloned.get_params()["alpha2"] == 0.2
    assert cloned.get_params()["eta"] == 0.0


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_hidden_layers", 0),
        ("n_hidden_units", 0),
        ("alpha", -1),
        ("max_iter", 0),
        ("activation", "linear"),
        ("alpha2", -0.01),
        ("alpha2", 1.5),
        ("eta", -0.1),
        ("eta", 1.1),
        ("tol", 0.0),
        ("tol", -1e-3),
        ("class_weight", "uniform"),
    ],
)
def test_mlp_triangular_hyperparameter_value_validation(
    X, y, param_name, invalid_value
):
    """Test that invalid hyperparameter values are rejected."""
    classifier = MLPTriangularClassifier(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_hidden_layers", 1.5),
        ("n_hidden_units", 2.5),
        ("alpha", "strong"),
        ("max_iter", 2.5),
        ("activation", 1),
        ("alpha2", "wide"),
        ("eta", None),
        ("tol", "small"),
        ("class_weight", [1, 2, 3]),
        ("random_state", "seed"),
    ],
)
def test_mlp_triangular_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that invalid hyperparameter types are rejected."""
    classifier = MLPTriangularClassifier(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_mlp_triangular_alpha2_upper_bound_depends_on_n_classes(X, y):
    """alpha2 above 2/9 is rejected for three classes but not for two."""
    with pytest.raises(ValueError, match="alpha2 must be a float in"):
        MLPTriangularClassifier(alpha2=0.3, max_iter=5).fit(X, y)

    binary = y[:6] > 0
    clf = MLPTriangularClassifier(alpha2=0.9, max_iter=5).fit(X[:6], binary.astype(int))
    assert clf.loss_fn_.alpha2 == 0.9


def test_mlp_triangular_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = MLPTriangularClassifier(max_iter=10, random_state=0)

    assert classifier.fit(X, y) is classifier


def test_mlp_triangular_fit_input_validation(X, y):
    """Test that input data is validated."""
    classifier = MLPTriangularClassifier(max_iter=1)

    with pytest.raises(ValueError):
        classifier.fit(X, y[:-1])
    with pytest.raises(ValueError):
        classifier.fit([], y)
    with pytest.raises(ValueError):
        classifier.fit(X, [])
    with pytest.raises(ValueError):
        classifier.fit(np.where(X > 3, np.nan, X), y)


def test_mlp_triangular_predict_raises_if_not_fitted(X):
    """Prediction methods raise NotFittedError before fitting."""
    classifier = MLPTriangularClassifier()

    with pytest.raises(NotFittedError):
        classifier.predict(X)
    with pytest.raises(NotFittedError):
        classifier.predict_proba(X)


def test_mlp_triangular_predict_rejects_wrong_n_features(X, y):
    """Prediction rejects data with a different number of features."""
    classifier = MLPTriangularClassifier(max_iter=5).fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])
    with pytest.raises(ValueError):
        classifier.predict_proba(np.hstack([X, X]))


def test_mlp_triangular_feature_names_in_when_dataframe(X, y):
    """feature_names_in_ is set when fitting a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = MLPTriangularClassifier(max_iter=5, random_state=0).fit(df, y)

    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_mlp_triangular_refit_resets_state(X, y):
    """Refitting on a problem of a different size rebuilds the layers."""
    clf = MLPTriangularClassifier(n_hidden_units=3, max_iter=5, random_state=0)
    clf.fit(X, y)
    X_new = np.hstack([X, X])
    y_new = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])

    clf.fit(X_new, y_new)

    assert clf.n_features_in_ == 4
    assert clf.loss_fn_.n_classes == 4
    assert clf.W_[0].shape == (4, 3)
    assert clf.W_out_.shape == (3, 4)
    assert clf.predict_proba(X_new).shape == (9, 4)


# -----------------
# Fitted state
# -----------------


def test_mlp_triangular_sets_fitted_attributes_after_fit(X, y):
    """Test that fitted attributes have the expected shapes and values."""
    clf = MLPTriangularClassifier(
        n_hidden_layers=2, n_hidden_units=4, max_iter=5, random_state=0
    ).fit(X, y)

    for attr in [
        "classes_",
        "n_classes_",
        "n_features_in_",
        "loss_fn_",
        "W_",
        "b_",
        "W_out_",
        "b_out_",
        "loss_",
        "n_iter_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert np.array_equal(clf.classes_, np.unique(y))
    assert clf.n_features_in_ == X.shape[1]
    assert clf.n_classes_ == 3
    assert [W.shape for W in clf.W_] == [(2, 4), (4, 4)]
    assert [b.shape for b in clf.b_] == [(4,), (4,)]
    assert clf.W_out_.shape == (4, 3)
    assert clf.b_out_.shape == (3,)
    assert np.isfinite(clf.loss_)
    assert clf.n_iter_ >= 0


@pytest.mark.parametrize("alpha2", [0.0, 0.05, 0.2])
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_mlp_triangular_loss_matches_parameters(X, y, alpha2, eta):
    """The fitted loss is configured from the estimator parameters."""
    clf = MLPTriangularClassifier(
        alpha2=alpha2, eta=eta, max_iter=5, random_state=0
    ).fit(X, y)

    assert isinstance(clf.loss_fn_, TriangularLoss)
    assert clf.loss_fn_.n_classes == clf.n_classes_
    assert clf.loss_fn_.alpha2 == alpha2
    assert clf.loss_fn_.eta == eta
    np.testing.assert_array_equal(
        clf.loss_fn_.cls_probs, get_triangular_soft_labels(3, alpha2)
    )


@pytest.mark.parametrize("n_hidden_layers", [1, 2, 3])
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_mlp_triangular_initial_parameter_layout(n_hidden_layers, n_classes):
    """Initial parameters pack every layer in order with the expected sizes."""
    n_features, n_hidden_units = 3, 5
    clf = MLPTriangularClassifier(
        n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units
    )
    clf.n_features_in_ = n_features
    clf.n_classes_ = n_classes
    params = clf._initialize_parameters(np.random.RandomState(0))

    assert params.shape == (
        _n_params(n_features, n_classes, n_hidden_layers, n_hidden_units),
    )

    W, b, W_out, b_out = clf._split_parameters(params)
    assert [w.shape for w in W] == [(n_features, n_hidden_units)] + [
        (n_hidden_units, n_hidden_units)
    ] * (n_hidden_layers - 1)
    assert all(np.array_equal(b_i, np.zeros(n_hidden_units)) for b_i in b)
    assert W_out.shape == (n_hidden_units, n_classes)
    np.testing.assert_array_equal(b_out, np.zeros(n_classes))


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_mlp_triangular_initial_weights_within_glorot_or_he_limits(activation):
    """Initial weights are drawn within the Glorot (or He for ReLU) limits."""
    n_features, n_hidden_units, n_classes = 6, 10, 3
    clf = MLPTriangularClassifier(n_hidden_units=n_hidden_units, activation=activation)
    clf.n_features_in_ = n_features
    clf.n_classes_ = n_classes
    W, _, W_out, _ = clf._split_parameters(
        clf._initialize_parameters(np.random.RandomState(0))
    )

    if activation == "relu":
        limit = np.sqrt(6 / n_features)
    else:
        limit = np.sqrt(6 / (n_features + n_hidden_units))
    assert np.all(np.abs(W[0]) <= limit)
    assert np.all(np.abs(W_out) <= np.sqrt(6 / (n_hidden_units + n_classes)))


def test_mlp_triangular_split_parameters_rejects_wrong_size():
    """A parameter vector of the wrong length is rejected."""
    clf = MLPTriangularClassifier(n_hidden_units=3)
    clf.n_features_in_ = 2
    clf.n_classes_ = 3
    params = clf._initialize_parameters(np.random.RandomState(0))

    with pytest.raises(ValueError, match="Expected"):
        clf._split_parameters(params[:-1])
    with pytest.raises(ValueError, match="Expected"):
        clf._split_parameters(np.concatenate([params, [0.0]]))


def test_mlp_triangular_objective_has_no_side_effects():
    """Evaluating the objective does not touch the fitted attributes."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 2))
    Y = np.eye(3)[rng.integers(0, 3, size=10)]
    clf = MLPTriangularClassifier(n_hidden_units=3)
    params = _prepare(clf, X, Y)

    clf._cost_and_grad(params, X, Y, np.ones(10))

    assert not hasattr(clf, "W_")
    assert not hasattr(clf, "W_out_")


# -----------------
# Objective and gradient
# -----------------


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("n_hidden_layers", [1, 2])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_mlp_triangular_objective_gradient_matches_finite_difference(
    activation, n_hidden_layers, eta
):
    """The analytic objective gradient matches a finite-difference approximation."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    Y = np.eye(4)[rng.integers(0, 4, size=X.shape[0])]
    sample_weight = rng.uniform(0.5, 3.0, size=X.shape[0])

    clf = MLPTriangularClassifier(
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=3,
        alpha=0.1,
        activation=activation,
        alpha2=0.1,
        eta=eta,
    )
    params = _prepare(clf, X, Y)
    # Move away from the zero biases so ReLU units are not at their kink
    params += rng.normal(scale=0.3, size=params.shape)

    error = check_grad(
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[0],
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[1],
        params,
    )

    assert error < 1e-5


def test_mlp_triangular_binary_objective_gradient_matches_finite_difference():
    """The gradient is also correct for a binary problem."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((15, 2))
    Y = np.eye(2)[rng.integers(0, 2, size=X.shape[0])]
    clf = MLPTriangularClassifier(n_hidden_units=3, alpha2=0.5)
    params = _prepare(clf, X, Y)

    error = check_grad(
        lambda p: clf._cost_and_grad(p, X, Y, np.ones(15))[0],
        lambda p: clf._cost_and_grad(p, X, Y, np.ones(15))[1],
        params,
    )

    assert error < 1e-5


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_mlp_triangular_objective_is_soft_cross_entropy_plus_l2(eta):
    """The objective equals the soft-target cross-entropy plus the penalty."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((25, 3))
    y_encoded = rng.integers(0, 4, size=X.shape[0])
    Y = np.eye(4)[y_encoded]
    sample_weight = rng.uniform(0.5, 2.0, size=X.shape[0])
    alpha = 0.3

    clf = MLPTriangularClassifier(
        n_hidden_layers=2, n_hidden_units=4, alpha=alpha, alpha2=0.1, eta=eta
    )
    params = _prepare(clf, X, Y) + rng.normal(scale=0.5, size=_n_params(3, 4, 2, 4))
    J, _ = clf._cost_and_grad(params, X, Y, sample_weight)

    clf._unpack_parameters(params)
    proba = clf._predict_proba(X)
    T = (1 - eta) * Y + eta * get_triangular_soft_labels(4, 0.1)[y_encoded]
    nll = -np.sum(sample_weight[:, None] * T * np.log(proba))
    l2 = sum(np.sum(W**2) for W in clf.W_) + np.sum(clf.W_out_**2)
    expected = (nll + 0.5 * alpha * l2) / np.sum(sample_weight)

    assert J == pytest.approx(expected, rel=1e-12)


def test_mlp_triangular_eta_zero_is_one_hot_cross_entropy():
    """With eta=0 the data term is the standard cross-entropy."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((15, 2))
    y_encoded = rng.integers(0, 3, size=15)
    Y = np.eye(3)[y_encoded]
    clf = MLPTriangularClassifier(n_hidden_units=3, alpha=0.0, alpha2=0.2, eta=0.0)
    params = _prepare(clf, X, Y)

    J, _ = clf._cost_and_grad(params, X, Y, np.ones(15))
    _, logits = clf._forward(X, *clf._split_parameters(params))
    expected = -np.mean(log_softmax(logits, axis=1)[np.arange(15), y_encoded])

    assert J == pytest.approx(expected, rel=1e-12)


def test_mlp_triangular_alpha2_has_no_effect_when_eta_is_zero(X, y):
    """With one-hot targets the soft label width is irrelevant."""
    common = dict(n_hidden_units=3, eta=0.0, max_iter=20, random_state=0)
    clf_a = MLPTriangularClassifier(alpha2=0.01, **common).fit(X, y)
    clf_b = MLPTriangularClassifier(alpha2=0.2, **common).fit(X, y)

    np.testing.assert_allclose(clf_a.W_out_, clf_b.W_out_)
    np.testing.assert_allclose(clf_a.predict_proba(X), clf_b.predict_proba(X))


def test_mlp_triangular_penalty_excludes_biases():
    """Changing biases never changes the L2 penalty."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((10, 2))
    Y = np.eye(3)[rng.integers(0, 3, size=10)]
    sw = np.ones(10)
    clf_reg = MLPTriangularClassifier(n_hidden_units=3, alpha=5.0)
    clf_free = MLPTriangularClassifier(n_hidden_units=3, alpha=0.0)
    params = _prepare(clf_reg, X, Y)
    _prepare(clf_free, X, Y)

    def penalty(p):
        return (
            clf_reg._cost_and_grad(p, X, Y, sw)[0]
            - clf_free._cost_and_grad(p, X, Y, sw)[0]
        )

    shifted = params.copy()
    shifted[6:9] += 1.0  # hidden biases
    shifted[-3:] += 1.0  # output biases

    assert penalty(shifted) == pytest.approx(penalty(params), rel=1e-12)


def test_mlp_triangular_objective_finite_for_confident_mistakes():
    """A confidently wrong prediction yields a finite loss and gradient."""
    X = np.array([[1e4, 1e4]])
    Y = np.array([[1.0, 0.0, 0.0]])
    clf = MLPTriangularClassifier(n_hidden_units=2, activation="relu", eta=0.0)
    params = _prepare(clf, X, Y)
    params[:4] = 1.0
    params[6:12] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        J, grad = clf._cost_and_grad(params, X, Y, np.ones(1))

    assert np.isfinite(J)
    assert np.isfinite(grad).all()


# -----------------
# Prediction
# -----------------


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_mlp_triangular_predict_proba_shape_bounds_and_normalization(
    X, y, activation, eta
):
    """Predicted probabilities form valid distributions for every configuration."""
    clf = MLPTriangularClassifier(
        n_hidden_units=4, activation=activation, eta=eta, max_iter=10, random_state=0
    ).fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], 3)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_mlp_triangular_predict_proba_matches_manual_forward_pass(X, y):
    """predict_proba is the softmax of the network's logits."""
    clf = MLPTriangularClassifier(
        n_hidden_layers=2, activation="tanh", max_iter=20
    ).fit(X, y)

    A = X
    for W, b in zip(clf.W_, clf.b_):
        A = np.tanh(A @ W + b)
    logits = A @ clf.W_out_ + clf.b_out_
    expected = np.exp(log_softmax(logits, axis=1))

    np.testing.assert_allclose(clf.predict_proba(X), expected, atol=1e-14)


def test_mlp_triangular_predict_returns_known_labels(X, y):
    """Predictions use the original labels and agree with argmax probabilities."""
    clf = MLPTriangularClassifier(max_iter=10, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)

    assert set(clf.predict(X)).issubset(set(clf.classes_))
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[proba.argmax(axis=1)])


@pytest.mark.parametrize("labels", [[1, 2, 3], [-1, 0, 1], [3, 5, 7], [10, 20, 30, 40]])
def test_mlp_triangular_label_roundtrip(labels):
    """Arbitrary ordinal labels are preserved through fit and predict."""
    labels_array = np.array(labels)
    y = np.repeat(labels_array, 3)
    X = np.arange(len(y), dtype=float).reshape(-1, 1).repeat(2, axis=1)

    classifier = MLPTriangularClassifier(
        n_hidden_units=4, max_iter=10, random_state=0
    ).fit(X, y)

    np.testing.assert_array_equal(classifier.classes_, labels_array)
    assert set(classifier.predict(X)).issubset(set(labels_array))


def test_mlp_triangular_binary_problem():
    """A binary problem fits and predicts two probabilities."""
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    clf = MLPTriangularClassifier(n_hidden_units=3, max_iter=100, random_state=0)
    clf.fit(X, y)

    assert clf.predict_proba(X).shape == (len(X), 2)
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0)
    np.testing.assert_array_equal(clf.predict(X), y)


def test_mlp_triangular_large_magnitude_inputs_remain_finite():
    """Large feature magnitudes do not produce NaNs or invalid probabilities."""
    X, y = make_ordinal_classification(
        n_samples=60, n_features=4, n_classes=3, n_informative=4, random_state=0
    )
    X *= 1000.0

    clf = MLPTriangularClassifier(n_hidden_units=4, max_iter=20, random_state=0)
    proba = clf.fit(X, y).predict_proba(X)

    assert np.isfinite(clf.loss_)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_mlp_triangular_soft_labels_spread_probability(ordinal_data):
    """Training on soft labels leaves more mass on neighbours than one-hot."""
    X, y = ordinal_data
    common = dict(n_hidden_units=8, max_iter=300, random_state=0, alpha=1e-4)
    hard = MLPTriangularClassifier(eta=0.0, **common).fit(X, y)
    soft = MLPTriangularClassifier(alpha2=0.2, eta=1.0, **common).fit(X, y)

    assert soft.predict_proba(X).max(axis=1).mean() < (
        hard.predict_proba(X).max(axis=1).mean()
    )


# -----------------
# Randomness and weights
# -----------------


def test_mlp_triangular_random_state_reproducibility(X, y):
    """Two fits with the same random state produce identical parameters."""
    clf_a = MLPTriangularClassifier(n_hidden_units=4, max_iter=10, random_state=0)
    clf_b = MLPTriangularClassifier(n_hidden_units=4, max_iter=10, random_state=0)
    clf_a.fit(X, y)
    clf_b.fit(X, y)

    for weights_a, weights_b in zip(clf_a.W_, clf_b.W_):
        np.testing.assert_array_equal(weights_a, weights_b)
    for biases_a, biases_b in zip(clf_a.b_, clf_b.b_):
        np.testing.assert_array_equal(biases_a, biases_b)
    np.testing.assert_array_equal(clf_a.W_out_, clf_b.W_out_)
    np.testing.assert_array_equal(clf_a.b_out_, clf_b.b_out_)


def test_mlp_triangular_random_state_different_seeds_differ(X, y):
    """Different seeds produce different initial network parameters."""
    clf_a = MLPTriangularClassifier(n_hidden_units=4, max_iter=1, random_state=0)
    clf_b = MLPTriangularClassifier(n_hidden_units=4, max_iter=1, random_state=1)

    assert not np.array_equal(clf_a.fit(X, y).W_[0], clf_b.fit(X, y).W_[0])


def test_mlp_triangular_random_state_accepts_random_state_instance(X, y):
    """A RandomState instance matches the equivalent integer seed."""
    clf_seed = MLPTriangularClassifier(n_hidden_units=4, max_iter=5, random_state=42)
    clf_instance = MLPTriangularClassifier(
        n_hidden_units=4, max_iter=5, random_state=np.random.RandomState(42)
    )
    clf_seed.fit(X, y)
    clf_instance.fit(X, y)

    for weights_seed, weights_instance in zip(clf_seed.W_, clf_instance.W_):
        np.testing.assert_array_equal(weights_seed, weights_instance)
    np.testing.assert_array_equal(clf_seed.W_out_, clf_instance.W_out_)


def test_mlp_triangular_class_weight_balanced_matches_equivalent_dict():
    """The balanced class weight and its explicit dictionary are equivalent."""
    X, y = make_ordinal_classification(
        n_samples=60,
        n_features=3,
        n_classes=3,
        n_informative=3,
        weights=[0.1, 0.3, 0.6],
        random_state=1,
    )
    common = dict(n_hidden_units=4, max_iter=10, random_state=0, alpha=1.0)
    clf_none = MLPTriangularClassifier(class_weight=None, **common).fit(X, y)
    clf_balanced = MLPTriangularClassifier(class_weight="balanced", **common)
    clf_balanced.fit(X, y)
    weights = compute_class_weight("balanced", classes=clf_balanced.classes_, y=y)
    class_weight = dict(zip(clf_balanced.classes_, weights))
    clf_dict = MLPTriangularClassifier(class_weight=class_weight, **common).fit(X, y)

    assert not np.allclose(clf_none.W_out_, clf_balanced.W_out_)
    for weights_balanced, weights_dict in zip(clf_balanced.W_, clf_dict.W_):
        np.testing.assert_allclose(weights_balanced, weights_dict, atol=1e-8)
    np.testing.assert_allclose(clf_balanced.W_out_, clf_dict.W_out_, atol=1e-8)
    np.testing.assert_allclose(clf_balanced.b_out_, clf_dict.b_out_, atol=1e-8)


def test_mlp_triangular_uniform_class_weight_matches_none(X, y):
    """Unit class weights are the same as no class weights."""
    common = dict(n_hidden_units=3, max_iter=20, random_state=0)
    clf_none = MLPTriangularClassifier(**common).fit(X, y)
    clf_ones = MLPTriangularClassifier(class_weight={0: 1, 1: 1, 2: 1}, **common)
    clf_ones.fit(X, y)

    np.testing.assert_allclose(clf_none.W_out_, clf_ones.W_out_)
    np.testing.assert_allclose(clf_none.b_out_, clf_ones.b_out_)


def test_mlp_triangular_integer_class_weight_matches_duplicated_samples(X, y):
    """Weighting a class by k equals repeating its samples k times."""
    common = dict(n_hidden_units=3, alpha=0.5, max_iter=30, random_state=0)
    clf_weighted = MLPTriangularClassifier(class_weight={0: 1, 1: 3, 2: 1}, **common)
    clf_weighted.fit(X, y)
    repeats = np.where(y == 1, 3, 1)
    clf_repeated = MLPTriangularClassifier(**common).fit(
        np.repeat(X, repeats, axis=0), np.repeat(y, repeats)
    )

    assert clf_weighted.loss_ == pytest.approx(clf_repeated.loss_, rel=1e-6)
    np.testing.assert_allclose(clf_weighted.W_out_, clf_repeated.W_out_, atol=1e-5)


# -----------------
# Optimisation and behaviour
# -----------------


def test_mlp_triangular_warns_when_not_converged(X, y):
    """Stopping after one iteration raises a ConvergenceWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("default", ConvergenceWarning)
        with pytest.warns(ConvergenceWarning, match=r"max_iter=1\b"):
            MLPTriangularClassifier(max_iter=1, random_state=0).fit(X, y)


def test_mlp_triangular_tol_is_forwarded_to_the_solver(ordinal_data):
    """A looser tolerance stops L-BFGS-B earlier than a tighter one."""
    X, y = ordinal_data
    n_iters = [
        MLPTriangularClassifier(tol=tol, max_iter=1000, random_state=0)
        .fit(X, y)
        .n_iter_
        for tol in (1e-1, 1e-8)
    ]

    assert n_iters[0] < n_iters[1]


def test_mlp_triangular_training_decreases_loss(ordinal_data):
    """More iterations reach a lower objective."""
    X, y = ordinal_data
    losses = [
        MLPTriangularClassifier(max_iter=n_iter, random_state=0).fit(X, y).loss_
        for n_iter in (1, 10, 100)
    ]

    assert losses[0] > losses[1] > losses[2]


def test_mlp_triangular_stronger_regularization_shrinks_weights(ordinal_data):
    """A larger alpha yields smaller output weights."""
    X, y = ordinal_data
    norms = [
        np.linalg.norm(
            MLPTriangularClassifier(alpha=alpha, max_iter=200, random_state=0)
            .fit(X, y)
            .W_out_
        )
        for alpha in (1e-4, 10.0)
    ]

    assert norms[0] > norms[1]


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_mlp_triangular_behaviour_on_ordinal_data(ordinal_data, activation, eta):
    """The model learns useful predictions on a separable ordinal dataset."""
    X, y = ordinal_data
    clf = MLPTriangularClassifier(
        n_hidden_units=8, activation=activation, eta=eta, max_iter=200, random_state=0
    ).fit(X, y)
    y_pred = clf.predict(X)

    assert (y_pred == y).mean() >= 0.85
    assert np.abs(y_pred - y).mean() <= 0.2


def test_mlp_triangular_in_pipeline_and_grid_search(ordinal_data):
    """The classifier works inside pipelines, cross-validation and grid search."""
    X, y = ordinal_data
    pipe = make_pipeline(
        StandardScaler(), MLPTriangularClassifier(max_iter=100, random_state=0)
    )

    scores = cross_val_score(pipe, X, y, cv=3)
    assert scores.shape == (3,)
    assert np.all(scores > 0.5)

    search = GridSearchCV(
        pipe,
        {
            "mlptriangularclassifier__alpha2": [0.01, 0.1],
            "mlptriangularclassifier__eta": [0.5, 1.0],
        },
        cv=3,
    ).fit(X, y)
    assert search.best_params_["mlptriangularclassifier__alpha2"] in (0.01, 0.1)
    assert search.predict(X).shape == y.shape
