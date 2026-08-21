"""Tests for the NNPOM classifier."""

import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from skordinal.classifiers import NNPOM
from skordinal.datasets import make_ordinal_classification


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.fixture
def ordinal_data():
    """Create a synthetic 3-class ordinal dataset for behavioural tests."""
    return make_ordinal_classification(
        n_samples=90,
        n_features=4,
        n_classes=3,
        n_informative=4,
        noise=0.1,
        random_state=0,
    )


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("epsilon_init", 0),
        ("epsilon_init", -1),
        ("n_hidden", -1),
        ("max_iter", -1),
        ("alpha", -1e-5),
    ],
)
def test_nnpom_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that NNPOM raises ValueError for invalid of hyperparameters."""
    classifier = NNPOM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("epsilon_init", "high"),
        ("n_hidden", 5.5),
        ("max_iter", 2.5),
        ("alpha", "tight"),
        ("random_state", "seed"),
        ("random_state", 1.5),
    ],
)
def test_nnpom_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that NNPOM raises ValueError for invalid types of hyperparameters."""
    classifier = NNPOM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_nnpom_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = NNPOM()
    model = classifier.fit(X, y)
    assert model is classifier


def test_nnpom_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = NNPOM()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_nnpom_sets_fitted_attributes_after_fit(X, y):
    """Test than NNPOM exposes fitted attributes aligned con sklearn-style."""
    clf = NNPOM(n_hidden=4, max_iter=5)
    clf.fit(X, y)

    for attr in [
        "classes_",
        "n_features_in_",
        "theta1_",
        "theta2_",
        "loss_",
        "n_iter_",
        "n_layers_",
        "n_outputs_",
        "out_activation_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert isinstance(clf.classes_, np.ndarray) and np.array_equal(
        clf.classes_, np.unique(y)
    )
    assert isinstance(clf.n_features_in_, int) and clf.n_features_in_ == X.shape[1]
    assert isinstance(clf.loss_, (float, np.floating)) and clf.loss_ >= 0
    assert isinstance(clf.n_iter_, int) and 1 <= clf.n_iter_ <= 5
    assert isinstance(clf.n_layers_, int) and clf.n_layers_ == 3
    assert isinstance(clf.n_outputs_, int) and clf.n_outputs_ == len(np.unique(y)) - 1
    assert isinstance(clf.out_activation_, str) and clf.out_activation_ == "logistic"


def test_nnpom_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = NNPOM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_nnpom_predict_raises_if_not_fitted(X):
    """predict and predict_projection raise NotFittedError before fit."""
    classifier = NNPOM()
    with pytest.raises(NotFittedError):
        classifier.predict(X)
    with pytest.raises(NotFittedError):
        classifier.predict_projection(X)


def test_nnpom_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = NNPOM(n_hidden=4, max_iter=5).fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_nnpom_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(NNPOM.__init__).parameters) - {"self"}
    assert set(NNPOM._parameter_constraints) == init_params


def test_nnpom_predict_rejects_wrong_n_features(X, y):
    """predict and predict_projection reject a mismatched n_features."""
    classifier = NNPOM(n_hidden=4, max_iter=5).fit(X, y)
    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])
    with pytest.raises(ValueError):
        classifier.predict_projection(X[:, :-1])


@pytest.mark.parametrize(
    "labels",
    [
        [1, 2, 3],  # standard 1-indexed
        [0, 1, 2],  # 0-indexed
        [-1, 0, 1],  # negative labels
        [3, 5, 7],  # non-contiguous with gaps
    ],
)
def test_nnpom_label_roundtrip(labels):
    """Test that NNPOM preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = NNPOM(n_hidden=4, max_iter=10)
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_nnpom_random_state_reproducibility(X, y):
    """Two fits with the same seed produce identical theta1_, theta2_, n_iter_."""
    clf_a = NNPOM(n_hidden=4, max_iter=10, random_state=0).fit(X, y)
    clf_b = NNPOM(n_hidden=4, max_iter=10, random_state=0).fit(X, y)

    np.testing.assert_array_equal(clf_a.theta1_, clf_b.theta1_)
    np.testing.assert_array_equal(clf_a.theta2_, clf_b.theta2_)
    assert clf_a.n_iter_ == clf_b.n_iter_


def test_nnpom_random_state_different_seeds_differ(X, y):
    """Different seeds produce different initial weights."""
    clf_a = NNPOM(n_hidden=4, max_iter=10, random_state=0).fit(X, y)
    clf_b = NNPOM(n_hidden=4, max_iter=10, random_state=1).fit(X, y)

    assert not np.array_equal(clf_a.theta1_, clf_b.theta1_)


def test_nnpom_random_state_accepts_random_state_instance(X, y):
    """RandomState instance gives the same result as the equivalent seed."""
    rs_seed = NNPOM(n_hidden=4, max_iter=10, random_state=42).fit(X, y)
    rs_instance = NNPOM(
        n_hidden=4, max_iter=10, random_state=np.random.RandomState(42)
    ).fit(X, y)

    np.testing.assert_array_equal(rs_seed.theta1_, rs_instance.theta1_)


def test_fit_no_nan_with_near_zero_probabilities():
    """Training on perfectly separated data must produce finite weights."""
    X_reg = np.vstack([np.ones((10, 2)) * 10, -np.ones((10, 2)) * 10])
    y_reg = np.array([0] * 10 + [1] * 10)
    clf = NNPOM(n_hidden=8, max_iter=50, random_state=0)
    clf.fit(X_reg, y_reg)
    assert np.isfinite(clf.theta1_).all()
    assert np.isfinite(clf.theta2_).all()
    assert np.isfinite(clf.thresholds_).all()
    assert set(clf.predict(X_reg)).issubset(set(clf.classes_))


def test_nnpom_objective_gradient_matches_finite_difference():
    """The analytic objective gradient matches a finite-difference approximation."""
    rng = np.random.default_rng(0)
    n_samples, n_features, n_hidden, n_classes = 30, 4, 3, 3
    X_data = rng.standard_normal((n_samples, n_features))
    y_encoded = rng.integers(0, n_classes, size=n_samples)
    Y = np.eye(n_classes)[y_encoded]

    n_theta1 = n_hidden * (n_features + 1)
    n_theta2 = n_hidden
    n_thresholds = n_classes - 1
    x0 = rng.standard_normal(n_theta1 + n_theta2 + n_thresholds) * 0.1

    clf = NNPOM()

    def cost(nn_params):
        J, _ = clf._objective(
            nn_params, n_features, n_hidden, n_classes, X_data, Y, 0.01
        )
        return J

    def grad(nn_params):
        _, g = clf._objective(
            nn_params, n_features, n_hidden, n_classes, X_data, Y, 0.01
        )
        return g

    err = check_grad(cost, grad, x0)
    assert err < 1e-4


def test_nnpom_objective_gradient_is_zero_in_clamped_region():
    """A fully clamped objective is flat, so its gradient must vanish."""
    n_samples, n_features, n_hidden, n_classes = 30, 4, 3, 3
    rng = np.random.default_rng(0)
    X_data = rng.standard_normal((n_samples, n_features))
    Y = np.zeros((n_samples, n_classes))
    Y[:, 0] = 1.0

    saturating_projection = 75.0
    theta1 = rng.standard_normal((n_hidden, n_features + 1)) * 0.01
    theta2 = np.full((1, n_hidden), saturating_projection / n_hidden)
    x0 = np.concatenate(
        [theta1.flatten(order="F"), theta2.flatten(order="F"), [0.0, 1.0]]
    )

    clf = NNPOM()
    cost, grad = clf._objective(x0, n_features, n_hidden, n_classes, X_data, Y, 0.0)

    np.testing.assert_allclose(cost, -np.log(1e-15))
    np.testing.assert_array_equal(grad, np.zeros_like(grad))


def test_nnpom_proba_and_cumproba_well_formed(ordinal_data):
    """predict_proba/predict_cumproba are well-formed and match predict."""
    X, y = ordinal_data
    clf = NNPOM(n_hidden=4, max_iter=50, random_state=0).fit(X, y)
    n_classes = len(clf.classes_)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], n_classes)
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    cumproba = clf.predict_cumproba(X)
    assert cumproba.shape == (X.shape[0], n_classes - 1)
    assert np.all(cumproba >= 0.0) and np.all(cumproba <= 1.0)
    assert np.all(np.diff(cumproba, axis=1) >= 0)

    expected_pred = clf.classes_[proba.argmax(axis=1)]
    np.testing.assert_array_equal(clf.predict(X), expected_pred)


def test_nnpom_convergence_warning_only_at_insufficient_max_iter(ordinal_data):
    """ConvergenceWarning fires at max_iter=1, not with the default budget."""
    X, y = ordinal_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        NNPOM(random_state=0).fit(X, y)

    with pytest.warns(ConvergenceWarning, match="did not converge") as record:
        clf = NNPOM(max_iter=1, random_state=0).fit(X, y)

    message = str(record[0].message)
    assert re.search(rf"stopped after {clf.n_iter_} iterations?\b", message), message


def test_nnpom_projection_well_formed(ordinal_data):
    """predict_projection is well-formed and consistent with predict."""
    X, y = ordinal_data
    clf = NNPOM(n_hidden=4, max_iter=50, random_state=0).fit(X, y)

    projection = clf.predict_projection(X)
    assert projection.shape == (len(X),)
    assert np.isfinite(projection).all()

    order = np.argsort(projection)
    assert np.all(np.diff(clf.predict(X)[order]) >= 0)
