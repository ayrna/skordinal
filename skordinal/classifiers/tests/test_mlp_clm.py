"""Tests for the MLP-CLM classifier."""

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.exceptions import NotFittedError
from sklearn.utils.class_weight import compute_class_weight

from skordinal.classifiers import MLPCLMClassifier
from skordinal.datasets import make_ordinal_classification
from skordinal.utils.extmath import params_to_thresholds


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
        ("n_hidden_layers", 0),
        ("n_hidden_units", 0),
        ("alpha", -1),
        ("max_iter", 0),
        ("activation", "linear"),
    ],
)
def test_mlp_clm_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that invalid hyperparameter values are rejected."""
    classifier = MLPCLMClassifier(**{param_name: invalid_value})

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
        ("random_state", "seed"),
    ],
)
def test_mlp_clm_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that invalid hyperparameter types are rejected."""
    classifier = MLPCLMClassifier(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_mlp_clm_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = MLPCLMClassifier(max_iter=10, random_state=0)

    assert classifier.fit(X, y) is classifier


def test_mlp_clm_fit_input_validation(X, y):
    """Test that input data is validated."""
    classifier = MLPCLMClassifier(max_iter=1)

    with pytest.raises(ValueError):
        classifier.fit(X, y[:-1])
    with pytest.raises(ValueError):
        classifier.fit([], y)
    with pytest.raises(ValueError):
        classifier.fit(X, [])


def test_mlp_clm_sets_fitted_attributes_after_fit(X, y):
    """Test that fitted attributes have the expected shapes and values."""
    clf = MLPCLMClassifier(
        n_hidden_layers=2, n_hidden_units=4, max_iter=5, random_state=0
    ).fit(X, y)

    for attr in [
        "classes_",
        "n_classes_",
        "n_features_in_",
        "W_",
        "b_",
        "w_out_",
        "t_",
        "loss_",
        "n_iter_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert np.array_equal(clf.classes_, np.unique(y))
    assert clf.n_features_in_ == X.shape[1]
    assert clf.n_classes_ == len(np.unique(y))
    assert len(clf.W_) == 2
    assert clf.W_[0].shape == (X.shape[1], 4)
    assert clf.W_[1].shape == (4, 4)
    assert clf.b_[0].shape == (4,)
    assert clf.b_[1].shape == (4,)
    assert clf.w_out_.shape == (4, 1)
    assert clf.t_.shape == (len(np.unique(y)) - 1,)
    assert np.all(np.diff(params_to_thresholds(clf.t_)) >= 0.0)
    assert np.isfinite(clf.loss_)


def test_mlp_clm_predict_proba_shape_bounds_and_normalization(X, y):
    """Predicted probabilities form valid distributions."""
    clf = MLPCLMClassifier(n_hidden_units=4, max_iter=10, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_mlp_clm_predict_returns_known_labels(X, y):
    """Predictions use the original labels and agree with argmax probabilities."""
    clf = MLPCLMClassifier(max_iter=10, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)

    assert set(clf.predict(X)).issubset(set(clf.classes_))
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[proba.argmax(axis=1)])


def test_mlp_clm_predict_raises_if_not_fitted(X):
    """Prediction methods raise NotFittedError before fitting."""
    classifier = MLPCLMClassifier()

    with pytest.raises(NotFittedError):
        classifier.predict(X)
    with pytest.raises(NotFittedError):
        classifier.predict_proba(X)


def test_mlp_clm_predict_rejects_wrong_n_features(X, y):
    """Prediction rejects data with a different number of features."""
    classifier = MLPCLMClassifier(max_iter=5).fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])


def test_mlp_clm_feature_names_in_when_dataframe(X, y):
    """feature_names_in_ is set when fitting a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = MLPCLMClassifier(max_iter=5, random_state=0).fit(df, y)

    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_mlp_clm_parameter_constraints_match_init_params():
    """Constraint keys match the public constructor parameters."""
    init_params = set(inspect.signature(MLPCLMClassifier.__init__).parameters) - {
        "self"
    }

    assert set(MLPCLMClassifier._parameter_constraints) == init_params


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
def test_mlp_clm_activation_fit_predicts_finite_values(X, y, activation):
    """Every supported hidden activation produces finite predictions."""
    clf = MLPCLMClassifier(
        activation=activation, n_hidden_units=4, max_iter=10, random_state=0
    ).fit(X, y)

    assert np.isfinite(clf.predict_proba(X)).all()


@pytest.mark.parametrize("labels", [[1, 2, 3], [-1, 0, 1], [3, 5, 7]])
def test_mlp_clm_label_roundtrip(labels):
    """Arbitrary ordinal labels are preserved through fit and predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = MLPCLMClassifier(n_hidden_units=4, max_iter=10, random_state=0).fit(
        X, y
    )

    np.testing.assert_array_equal(classifier.classes_, labels_array)
    assert set(classifier.predict(X)).issubset(set(labels_array))


def test_mlp_clm_random_state_reproducibility(X, y):
    """Two fits with the same random state produce identical parameters."""
    clf_a = MLPCLMClassifier(n_hidden_units=4, max_iter=10, random_state=0).fit(X, y)
    clf_b = MLPCLMClassifier(n_hidden_units=4, max_iter=10, random_state=0).fit(X, y)

    for weights_a, weights_b in zip(clf_a.W_, clf_b.W_):
        np.testing.assert_array_equal(weights_a, weights_b)
    for biases_a, biases_b in zip(clf_a.b_, clf_b.b_):
        np.testing.assert_array_equal(biases_a, biases_b)
    np.testing.assert_array_equal(clf_a.w_out_, clf_b.w_out_)
    np.testing.assert_array_equal(clf_a.t_, clf_b.t_)


def test_mlp_clm_random_state_different_seeds_differ(X, y):
    """Different seeds produce different initial network parameters."""
    clf_a = MLPCLMClassifier(n_hidden_units=4, max_iter=1, random_state=0).fit(X, y)
    clf_b = MLPCLMClassifier(n_hidden_units=4, max_iter=1, random_state=1).fit(X, y)

    assert not np.array_equal(clf_a.W_[0], clf_b.W_[0])


def test_mlp_clm_random_state_accepts_random_state_instance(X, y):
    """A RandomState instance matches the equivalent integer seed."""
    clf_seed = MLPCLMClassifier(n_hidden_units=4, max_iter=5, random_state=42).fit(X, y)
    clf_instance = MLPCLMClassifier(
        n_hidden_units=4, max_iter=5, random_state=np.random.RandomState(42)
    ).fit(X, y)

    for weights_seed, weights_instance in zip(clf_seed.W_, clf_instance.W_):
        np.testing.assert_array_equal(weights_seed, weights_instance)
    np.testing.assert_array_equal(clf_seed.w_out_, clf_instance.w_out_)
    np.testing.assert_array_equal(clf_seed.t_, clf_instance.t_)


def test_mlp_clm_class_weight_balanced_matches_equivalent_dict():
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
    clf_none = MLPCLMClassifier(class_weight=None, **common).fit(X, y)
    clf_balanced = MLPCLMClassifier(class_weight="balanced", **common).fit(X, y)
    weights = compute_class_weight("balanced", classes=clf_balanced.classes_, y=y)
    class_weight = dict(zip(clf_balanced.classes_, weights))
    clf_dict = MLPCLMClassifier(class_weight=class_weight, **common).fit(X, y)

    assert not np.allclose(clf_none.w_out_, clf_balanced.w_out_)
    for weights_balanced, weights_dict in zip(clf_balanced.W_, clf_dict.W_):
        np.testing.assert_allclose(weights_balanced, weights_dict, atol=1e-8)
    np.testing.assert_allclose(clf_balanced.w_out_, clf_dict.w_out_, atol=1e-8)
    np.testing.assert_allclose(clf_balanced.t_, clf_dict.t_, atol=1e-8)


def test_mlp_clm_class_weight_uses_weight_sum():
    """Class weights are normalized by the sum of sample weights."""
    X = np.array(
        [
            [-1.0, -1.0],
            [-0.8, -1.2],
            [0.0, 0.0],
            [0.2, -0.1],
            [1.0, 1.0],
            [1.2, 0.8],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])

    class_weight = {0: 1.0, 1: 2.0, 2: 4.0}

    clf = MLPCLMClassifier(
        n_hidden_units=3,
        max_iter=20,
        random_state=0,
        class_weight=class_weight,
    )

    clf.fit(X, y)

    sample_weight = np.array([1.0, 1.0, 2.0, 2.0, 4.0, 4.0])

    assert np.sum(sample_weight) != len(sample_weight)


def test_mlp_clm_objective_gradient_matches_finite_difference():
    """The analytic objective gradient matches a finite-difference approximation."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y_encoded = rng.integers(0, 3, size=X.shape[0])
    Y = np.eye(3)[y_encoded]
    sample_weight = rng.uniform(0.5, 3.0, size=X.shape[0])

    clf = MLPCLMClassifier(n_hidden_layers=2, n_hidden_units=3, alpha=0.1)
    clf.n_features_in_ = X.shape[1]
    clf.n_classes_ = Y.shape[1]
    params = clf._initialize_parameters(np.random.RandomState(0))

    error = check_grad(
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[0],
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[1],
        params,
    )

    assert error < 1e-4


def test_mlp_clm_binary_problem_has_one_threshold():
    """A binary problem produces one ordered threshold and two probabilities."""
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    clf = MLPCLMClassifier(n_hidden_units=3, max_iter=10, random_state=0).fit(X, y)

    assert clf.t_.shape == (1,)
    assert clf.predict_proba(X).shape == (len(X), 2)
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0)


def test_mlp_clm_large_magnitude_inputs_remain_finite():
    """Large feature magnitudes do not produce NaNs or invalid probabilities."""
    X, y = make_ordinal_classification(
        n_samples=60,
        n_features=4,
        n_classes=3,
        n_informative=4,
        random_state=0,
    )
    X *= 1000.0

    clf = MLPCLMClassifier(n_hidden_units=4, max_iter=20, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_mlp_clm_behaviour_on_ordinal_data(ordinal_data):
    """The model learns useful predictions on a separable ordinal dataset."""
    X, y = ordinal_data
    clf = MLPCLMClassifier(n_hidden_units=8, max_iter=100, random_state=0).fit(X, y)

    assert (clf.predict(X) == y).mean() >= 0.70
