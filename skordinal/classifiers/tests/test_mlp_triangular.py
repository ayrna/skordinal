"""Tests for the MLP triangular soft-labelling classifier."""

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.exceptions import NotFittedError
from sklearn.utils.class_weight import compute_class_weight

from skordinal.classifiers._mlp_triangular import MLPTriangularClassifier
from skordinal.datasets import make_ordinal_classification


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
        ("t_alpha", -1),
        ("eta", -0.1),
        ("eta", 1.1),
        ("max_iter", 0),
        ("activation", "linear"),
        ("class_weight", "unknown"),
    ],
)
def test_mlp_triangular_hyperparameter_value_validation(
    X, y, param_name, invalid_value
):
    """Invalid hyperparameter values are rejected."""
    classifier = MLPTriangularClassifier(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_hidden_layers", 1.5),
        ("n_hidden_units", 2.5),
        ("alpha", "strong"),
        ("t_alpha", "wide"),
        ("eta", "soft"),
        ("max_iter", 2.5),
        ("activation", 1),
        ("random_state", "seed"),
        ("class_weight", 1.0),
    ],
)
def test_mlp_triangular_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Invalid hyperparameter types are rejected."""
    classifier = MLPTriangularClassifier(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_mlp_triangular_fit_returns_self(X, y):
    """fit returns self for sklearn compatibility."""
    classifier = MLPTriangularClassifier(max_iter=10, random_state=0)

    assert classifier.fit(X, y) is classifier


def test_mlp_triangular_fit_input_validation(X, y):
    """Input arrays must have compatible sample and feature dimensions."""
    classifier = MLPTriangularClassifier(max_iter=1)

    with pytest.raises(ValueError):
        classifier.fit(X, y[:-1])
    with pytest.raises(ValueError):
        classifier.fit([], y)
    with pytest.raises(ValueError):
        classifier.fit(X, [])


def test_mlp_triangular_sets_fitted_attributes_after_fit(X, y):
    """Fitted parameters and metadata have the expected shapes."""
    clf = MLPTriangularClassifier(
        n_hidden_layers=2, n_hidden_units=4, max_iter=5, random_state=0
    ).fit(X, y)

    for attr in [
        "classes_",
        "n_classes_",
        "n_features_in_",
        "W_",
        "b_",
        "W_out_",
        "b_out_",
        "soft_label_matrix_",
        "loss_",
        "n_iter_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert np.array_equal(clf.classes_, np.unique(y))
    assert clf.n_classes_ == len(np.unique(y))
    assert clf.n_features_in_ == X.shape[1]
    assert len(clf.W_) == len(clf.b_) == 2
    assert clf.W_[0].shape == (X.shape[1], 4)
    assert clf.W_[1].shape == (4, 4)
    assert clf.b_[0].shape == clf.b_[1].shape == (4,)
    assert clf.W_out_.shape == (4, len(np.unique(y)))
    assert clf.b_out_.shape == (len(np.unique(y)),)
    assert clf.soft_label_matrix_.shape == (len(np.unique(y)), len(np.unique(y)))
    np.testing.assert_allclose(clf.soft_label_matrix_.sum(axis=1), 1.0)
    assert np.isfinite(clf.loss_)


def test_mlp_triangular_predict_proba_is_valid(X, y):
    """Predicted probabilities are finite distributions over all classes."""
    clf = MLPTriangularClassifier(n_hidden_units=4, max_iter=10, random_state=0).fit(
        X, y
    )
    proba = clf.predict_proba(X)

    assert proba.shape == (len(X), len(np.unique(y)))
    assert np.isfinite(proba).all()
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_mlp_triangular_predict_returns_known_labels(X, y):
    """predict returns original labels and the most probable class."""
    clf = MLPTriangularClassifier(max_iter=10, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)

    assert set(clf.predict(X)).issubset(set(clf.classes_))
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[proba.argmax(axis=1)])


def test_mlp_triangular_predict_raises_if_not_fitted(X):
    """Prediction methods raise NotFittedError before fitting."""
    classifier = MLPTriangularClassifier()

    with pytest.raises(NotFittedError):
        classifier.predict(X)
    with pytest.raises(NotFittedError):
        classifier.predict_proba(X)


def test_mlp_triangular_predict_rejects_wrong_n_features(X, y):
    """Prediction rejects a different number of input features."""
    classifier = MLPTriangularClassifier(max_iter=5).fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])


def test_mlp_triangular_feature_names_in_when_dataframe(X, y):
    """feature_names_in_ is populated for DataFrame input."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = MLPTriangularClassifier(max_iter=5, random_state=0).fit(df, y)

    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_mlp_triangular_parameter_constraints_match_init_params():
    """Constraint keys match the public constructor parameters."""
    init_params = set(
        inspect.signature(MLPTriangularClassifier.__init__).parameters
    ) - {"self"}

    assert set(MLPTriangularClassifier._parameter_constraints) == init_params


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
def test_mlp_triangular_activation_fit_predicts_finite_values(X, y, activation):
    """Every supported hidden activation produces finite probabilities."""
    clf = MLPTriangularClassifier(
        activation=activation, n_hidden_units=4, max_iter=10, random_state=0
    ).fit(X, y)

    assert np.isfinite(clf.predict_proba(X)).all()


@pytest.mark.parametrize("labels", [[1, 2, 3], [-1, 0, 1], [3, 5, 7]])
def test_mlp_triangular_label_roundtrip(labels):
    """Arbitrary ordinal labels are preserved through fit and predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = MLPTriangularClassifier(
        n_hidden_units=4, max_iter=10, random_state=0
    ).fit(X, y)

    np.testing.assert_array_equal(classifier.classes_, labels_array)
    assert set(classifier.predict(X)).issubset(set(labels_array))


def test_mlp_triangular_random_state_reproducibility(X, y):
    """The same random state produces identical fitted parameters."""
    clf_a = MLPTriangularClassifier(n_hidden_units=4, max_iter=10, random_state=0).fit(
        X, y
    )
    clf_b = MLPTriangularClassifier(n_hidden_units=4, max_iter=10, random_state=0).fit(
        X, y
    )

    for weights_a, weights_b in zip(clf_a.W_, clf_b.W_):
        np.testing.assert_array_equal(weights_a, weights_b)
    np.testing.assert_array_equal(clf_a.W_out_, clf_b.W_out_)
    np.testing.assert_array_equal(clf_a.b_out_, clf_b.b_out_)


def test_mlp_triangular_random_state_different_seeds_differ(X, y):
    """Different seeds produce different initial network parameters."""
    clf_a = MLPTriangularClassifier(n_hidden_units=4, max_iter=1, random_state=0).fit(
        X, y
    )
    clf_b = MLPTriangularClassifier(n_hidden_units=4, max_iter=1, random_state=1).fit(
        X, y
    )

    assert not np.array_equal(clf_a.W_[0], clf_b.W_[0])


def test_mlp_triangular_random_state_accepts_random_state_instance(X, y):
    """A RandomState instance matches the equivalent integer seed."""
    clf_seed = MLPTriangularClassifier(
        n_hidden_units=4, max_iter=5, random_state=42
    ).fit(X, y)
    clf_instance = MLPTriangularClassifier(
        n_hidden_units=4, max_iter=5, random_state=np.random.RandomState(42)
    ).fit(X, y)

    for weights_seed, weights_instance in zip(clf_seed.W_, clf_instance.W_):
        np.testing.assert_array_equal(weights_seed, weights_instance)
    np.testing.assert_array_equal(clf_seed.W_out_, clf_instance.W_out_)
    np.testing.assert_array_equal(clf_seed.b_out_, clf_instance.b_out_)


def test_mlp_triangular_soft_labels_match_expected_matrix():
    """Triangular soft labels use ordinal distances and eta interpolation."""
    clf = MLPTriangularClassifier(t_alpha=0.5, eta=1.0)
    clf.n_classes_ = 3
    clf._precompute_soft_labels()

    np.testing.assert_allclose(
        clf.soft_label_matrix_,
        [[2 / 3, 1 / 3, 0.0], [1 / 4, 1 / 2, 1 / 4], [0.0, 1 / 3, 2 / 3]],
    )


def test_mlp_triangular_eta_zero_uses_one_hot_labels():
    """eta=0 disables soft-label smoothing."""
    clf = MLPTriangularClassifier(t_alpha=0.1, eta=0.0)
    clf.n_classes_ = 4
    clf._precompute_soft_labels()

    np.testing.assert_array_equal(clf.soft_label_matrix_, np.eye(4))


def test_mlp_triangular_t_alpha_zero_uses_uniform_soft_labels():
    """t_alpha=0 distributes soft-label mass uniformly."""
    clf = MLPTriangularClassifier(t_alpha=0.0, eta=1.0)
    clf.n_classes_ = 3
    clf._precompute_soft_labels()

    np.testing.assert_allclose(clf.soft_label_matrix_, np.full((3, 3), 1 / 3))


def test_mlp_triangular_class_weight_balanced_matches_equivalent_dict():
    """Balanced weights match their explicit class-weight dictionary."""
    X, y = make_ordinal_classification(
        n_samples=60,
        n_features=3,
        n_classes=3,
        n_informative=3,
        weights=[0.1, 0.3, 0.6],
        random_state=1,
    )
    common = dict(n_hidden_units=4, max_iter=10, random_state=0, alpha=1.0)
    clf_balanced = MLPTriangularClassifier(class_weight="balanced", **common).fit(X, y)
    weights = compute_class_weight("balanced", classes=clf_balanced.classes_, y=y)
    class_weight = dict(zip(clf_balanced.classes_, weights))
    clf_dict = MLPTriangularClassifier(class_weight=class_weight, **common).fit(X, y)

    for weights_balanced, weights_dict in zip(clf_balanced.W_, clf_dict.W_):
        np.testing.assert_allclose(weights_balanced, weights_dict, atol=1e-8)
    np.testing.assert_allclose(clf_balanced.W_out_, clf_dict.W_out_, atol=1e-8)
    np.testing.assert_allclose(clf_balanced.b_out_, clf_dict.b_out_, atol=1e-8)


def test_mlp_triangular_gradient_with_non_uniform_sample_weights():
    """The gradient is correct even with non-uniform sample weights."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3))
    y_encoded = rng.integers(0, 3, size=len(X))
    Y = np.eye(3)[y_encoded]

    sample_weight = rng.uniform(0.1, 3.0, size=len(X))

    clf = MLPTriangularClassifier(
        n_hidden_layers=1, n_hidden_units=4, alpha=0.1, t_alpha=0.1, eta=0.5
    )
    clf.n_features_in_ = X.shape[1]
    clf.n_classes_ = Y.shape[1]
    params = clf._initialize_parameters(np.random.RandomState(42))

    error = check_grad(
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[0],
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[1],
        params,
    )

    assert error < 1e-4


def test_mlp_triangular_class_weight_changes_results(X, y):
    """Providing class weights actively changes the optimization outcome."""
    clf_unweighted = MLPTriangularClassifier(max_iter=10, random_state=0).fit(X, y)

    clf_weighted = MLPTriangularClassifier(
        class_weight={0: 1.0, 1: 1.0, 2: 10.0}, max_iter=10, random_state=0
    ).fit(X, y)

    assert not np.allclose(clf_unweighted.W_out_, clf_weighted.W_out_)


def test_mlp_triangular_t_alpha_clips_at_zero():
    """Large t_alpha values correctly clip distant class probabilities to 0."""
    clf = MLPTriangularClassifier(t_alpha=1.0, eta=1.0)
    clf.n_classes_ = 3

    clf._precompute_soft_labels()

    np.testing.assert_array_equal(clf.soft_label_matrix_, np.eye(3))


def test_mlp_triangular_binary_problem_has_two_probabilities():
    """A binary problem produces two output probabilities."""
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    clf = MLPTriangularClassifier(n_hidden_units=3, max_iter=10, random_state=0).fit(
        X, y
    )

    assert clf.W_out_.shape == (3, 2)
    assert clf.soft_label_matrix_.shape == (2, 2)
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0)


def test_mlp_triangular_objective_gradient_matches_finite_difference():
    """The analytic objective gradient matches a finite-difference approximation."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y_encoded = rng.integers(0, 3, size=len(X))
    Y = np.eye(3)[y_encoded]
    sample_weight = np.ones(len(X))

    clf = MLPTriangularClassifier(
        n_hidden_layers=2, n_hidden_units=3, alpha=0.1, t_alpha=0.2, eta=0.8
    )
    clf.n_features_in_ = X.shape[1]
    clf.n_classes_ = Y.shape[1]
    params = clf._initialize_parameters(np.random.RandomState(0))

    error = check_grad(
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[0],
        lambda p: clf._cost_and_grad(p, X, Y, sample_weight)[1],
        params,
    )

    assert error < 1e-4


def test_mlp_triangular_large_magnitude_inputs_remain_finite():
    """Large feature magnitudes do not produce invalid probabilities."""
    X, y = make_ordinal_classification(
        n_samples=60,
        n_features=4,
        n_classes=3,
        n_informative=4,
        random_state=0,
    )
    X *= 1000.0

    clf = MLPTriangularClassifier(n_hidden_units=4, max_iter=20, random_state=0).fit(
        X, y
    )
    proba = clf.predict_proba(X)

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_mlp_triangular_behaviour_on_ordinal_data(ordinal_data):
    """The model learns useful predictions on a separable ordinal dataset."""
    X, y = ordinal_data
    clf = MLPTriangularClassifier(n_hidden_units=8, max_iter=100, random_state=0).fit(
        X, y
    )

    assert (clf.predict(X) == y).mean() >= 0.70
