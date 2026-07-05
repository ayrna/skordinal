"""Tests for the ELMOP classifier."""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skordinal.classifiers import ELMOP


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_hidden", 0),
        ("n_hidden", -1),
        ("activation", "relu"),
    ],
)
def test_elmop_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that ELMOP raises ValueError for invalid of hyperparameters."""
    classifier = ELMOP(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_hidden", 5.5),
        ("activation", 42),
        ("random_state", "seed"),
        ("random_state", 1.5),
    ],
)
def test_elmop_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that ELMOP raises ValueError for invalid types of hyperparameters."""
    classifier = ELMOP(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_elmop_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = ELMOP()
    model = classifier.fit(X, y)
    assert model is classifier


def test_elmop_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = ELMOP()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_elmop_sets_fitted_attributes_after_fit(X, y):
    """Test than ELMOP exposes fitted attributes aligned con sklearn-style."""
    clf = ELMOP(n_hidden=4)
    clf.fit(X, y)

    for attr in [
        "classes_",
        "n_features_in_",
        "input_weights_",
        "input_biases_",
        "output_weights_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert isinstance(clf.classes_, np.ndarray) and np.array_equal(
        clf.classes_, np.unique(y)
    )
    assert isinstance(clf.n_features_in_, int) and clf.n_features_in_ == X.shape[1]
    assert clf.input_weights_.shape == (4, X.shape[1])
    assert clf.input_biases_.shape == (4,)
    assert clf.output_weights_.shape == (4, len(np.unique(y)) - 1)
    assert np.isfinite(clf.output_weights_).all()


def test_elmop_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = ELMOP()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_elmop_predict_raises_if_not_fitted(X):
    """Test that predict raises NotFittedError if called before fit."""
    classifier = ELMOP()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_elmop_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = ELMOP(n_hidden=4).fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_elmop_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(ELMOP.__init__).parameters) - {"self"}
    assert set(ELMOP._parameter_constraints) == init_params


def test_elmop_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = ELMOP(n_hidden=4).fit(X, y)
    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])


@pytest.mark.parametrize(
    "labels",
    [
        [1, 2, 3],  # standard 1-indexed
        [0, 1, 2],  # 0-indexed
        [-1, 0, 1],  # negative labels
        [3, 5, 7],  # non-contiguous with gaps
    ],
)
def test_elmop_label_roundtrip(labels):
    """Test that ELMOP preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = ELMOP(n_hidden=4)
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_elmop_random_state_reproducibility(X, y):
    """Two fits with the same seed produce identical weight matrices."""
    clf_a = ELMOP(n_hidden=4, random_state=0).fit(X, y)
    clf_b = ELMOP(n_hidden=4, random_state=0).fit(X, y)

    np.testing.assert_array_equal(clf_a.input_weights_, clf_b.input_weights_)
    np.testing.assert_array_equal(clf_a.input_biases_, clf_b.input_biases_)
    np.testing.assert_array_equal(clf_a.output_weights_, clf_b.output_weights_)


def test_elmop_random_state_different_seeds_differ(X, y):
    """Different seeds produce different random input weights."""
    clf_a = ELMOP(n_hidden=4, random_state=0).fit(X, y)
    clf_b = ELMOP(n_hidden=4, random_state=1).fit(X, y)

    assert not np.array_equal(clf_a.input_weights_, clf_b.input_weights_)


def test_elmop_random_state_accepts_random_state_instance(X, y):
    """RandomState instance gives the same result as the equivalent seed."""
    rs_seed = ELMOP(n_hidden=4, random_state=42).fit(X, y)
    rs_instance = ELMOP(n_hidden=4, random_state=np.random.RandomState(42)).fit(X, y)

    np.testing.assert_array_equal(rs_seed.input_weights_, rs_instance.input_weights_)


def test_elmop_predict_cumproba_shape_and_bounds(X, y):
    """predict_cumproba has shape (n, n_classes - 1) with values in [0, 1]."""
    classifier = ELMOP(n_hidden=4, random_state=0).fit(X, y)
    cumproba = classifier.predict_cumproba(X)

    n_classes = len(np.unique(y))
    assert cumproba.shape == (X.shape[0], n_classes - 1)
    assert np.all(cumproba >= 0.0) and np.all(cumproba <= 1.0)


def test_elmop_activation_changes_output_weights(X, y):
    """Different activation functions produce different output weights."""
    clf_sigmoid = ELMOP(activation="sigmoid", random_state=0).fit(X, y)
    clf_hardlim = ELMOP(activation="hardlim", random_state=0).fit(X, y)

    assert not np.allclose(clf_sigmoid.output_weights_, clf_hardlim.output_weights_)


def test_elmop_n_hidden_greater_than_n_samples_finite():
    """Fitting with more hidden units than samples yields finite weights."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    y = np.tile(np.arange(3), 20 // 3 + 1)[:20]

    classifier = ELMOP(n_hidden=100, random_state=0).fit(X, y)

    n_classes = len(np.unique(y))
    assert classifier.output_weights_.shape == (100, n_classes - 1)
    assert np.isfinite(classifier.output_weights_).all()


def test_elmop_large_magnitude_x_no_nan():
    """Large-magnitude inputs still yield finite, normalised probabilities."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4)) * 1000.0
    y = np.tile(np.arange(3), 20)

    classifier = ELMOP(random_state=0).fit(X, y)
    proba = classifier.predict_proba(X)

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
