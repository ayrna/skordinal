"""Tests for make_ordinal_classification."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.linear_model import LogisticRegression

from skordinal.datasets import make_ordinal_classification


@pytest.mark.parametrize(
    "n_samples, n_features, n_classes, n_informative, noise",
    [
        (100, 10, 5, 5, 0.1),
        (50, 4, 3, 3, 0.0),
        (200, 8, 10, 1, 1.0),
    ],
    ids=["defaults", "small_noise_free", "many_classes_one_informative"],
)
def test_output_shape_and_dtype(n_samples, n_features, n_classes, n_informative, noise):
    """X and y have the requested shapes and the documented float64/intp dtypes."""
    X, y = make_ordinal_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        noise=noise,
        random_state=0,
    )
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert X.dtype == np.float64
    assert y.dtype == np.intp


@pytest.mark.parametrize("n_classes", [2, 3, 5, 10])
def test_labels_form_complete_ordinal_set(n_classes):
    """y covers exactly the contiguous label set {0, ..., n_classes-1}."""
    _, y = make_ordinal_classification(
        n_samples=200, n_classes=n_classes, random_state=0
    )
    assert set(np.unique(y)) == set(range(n_classes))


@pytest.mark.parametrize(
    "seed_factory",
    [lambda: 0, lambda: np.random.RandomState(0)],
    ids=["int_seed", "random_state_instance"],
)
def test_same_seed_is_reproducible(seed_factory):
    """Equal int seeds or RandomState instances reproduce X and y exactly."""
    X1, y1 = make_ordinal_classification(random_state=seed_factory())
    X2, y2 = make_ordinal_classification(random_state=seed_factory())
    assert_array_equal(X1, X2)
    assert_array_equal(y1, y2)


def test_unseeded_calls_vary():
    """Two unseeded calls draw different feature matrices."""
    X1, _ = make_ordinal_classification(random_state=None)
    X2, _ = make_ordinal_classification(random_state=None)
    assert not np.array_equal(X1, X2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_features": 0}, "n_features"),
        ({"n_classes": 1}, "n_classes"),
        ({"n_informative": 0}, "n_informative"),
        ({"n_features": 4, "n_informative": 5}, "n_informative"),
        ({"noise": -0.1}, "noise"),
        ({"n_classes": 3, "weights": [0.25, 0.25, 0.25, 0.25]}, "weights"),
        ({"n_classes": 3, "weights": [0.5, -0.2, 0.7]}, "weights"),
    ],
    ids=[
        "n_samples_zero",
        "n_features_zero",
        "n_classes_below_2",
        "n_informative_zero",
        "n_informative_exceeds_features",
        "negative_noise",
        "weights_wrong_length",
        "weights_negative",
    ],
)
def test_invalid_parameters_raise(kwargs, match):
    """Out-of-range parameters raise ValueError naming the offending parameter."""
    with pytest.raises(ValueError, match=match):
        make_ordinal_classification(**kwargs)


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([0.1, 0.2, 0.7], [0.1, 0.2, 0.7]),
        ([0.1, 0.2], [0.1, 0.2, 0.7]),
    ],
    ids=["full_length", "inferred_last"],
)
def test_weights_control_class_proportions(weights, expected):
    """Realised class frequencies follow weights, inferring the last when omitted."""
    _, y = make_ordinal_classification(
        n_samples=2000, n_classes=3, weights=weights, random_state=0
    )
    proportions = np.bincount(y, minlength=3) / y.size
    np.testing.assert_allclose(proportions, expected, atol=0.02)


def test_output_trains_sklearn_classifier():
    """Generated data is valid input for a scikit-learn classifier."""
    X, y = make_ordinal_classification(n_samples=120, n_classes=4, random_state=0)
    y_pred = LogisticRegression(max_iter=500).fit(X, y).predict(X)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)) <= set(range(4))
