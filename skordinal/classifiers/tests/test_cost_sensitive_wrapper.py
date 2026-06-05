"""Tests for the CostSensitiveWrapper classifier."""

import inspect

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from skordinal.classifiers import CostSensitiveWrapper


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


def test_cost_sensitive_wrapper_predict_matches_expected():
    """Test that predictions match expected values."""
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    classifier = CostSensitiveWrapper(LogisticRegression()).fit(X, y)
    npt.assert_array_equal(classifier.predict(X), y)


@pytest.mark.parametrize("invalid_value", [5, "logistic", object()])
def test_cost_sensitive_wrapper_hyperparameter_type_validation(X, y, invalid_value):
    """Test that CostSensitiveWrapper raises ValueError for invalid types of hyperparameters."""
    classifier = CostSensitiveWrapper(estimator=invalid_value)

    with pytest.raises(ValueError, match=r"The 'estimator' parameter.*"):
        classifier.fit(X, y)


def test_cost_sensitive_wrapper_fit_input_validation(X, y):
    """Test that input data is validated."""
    classifier = CostSensitiveWrapper()

    with pytest.raises(ValueError):
        classifier.fit(X, y[:-1])

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])


def test_cost_sensitive_wrapper_requires_sample_weight_support(X, y):
    """Test that a base estimator without sample_weight support is rejected."""
    classifier = CostSensitiveWrapper(KNeighborsClassifier())

    with pytest.raises(ValueError, match="sample_weight"):
        classifier.fit(X, y)


def test_cost_sensitive_wrapper_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = CostSensitiveWrapper().fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_cost_sensitive_wrapper_predict_proba_sums_to_one(X, y):
    """Test that predict_proba returns rows that sum to one."""
    classifier = CostSensitiveWrapper().fit(X, y)
    proba = classifier.predict_proba(X)

    assert proba.shape == (X.shape[0], classifier.classes_.size)
    npt.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]))
    assert np.all(proba >= 0.0)


def test_cost_sensitive_wrapper_sets_classes_and_n_features_in_after_fit(X, y):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = CostSensitiveWrapper().fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_cost_sensitive_wrapper_predict_raises_if_not_fitted(X):
    """Test that predict raises when the model is not fitted."""
    from sklearn.exceptions import NotFittedError

    classifier = CostSensitiveWrapper()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_cost_sensitive_wrapper_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = CostSensitiveWrapper().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_cost_sensitive_wrapper_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(CostSensitiveWrapper.__init__).parameters) - {
        "self"
    }
    assert set(CostSensitiveWrapper._parameter_constraints) == init_params


def test_cost_sensitive_wrapper_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = CostSensitiveWrapper().fit(X, y)
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
def test_cost_sensitive_wrapper_label_roundtrip(labels):
    """Test that CostSensitiveWrapper preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = CostSensitiveWrapper().fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))
