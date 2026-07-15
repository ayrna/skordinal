"""Tests for the SVOREX classifier."""

import inspect
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from skordinal.classifiers import SVOREX
from skordinal.utils._testing import make_balance_scale_split

PREDICTIONS_DIR = Path(__file__).parent / "data" / "SVOREX"


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.mark.parametrize(
    "kernel",
    [
        "rbf",
        "linear",
        "poly",
    ],
)
def test_svorex_predict_matches_expected(kernel):
    """Test that predictions match expected values."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    classifier = SVOREX(C=0.5, kernel=kernel, degree=4, tol=0.002, gamma=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_expected = np.loadtxt(PREDICTIONS_DIR / f"predictions_{kernel}.csv", dtype=int)

    npt.assert_equal(
        y_pred, y_expected, "The prediction doesnt match with the desired values"
    )


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", 0),
        ("C", -1),
        ("degree", -1),
        ("degree", 0),
        ("tol", 0),
        ("tol", -1e-5),
        ("kernel", "unknown"),
        ("gamma", -1),
    ],
)
def test_svorex_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that SVOREX raises ValueError for invalid of hyperparameters."""
    classifier = SVOREX(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", "high"),
        ("kernel", 5),
        ("degree", 2.5),
        ("tol", "tight"),
        ("gamma", "low"),
    ],
)
def test_svorex_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that SVOREX raises ValueError for invalid types of hyperparameters."""
    classifier = SVOREX(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_svorex_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = SVOREX()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The SVOREX fit method doesnt return Null on error"


def test_svorex_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = SVOREX()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_svorex_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = SVOREX()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_svorex_sets_classes_and_n_features_in_after_fit(X, y):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = SVOREX().fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_svorex_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = SVOREX().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_svorex_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(SVOREX.__init__).parameters) - {"self"}
    assert set(SVOREX._parameter_constraints) == init_params


def test_svorex_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = SVOREX().fit(X, y)
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
def test_svorex_label_roundtrip(labels):
    """Test that SVOREX preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = SVOREX(C=0.5, kernel="linear")
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))
