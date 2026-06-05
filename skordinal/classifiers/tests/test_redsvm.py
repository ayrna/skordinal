"""Tests for the REDSVM classifier."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from skordinal.classifiers import REDSVM
from skordinal.utils._testing import make_balance_scale_split

PREDICTIONS_DIR = Path(__file__).parent / "data" / "REDSVM"


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
        "linear",
        "poly",
        "rbf",
        "sigmoid",
        "stump",
        "perceptron",
        "laplacian",
        "exponential",
    ],
)
def test_redsvm_predict_matches_expected(kernel):
    """Test that predictions match expected values."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    classifier = REDSVM(
        C=0.1,
        kernel=kernel,
        degree=2,
        gamma=0.1,
        coef0=0.5,
        shrinking=False,
        tol=0.005,
        cache_size=150,
    )

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
        ("gamma", -0.5),
        ("tol", -1e-5),
        ("cache_size", 0),
        ("kernel", "unknown"),
        ("gamma", "invalid_string"),
    ],
)
def test_redsvm_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that REDSVM raises ValueError for invalid of hyperparameters."""
    classifier = REDSVM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", "high"),
        ("kernel", 5),
        ("degree", 2.5),
        ("coef0", "bias"),
        ("shrinking", "yes"),
        ("shrinking", 2),
        ("tol", "tight"),
        ("cache_size", "big"),
    ],
)
def test_redsvm_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that REDSVM raises ValueError for invalid types of hyperparameters."""
    classifier = REDSVM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_redsvm_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = REDSVM()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"


def test_redsvm_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = REDSVM()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_redsvm_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = REDSVM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_redsvm_sets_classes_and_n_features_in_after_fit(X, y):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = REDSVM().fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_redsvm_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = REDSVM().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_redsvm_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(REDSVM.__init__).parameters) - {"self"}
    assert set(REDSVM._parameter_constraints) == init_params


def test_redsvm_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = REDSVM().fit(X, y)
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
def test_redsvm_label_roundtrip(labels):
    """Test that REDSVM preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = REDSVM(C=0.1, kernel="linear")
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_redsvm_gamma_scale_zero_variance(X, y):
    """Test that gamma='scale' handles zero-variance input without dividing by zero."""
    X_constant = np.ones_like(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        classifier = REDSVM(gamma="scale", kernel="rbf").fit(X_constant, y)

    assert set(classifier.predict(X_constant)).issubset(set(classifier.classes_))
