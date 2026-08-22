"""Tests for the SVORIM classifier."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from skordinal.classifiers import SVOREX, SVORIM
from skordinal.utils._testing import make_balance_scale_split

PREDICTIONS_DIR = Path(__file__).parent / "data" / "SVORIM"


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
def test_svorim_predict_matches_expected(kernel):
    """Test that predictions match expected values."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    classifier = SVORIM(C=0.5, kernel=kernel, degree=4, tol=0.002, gamma=0.1)
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
        ("gamma", "low"),
    ],
)
def test_svorim_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that SVORIM raises ValueError for invalid of hyperparameters."""
    classifier = SVORIM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", "high"),
        ("kernel", 5),
        ("degree", 2.5),
        ("tol", "tight"),
    ],
)
def test_svorim_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that SVORIM raises ValueError for invalid types of hyperparameters."""
    classifier = SVORIM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_svorim_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = SVORIM()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The SVORIM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The SVORIM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The SVORIM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The SVORIM fit method doesnt return Null on error"


def test_svorim_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = SVORIM()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_svorim_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = SVORIM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_svorim_sets_classes_and_n_features_in_after_fit(X, y):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = SVORIM().fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_svorim_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = SVORIM().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_svorim_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(SVORIM.__init__).parameters) - {"self"}
    assert set(SVORIM._parameter_constraints) == init_params


def test_svorim_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = SVORIM().fit(X, y)
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
def test_svorim_label_roundtrip(labels):
    """Test that SVORIM preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = SVORIM(C=0.5, kernel="linear")
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


@pytest.mark.parametrize(
    "gamma, expected_gamma",
    [
        ("auto", lambda X: 1.0 / X.shape[1]),
        ("scale", lambda X: 1.0 / (X.shape[1] * X.var())),
    ],
    ids=["auto", "scale"],
)
def test_svorim_gamma_string_resolves_like_numeric(gamma, expected_gamma):
    """A string gamma predicts the same as the numeric value it resolves to."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    resolved = SVORIM(gamma=gamma).fit(X_train, y_train)
    explicit = SVORIM(gamma=expected_gamma(X_train)).fit(X_train, y_train)

    np.testing.assert_array_equal(resolved.predict(X_test), explicit.predict(X_test))


def test_svorim_gamma_scale_zero_variance(X, y):
    """Test that gamma='scale' handles zero-variance input without dividing by zero."""
    X_constant = np.ones_like(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        classifier = SVORIM(gamma="scale", kernel="rbf").fit(X_constant, y)

    assert set(classifier.predict(X_constant)).issubset(set(classifier.classes_))


def test_svorim_thresholds_are_ordered():
    """Test that fitted thresholds are non-decreasing, as SVORIM guarantees by
    construction (no explicit ordering constraint is needed, unlike SVOREX)."""
    X_train, _, y_train, _ = make_balance_scale_split()

    classifier = SVORIM(C=0.5, kernel="rbf", gamma=0.1, tol=0.002).fit(X_train, y_train)

    thresholds = classifier.model_["biasj"]
    assert list(thresholds) == sorted(thresholds)


def test_svorim_matches_svorex_when_binary():
    """With exactly two classes there is a single threshold and no ordering
    constraint applies, so SVOREX (explicit) and SVORIM (implicit) solve the
    same QP and should agree up to solver tolerance."""
    X_train, X_test, y_train, y_test = make_balance_scale_split()
    mask_train = y_train != 1
    mask_test = y_test != 1

    kwargs = {"C": 0.5, "kernel": "rbf", "gamma": 0.1, "tol": 0.001}
    svorex_pred = (
        SVOREX(**kwargs)
        .fit(X_train[mask_train], y_train[mask_train])
        .predict(X_test[mask_test])
    )
    svorim_pred = (
        SVORIM(**kwargs)
        .fit(X_train[mask_train], y_train[mask_train])
        .predict(X_test[mask_test])
    )

    npt.assert_array_equal(svorex_pred, svorim_pred)


def test_svorex_and_svorim_coexist_in_the_same_process():
    """Fitting both C extensions in one process must not corrupt either one:
    guards against symbol collisions between the two independently vendored
    C backends."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    svorex_pred = (
        SVOREX(C=0.5, kernel="rbf", gamma=0.1, tol=0.002)
        .fit(X_train, y_train)
        .predict(X_test)
    )
    y_expected = np.loadtxt(
        Path(__file__).parent / "data" / "SVOREX" / "predictions_rbf.csv", dtype=int
    )

    npt.assert_equal(svorex_pred, y_expected)
