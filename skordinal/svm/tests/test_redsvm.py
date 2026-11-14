"""Tests for the REDSVM classifier."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import get_tags

from skordinal.svm import REDSVM
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
    """predict and predict_projection reject a mismatched n_features."""
    classifier = REDSVM().fit(X, y)
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


@pytest.mark.parametrize(
    "gamma, expected_gamma",
    [
        ("auto", lambda X: 1.0 / X.shape[1]),
        ("scale", lambda X: 1.0 / (X.shape[1] * X.var())),
    ],
    ids=["auto", "scale"],
)
def test_redsvm_gamma_string_resolves_like_numeric(gamma, expected_gamma):
    """A string gamma predicts the same as the numeric value it resolves to."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    resolved = REDSVM(gamma=gamma).fit(X_train, y_train)
    explicit = REDSVM(gamma=expected_gamma(X_train)).fit(X_train, y_train)

    np.testing.assert_array_equal(resolved.predict(X_test), explicit.predict(X_test))


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_redsvm_projection_well_formed(kernel):
    """predict_projection is well-formed and reproduces predict via thresholds_."""
    X_train, X_test, y_train, _ = make_balance_scale_split()
    clf = REDSVM(kernel=kernel).fit(X_train, y_train)

    projection = clf.predict_projection(X_test)
    assert projection.shape == (len(X_test),)
    assert np.isfinite(projection).all()

    assert clf.thresholds_.shape == (clf.classes_.size - 1,)
    assert np.all(np.diff(clf.thresholds_) >= 0)

    # recomputing predict cross-checks what the C++ actually returned
    n_reached = (projection[:, np.newaxis] >= clf.thresholds_[np.newaxis, :]).sum(
        axis=1
    )
    npt.assert_array_equal(clf.predict(X_test), clf.classes_[n_reached])


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_redsvm_precomputed_kernel_matches_explicit_kernel(kernel):
    """A standard Gram matrix reproduces native kernel labels and projections."""
    X = np.arange(36, dtype=float).reshape(18, 2) / 12.0 - 1.5
    y = np.repeat([-3, 2, 9], 6)
    order = np.random.default_rng(42).permutation(len(X))
    X, y = X[order], y[order]
    X_test = X[::3] + 0.1
    params = {"gamma": 0.3, "degree": 2, "coef0": 0.4}
    K = pairwise_kernels(X, metric=kernel, filter_params=True, **params)
    K_test = pairwise_kernels(X_test, X, metric=kernel, filter_params=True, **params)
    expected = REDSVM(kernel=kernel, tol=1e-6, **params).fit(X, y)
    clf = REDSVM(kernel="precomputed", tol=1e-6).fit(K, y)

    npt.assert_array_equal(clf.predict(K_test), expected.predict(X_test))
    npt.assert_allclose(
        clf.predict_projection(K_test), expected.predict_projection(X_test), atol=1e-5
    )
    with pytest.raises(ValueError, match="features"):
        clf.predict(K_test[:, :-1])


def test_redsvm_precomputed_kernel_requires_square_training_matrix():
    """Reject rectangular training matrices before entering the native bridge."""
    with pytest.raises(ValueError, match="square"):
        REDSVM(kernel="precomputed").fit(np.ones((6, 5)), [0, 0, 1, 1, 2, 2])


def test_redsvm_precomputed_kernel_cross_validation_slices_training_columns():
    """The pairwise tag makes cross validation select matching Gram submatrices."""
    X = np.arange(36, dtype=float).reshape(18, 2) / 12.0 - 1.5
    y = np.repeat([-3, 2, 9], 6)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    K = pairwise_kernels(X, metric="rbf", gamma=0.3)

    expected = cross_val_score(REDSVM(gamma=0.3), X, y, cv=splitter)
    actual = cross_val_score(REDSVM(kernel="precomputed"), K, y, cv=splitter)

    npt.assert_array_equal(actual, expected)
    assert get_tags(REDSVM(kernel="precomputed")).input_tags.pairwise
    assert not get_tags(REDSVM()).input_tags.pairwise


@pytest.mark.parametrize("kernel", ["linear", "rbf", "precomputed"])
def test_redsvm_legacy_fitted_state_retains_input_convention(kernel):
    """Previously serialized models keep their original prediction input format."""
    X = np.arange(18, dtype=float).reshape(9, 2) / 10.0
    y = np.repeat([-3, 2, 9], 3)
    inputs = pairwise_kernels(X, metric="rbf") if kernel == "precomputed" else X
    clf = REDSVM(kernel=kernel).fit(inputs, y)
    expected = clf.predict(inputs)
    expected_projection = clf.predict_projection(inputs)
    del clf._precomputed_
    if kernel == "precomputed":
        # Earlier models counted the caller-supplied serial column as a feature.
        clf.n_features_in_ += 1
        inputs = np.column_stack((np.zeros(len(inputs)), inputs))

    npt.assert_array_equal(clf.predict(inputs), expected)
    npt.assert_array_equal(clf.predict_projection(inputs), expected_projection)
