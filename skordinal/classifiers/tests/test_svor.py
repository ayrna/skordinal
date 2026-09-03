"""Tests for the SVOR classifier."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skordinal.classifiers import SVOR
from skordinal.utils._testing import make_balance_scale_split

# The frozen goldens are per formulation, not per class
PREDICTIONS_DIR = {
    "explicit": Path(__file__).parent / "data" / "SVOREX",
    "implicit": Path(__file__).parent / "data" / "SVORIM",
}

CONSTRAINTS = ["explicit", "implicit"]


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.mark.parametrize("constraints", CONSTRAINTS)
@pytest.mark.parametrize(
    "kernel",
    [
        "rbf",
        "linear",
        "poly",
    ],
)
def test_svor_predict_matches_expected(kernel, constraints):
    """Test that predictions match expected values."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    classifier = SVOR(
        C=0.5, kernel=kernel, degree=4, tol=0.002, gamma=0.1, constraints=constraints
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_expected = np.loadtxt(
        PREDICTIONS_DIR[constraints] / f"predictions_{kernel}.csv", dtype=int
    )

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
        ("constraints", "unknown"),
    ],
)
def test_svor_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that SVOR raises ValueError for invalid values of hyperparameters."""
    classifier = SVOR(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", "high"),
        ("kernel", 5),
        ("degree", 2.5),
        ("tol", "tight"),
        ("constraints", 5),
    ],
)
def test_svor_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that SVOR raises ValueError for invalid types of hyperparameters."""
    classifier = SVOR(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_svor_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = SVOR()
    for X_bad, y_bad in ((X, y_invalid), ([], y), (X, []), (X_invalid, y)):
        with pytest.raises(ValueError):
            classifier.fit(X_bad, y_bad)


def test_svor_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = SVOR()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_svor_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = SVOR()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_svor_fit_returns_self(X, y):
    """Test that fit returns the estimator itself."""
    classifier = SVOR()

    assert classifier.fit(X, y) is classifier


def test_svor_predict_raises_if_not_fitted(X):
    """Test that predict and predict_projection require a fitted estimator."""
    classifier = SVOR()

    with pytest.raises(NotFittedError):
        classifier.predict(X)
    with pytest.raises(NotFittedError):
        classifier.predict_projection(X)


@pytest.mark.parametrize("constraints", CONSTRAINTS)
def test_svor_sets_classes_and_n_features_in_after_fit(X, y, constraints):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = SVOR(constraints=constraints).fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_svor_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = SVOR().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_svor_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(SVOR.__init__).parameters) - {"self"}
    assert set(SVOR._parameter_constraints) == init_params


def test_svor_predict_rejects_wrong_n_features(X, y):
    """predict and predict_projection reject a mismatched n_features."""
    classifier = SVOR().fit(X, y)
    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])
    with pytest.raises(ValueError):
        classifier.predict_projection(X[:, :-1])


@pytest.mark.parametrize("constraints", CONSTRAINTS)
@pytest.mark.parametrize(
    "labels",
    [
        [1, 2, 3],  # Standard 1-indexed
        [0, 1, 2],  # 0-indexed
        [-1, 0, 1],  # Negative labels
        [3, 5, 7],  # Non-contiguous with gaps
    ],
)
def test_svor_label_roundtrip(labels, constraints):
    """Test that SVOR preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = SVOR(C=0.5, kernel="linear", constraints=constraints)
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


@pytest.mark.parametrize("constraints", CONSTRAINTS)
@pytest.mark.parametrize(
    "gamma, expected_gamma",
    [
        ("auto", lambda X: 1.0 / X.shape[1]),
        ("scale", lambda X: 1.0 / (X.shape[1] * X.var())),
    ],
    ids=["auto", "scale"],
)
def test_svor_gamma_string_resolves_like_numeric(gamma, expected_gamma, constraints):
    """A string gamma predicts the same as the numeric value it resolves to."""
    X_train, X_test, y_train, _ = make_balance_scale_split()

    resolved = SVOR(gamma=gamma, constraints=constraints).fit(X_train, y_train)
    explicit = SVOR(gamma=expected_gamma(X_train), constraints=constraints).fit(
        X_train, y_train
    )

    np.testing.assert_array_equal(resolved.predict(X_test), explicit.predict(X_test))


def test_svor_gamma_scale_zero_variance(X, y):
    """Test that gamma='scale' handles zero-variance input without dividing by zero."""
    X_constant = np.ones_like(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        classifier = SVOR(gamma="scale", kernel="rbf").fit(X_constant, y)

    assert set(classifier.predict(X_constant)).issubset(set(classifier.classes_))


@pytest.mark.parametrize("constraints", CONSTRAINTS)
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_svor_projection_well_formed(kernel, constraints):
    """predict_projection is well-formed and reproduces predict via thresholds_."""
    X_train, X_test, y_train, _ = make_balance_scale_split()
    clf = SVOR(kernel=kernel, constraints=constraints).fit(X_train, y_train)

    projection = clf.predict_projection(X_test)
    assert projection.shape == (len(X_test),)
    assert np.isfinite(projection).all()

    assert clf.thresholds_.shape == (clf.classes_.size - 1,)
    assert np.all(np.diff(clf.thresholds_) >= 0)

    # Recomputing predict cross-checks what the C actually returned
    n_exceeded = (projection[:, np.newaxis] > clf.thresholds_[np.newaxis, :]).sum(
        axis=1
    )
    npt.assert_array_equal(clf.predict(X_test), clf.classes_[n_exceeded])


# The poly goldens happen to coincide at these hyperparameters, so they cannot
# separate the two formulations on their own
@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_svor_constraints_change_predictions(kernel):
    """The two formulations solve different problems and must not coincide."""
    X_train, X_test, y_train, _ = make_balance_scale_split()
    kwargs = {"C": 0.5, "kernel": kernel, "degree": 4, "tol": 0.002, "gamma": 0.1}

    explicit = SVOR(constraints="explicit", **kwargs).fit(X_train, y_train)
    implicit = SVOR(constraints="implicit", **kwargs).fit(X_train, y_train)

    assert not np.array_equal(explicit.predict(X_test), implicit.predict(X_test))
    assert not np.array_equal(explicit.thresholds_, implicit.thresholds_)


def test_svor_modes_agree_when_binary():
    """With exactly two classes there is a single threshold and no ordering
    constraint applies, so the explicit and implicit formulations solve the
    same QP and should agree up to solver tolerance."""
    X_train, X_test, y_train, y_test = make_balance_scale_split()
    mask_train = y_train != 1
    mask_test = y_test != 1

    kwargs = {"C": 0.5, "kernel": "rbf", "gamma": 0.1, "tol": 0.001}
    predictions = [
        SVOR(constraints=constraints, **kwargs)
        .fit(X_train[mask_train], y_train[mask_train])
        .predict(X_test[mask_test])
        for constraints in CONSTRAINTS
    ]

    npt.assert_array_equal(*predictions)


def test_svor_modes_coexist_in_the_same_process():
    """Fitting both formulations in one process must not corrupt either one:
    they share one Alphas layout and one allocator, and only one of them
    allocates the per-threshold arrays."""
    X_train, X_test, y_train, _ = make_balance_scale_split()
    kwargs = {"C": 0.5, "kernel": "rbf", "gamma": 0.1, "tol": 0.002}

    # Interleave so each mode allocates while the other holds live state
    implicit = SVOR(constraints="implicit", **kwargs).fit(X_train, y_train)
    explicit = SVOR(constraints="explicit", **kwargs).fit(X_train, y_train)
    implicit_pred = implicit.predict(X_test)
    explicit_pred = explicit.predict(X_test)

    for constraints, y_pred in (
        ("explicit", explicit_pred),
        ("implicit", implicit_pred),
    ):
        npt.assert_equal(
            y_pred,
            np.loadtxt(PREDICTIONS_DIR[constraints] / "predictions_rbf.csv", dtype=int),
        )
