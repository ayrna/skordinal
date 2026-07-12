"""Tests for the KDLOR classifier."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.optimize
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from skordinal.classifiers import KDLOR


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


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", 0),
        ("C", -1),
        ("u", 0),
        ("u", -1e-5),
        ("degree", 0),
        ("tol", 0),
        ("gamma", 0),
        ("gamma", -0.1),
        ("max_iter", 0),
    ],
)
def test_kdlor_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that KDLOR raises ValueError for invalid hyperparameter values."""
    classifier = KDLOR(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("C", "high"),
        ("u", "tight"),
        ("kernel", 42),
        ("gamma", "fast"),
        ("degree", 1.5),
        ("coef0", "x"),
        ("tol", "eps"),
    ],
)
def test_kdlor_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that KDLOR raises ValueError for invalid hyperparameter types."""
    classifier = KDLOR(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_kdlor_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = KDLOR()
    model = classifier.fit(X, y)
    assert model is classifier


def test_kdlor_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = KDLOR()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_kdlor_sets_fitted_attributes_after_fit(X, y):
    """Test that KDLOR exposes fitted attributes aligned with sklearn style."""
    clf = KDLOR().fit(X, y)

    for attr in [
        "classes_",
        "n_features_in_",
        "gamma_",
        "X_fit_",
        "dual_coef_",
        "thresholds_",
        "n_iter_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert isinstance(clf.classes_, np.ndarray) and np.array_equal(
        clf.classes_, np.unique(y)
    )
    assert isinstance(clf.n_features_in_, int) and clf.n_features_in_ == X.shape[1]
    assert isinstance(clf.gamma_, float) and clf.gamma_ > 0.0
    assert clf.X_fit_.shape == X.shape
    assert clf.dual_coef_.shape == (X.shape[0],)
    assert clf.thresholds_.shape == (len(np.unique(y)) - 1,)
    assert isinstance(clf.n_iter_, int) and clf.n_iter_ >= 1


def test_kdlor_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = KDLOR()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_kdlor_predict_raises_if_not_fitted(X):
    """Test that predict raises NotFittedError if called before fit."""
    classifier = KDLOR()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_kdlor_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = KDLOR().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_kdlor_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(KDLOR.__init__).parameters) - {"self"}
    assert set(KDLOR._parameter_constraints) == init_params


def test_kdlor_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = KDLOR().fit(X, y)
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
def test_kdlor_label_roundtrip(labels):
    """Test that KDLOR preserves ordinal labels through fit and predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = KDLOR(kernel="linear")
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_kdlor_deterministic_two_fits(X, y):
    """Two fits on the same data give identical dual_coef_ and thresholds_."""
    clf_a = KDLOR().fit(X, y)
    clf_b = KDLOR().fit(X, y)

    np.testing.assert_array_equal(clf_a.dual_coef_, clf_b.dual_coef_)
    np.testing.assert_array_equal(clf_a.thresholds_, clf_b.thresholds_)


def test_kdlor_predictions_invariant_to_c(X, y):
    """Predictions do not depend on C; dual_coef_ scales linearly with it."""
    ref = KDLOR(C=0.1).fit(X, y)

    for c_val in (1.0, 1000.0):
        clf = KDLOR(C=c_val).fit(X, y)
        np.testing.assert_array_equal(clf.predict(X), ref.predict(X))
        ratio = clf.dual_coef_ / ref.dual_coef_
        np.testing.assert_allclose(ratio, c_val / 0.1, rtol=1e-6)


@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid", "laplacian"])
def test_kdlor_thresholds_non_decreasing_across_kernels(X, y, kernel):
    """thresholds_ are non-decreasing and predict stays within classes_."""
    classifier = KDLOR(kernel=kernel).fit(X, y)

    assert np.all(np.diff(classifier.thresholds_) >= 0)
    assert set(classifier.predict(X)).issubset(set(classifier.classes_))


@pytest.mark.parametrize(
    "gamma, expected_gamma",
    [
        ("auto", lambda X: 1.0 / X.shape[1]),
        ("scale", lambda X: 1.0 / (X.shape[1] * X.var())),
        (0.5, lambda X: 0.5),
    ],
    ids=["auto", "scale", "numeric"],
)
def test_kdlor_gamma_resolves_to_expected_value(X, y, gamma, expected_gamma):
    """gamma_ resolves correctly for 'auto', 'scale', and a numeric value."""
    classifier = KDLOR(gamma=gamma).fit(X, y)
    np.testing.assert_allclose(classifier.gamma_, expected_gamma(X))


def test_kdlor_predict_proba_rows_sum_to_one(X, y):
    """predict_proba rows are non-negative and sum to one."""
    classifier = KDLOR().fit(X, y)
    proba = classifier.predict_proba(X)

    n_classes = len(np.unique(y))
    assert proba.shape == (X.shape[0], n_classes)
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_kdlor_predict_cumproba_shape_and_bounds(X, y):
    """predict_cumproba has shape (n, n_classes - 1), rows non-decreasing."""
    classifier = KDLOR().fit(X, y)
    cumproba = classifier.predict_cumproba(X)

    n_classes = len(np.unique(y))
    assert cumproba.shape == (X.shape[0], n_classes - 1)
    assert np.all(cumproba >= 0.0) and np.all(cumproba <= 1.0)
    assert np.all(np.diff(cumproba, axis=1) >= 0)


def test_kdlor_singular_scatter_raises_linalg_error():
    """A near-singular scatter matrix raises at tiny u, fits at default u."""
    rng = np.random.default_rng(0)
    X_base = rng.standard_normal((3, 2))
    # Duplicate each base pattern, then interleave labels so the resulting
    # kernel matrix collapses to rank 3, making the scatter matrix singular
    X = np.repeat(X_base, 10, axis=0)
    y = np.tile(np.array([0, 1, 2]), 10)

    with pytest.raises(scipy.linalg.LinAlgError, match="increasing the parameter u"):
        KDLOR(u=1e-300).fit(X, y)

    clf = KDLOR().fit(X, y)
    assert np.all(np.isfinite(clf.dual_coef_))


def test_kdlor_convergence_warning_only_at_insufficient_max_iter(X, y):
    """ConvergenceWarning fires at max_iter=1 on hard data, not by default."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        KDLOR().fit(X, y)

    rng = np.random.default_rng(0)
    X_hard = rng.standard_normal((100, 4))
    y_hard = np.tile(np.arange(4), 25)

    with pytest.warns(ConvergenceWarning, match="did not converge"):
        clf = KDLOR(max_iter=1).fit(X_hard, y_hard)
    assert clf.n_iter_ <= clf.max_iter


def test_kdlor_no_convergence_warning_when_solution_is_kkt_optimal(X, y, monkeypatch):
    """No ConvergenceWarning when SLSQP reports failure at a KKT-optimal point."""
    real_minimize = scipy.optimize.minimize

    def force_failure(*args, **kwargs):
        """Return the real SLSQP solution but flag it as unsuccessful."""
        result = real_minimize(*args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", force_failure)
    # kernel="linear" on this fixture gives a boundary QP optimum (a zero
    # dual entry), also exercising the off-support residual term
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        KDLOR(kernel="linear").fit(X, y)


def test_kdlor_kernel_scale_invariance_linear_kernel(X, y):
    """Rescaling X leaves predictions and thresholds_ unchanged (linear kernel)."""
    clf = KDLOR(kernel="linear").fit(X, y)
    clf_scaled = KDLOR(kernel="linear").fit(X * 10.0, y)

    np.testing.assert_array_equal(clf.predict(X), clf_scaled.predict(X * 10.0))
    # Locks in the relative (not absolute) ridge: an absolute ridge would
    # shift thresholds_ under rescaling even though predict stays correct
    np.testing.assert_allclose(clf.thresholds_, clf_scaled.thresholds_, rtol=1e-6)


def test_kdlor_constant_x_gamma_scale_fallback():
    """Constant X triggers the gamma='scale' fallback and predicts class 0."""
    X = np.full((12, 3), 5.0)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    with warnings.catch_warnings():
        # Constant X collapses the thresholds to a single value, the
        # expected degenerate case that raises the ordering RuntimeWarning
        warnings.simplefilter("ignore", RuntimeWarning)
        clf = KDLOR(kernel="rbf", gamma="scale").fit(X, y)

    np.testing.assert_allclose(clf.gamma_, 1.0)
    preds = clf.predict(X)
    np.testing.assert_array_equal(preds, np.zeros(len(y), dtype=preds.dtype))


def test_kdlor_poly_kernel_overflow_raises_value_error():
    """A poly-kernel overflow raises ValueError instead of silent NaN."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((9, 3)) * 1e10
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    with warnings.catch_warnings():
        # Suppress the expected matmul overflow warning; the ValueError
        # raised from the non-finite scatter matrix is what is under test
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="non-finite"):
            KDLOR(kernel="poly", gamma=1e10, degree=5).fit(X, y)


def test_kdlor_large_magnitude_x_finite_probabilities():
    """Large-magnitude inputs still yield finite, normalised probabilities."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4)) * 1000.0
    y = np.tile(np.arange(3), 20)

    classifier = KDLOR().fit(X, y)
    proba = classifier.predict_proba(X)

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
