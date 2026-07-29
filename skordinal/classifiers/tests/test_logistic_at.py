"""Tests for the LogisticAT classifier."""

import inspect

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
from sklearn.exceptions import NotFittedError
from sklearn.utils.class_weight import compute_class_weight

from skordinal.classifiers import LogisticAT
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
    """Create a synthetic 3-class ordinal dataset for behavioural tests."""
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
        ("alpha", -1),
        ("max_iter", 0),
        ("tol", 0),
        ("class_weight", "unknown"),
    ],
)
def test_logistic_at_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Invalid hyperparameter values raise ValueError."""
    classifier = LogisticAT(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("alpha", "strong"),
        ("max_iter", 1.5),
        ("tol", "eps"),
        ("class_weight", 1.0),
    ],
)
def test_logistic_at_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Invalid hyperparameter types raise ValueError."""
    classifier = LogisticAT(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_logistic_at_fit_returns_self(X, y):
    """fit returns self (sklearn contract)."""
    classifier = LogisticAT()
    model = classifier.fit(X, y)
    assert model is classifier


def test_logistic_at_fit_input_validation(X, y):
    """fit rejects mismatched or empty X and y."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = LogisticAT()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_logistic_at_sets_fitted_attributes_after_fit(X, y):
    """fit sets the sklearn fitted attributes."""
    clf = LogisticAT().fit(X, y)

    for attr in [
        "classes_",
        "n_features_in_",
        "coef_",
        "thresholds_",
        "n_iter_",
        "loss_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert isinstance(clf.classes_, np.ndarray) and np.array_equal(
        clf.classes_, np.unique(y)
    )
    assert isinstance(clf.n_features_in_, int) and clf.n_features_in_ == X.shape[1]
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.thresholds_.shape == (len(np.unique(y)) - 1,)
    assert type(clf.n_iter_) is int
    assert type(clf.loss_) is float
    assert np.isfinite(clf.loss_)
    assert clf.loss_ >= 0.0


def test_logistic_at_predict_invalid_input_raises_error(X, y):
    """predict rejects invalid input."""
    classifier = LogisticAT()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_logistic_at_predict_raises_if_not_fitted(X):
    """predict raises NotFittedError before fit."""
    classifier = LogisticAT()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_logistic_at_feature_names_in_when_dataframe(X, y):
    """feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = LogisticAT().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_logistic_at_parameter_constraints_match_init_params():
    """_parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(LogisticAT.__init__).parameters) - {"self"}
    assert set(LogisticAT._parameter_constraints) == init_params


def test_logistic_at_predict_rejects_wrong_n_features(X, y):
    """predict rejects a mismatched n_features."""
    classifier = LogisticAT().fit(X, y)
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
def test_logistic_at_label_roundtrip(labels):
    """fit preserves arbitrary ordinal labels in classes_."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = LogisticAT()
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_logistic_at_deterministic_two_fits(X, y):
    """Two fits on the same data give bit-identical coef_ and thresholds_."""
    clf_a = LogisticAT().fit(X, y)
    clf_b = LogisticAT().fit(X, y)

    np.testing.assert_array_equal(clf_a.coef_, clf_b.coef_)
    np.testing.assert_array_equal(clf_a.thresholds_, clf_b.thresholds_)


def test_logistic_at_thresholds_non_decreasing(ordinal_data):
    """thresholds_ are non-decreasing after fit."""
    X, y = ordinal_data
    clf = LogisticAT().fit(X, y)

    assert np.all(np.diff(clf.thresholds_) >= 0)
    assert set(clf.predict(X)).issubset(set(clf.classes_))


def test_logistic_at_proba_and_cumproba_well_formed(ordinal_data):
    """predict_proba/predict_cumproba are well-formed and match predict."""
    X, y = ordinal_data
    clf = LogisticAT().fit(X, y)
    n_classes = len(clf.classes_)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], n_classes)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    cumproba = clf.predict_cumproba(X)
    assert cumproba.shape == (X.shape[0], n_classes - 1)
    assert np.all(cumproba >= 0.0) and np.all(cumproba <= 1.0)
    assert np.all(np.diff(cumproba, axis=1) >= 0)

    expected_pred = clf.classes_[(cumproba < 0.5).sum(axis=1)]
    np.testing.assert_array_equal(clf.predict(X), expected_pred)


def test_logistic_at_alpha_regularisation_reduces_coef_norm(ordinal_data):
    """Higher alpha yields a coef_ with a smaller L2 norm than alpha=0."""
    X, y = ordinal_data
    clf_noreg = LogisticAT(alpha=0.0).fit(X, y)
    clf_reg = LogisticAT(alpha=10.0).fit(X, y)

    assert np.linalg.norm(clf_reg.coef_) < np.linalg.norm(clf_noreg.coef_)


def test_logistic_at_class_weight_balanced_differs_from_none_and_matches_dict():
    """class_weight='balanced' differs from None and matches a dict."""
    X, y = make_ordinal_classification(
        n_samples=90,
        n_features=4,
        n_classes=3,
        n_informative=4,
        weights=[0.1, 0.3, 0.6],
        random_state=1,
    )

    clf_none = LogisticAT(alpha=1.0).fit(X, y)
    clf_bal = LogisticAT(alpha=1.0, class_weight="balanced").fit(X, y)
    assert not (
        np.allclose(clf_bal.coef_, clf_none.coef_)
        and np.allclose(clf_bal.thresholds_, clf_none.thresholds_)
    )

    weights = compute_class_weight("balanced", classes=clf_bal.classes_, y=y)
    cw_dict = dict(zip(clf_bal.classes_, weights))
    clf_dict = LogisticAT(alpha=1.0, class_weight=cw_dict).fit(X, y)

    np.testing.assert_allclose(clf_dict.coef_, clf_bal.coef_, atol=1e-8)
    np.testing.assert_allclose(clf_dict.thresholds_, clf_bal.thresholds_, atol=1e-8)


def test_logistic_at_binary_ordinal_data():
    """LogisticAT fits and predicts on a 2-class ordinal dataset."""
    X, y = make_ordinal_classification(
        n_samples=60,
        n_features=3,
        n_classes=2,
        n_informative=3,
        noise=0.1,
        random_state=2,
    )
    clf = LogisticAT().fit(X, y)

    assert clf.thresholds_.shape == (1,)
    assert set(clf.predict(X)).issubset(set(clf.classes_))
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def _build_logistic_at_objective_callables(n, p, K, seed):
    """Return (fun, jac, params0) for check_grad on LogisticAT._objective."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y_enc = rng.integers(0, K, size=n)

    clf = LogisticAT(alpha=0.3)
    w = rng.standard_normal(p) * 0.3
    t = np.zeros(K - 1)
    t[0] = -1.0
    if K > 2:
        t[1:] = 0.5
    params = np.concatenate([w, t])
    sample_weight = np.ones(n)

    def fun(params_):
        """Return the scalar objective value for params_."""
        value, _ = clf._objective(params_, X, y_enc, sample_weight, p, K)
        return value

    def jac(params_):
        """Return the gradient vector for params_."""
        _, grad = clf._objective(params_, X, y_enc, sample_weight, p, K)
        return grad

    return fun, jac, params


@pytest.mark.parametrize("K", [3, 5])
def test_logistic_at_objective_gradient_matches_numerical(K):
    """Analytic gradient of _objective matches finite differences."""
    fun, jac, params = _build_logistic_at_objective_callables(n=80, p=3, K=K, seed=42)
    abs_err = scipy.optimize.check_grad(fun, jac, params)
    g_num = scipy.optimize.approx_fprime(params, fun, 1e-5)
    rel_err = abs_err / (1.0 + np.max(np.abs(g_num)))
    assert rel_err < 1e-5


def test_logistic_at_large_magnitude_x_finite_probabilities():
    """Large-magnitude inputs still yield finite, normalised probabilities."""
    X, y = make_ordinal_classification(
        n_samples=60, n_features=4, n_classes=3, n_informative=4, random_state=0
    )
    X = X * 1000.0

    classifier = LogisticAT().fit(X, y)
    proba = classifier.predict_proba(X)

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
