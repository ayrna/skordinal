"""Tests for the POM classifier."""

import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.class_weight import compute_class_weight

from skordinal.classifiers import POM
from skordinal.datasets import make_ordinal_classification

_ALL_LINKS = ["logit", "probit", "cloglog", "cauchit", "loglog"]


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
        ("link", "bogus"),
        ("solver", "adam"),
        ("class_weight", "unknown"),
    ],
)
def test_pom_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that POM raises ValueError for invalid hyperparameter values."""
    classifier = POM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("link", 42),
        ("alpha", "strong"),
        ("solver", 3),
        ("max_iter", 1.5),
        ("tol", "eps"),
        ("class_weight", 1.0),
    ],
)
def test_pom_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that POM raises ValueError for invalid hyperparameter types."""
    classifier = POM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_pom_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = POM()
    model = classifier.fit(X, y)
    assert model is classifier


def test_pom_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = POM()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_pom_sets_fitted_attributes_after_fit(X, y):
    """Test that POM exposes fitted attributes aligned with sklearn style."""
    clf = POM().fit(X, y)

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


def test_pom_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = POM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_pom_predict_raises_if_not_fitted(X):
    """Test that predict raises NotFittedError if called before fit."""
    classifier = POM()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_pom_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = POM().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_pom_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(POM.__init__).parameters) - {"self"}
    assert set(POM._parameter_constraints) == init_params


def test_pom_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = POM().fit(X, y)
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
def test_pom_label_roundtrip(labels):
    """Test that POM preserves arbitrary ordinal labels through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = POM()
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_pom_deterministic_two_fits(X, y):
    """Two fits on the same data give bit-identical coef_ and thresholds_."""
    clf_a = POM().fit(X, y)
    clf_b = POM().fit(X, y)

    np.testing.assert_array_equal(clf_a.coef_, clf_b.coef_)
    np.testing.assert_array_equal(clf_a.thresholds_, clf_b.thresholds_)


@pytest.mark.parametrize("link", _ALL_LINKS)
def test_pom_link_fit_predict_thresholds_ordered(ordinal_data, link):
    """fit converges without warning; predict is accurate; thresholds valid."""
    X, y = ordinal_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf = POM(link=link).fit(X, y)

    assert set(clf.predict(X)).issubset(set(clf.classes_))
    assert np.all(np.diff(clf.thresholds_) >= 0)
    # the fixture is well-separated 3-class data; every link should fit it
    # accurately, so a wrong predict path (e.g. flipped eta sign) stands out
    assert (clf.predict(X) == y).mean() >= 0.85


def test_pom_alpha_regularisation_reduces_coef_norm(ordinal_data):
    """Higher alpha yields a coef_ with a smaller L2 norm than alpha=0."""
    X, y = ordinal_data
    clf_noreg = POM(alpha=0.0).fit(X, y)
    clf_reg = POM(alpha=10.0).fit(X, y)

    assert np.linalg.norm(clf_reg.coef_) < np.linalg.norm(clf_noreg.coef_)


def test_pom_class_weight_balanced_differs_from_none_and_matches_dict():
    """class_weight='balanced' differs from None and matches a dict."""
    X, y = make_ordinal_classification(
        n_samples=90,
        n_features=4,
        n_classes=3,
        n_informative=4,
        weights=[0.1, 0.3, 0.6],
        random_state=1,
    )

    clf_none = POM(alpha=1.0).fit(X, y)
    clf_bal = POM(alpha=1.0, class_weight="balanced").fit(X, y)
    assert not (
        np.allclose(clf_bal.coef_, clf_none.coef_)
        and np.allclose(clf_bal.thresholds_, clf_none.thresholds_)
    )

    weights = compute_class_weight("balanced", classes=clf_bal.classes_, y=y)
    cw_dict = dict(zip(clf_bal.classes_, weights))
    clf_dict = POM(alpha=1.0, class_weight=cw_dict).fit(X, y)

    np.testing.assert_allclose(clf_dict.coef_, clf_bal.coef_, atol=1e-8)
    np.testing.assert_allclose(clf_dict.thresholds_, clf_bal.thresholds_, atol=1e-8)


def test_pom_proba_and_cumproba_well_formed(ordinal_data):
    """predict_proba/predict_cumproba are well-formed and match predict."""
    X, y = ordinal_data
    clf = POM().fit(X, y)
    n_classes = len(clf.classes_)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], n_classes)
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    cumproba = clf.predict_cumproba(X)
    assert cumproba.shape == (X.shape[0], n_classes - 1)
    assert np.all(cumproba >= 0.0) and np.all(cumproba <= 1.0)
    assert np.all(np.diff(cumproba, axis=1) >= 0)

    expected_pred = clf.classes_[proba.argmax(axis=1)]
    np.testing.assert_array_equal(clf.predict(X), expected_pred)


def test_pom_solvers_reach_same_optimum(ordinal_data):
    """newton-cg and bfgs converge to the same fit as the lbfgs reference."""
    X, y = ordinal_data
    ref = POM(solver="lbfgs", tol=1e-8, max_iter=5000).fit(X, y)

    for solver in ("newton-cg", "bfgs"):
        other = POM(solver=solver, tol=1e-8, max_iter=5000).fit(X, y)
        np.testing.assert_allclose(other.coef_, ref.coef_, atol=1e-3)
        np.testing.assert_allclose(other.thresholds_, ref.thresholds_, atol=1e-3)


def test_pom_convergence_warning_names_true_iteration_count(ordinal_data):
    """POM(max_iter=1).fit warns, naming the true n_iter_ in the message."""
    X, y = ordinal_data
    with pytest.warns(ConvergenceWarning, match="did not converge") as record:
        clf = POM(max_iter=1).fit(X, y)

    message = str(record[0].message)
    assert re.search(rf"stopped after {clf.n_iter_} iterations?\b", message), message


def test_pom_convergence_warning_reports_nit_not_max_iter(ordinal_data, monkeypatch):
    """The warning names the true result.nit, never the max_iter value."""
    X, y = ordinal_data
    real_minimize = scipy.optimize.minimize

    def forced_failure(*args, **kwargs):
        """Run the real optimizer, then force a failed result with nit=7."""
        result = real_minimize(*args, **kwargs)
        result.success = False
        result.nit = 7
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", forced_failure)

    with pytest.warns(ConvergenceWarning, match="did not converge") as record:
        clf = POM(max_iter=1000).fit(X, y)

    message = str(record[0].message)
    assert "7" in message
    assert "1000" not in message
    # loss_ must be set even though the solver did not converge
    assert np.isfinite(clf.loss_)


def test_pom_exact_proba_tie_predicts_lower_class(X, y):
    """An exact predict_proba tie resolves to the lower class."""
    clf = POM(link="logit").fit(X, y)
    # Overwrite the fitted state with round numbers so a probe row lands on
    # an exact floating-point tie, which a real fit cannot guarantee
    clf.n_features_in_ = 1
    clf.coef_ = np.array([1.0])
    clf.thresholds_ = np.array([0.0])
    clf.classes_ = np.array([0, 1])

    probe = np.array([[0.0]])
    proba = clf.predict_proba(probe)
    np.testing.assert_allclose(proba, [[0.5, 0.5]], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(clf.predict(probe), clf.classes_[[0]])


def test_pom_loglog_matches_reversed_cloglog_fit(ordinal_data):
    """POM('loglog') on y matches POM('cloglog') on reversed labels."""
    X, y = ordinal_data
    n_classes = len(np.unique(y))
    y_reversed = (n_classes - 1) - y

    # alpha=0.1 keeps both fits away from the flat near-separable regime; the
    # L2 penalty is sign-symmetric so the reversal identity still holds
    clf_loglog = POM(link="loglog", alpha=0.1, tol=1e-10, max_iter=5000).fit(X, y)
    clf_cloglog = POM(link="cloglog", alpha=0.1, tol=1e-10, max_iter=5000).fit(
        X, y_reversed
    )

    mapped_coef = -clf_cloglog.coef_
    mapped_thresholds = -clf_cloglog.thresholds_[::-1]

    np.testing.assert_allclose(clf_loglog.coef_, mapped_coef, atol=1e-4)
    np.testing.assert_allclose(clf_loglog.thresholds_, mapped_thresholds, atol=1e-4)


def _build_pom_objective_callables(link, n, p, K, seed):
    """Return (fun, jac, params0) for check_grad on POM._objective."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y_enc = rng.integers(0, K, size=n)

    clf = POM(link=link, alpha=0.3)
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


@pytest.mark.parametrize("link", _ALL_LINKS)
@pytest.mark.parametrize("K", [2, 4])
def test_pom_objective_gradient_matches_numerical(link, K):
    """Analytic gradient of _objective matches finite differences."""
    fun, jac, params = _build_pom_objective_callables(link, n=80, p=3, K=K, seed=42)

    abs_err = scipy.optimize.check_grad(fun, jac, params)
    g_num = scipy.optimize.approx_fprime(params, fun, 1e-5)
    rel_err = abs_err / (1.0 + np.max(np.abs(g_num)))

    assert rel_err < 1e-5


def test_pom_large_magnitude_x_finite_probabilities():
    """Large-magnitude inputs still yield finite, normalised probabilities."""
    X, y = make_ordinal_classification(
        n_samples=60, n_features=4, n_classes=3, n_informative=4, random_state=0
    )
    X = X * 1000.0

    # no RuntimeWarning (e.g. log of a non-positive value) may escape fit
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        classifier = POM().fit(X, y)
        proba = classifier.predict_proba(X)

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
