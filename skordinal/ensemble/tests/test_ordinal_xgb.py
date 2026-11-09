"""Tests for the OrdinalXGBClassifier classifier."""

import inspect
import json
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import scipy.special
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils import get_tags
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from threadpoolctl import threadpool_limits

from skordinal.datasets import make_ordinal_classification
from skordinal.ensemble import OrdinalXGBClassifier

xgb = pytest.importorskip("xgboost")

THRESHOLD_UPDATES = ["fixed", "refit", "input_median", "input_mean", "latent_median"]


def test_predict_repeated_thresholds_and_adjacent_floats(monkeypatch):
    """Hard predictions preserve lower ties with repeated thresholds."""
    classifier = OrdinalXGBClassifier()
    classifier.classes_ = np.array([-10, 20, 50, 80])
    classifier.thresholds_ = np.array([-0.5, 0.5, 0.5])
    classifier.n_features_in_ = 1
    projection = np.array(
        [-1e100, -0.5, np.nextafter(-0.5, np.inf), 0.5, np.nextafter(0.5, np.inf), 1e100]
    )
    monkeypatch.setattr(classifier, "_project", lambda X: projection)

    np.testing.assert_array_equal(
        classifier.predict(np.zeros((len(projection), 1))), [-10, -10, 20, 20, 80, 80]
    )


@pytest.fixture(autouse=True)
def single_openmp_thread():
    """Run XGBoost on one OpenMP thread, as all cores slow it down on small data."""
    with threadpool_limits(limits=1, user_api="openmp"):
        yield


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
    """Create a synthetic 4-class ordinal dataset for behavioural tests."""
    return make_ordinal_classification(
        n_samples=200,
        n_features=4,
        n_classes=4,
        n_informative=4,
        noise=0.5,
        random_state=0,
    )


@pytest.fixture
def missing_data(ordinal_data):
    """Create the 4-class dataset with about 10% of its entries missing."""
    X, y = ordinal_data
    X = X.copy()
    X[np.random.default_rng(0).random(X.shape) < 0.1] = np.nan
    return X, y


@pytest.fixture
def captured_objective(monkeypatch):
    """Record the custom objective that fit passes to xgboost.train."""
    captured = {}
    train = xgb.train

    def spy(*args, **kwargs):
        """Store the objective and train as xgboost does."""
        captured["obj"] = kwargs["obj"]
        return train(*args, **kwargs)

    monkeypatch.setattr(xgb, "train", spy)
    return captured


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_estimators", 0),
        ("learning_rate", 0),
        ("max_depth", -1),
        ("max_leaves", -1),
        ("max_bin", 1),
        ("grow_policy", "leafwise"),
        ("tree_method", "gpu_hist"),
        ("verbosity", 4),
        ("gamma", -1.0),
        ("min_child_weight", -1.0),
        ("subsample", 0),
        ("sampling_method", "random"),
        ("colsample_bytree", 1.5),
        ("colsample_bynode", 0),
        ("reg_lambda", -1.0),
        ("num_parallel_tree", 0),
        ("importance_type", "split"),
        ("threshold_update", "quantile"),
        ("threshold_solver", "newton"),
        ("threshold_learning_rate", 0),
        ("class_weight", "unknown"),
    ],
)
def test_ordinal_xgb_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Invalid hyperparameter values raise ValueError."""
    classifier = OrdinalXGBClassifier(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_estimators", 1.5),
        ("learning_rate", "fast"),
        ("max_depth", 2.5),
        ("verbosity", "loud"),
        ("n_jobs", 1.5),
        ("gamma", "none"),
        ("random_state", "seed"),
        ("missing", "nan"),
        ("monotone_constraints", 1),
        ("interaction_constraints", 1),
        ("device", 0),
        ("feature_weights", 1.0),
        ("threshold_update", 1),
        ("threshold_solver", True),
        ("threshold_learning_rate", "same"),
        ("class_weight", 1.0),
    ],
)
def test_ordinal_xgb_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Invalid hyperparameter types raise ValueError."""
    classifier = OrdinalXGBClassifier(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_ordinal_xgb_fit_returns_self(X, y):
    """fit returns self (sklearn contract)."""
    classifier = OrdinalXGBClassifier()
    model = classifier.fit(X, y)
    assert model is classifier


def test_ordinal_xgb_fit_input_validation(X, y):
    """fit rejects mismatched or empty X and y."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = OrdinalXGBClassifier()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_ordinal_xgb_sets_fitted_attributes_after_fit(X, y):
    """fit sets the sklearn fitted attributes."""
    clf = OrdinalXGBClassifier().fit(X, y)

    for attr in [
        "classes_",
        "n_features_in_",
        "thresholds_",
        "booster_",
        "feature_importances_",
    ]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert isinstance(clf.classes_, np.ndarray) and np.array_equal(
        clf.classes_, np.unique(y)
    )
    assert isinstance(clf.n_features_in_, int) and clf.n_features_in_ == X.shape[1]
    assert clf.thresholds_.shape == (len(np.unique(y)) - 1,)
    assert clf.booster_.num_boosted_rounds() == clf.n_estimators
    assert clf.feature_importances_.shape == (X.shape[1],)


def test_ordinal_xgb_predict_invalid_input_raises_error(X, y):
    """predict rejects invalid input."""
    classifier = OrdinalXGBClassifier()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_ordinal_xgb_predict_raises_if_not_fitted(X):
    """Fitted methods and feature_importances_ raise NotFittedError before fit."""
    classifier = OrdinalXGBClassifier()
    for method in [
        classifier.predict,
        classifier.predict_proba,
        classifier.predict_cumproba,
        classifier.predict_projection,
    ]:
        with pytest.raises(NotFittedError):
            method(X)
    with pytest.raises(NotFittedError):
        classifier.get_booster()
    with pytest.raises(NotFittedError):
        classifier.feature_importances_


def test_ordinal_xgb_feature_names_in_when_dataframe(X, y):
    """feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = OrdinalXGBClassifier().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_ordinal_xgb_parameter_constraints_match_init_params():
    """_parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(OrdinalXGBClassifier.__init__).parameters)
    assert set(OrdinalXGBClassifier._parameter_constraints) == init_params - {"self"}


def test_ordinal_xgb_predict_rejects_wrong_n_features(X, y):
    """predict and predict_projection reject a mismatched n_features."""
    classifier = OrdinalXGBClassifier().fit(X, y)
    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])
    with pytest.raises(ValueError):
        classifier.predict_projection(X[:, :-1])


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([1, 2, 3], id="one-indexed"),
        pytest.param([0, 1, 2], id="zero-indexed"),
        pytest.param([-1, 0, 1], id="negative"),
        pytest.param([3, 5, 7], id="gaps"),
    ],
)
def test_ordinal_xgb_label_roundtrip(labels):
    """fit preserves arbitrary ordinal labels in classes_."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = OrdinalXGBClassifier()
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_ordinal_xgb_deterministic_two_fits(X, y):
    """Two fits on the same data give bit-identical projections and thresholds_."""
    clf_a = OrdinalXGBClassifier().fit(X, y)
    clf_b = OrdinalXGBClassifier().fit(X, y)

    np.testing.assert_array_equal(
        clf_a.predict_projection(X), clf_b.predict_projection(X)
    )
    np.testing.assert_array_equal(clf_a.thresholds_, clf_b.thresholds_)


def test_ordinal_xgb_thresholds_non_decreasing():
    """Out-of-order thresholds are stored sorted with predict unchanged."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (300, 2))
    y = np.digitize(np.abs(X[:, 0]), [0.25, 0.5, 0.75])
    clf = OrdinalXGBClassifier(n_estimators=20, threshold_update="input_median")
    clf.fit(X, y)

    centers = np.array([np.median(X[y == k], axis=0) for k in clf.classes_])
    f = clf.predict_projection(centers)
    unsorted = (f[:-1] + f[1:]) / 2
    assert np.any(np.diff(unsorted) < 0)

    np.testing.assert_allclose(clf.thresholds_, np.sort(unsorted))
    projection = clf.predict_projection(X)
    n_exceeded = (projection[:, None] > unsorted[None, :]).sum(axis=1)
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[n_exceeded])


def test_ordinal_xgb_proba_and_cumproba_well_formed(ordinal_data):
    """predict_proba/predict_cumproba are well-formed and match predict."""
    X, y = ordinal_data
    clf = OrdinalXGBClassifier().fit(X, y)
    n_classes = len(clf.classes_)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], n_classes)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    cumproba = clf.predict_cumproba(X)
    assert cumproba.shape == (X.shape[0], n_classes - 1)
    assert np.all(np.diff(cumproba, axis=1) >= 0)
    np.testing.assert_allclose(np.cumsum(proba, axis=1)[:, :-1], cumproba)
    f = clf.predict_projection(X)
    expected_cumproba = scipy.special.expit(clf.thresholds_[None, :] - f[:, None])
    np.testing.assert_allclose(cumproba, expected_cumproba, atol=1e-12)

    expected_pred = clf.classes_[(cumproba < 0.5).sum(axis=1)]
    np.testing.assert_array_equal(clf.predict(X), expected_pred)


def test_ordinal_xgb_exact_proba_tie_predicts_lower_class(X, y):
    """An exact predict_proba tie resolves to the lower class."""
    clf = OrdinalXGBClassifier(n_estimators=5).fit(X[y < 2], y[y < 2])
    probe = X[:1]
    # Put the threshold on the probe's projection, which no real fit guarantees
    clf.thresholds_ = clf.predict_projection(probe)

    proba = clf.predict_proba(probe)
    np.testing.assert_allclose(proba, [[0.5, 0.5]], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(clf.predict(probe), clf.classes_[[0]])


def test_ordinal_xgb_class_weight_balanced_differs_from_none_and_matches_dict():
    """class_weight='balanced' differs from None and matches a dict."""
    X, y = make_ordinal_classification(
        n_samples=90,
        n_features=4,
        n_classes=3,
        n_informative=4,
        weights=[0.1, 0.3, 0.6],
        random_state=1,
    )

    clf_none = OrdinalXGBClassifier(n_estimators=5).fit(X, y)
    clf_bal = OrdinalXGBClassifier(n_estimators=5, class_weight="balanced").fit(X, y)
    assert not (
        np.allclose(clf_bal.predict_projection(X), clf_none.predict_projection(X))
        and np.allclose(clf_bal.thresholds_, clf_none.thresholds_)
    )

    weights = compute_class_weight("balanced", classes=clf_bal.classes_, y=y)
    cw_dict = dict(zip(clf_bal.classes_, weights))
    clf_dict = OrdinalXGBClassifier(n_estimators=5, class_weight=cw_dict).fit(X, y)

    np.testing.assert_allclose(
        clf_dict.predict_projection(X), clf_bal.predict_projection(X)
    )
    np.testing.assert_allclose(clf_dict.thresholds_, clf_bal.thresholds_)


@pytest.mark.parametrize("K", [3, 5])
def test_ordinal_xgb_objective_gradient_matches_numerical(K, captured_objective):
    """The objective's gradient and Hessian match finite differences."""
    rng = np.random.default_rng(42)
    n = 40
    X = rng.standard_normal((n, 2))
    y = np.arange(n) % K
    class_weight = dict(enumerate(rng.uniform(0.5, 2.0, size=K)))
    clf = OrdinalXGBClassifier(
        n_estimators=1, threshold_update="fixed", class_weight=class_weight
    ).fit(X, y)
    objective = captured_objective["obj"]
    sample_weight = compute_sample_weight(class_weight, y)
    s = np.where(y[:, None] <= np.arange(K - 1), 1.0, -1.0)
    f = rng.standard_normal(n)

    def loss(f_):
        """Return the weighted all-threshold loss at f_."""
        margin = s * (clf.thresholds_[None, :] - f_[:, None])
        return (np.logaddexp(0.0, -margin).sum(axis=1) * sample_weight).sum()

    def gradient(f_):
        """Return the objective's gradient at f_."""
        return objective(f_, None)[0]

    grad, hess = objective(f, None)
    grad_num = scipy.optimize.approx_fprime(f, loss, 1e-6)
    np.testing.assert_allclose(grad, grad_num, rtol=1e-5, atol=1e-6)
    hess_num = np.diag(scipy.optimize.approx_fprime(f, gradient, 1e-6))
    np.testing.assert_allclose(hess, hess_num, rtol=1e-4, atol=1e-6)


def test_ordinal_xgb_objective_hessian_finite_when_saturated(captured_objective):
    """The objective's Hessian stays finite when the margins are large."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])
    OrdinalXGBClassifier(n_estimators=1, threshold_update="fixed").fit(X, y)

    grad, hess = captured_objective["obj"](np.array([-800.0, 0.0, 800.0]), None)

    assert np.isfinite(grad).all() and np.isfinite(hess).all()
    assert np.all(hess >= 0.0)


def test_ordinal_xgb_convergence_warning(ordinal_data, monkeypatch):
    """Only an L-BFGS refit that stops early warns with ConvergenceWarning."""
    X, y = ordinal_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        OrdinalXGBClassifier(threshold_solver="lbfgs").fit(X, y)

    minimize = scipy.optimize.minimize

    def minimize_one_iteration(*args, **kwargs):
        """Run scipy's minimize capped at one iteration."""
        kwargs["options"] = {**kwargs["options"], "maxiter": 1}
        return minimize(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", minimize_one_iteration)
    with pytest.warns(ConvergenceWarning, match="failed to converge"):
        OrdinalXGBClassifier(threshold_solver="lbfgs").fit(X, y)


def test_ordinal_xgb_projection_well_formed(ordinal_data):
    """predict_projection is well-formed and consistent with predict."""
    X, y = ordinal_data
    clf = OrdinalXGBClassifier().fit(X, y)

    projection = clf.predict_projection(X)
    assert projection.shape == (len(X),)
    assert np.isfinite(projection).all()

    n_exceeded = (projection[:, np.newaxis] > clf.thresholds_[np.newaxis, :]).sum(
        axis=1
    )
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[n_exceeded])


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="defaults"),
        pytest.param(
            dict(subsample=0.7, colsample_bytree=0.8, random_state=1), id="subsample"
        ),
    ],
)
def test_ordinal_xgb_binary_matches_xgboost_logistic(params):
    """With two classes and fixed thresholds it is XGBoost's logistic booster."""
    X, y = make_ordinal_classification(
        n_samples=300, n_features=5, n_classes=2, weights=[0.3, 0.7], random_state=0
    )
    clf = OrdinalXGBClassifier(n_estimators=50, threshold_update="fixed", **params)
    clf.fit(X, y)

    # The custom objective starts at f = 0.5, the logistic one at base_score
    start = scipy.special.expit(0.5 - clf.thresholds_[0])
    reference = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=50, base_score=start, **params
    ).fit(X, y)

    np.testing.assert_allclose(
        clf.predict_proba(X), reference.predict_proba(X), atol=1e-6
    )
    np.testing.assert_array_equal(clf.predict(X), reference.predict(X))


def test_ordinal_xgb_booster_params_reach_xgboost(X, y):
    """The booster hyperparameters reach the booster configuration."""
    clf = OrdinalXGBClassifier(
        max_depth=2,
        max_leaves=4,
        max_bin=64,
        grow_policy="lossguide",
        learning_rate=0.05,
        tree_method="approx",
        n_jobs=2,
        gamma=0.5,
        min_child_weight=2.0,
        max_delta_step=1.0,
        subsample=0.9,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        colsample_bynode=0.6,
        reg_alpha=0.3,
        reg_lambda=2.0,
        num_parallel_tree=2,
        interaction_constraints="[[0, 1]]",
    ).fit(X, y)

    config = json.loads(clf.booster_.save_config())["learner"]
    train_param = config["gradient_booster"]["tree_train_param"]
    expected = {
        "max_depth": 2,
        "max_leaves": 4,
        "max_bin": 64,
        "learning_rate": 0.05,
        "min_split_loss": 0.5,
        "min_child_weight": 2.0,
        "max_delta_step": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.7,
        "colsample_bynode": 0.6,
        "alpha": 0.3,
        "lambda": 2.0,
    }
    for name, value in expected.items():
        assert float(train_param[name]) == pytest.approx(value), name
    assert train_param["grow_policy"] == "lossguide"
    assert train_param["interaction_constraints"] == "[[0, 1]]"
    assert config["gradient_booster"]["gbtree_model_param"]["num_parallel_tree"] == "2"
    assert config["gradient_booster"]["gbtree_train_param"]["tree_method"] == "approx"
    assert int(config["generic_param"]["nthread"]) == 2


@pytest.mark.parametrize(
    "method",
    ["fit", "predict", "predict_proba", "predict_cumproba", "predict_projection"],
)
def test_ordinal_xgb_scopes_verbosity(X, y, monkeypatch, method):
    """Each method runs XGBoost with its verbosity and keeps the global one."""
    clf = OrdinalXGBClassifier(n_estimators=1, verbosity=0).fit(X, y)
    train, inplace_predict = xgb.train, xgb.Booster.inplace_predict
    verbosities = []

    def recording_train(*args, **kwargs):
        """Record XGBoost's global verbosity, then train."""
        verbosities.append(xgb.get_config()["verbosity"])
        return train(*args, **kwargs)

    def recording_inplace_predict(*args, **kwargs):
        """Record XGBoost's global verbosity, then predict."""
        verbosities.append(xgb.get_config()["verbosity"])
        return inplace_predict(*args, **kwargs)

    monkeypatch.setattr(xgb, "train", recording_train)
    monkeypatch.setattr(xgb.Booster, "inplace_predict", recording_inplace_predict)
    args = (X, y) if method == "fit" else (X,)
    with xgb.config_context(verbosity=1):
        getattr(clf, method)(*args)
        assert xgb.get_config()["verbosity"] == 1
    assert verbosities and set(verbosities) == {0}


@pytest.mark.parametrize(
    "make_random_state",
    [
        pytest.param(lambda: None, id="None"),
        pytest.param(lambda: 3, id="int"),
        pytest.param(lambda: np.random.RandomState(0), id="RandomState"),
        pytest.param(lambda: np.random.default_rng(0), id="Generator"),
    ],
)
def test_ordinal_xgb_random_state_seeds_booster_as_xgboost(X, y, make_random_state):
    """random_state gives the booster the seed that XGBClassifier would."""
    clf = OrdinalXGBClassifier(n_estimators=1, random_state=make_random_state())
    clf.fit(X, y)

    reference = xgb.XGBClassifier(random_state=make_random_state())
    expected_seed = reference.get_xgb_params()["random_state"]
    seed = json.loads(clf.booster_.save_config())["learner"]["generic_param"]["seed"]
    # XGBoost uses seed 0 when none is given
    assert int(seed) == (0 if expected_seed is None else expected_seed)


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_ordinal_xgb_fixed_keeps_prior_thresholds(ordinal_data, class_weight):
    """threshold_update='fixed' keeps the unweighted cumulative prior logits."""
    X, y = ordinal_data
    clf = OrdinalXGBClassifier(threshold_update="fixed", class_weight=class_weight)
    clf.fit(X, y)

    prior = np.bincount(y)[:-1].cumsum() / len(y)
    np.testing.assert_allclose(clf.thresholds_, scipy.special.logit(prior))


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_ordinal_xgb_lbfgs_refit_minimises_weighted_loss(class_weight):
    """The L-BFGS refit leaves a zero weighted threshold gradient."""
    X, y = make_ordinal_classification(
        n_samples=200, n_classes=4, weights=[0.1, 0.2, 0.3, 0.4], random_state=0
    )
    clf = OrdinalXGBClassifier(
        n_estimators=5, threshold_solver="lbfgs", class_weight=class_weight
    ).fit(X, y)

    f = clf.predict_projection(X)
    weights = compute_sample_weight(class_weight, y)
    lower = np.arange(len(clf.thresholds_))[None, :] < y[:, None]
    sigma = scipy.special.expit(f[:, None] - clf.thresholds_)
    grad = (weights[:, None] * (lower - sigma)).sum(axis=0)

    np.testing.assert_allclose(grad / len(y), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "threshold_learning_rate, class_weight",
    [
        pytest.param(None, None, id="learning_rate"),
        pytest.param(0.2, "balanced", id="threshold_learning_rate-balanced"),
    ],
)
def test_ordinal_xgb_sgd_refit_steps_once_per_sample(
    threshold_learning_rate, class_weight
):
    """The SGD refit takes one weighted step per sample, in row order."""
    X, y = make_ordinal_classification(
        n_samples=200, n_classes=4, weights=[0.1, 0.2, 0.3, 0.4], random_state=0
    )
    learning_rate = 0.05
    clf = OrdinalXGBClassifier(
        n_estimators=10,
        learning_rate=learning_rate,
        threshold_solver="sgd",
        threshold_learning_rate=threshold_learning_rate,
        class_weight=class_weight,
    ).fit(X, y)

    step = threshold_learning_rate or learning_rate
    f = clf.predict_projection(X)
    weights = compute_sample_weight(class_weight, y)
    expected = scipy.special.logit(np.bincount(y)[:-1].cumsum() / len(y))
    for f_i, y_i, w_i in zip(f, y, weights):
        for k in range(len(expected)):
            if k < y_i:
                expected[k] -= step * w_i / (1.0 + np.exp(f_i - expected[k]))
            else:
                expected[k] += step * w_i / (1.0 + np.exp(expected[k] - f_i))

    np.testing.assert_allclose(clf.thresholds_, np.sort(expected))


@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize(
    "threshold_update, center",
    [("input_median", np.median), ("input_mean", np.mean)],
)
def test_ordinal_xgb_input_updates_bisect_projected_class_centers(
    ordinal_data, threshold_update, center, class_weight
):
    """The input updates bisect the projections of consecutive class centers."""
    X, y = ordinal_data
    clf = OrdinalXGBClassifier(
        threshold_update=threshold_update, class_weight=class_weight
    ).fit(X, y)

    centers = np.array([center(X[y == k], axis=0) for k in clf.classes_])
    f = clf.predict_projection(centers)
    np.testing.assert_allclose(clf.thresholds_, np.sort((f[:-1] + f[1:]) / 2))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "threshold_update, center",
    [("input_median", np.nanmedian), ("input_mean", np.nanmean)],
)
def test_ordinal_xgb_input_updates_skip_missing_values(
    missing_data, threshold_update, center
):
    """The input updates skip missing entries and keep a column missing in a class."""
    X, y = missing_data
    X = X.copy()
    X[y == 0, 0] = np.nan
    clf = OrdinalXGBClassifier(threshold_update=threshold_update).fit(X, y)

    centers = np.array([center(X[y == k], axis=0) for k in clf.classes_])
    assert np.isnan(centers[0, 0])
    f = clf.predict_projection(centers)
    np.testing.assert_allclose(clf.thresholds_, np.sort((f[:-1] + f[1:]) / 2))


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_ordinal_xgb_latent_median_bisects_class_median_projections(
    ordinal_data, class_weight
):
    """threshold_update='latent_median' bisects consecutive class median projections."""
    X, y = ordinal_data
    clf = OrdinalXGBClassifier(
        threshold_update="latent_median", class_weight=class_weight
    ).fit(X, y)

    f = clf.predict_projection(X)
    medians = np.array([np.median(f[y == k]) for k in clf.classes_])
    np.testing.assert_allclose(
        clf.thresholds_, np.sort((medians[:-1] + medians[1:]) / 2)
    )


@pytest.mark.parametrize("threshold_update", THRESHOLD_UPDATES)
def test_ordinal_xgb_fits_and_predicts_with_missing_values(
    missing_data, threshold_update
):
    """NaN in X fits and predicts with every threshold_update."""
    X, y = missing_data
    clf = OrdinalXGBClassifier(threshold_update=threshold_update).fit(X, y)

    assert np.isfinite(clf.thresholds_).all()
    assert np.isfinite(clf.predict_projection(X)).all()
    assert set(clf.predict(X)).issubset(set(clf.classes_))
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("threshold_update", ["input_median", "input_mean"])
def test_ordinal_xgb_missing_value_matches_nan(missing_data, threshold_update):
    """A non-NaN missing value gives the fit of the same data with NaN."""
    X_nan, y = missing_data
    X_marked = np.where(np.isnan(X_nan), -999.0, X_nan)

    clf_nan = OrdinalXGBClassifier(threshold_update=threshold_update).fit(X_nan, y)
    clf_marked = OrdinalXGBClassifier(
        threshold_update=threshold_update, missing=-999.0
    ).fit(X_marked, y)

    np.testing.assert_allclose(clf_marked.thresholds_, clf_nan.thresholds_)
    np.testing.assert_allclose(
        clf_marked.predict_projection(X_marked), clf_nan.predict_projection(X_nan)
    )


def test_ordinal_xgb_feature_weights_reach_dmatrix(ordinal_data):
    """With colsample_bynode a feature of weight 0 is never used."""
    X, y = ordinal_data
    params = dict(n_estimators=20, colsample_bynode=0.5)
    clf_free = OrdinalXGBClassifier(**params).fit(X, y)
    clf_weighted = OrdinalXGBClassifier(
        feature_weights=[1.0, 1.0, 1.0, 0.0], **params
    ).fit(X, y)

    assert clf_free.feature_importances_[3] > 0.0
    assert clf_weighted.feature_importances_[3] == 0.0


@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_ordinal_xgb_monotone_constraints_by_feature_name(ordinal_data):
    """Constraints by feature name bind and predict on a DataFrame or an array."""
    X, y = ordinal_data
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    clf = OrdinalXGBClassifier(
        n_estimators=20, monotone_constraints={"a": 1, "b": -1}
    ).fit(df, y)

    np.testing.assert_array_equal(clf.predict(df), clf.predict(X))

    grid = np.tile(X.mean(axis=0), (50, 1))
    grid[:, 0] = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    assert np.all(np.diff(clf.predict_projection(grid)) >= 0)
    grid[:, 0] = X[:, 0].mean()
    grid[:, 1] = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    assert np.all(np.diff(clf.predict_projection(grid)) <= 0)


@pytest.mark.parametrize(
    "importance_type, as_dataframe, gamma",
    [
        pytest.param(importance_type, False, 0.0, id=importance_type)
        for importance_type in ["gain", "weight", "cover", "total_gain", "total_cover"]
    ]
    + [
        pytest.param("gain", True, 0.0, id="dataframe"),
        pytest.param("gain", False, 1e6, id="no-splits"),
    ],
)
def test_ordinal_xgb_feature_importances_match_xgboost(
    ordinal_data, importance_type, as_dataframe, gamma
):
    """feature_importances_ equals XGBClassifier's on the same booster."""
    X, y = ordinal_data
    if as_dataframe:
        X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    clf = OrdinalXGBClassifier(
        n_estimators=20, gamma=gamma, importance_type=importance_type
    ).fit(X, y)

    xgboost_model = SimpleNamespace(
        get_booster=clf.get_booster,
        booster="gbtree",
        importance_type=importance_type,
        n_features_in_=clf.n_features_in_,
    )
    expected = xgb.XGBClassifier.feature_importances_.fget(xgboost_model)

    assert clf.feature_importances_.dtype == expected.dtype
    np.testing.assert_array_equal(clf.feature_importances_, expected)


def test_ordinal_xgb_get_booster_returns_booster(X, y):
    """get_booster returns booster_."""
    clf = OrdinalXGBClassifier(n_estimators=1).fit(X, y)

    assert clf.get_booster() is clf.booster_


def test_ordinal_xgb_allows_nan_tag():
    """The allow_nan input tag is set."""
    assert get_tags(OrdinalXGBClassifier()).input_tags.allow_nan


def test_ordinal_xgb_fit_without_xgboost_raises_import_error(X, y, monkeypatch):
    """fit raises ImportError naming the xgboost extra when xgboost is missing."""
    monkeypatch.setitem(sys.modules, "xgboost", None)

    with pytest.raises(ImportError, match=r"skordinal\[xgboost\]"):
        OrdinalXGBClassifier().fit(X, y)
