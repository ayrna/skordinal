"""Tests for the OPBE classifier."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from skordinal.datasets import make_ordinal_classification
from skordinal.linear_model import POM
from skordinal.reduction import OPBE
from skordinal.svm import SVOR


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


def test_opbe_predict_matches_expected():
    """predict recovers the labels of a separable toy problem."""
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0], [16.0], [17.0], [18.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    classifier = OPBE().fit(X, y)
    np.testing.assert_array_equal(classifier.predict(X), y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("min_spread", -1.0),
    ],
)
def test_opbe_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Invalid hyperparameter values raise ValueError."""
    classifier = OPBE(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("estimator", 3),
        ("estimator", LogisticRegression()),
        ("min_spread", "wide"),
        ("n_jobs", 1.5),
    ],
)
def test_opbe_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Invalid hyperparameter types raise ValueError."""
    classifier = OPBE(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_opbe_fit_returns_self(X, y):
    """fit returns self (sklearn contract)."""
    classifier = OPBE()
    model = classifier.fit(X, y)
    assert model is classifier


def test_opbe_fit_input_validation(X, y):
    """fit rejects mismatched or empty X and y."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = OPBE()
    with pytest.raises(ValueError):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_opbe_sets_fitted_attributes_after_fit(X, y):
    """fit sets the fitted attributes, including the default estimator."""
    clf = OPBE().fit(X, y)
    n_classes = len(np.unique(y))

    for attr in ["classes_", "n_features_in_", "estimators_", "fusion_weights_"]:
        assert hasattr(clf, attr), f"Missing fitted attribute: {attr}"

    assert isinstance(clf.classes_, np.ndarray) and np.array_equal(
        clf.classes_, np.unique(y)
    )
    assert isinstance(clf.n_features_in_, int) and clf.n_features_in_ == X.shape[1]
    assert clf.estimator is None
    assert len(clf.estimators_) == n_classes
    assert all(isinstance(est, SVOR) for est in clf.estimators_)
    assert clf.fusion_weights_.shape == (n_classes,)


def test_opbe_predict_invalid_input_raises_error(X, y):
    """predict rejects invalid input."""
    classifier = OPBE()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_opbe_predict_raises_if_not_fitted(X):
    """predict and predict_proba raise NotFittedError before fit."""
    classifier = OPBE()
    with pytest.raises(NotFittedError):
        classifier.predict(X)
    with pytest.raises(NotFittedError):
        classifier.predict_proba(X)


def test_opbe_feature_names_in_when_dataframe(X, y):
    """feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = OPBE().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_opbe_parameter_constraints_match_init_params():
    """_parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(OPBE.__init__).parameters) - {"self"}
    assert set(OPBE._parameter_constraints) == init_params


def test_opbe_predict_rejects_wrong_n_features(X, y):
    """predict and predict_proba reject a mismatched n_features."""
    classifier = OPBE().fit(X, y)
    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])
    with pytest.raises(ValueError):
        classifier.predict_proba(X[:, :-1])


@pytest.mark.parametrize(
    "labels",
    [
        [1, 2, 3],
        [0, 1, 2],
        [-1, 0, 1],
        [3, 5, 7],
    ],
)
def test_opbe_label_roundtrip(labels):
    """fit preserves arbitrary ordinal labels in classes_."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = OPBE()
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_opbe_deterministic_two_fits(X, y):
    """Two fits on the same data give bit-identical predict_proba."""
    clf_a = OPBE().fit(X, y)
    clf_b = OPBE().fit(X, y)

    np.testing.assert_array_equal(clf_a.predict_proba(X), clf_b.predict_proba(X))


def test_opbe_proba_well_formed(ordinal_data):
    """predict_proba is well-formed and matches predict."""
    X, y = ordinal_data
    clf = OPBE().fit(X, y)
    n_classes = len(clf.classes_)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], n_classes)
    assert np.all(proba >= 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    expected_pred = clf.classes_[proba.argmax(axis=1)]
    np.testing.assert_array_equal(clf.predict(X), expected_pred)


def test_opbe_binary_ordinal_data():
    """OPBE fits and predicts on a 2-class ordinal dataset."""
    X, y = make_ordinal_classification(
        n_samples=60,
        n_features=3,
        n_classes=2,
        n_informative=3,
        noise=0.1,
        random_state=2,
    )
    clf = OPBE().fit(X, y)

    assert len(clf.estimators_) == 2
    assert set(clf.predict(X)).issubset(set(clf.classes_))
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_opbe_exact_proba_tie_predicts_lower_class(X, y):
    """An exact predict_proba tie resolves to the lower class."""
    clf = OPBE(estimator=POM()).fit(X[y < 2], y[y < 2])
    # Round fitted state, since no real fit puts the probe on an exact tie
    for est in clf.estimators_:
        est.coef_ = np.array([1.0, 0.0])
        est.thresholds_ = np.array([0.0])

    probe = np.array([[0.0, 0.0]])
    proba = clf.predict_proba(probe)
    np.testing.assert_allclose(proba, [[0.5, 0.5]], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(clf.predict(probe), clf.classes_[[0]])


def test_opbe_large_magnitude_x_finite_probabilities():
    """Large-magnitude inputs still yield finite, normalised probabilities."""
    X, y = make_ordinal_classification(
        n_samples=60, n_features=4, n_classes=3, n_informative=4, random_state=0
    )
    X = X * 1000.0

    # No RuntimeWarning, such as a log of zero, may escape fit or predict_proba
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        classifier = OPBE().fit(X, y)
        proba = classifier.predict_proba(X)

    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_opbe_clones_the_given_estimator(X, y):
    """fit clones the base estimator instead of fitting it in place."""
    base = POM()
    classifier = OPBE(estimator=base).fit(X, y)

    assert classifier.estimator is base
    assert not hasattr(base, "coef_")
    assert len(set(map(id, classifier.estimators_))) == len(classifier.estimators_)


def test_opbe_grid_search_tunes_nested_estimator(ordinal_data):
    """GridSearchCV reaches the base estimator through estimator__*."""
    X, y = ordinal_data
    search = GridSearchCV(
        OPBE(estimator=SVOR()),
        param_grid={"estimator__C": [0.1, 10.0]},
        cv=3,
        error_score="raise",
    ).fit(X, y)

    best_C = search.best_params_["estimator__C"]
    assert all(est.C == best_C for est in search.best_estimator_.estimators_)


def test_opbe_validates_nested_estimator_params(X, y):
    """An invalid base-estimator hyperparameter is reported at fit."""
    classifier = OPBE(estimator=SVOR(C=-1))

    with pytest.raises(ValueError, match=r"The 'C' parameter of SVOR"):
        classifier.fit(X, y)


def test_opbe_submodels_separate_lower_equal_upper(X, y):
    """Each submodel learns its class (1) against the lower (0) and upper (2) ones."""
    clf = OPBE().fit(X, y)

    np.testing.assert_array_equal(clf.estimators_[0].classes_, [1, 2])
    np.testing.assert_array_equal(clf.estimators_[1].classes_, [0, 1, 2])
    np.testing.assert_array_equal(clf.estimators_[2].classes_, [0, 1])


@pytest.mark.parametrize(
    "n_classes, expected_weights",
    [
        (2, [2.0, 2.0]),
        (3, [2.5, 2.0, 2.5]),
        (5, [37 / 12, 7 / 3, 13 / 6, 7 / 3, 37 / 12]),
    ],
)
def test_opbe_fusion_weights_match_paper(n_classes, expected_weights):
    """fusion_weights_ are the w_j of the paper for each number of classes."""
    X, y = make_ordinal_classification(
        n_samples=30 * n_classes,
        n_features=3,
        n_classes=n_classes,
        n_informative=3,
        random_state=0,
    )
    clf = OPBE().fit(X, y)

    np.testing.assert_allclose(clf.fusion_weights_, expected_weights)


@pytest.mark.parametrize(
    "min_spread, expected_proba",
    [
        (4.0, [0.0319811017, 0.4750378062, 0.4929810921]),
        (0.0, [0.1097546823, 0.2728040148, 0.6174413029]),
    ],
)
def test_opbe_predict_proba_matches_paper(X, y, min_spread, expected_proba):
    """predict_proba is the paper's normalised product of the submodels."""
    clf = OPBE(estimator=POM(), min_spread=min_spread).fit(X, y)
    # The first feature as projection and an inner spread of 2, doubled by min_spread=4
    for est, thresholds in zip(clf.estimators_, [[0.0], [-1.0, 1.0], [0.0]]):
        est.coef_ = np.array([1.0, 0.0])
        est.thresholds_ = np.array(thresholds)

    probe = np.array([[0.5, 0.0]])
    np.testing.assert_allclose(clf.predict_proba(probe), [expected_proba], atol=1e-10)


def test_opbe_log_domain_avoids_underflow():
    """Fused scores below the float64 range keep their ratio in the log domain."""
    X, y = make_ordinal_classification(
        n_samples=40, n_features=2, n_classes=4, n_informative=2, random_state=0
    )
    clf = OPBE(estimator=POM()).fit(X, y)
    # Every class takes at least one factor exp(-800), so each product underflows
    for est, thresholds in zip(
        clf.estimators_, [[0.0], [-800.0, 800.0], [-800.0, 800.0], [0.0]]
    ):
        est.coef_ = np.array([1.0, 0.0])
        est.thresholds_ = np.array(thresholds)

    probe = np.array([[0.0, 0.0]])
    np.testing.assert_allclose(
        clf.predict_proba(probe), [[0.0, 0.5, 0.5, 0.0]], atol=1e-12
    )
    np.testing.assert_array_equal(clf.predict(probe), clf.classes_[[1]])


@pytest.mark.parametrize("min_spread", [4.0, 0.0])
def test_opbe_equal_thresholds_warn_and_exclude_the_class(
    ordinal_data, monkeypatch, min_spread
):
    """Equal inner thresholds warn at fit and give their class probability 0."""
    X, y = ordinal_data
    pom_fit = POM.fit

    def fit_with_equal_thresholds(self, X, y):
        pom_fit(self, X, y)
        self.thresholds_ = np.full_like(self.thresholds_, self.thresholds_[0])
        return self

    monkeypatch.setattr(POM, "fit", fit_with_equal_thresholds)
    with pytest.warns(UserWarning, match="class 1 in OPBE has equal thresholds"):
        clf = OPBE(estimator=POM(), min_spread=min_spread).fit(X, y)

    np.testing.assert_array_equal(clf.predict_proba(X)[:, 1], 0.0)
    assert 1 not in clf.predict(X)


def test_opbe_decreasing_thresholds_raise(ordinal_data, monkeypatch):
    """A submodel whose thresholds decrease is rejected at fit."""
    X, y = ordinal_data
    pom_fit = POM.fit

    def fit_with_reversed_thresholds(self, X, y):
        pom_fit(self, X, y)
        self.thresholds_ = self.thresholds_[::-1]
        return self

    monkeypatch.setattr(POM, "fit", fit_with_reversed_thresholds)
    with pytest.raises(ValueError, match="class 1 in OPBE has decreasing thresholds"):
        OPBE(estimator=POM()).fit(X, y)


def test_opbe_n_jobs_matches_serial(ordinal_data):
    """n_jobs=2 gives the same predict_proba as a serial fit."""
    X, y = ordinal_data
    serial = OPBE().fit(X, y)
    parallel = OPBE(n_jobs=2).fit(X, y)

    np.testing.assert_array_equal(parallel.predict_proba(X), serial.predict_proba(X))
