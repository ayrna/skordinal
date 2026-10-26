"""Tests for the EBANO probability-averaging classifier."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from skordinal.classifiers import EBANO, LogisticAT, LogisticIT, _ebano


def _expected_failed_sklearn_checks(estimator):
    return {
        "check_classifiers_classes": (
            "EBANO supports numeric ordinal target labels, not string labels."
        )
    }


@pytest.fixture
def ordinal_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(90, 4))
    y = np.digitize(X[:, 0] + X[:, 1] * 0.4, [-0.7, 0.7])
    return X, y


def _estimators():
    return [
        ("lr1", LogisticRegression(max_iter=300, C=0.5)),
        ("lr2", LogisticRegression(max_iter=300, C=2.0)),
    ]


class _WrongProbabilityShapeClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.full((len(X), 4), 0.25)

    def predict(self, X):
        return self.classes_[np.zeros(len(X), dtype=int)]


class _WrongClassesClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.full((len(X), 3), 1 / 3)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _ObjectTargetLogisticRegression(LogisticRegression):
    """LogisticRegression that accepts object-dtype integer targets."""

    def fit(self, X, y, **fit_params):
        return super().fit(X, np.asarray(y).astype(int), **fit_params)


@parametrize_with_checks(
    [EBANO(_estimators(), n_iter=5, cv=3, random_state=0)],
    expected_failed_checks=_expected_failed_sklearn_checks,
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_fit_and_predict_are_weighted_sum_of_estimator_probabilities(ordinal_data):
    X, y = ordinal_data
    model = EBANO(_estimators(), n_iter=25, cv=3, random_state=7)

    assert model.fit(X, y) is model
    assert model.classes_.tolist() == [0, 1, 2]
    assert model.weights_.shape == (2,)
    assert np.all(model.weights_ >= 0)
    assert model.weights_.sum() == pytest.approx(1.0)
    assert np.isfinite(model.best_score_)
    assert model.named_estimators_.lr1 is model.estimators_[0]

    X_new = np.array([[3.0, -1.0, 0.0, 0.5], [-2.0, 0.5, 1.0, 0.0]])
    expected = sum(
        weight * estimator.predict_proba(X_new)
        for weight, estimator in zip(model.weights_, model.estimators_)
    )
    np.testing.assert_allclose(model.predict_proba(X_new), expected)
    np.testing.assert_allclose(model.predict_proba(X_new).sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        model.predict(X_new), model.classes_[np.argmax(expected, axis=1)]
    )


def test_default_scoring_is_ordinal_scorer(ordinal_data):
    X, y = ordinal_data
    model = EBANO(_estimators(), n_iter=5, cv=3, random_state=0).fit(X, y)

    assert model.scoring == "neg_average_mean_absolute_error"
    assert model.best_score_ <= 0


def test_works_with_real_ordinal_classifiers(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        [("at", LogisticAT()), ("it", LogisticIT())],
        n_iter=5,
        cv=3,
        random_state=0,
    ).fit(X, y)

    assert model.predict(X).shape == (len(X),)
    assert set(model.predict(X)).issubset({0, 1, 2})
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_ebano_works_in_pipeline_and_grid_search(ordinal_data):
    X, y = ordinal_data
    pipeline = make_pipeline(
        StandardScaler(), EBANO(_estimators(), n_iter=5, cv=3, random_state=0)
    )
    search = GridSearchCV(
        pipeline,
        {"ebano__n_iter": [3, 5], "ebano__lr1__C": [0.5, 1.0]},
        cv=3,
        scoring="accuracy",
    )

    search.fit(X, y)

    assert search.best_estimator_.predict_proba(X[:4]).shape == (4, 3)


def test_accepts_pipeline_as_base_estimator(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        [("pipe", make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)))],
        n_iter=3,
        cv=3,
        random_state=0,
    ).fit(X, y)

    assert model.predict(X).shape == (len(X),)


def test_accepts_non_contiguous_numeric_classes(ordinal_data):
    X, y = ordinal_data
    labels = np.array([10, 30, 80])[y]
    model = EBANO(_estimators(), n_iter=5, cv=3, random_state=0).fit(X, labels)

    assert model.classes_.tolist() == [10, 30, 80]
    assert set(model.predict(X[:5])).issubset({10, 30, 80})


def test_accepts_object_dtype_targets(ordinal_data):
    # Regression test: object-dtype ``y`` must not reach the scorer next to
    # integer predictions ("mix of unknown and multiclass targets").
    X, y = ordinal_data
    model = EBANO(
        [("lr", _ObjectTargetLogisticRegression(max_iter=300))],
        n_iter=3,
        cv=3,
        random_state=0,
    )

    model.fit(X, y.astype(object))

    assert model.classes_.tolist() == [0, 1, 2]


def test_handles_folds_missing_a_class(ordinal_data):
    # With sorted targets and unshuffled KFold, one training fold lacks the
    # highest class. cross_val_predict pads the probability matrix.
    X, y = ordinal_data
    order = np.argsort(y, kind="stable")
    model = EBANO(_estimators(), n_iter=5, cv=KFold(3), random_state=0)

    model.fit(X[order], y[order])

    assert model.predict_proba(X).shape == (len(X), 3)


def test_drop_estimators_are_excluded_but_preserved_by_name(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        [("lr", LogisticRegression(max_iter=300)), ("disabled", "drop")],
        n_iter=3,
        cv=3,
        random_state=0,
    ).fit(X, y)

    assert len(model.estimators_) == 1
    assert model.named_estimators_.lr is model.estimators_[0]
    assert model.named_estimators_.disabled == "drop"
    assert model.weights_.shape == (1,)


def test_set_params_supports_named_estimators_and_nested_params():
    model = EBANO(_estimators())

    model.set_params(lr1__C=4.0, lr2="drop")

    assert model.estimators[0][1].C == 4.0
    assert model.estimators[1][1] == "drop"
    assert model.get_params()["lr1__C"] == 4.0


def test_random_state_reproduces_weight_search(ordinal_data):
    X, y = ordinal_data
    first = EBANO(_estimators(), n_iter=20, cv=3, random_state=42).fit(X, y)
    second = EBANO(_estimators(), n_iter=20, cv=3, random_state=42).fit(X, y)

    np.testing.assert_array_equal(first.weights_, second.weights_)
    assert first.best_score_ == second.best_score_


def test_random_state_accepts_random_state_instance(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        _estimators(), n_iter=5, cv=3, random_state=np.random.RandomState(0)
    ).fit(X, y)

    assert model.weights_.sum() == pytest.approx(1.0)


def test_weight_search_keeps_highest_scoring_candidate(ordinal_data, monkeypatch):
    X, y = ordinal_data
    first_probabilities = np.tile([0.9, 0.05, 0.05], (len(X), 1))
    second_probabilities = np.tile([0.1, 0.4, 0.5], (len(X), 1))
    monkeypatch.setattr(
        _ebano,
        "cross_val_predict",
        lambda estimator, X, y, **kwargs: (
            first_probabilities if estimator.C == 0.5 else second_probabilities
        ),
    )
    random_state = 12
    candidates = np.random.RandomState(random_state).uniform(size=(10, 2))
    candidates /= candidates.sum(axis=1, keepdims=True)
    expected_weights = candidates[
        np.argmax(candidates[:, 0] * 0.9 + candidates[:, 1] * 0.1)
    ]
    model = EBANO(
        _estimators(),
        n_iter=10,
        cv=3,
        scoring=lambda estimator, X, y: estimator.predict_proba(X)[0, 0],
        random_state=random_state,
    ).fit(X, y)

    np.testing.assert_allclose(model.weights_, expected_weights)
    assert model.best_score_ == pytest.approx(
        0.9 * expected_weights[0] + 0.1 * expected_weights[1]
    )


def test_fit_accepts_cv_splitter_and_sklearn_scorer(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        _estimators(),
        n_iter=5,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=3),
        scoring="accuracy",
        random_state=3,
    ).fit(X, y)

    assert 0 <= model.best_score_ <= 1


def test_fit_accepts_cv_iterable(ordinal_data):
    X, y = ordinal_data
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=3).split(X, y))
    model = EBANO(_estimators(), n_iter=5, cv=cv, random_state=3).fit(X, y)

    assert model.weights_.shape == (2,)


def test_fit_accepts_callable_scorer(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        _estimators(),
        n_iter=5,
        cv=3,
        scoring=lambda estimator, X, y: accuracy_score(y, estimator.predict(X)),
        random_state=1,
    ).fit(X, y)

    assert 0 <= model.best_score_ <= 1


def test_cv_must_be_a_partition(ordinal_data):
    # cross_val_predict only supports splitters that partition the data.
    X, y = ordinal_data
    model = EBANO(_estimators(), n_iter=2, cv=ShuffleSplit(n_splits=3, random_state=0))

    with pytest.raises(ValueError, match="partition"):
        model.fit(X, y)


def test_fit_with_parallel_jobs(ordinal_data):
    X, y = ordinal_data
    model = EBANO(_estimators(), n_iter=5, cv=3, n_jobs=2, random_state=0).fit(X, y)

    assert model.weights_.shape == (2,)


def test_prefit_frozen_estimators_remain_fitted_when_cloned(ordinal_data):
    X, y = ordinal_data
    fitted = LogisticRegression(max_iter=300).fit(X[::2], y[::2])
    model = EBANO(
        [("frozen", FrozenEstimator(fitted))],
        n_iter=3,
        cv="prefit",
        random_state=0,
    )

    model.fit(X[1::2], y[1::2])

    check_is_fitted(model.named_estimators_.frozen.estimator)
    assert model.named_estimators_.frozen.estimator is fitted
    np.testing.assert_allclose(model.predict_proba(X[:5]), fitted.predict_proba(X[:5]))


def test_prefit_frozen_estimators_work_inside_grid_search(ordinal_data):
    X, y = ordinal_data
    fitted = LogisticRegression(max_iter=300).fit(X[::2], y[::2])
    model = EBANO(
        [("frozen", FrozenEstimator(fitted))],
        n_iter=3,
        cv="prefit",
        random_state=0,
    )
    search = GridSearchCV(model, {"n_iter": [2, 3]}, cv=3, scoring="accuracy")

    search.fit(X[1::2], y[1::2])

    check_is_fitted(fitted)
    assert search.best_estimator_.named_estimators_.frozen.estimator is fitted


def test_prefit_with_dropped_estimator(ordinal_data):
    X, y = ordinal_data
    fitted = LogisticRegression(max_iter=300).fit(X[::2], y[::2])
    model = EBANO(
        [("frozen", FrozenEstimator(fitted)), ("off", "drop")],
        n_iter=3,
        cv="prefit",
        random_state=0,
    ).fit(X[1::2], y[1::2])

    assert len(model.estimators_) == 1
    assert model.named_estimators_.off == "drop"


def test_prefit_requires_fitted_estimators(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        [("unfitted", FrozenEstimator(LogisticRegression()))],
        n_iter=2,
        cv="prefit",
    )

    with pytest.raises(NotFittedError):
        model.fit(X, y)


def test_accepts_nan_when_base_estimators_do(ordinal_data):
    # Checks that X is not converted and that the tags propagate allow_nan.
    X, y = ordinal_data
    X_nan = X.copy()
    X_nan[::7, 0] = np.nan
    model = EBANO(
        [("hgb", HistGradientBoostingClassifier(max_iter=20))],
        n_iter=3,
        cv=3,
        random_state=0,
    ).fit(X_nan, y)

    assert model.predict(X_nan).shape == (len(X),)


# --------------------------------------------------------------------------- #
# Error handling (EBANO-specific)
# --------------------------------------------------------------------------- #


def test_fit_rejects_wrong_probability_shape(ordinal_data):
    X, y = ordinal_data
    model = EBANO([("bad", _WrongProbabilityShapeClassifier())], n_iter=2, cv=3)

    with pytest.raises(ValueError, match="returned probabilities with shape"):
        model.fit(X, y)


def test_fit_rejects_estimator_with_different_classes(ordinal_data, monkeypatch):
    X, y = ordinal_data
    monkeypatch.setattr(
        _ebano,
        "cross_val_predict",
        lambda estimator, X, y, **kwargs: np.full((len(X), 3), 1 / 3),
    )
    model = EBANO([("bad", _WrongClassesClassifier())], n_iter=2, cv=3)

    with pytest.raises(ValueError, match="has classes"):
        model.fit(X, y)


def test_fit_rejects_non_finite_scorer_result(ordinal_data):
    X, y = ordinal_data
    model = EBANO(
        _estimators(),
        n_iter=2,
        cv=3,
        scoring=lambda estimator, X, y: np.nan,
    )

    with pytest.raises(ValueError, match="non-finite value"):
        model.fit(X, y)


def test_fit_rejects_unknown_scoring_name(ordinal_data):
    X, y = ordinal_data
    model = EBANO(_estimators(), n_iter=2, cv=2, scoring="not-a-scorer")

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fit_rejects_single_class_targets(ordinal_data):
    X, _ = ordinal_data
    model = EBANO(_estimators(), n_iter=2, cv=2)

    with pytest.raises(ValueError, match="at least 2 unique classes"):
        model.fit(X, np.zeros(len(X), dtype=int))


@pytest.mark.parametrize(
    "estimators",
    [
        ("lr", LogisticRegression()),
        [("bad__name", LogisticRegression())],
        [(1, LogisticRegression())],
        [("lr", object())],
    ],
    ids=["not-a-list", "double-underscore-name", "non-string-name", "not-an-estimator"],
)
def test_fit_rejects_malformed_estimator_specification(estimators, ordinal_data):
    X, y = ordinal_data
    model = EBANO(estimators, n_iter=2, cv=2)

    with pytest.raises((TypeError, ValueError)):
        model.fit(X, y)


@pytest.mark.parametrize(
    "estimators",
    [
        [],
        [("invalid", SVC(probability=False))],
        [("one", LogisticRegression()), ("one", LogisticRegression())],
        [("only_drop", "drop")],
    ],
    ids=["empty", "no-predict-proba", "duplicate-names", "only-drop"],
)
def test_fit_rejects_invalid_estimators(estimators, ordinal_data):
    X, y = ordinal_data
    model = EBANO(estimators, n_iter=2, cv=2)

    with pytest.raises(ValueError):
        model.fit(X, y)
