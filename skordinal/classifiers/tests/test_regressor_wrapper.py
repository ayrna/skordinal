"""Tests for the RegressorWrapper classifier."""

import inspect

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from skordinal.classifiers import RegressorWrapper


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]], dtype=float)


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


def test_regressor_wrapper_predict_matches_expected():
    """Test that predictions match expected values."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1, 2, 2])

    classifier = RegressorWrapper(LinearRegression()).fit(X, y)
    npt.assert_array_equal(classifier.predict(X), y)


@pytest.mark.parametrize("invalid_value", [5, "svr", object()])
def test_regressor_wrapper_hyperparameter_type_validation(X, y, invalid_value):
    """Test that RegressorWrapper raises ValueError for invalid types of hyperparameters."""
    classifier = RegressorWrapper(estimator=invalid_value)

    with pytest.raises(ValueError, match=r"The 'estimator' parameter.*"):
        classifier.fit(X, y)


def test_regressor_wrapper_fit_input_validation(X, y):
    """Test that input data is validated."""
    classifier = RegressorWrapper()

    with pytest.raises(ValueError):
        classifier.fit(X, y[:-1])

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])


def test_regressor_wrapper_rejects_non_regressor(X, y):
    """Test that a base estimator that is not a regressor is rejected."""
    classifier = RegressorWrapper(LogisticRegression())

    with pytest.raises(ValueError, match="must be a regressor"):
        classifier.fit(X, y)


def test_regressor_wrapper_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = RegressorWrapper().fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_regressor_wrapper_sets_classes_and_n_features_in_after_fit(X, y):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = RegressorWrapper().fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_regressor_wrapper_predict_raises_if_not_fitted(X):
    """Test that predict raises when the model is not fitted."""
    from sklearn.exceptions import NotFittedError

    classifier = RegressorWrapper()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_regressor_wrapper_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = RegressorWrapper().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_regressor_wrapper_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(RegressorWrapper.__init__).parameters) - {
        "self"
    }
    assert set(RegressorWrapper._parameter_constraints) == init_params


def test_regressor_wrapper_predict_rejects_wrong_n_features(X, y):
    """predict and predict_projection reject a mismatched n_features."""
    classifier = RegressorWrapper().fit(X, y)
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
def test_regressor_wrapper_label_roundtrip(labels):
    """Test that RegressorWrapper preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = RegressorWrapper(LinearRegression()).fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_regressor_wrapper_predict_raises_on_nan_output(X, y, monkeypatch):
    """Test that a NaN regressor prediction raises a ValueError."""

    def predict_nan(self, X):
        return np.full(len(X), np.nan)

    monkeypatch.setattr(LinearRegression, "predict", predict_nan)
    classifier = RegressorWrapper(LinearRegression()).fit(X, y)

    with pytest.raises(ValueError, match="cannot map NaN to a rank"):
        classifier.predict(X)


@pytest.mark.parametrize(
    "raw_output, expected_class_fn",
    [
        (np.inf, lambda classes: classes.max()),
        (-np.inf, lambda classes: classes.min()),
    ],
)
def test_regressor_wrapper_predict_clips_infinite_output_to_extreme_class(
    X, y, monkeypatch, raw_output, expected_class_fn
):
    """Test that an infinite regressor prediction maps to the nearest extreme class."""

    def predict_constant(self, X):
        return np.full(len(X), raw_output)

    monkeypatch.setattr(LinearRegression, "predict", predict_constant)
    classifier = RegressorWrapper(LinearRegression()).fit(X, y)

    predictions = classifier.predict(X)

    npt.assert_array_equal(
        predictions, np.full(len(X), expected_class_fn(classifier.classes_))
    )


def test_regressor_wrapper_projection_is_the_wrapped_regressor_output(X, y):
    """predict_projection returns the wrapped regressor's own output."""
    clf = RegressorWrapper(LinearRegression()).fit(X, y)

    projection = clf.predict_projection(X)
    assert projection.shape == (len(X),)
    assert np.isfinite(projection).all()

    # the projection is the regressor output on the rank scale, unclipped
    npt.assert_allclose(projection, clf.estimator_.predict(X))

    # predict clips then rounds to nearest rank; ties fall to the lower class
    max_rank = clf.classes_.size - 1
    clipped = np.clip(projection, 0.0, max_rank)
    n_exceeded = (clipped[:, np.newaxis] > clf.thresholds_[np.newaxis, :]).sum(axis=1)
    npt.assert_array_equal(clf.predict(X), clf.classes_[n_exceeded])
