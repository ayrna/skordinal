"""Tests for the ORBoost classifier."""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skordinal.classifiers import ORBoost
from skordinal.datasets import load_era, make_ordinal_classification


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.fixture
def ordinal_dataset():
    """Create a synthetic, separable ordinal dataset for behavioral tests."""
    return make_ordinal_classification(n_samples=120, random_state=0)


def test_orboost_default_params():
    """Test that ORBoost uses the documented default hyperparameters."""
    classifier = ORBoost()

    assert classifier.n_estimators == 200
    assert classifier.loss_form == "lr"
    assert classifier.base_learner == "stump"
    assert classifier.weight_reg is True


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_estimators", 0),
        ("loss_form", "unknown"),
        ("base_learner", "svm"),
    ],
)
def test_orboost_hyperparameter_value_validation(X, y, param_name, invalid_value):
    """Test that ORBoost raises ValueError for invalid of hyperparameters."""
    classifier = ORBoost(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("n_estimators", 2.5),
        ("loss_form", 5),
        ("base_learner", 5),
        ("weight_reg", "yes"),
    ],
)
def test_orboost_hyperparameter_type_validation(X, y, param_name, invalid_value):
    """Test that ORBoost raises ValueError for invalid types of hyperparameters."""
    classifier = ORBoost(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_orboost_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(ORBoost.__init__).parameters) - {"self"}
    assert set(ORBoost._parameter_constraints) == init_params


def test_orboost_fit_returns_self(X, y):
    """fit should return self for sklearn compatibility."""
    classifier = ORBoost(n_estimators=5)
    model = classifier.fit(X, y)
    assert model is classifier


def test_orboost_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = ORBoost(n_estimators=5)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        classifier.fit(X, y_invalid)

    with pytest.raises(ValueError):
        classifier.fit([], y)

    with pytest.raises(ValueError):
        classifier.fit(X, [])

    with pytest.raises(ValueError):
        classifier.fit(X_invalid, y)


def test_orboost_sets_classes_and_n_features_in_after_fit(X, y):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = ORBoost(n_estimators=5).fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_orboost_model_dict_present_after_fit(X, y):
    """Test that model_ is a dict with 'model' and 'params' keys after fit."""
    classifier = ORBoost(n_estimators=5).fit(X, y)

    assert isinstance(classifier.model_, dict)
    assert "model" in classifier.model_
    assert "params" in classifier.model_


def test_orboost_predict_raises_if_not_fitted(X):
    """predict and predict_projection raise NotFittedError before fit."""
    classifier = ORBoost()
    with pytest.raises(NotFittedError):
        classifier.predict(X)
    with pytest.raises(NotFittedError):
        classifier.predict_projection(X)


def test_orboost_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = ORBoost(n_estimators=5).fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_orboost_predict_rejects_wrong_n_features(X, y):
    """predict and predict_projection reject a mismatched n_features."""
    classifier = ORBoost(n_estimators=5).fit(X, y)
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
def test_orboost_label_roundtrip(labels):
    """Test that ORBoost preserves arbitrary ordinal label sets through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = ORBoost(n_estimators=10)
    classifier.fit(X, y)

    assert np.array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


@pytest.mark.parametrize("loss_form", ["lr", "full"])
def test_orboost_loss_form_produces_valid_predictions(ordinal_dataset, loss_form):
    """Test that both loss_form values produce predictions drawn from classes_."""
    X, y = ordinal_dataset
    classifier = ORBoost(n_estimators=20, loss_form=loss_form).fit(X, y)

    preds = classifier.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset(set(classifier.classes_))


@pytest.mark.parametrize("base_learner", ["stump", "perceptron"])
def test_orboost_base_learner_produces_valid_predictions(ordinal_dataset, base_learner):
    """Test that both base_learner types produce predictions drawn from classes_."""
    X, y = ordinal_dataset
    classifier = ORBoost(n_estimators=20, base_learner=base_learner).fit(X, y)

    preds = classifier.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset(set(classifier.classes_))


def test_orboost_weight_reg_false_fits_and_predicts(ordinal_dataset):
    """Test that weight_reg=False fits and predicts without error."""
    X, y = ordinal_dataset
    classifier = ORBoost(n_estimators=20, weight_reg=False).fit(X, y)

    preds = classifier.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset(set(classifier.classes_))


def test_orboost_full_loss_form_weight_reg_false_fits_and_predicts(ordinal_dataset):
    """Test that loss_form='full' combined with weight_reg=False fits and predicts."""
    X, y = ordinal_dataset
    classifier = ORBoost(n_estimators=20, loss_form="full", weight_reg=False)
    classifier.fit(X, y)

    preds = classifier.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset(set(classifier.classes_))


def test_orboost_stump_predictions_are_deterministic(ordinal_dataset):
    """Test that two default (stump) fits on the same data give equal predicts."""
    X, y = ordinal_dataset
    classifier_a = ORBoost(n_estimators=20).fit(X, y)
    classifier_b = ORBoost(n_estimators=20).fit(X, y)

    np.testing.assert_array_equal(classifier_a.predict(X), classifier_b.predict(X))


def test_orboost_predictions_are_nontrivial(ordinal_dataset):
    """Test that predictions use multiple classes and beat the majority baseline."""
    X, y = ordinal_dataset
    classifier = ORBoost(n_estimators=20).fit(X, y)

    preds = classifier.predict(X)
    assert len(np.unique(preds)) > 1

    majority_baseline = np.unique(y, return_counts=True)[1].max() / len(y)
    training_accuracy = np.mean(preds == y)
    assert training_accuracy > majority_baseline


def test_orboost_fits_on_era_dataset():
    """Test that ORBoost fits and predicts on the real-world ERA dataset."""
    X, y = load_era(return_X_y=True)
    classifier = ORBoost(n_estimators=10).fit(X, y)

    preds = classifier.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset(set(classifier.classes_))


def test_orboost_projection_well_formed(ordinal_dataset):
    """predict_projection is well-formed and reproduces predict via thresholds_."""
    X, y = ordinal_dataset
    clf = ORBoost(n_estimators=20).fit(X, y)

    projection = clf.predict_projection(X)
    assert projection.shape == (len(X),)
    assert np.isfinite(projection).all()

    assert clf.thresholds_.shape == (clf.classes_.size - 1,)
    assert np.all(np.diff(clf.thresholds_) >= 0)

    # recomputing predict cross-checks what the C++ actually returned
    n_reached = (projection[:, np.newaxis] >= clf.thresholds_[np.newaxis, :]).sum(
        axis=1
    )
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[n_reached])
