"Tests for model selection and estimator loading utilities."

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from skordinal.classifiers import NNOP, NNPOM, REDSVM, SVOREX, OrdinalDecomposition
from skordinal.metrics import get_ordinal_scorer
from skordinal.model_selection import get_classifier_by_name, load_classifier
from skordinal.utils._testing import TEST_RANDOM_STATE


def test_get_classifier_by_name_correct():
    """Test that get_classifier_by_name returns the correct classifier."""
    # skordinal classifiers
    assert get_classifier_by_name("NNOP") == NNOP
    assert get_classifier_by_name("NNPOM") == NNPOM
    assert get_classifier_by_name("OrdinalDecomposition") == OrdinalDecomposition
    assert get_classifier_by_name("REDSVM") == REDSVM
    assert get_classifier_by_name("SVOREX") == SVOREX

    # Scikit-learn classifiers
    assert get_classifier_by_name("SVC") == SVC
    assert get_classifier_by_name("LogisticRegression") == LogisticRegression


def test_get_classifier_by_name_incorrect():
    """Test that get_classifier_by_name raises ValueError for unknown classifiers."""
    with pytest.raises(ValueError, match="Unknown classifier 'RandomForest'"):
        get_classifier_by_name("RandomForest")

    with pytest.raises(ValueError, match="Unknown classifier 'SVR'"):
        get_classifier_by_name("SVR")


def test_load_classifier_without_parameters():
    """Test that load_classifier correctly instantiates classifiers without
    parameters."""
    assert isinstance(load_classifier("NNOP"), NNOP)
    assert isinstance(load_classifier("NNPOM"), NNPOM)
    assert isinstance(load_classifier("OrdinalDecomposition"), OrdinalDecomposition)
    assert isinstance(load_classifier("REDSVM"), REDSVM)
    assert isinstance(load_classifier("SVOREX"), SVOREX)
    assert isinstance(load_classifier("SVC"), SVC)
    assert isinstance(load_classifier("LogisticRegression"), LogisticRegression)


def test_load_classifier_with_parameters():
    """Test that load_classifier correctly instantiates classifiers with
    parameters."""
    param_grid = {
        "epsilon_init": 0.5,
        "n_hidden": 10,
        "max_iter": 500,
        "alpha": 0.01,
    }
    classifier = load_classifier("NNPOM", param_grid=param_grid)
    assert isinstance(classifier, NNPOM)
    assert classifier.epsilon_init == 0.5
    assert classifier.n_hidden == 10
    assert classifier.max_iter == 500
    assert classifier.alpha == 0.01


def test_load_classifier_with_searchcv():
    """Test that load_classifier correctly returns a GridSearchCV when param_grid has multiple values."""
    param_grid = {"C": [0.1, 1.0], "probability": "True"}

    classifier = load_classifier(
        "SVC",
        param_grid=param_grid,
        random_state=TEST_RANDOM_STATE,
        cv_n_folds=5,
        cv_metric="neg_mean_absolute_error",
        n_jobs=1,
    )

    expected_param_grid = {
        "C": [0.1, 1.0],
        "probability": ["True"],
        "random_state": [TEST_RANDOM_STATE],
    }

    assert isinstance(classifier, GridSearchCV)
    assert classifier.cv.n_splits == 5
    assert classifier.param_grid == expected_param_grid


def test_load_classifier_with_ensemble_method():
    """Test that load_classifier correctly handles ensemble methods."""
    param_grid = {
        "dtype": "ordered_partitions",
        "decision_method": "frank_hall",
        "base_classifier": "SVC",
        "parameters": {
            "C": [0.01, 0.1, 1, 10],
            "gamma": [0.01, 0.1, 1, 10],
            "probability": ["True"],
        },
    }
    classifier = load_classifier(
        classifier_name="OrdinalDecomposition",
        param_grid=param_grid,
        n_jobs=10,
        cv_n_folds=3,
        cv_metric=get_ordinal_scorer("neg_mean_absolute_error"),
        random_state=TEST_RANDOM_STATE,
    )
    assert isinstance(classifier, GridSearchCV)
    assert classifier.param_grid["decision_method"] == [param_grid["decision_method"]]
    assert classifier.param_grid["base_classifier"] == [param_grid["base_classifier"]]
    for params in classifier.param_grid["parameters"]:
        assert params["random_state"] == TEST_RANDOM_STATE
    assert classifier.cv.n_splits == 3


def test_load_classifier_with_invalid_param():
    """Test that load_classifier raises error with invalid parameter key."""
    error_msg = "Invalid parameter 'T' for classifier 'SVC'."

    with pytest.raises(ValueError, match=error_msg):
        load_classifier(classifier_name="SVC", param_grid={"T": 0.1})
