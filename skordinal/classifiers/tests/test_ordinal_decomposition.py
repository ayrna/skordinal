"""Tests for the OrdinalDecomposition ensemble."""

import inspect

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skordinal.classifiers import OrdinalDecomposition


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([1, 1, 1, 2, 2, 2])


def test_ordinal_decomposition(X, y):
    """Check if this algorithm can correctly classify a toy problem."""
    classifier = OrdinalDecomposition(
        dtype="ordered_partitions",
        decision_method="frank_hall",
        base_classifier="SVC",
        parameters={"C": 1.0, "gamma": "scale", "probability": True},
    )

    y_pred = classifier.fit(X, y).predict(X)
    npt.assert_array_equal(y_pred, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("dtype", "one_vs_all"),
        ("dtype", "frank_hall"),
        ("decision_method", "invalid"),
        ("decision_method", "one_vs_next"),
    ],
)
def test_ordinal_decomposition_hyperparameter_value_validation(
    X, y, param_name, invalid_value
):
    """Test that OrdinalDecomposition raises ValueError for invalid of
    hyperparameters."""
    classifier = OrdinalDecomposition(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("dtype", ["ordered_partitions"]),
        ("decision_method", 0),
        ("base_classifier", 3),
        ("parameters", "tol"),
        ("parameters", []),
    ],
)
def test_ordinal_decomposition_hyperparameter_type_validation(
    X, y, param_name, invalid_value
):
    """Test that OrdinalDecomposition raises ValueError for invalid types of hyperparameters."""
    classifier = OrdinalDecomposition(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        classifier.fit(X, y)


def test_ordinal_decomposition_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = OrdinalDecomposition()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The fit method doesnt return Null on error"


@pytest.mark.parametrize(
    "dtype, expected_cm",
    [
        (
            "ordered_partitions",
            np.array(
                [
                    [-1, -1, -1, -1],
                    [1, -1, -1, -1],
                    [1, 1, -1, -1],
                    [1, 1, 1, -1],
                    [1, 1, 1, 1],
                ]
            ),
        ),
        (
            "one_vs_next",
            np.array(
                [
                    [-1, 0, 0, 0],
                    [1, -1, 0, 0],
                    [0, 1, -1, 0],
                    [0, 0, 1, -1],
                    [0, 0, 0, 1],
                ]
            ),
        ),
        (
            "one_vs_followers",
            np.array(
                [
                    [-1, 0, 0, 0],
                    [1, -1, 0, 0],
                    [1, 1, -1, 0],
                    [1, 1, 1, -1],
                    [1, 1, 1, 1],
                ]
            ),
        ),
        (
            "one_vs_previous",
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, -1],
                    [1, 1, -1, 0],
                    [1, -1, 0, 0],
                    [-1, 0, 0, 0],
                ]
            ),
        ),
    ],
)
def test_coding_matrix(dtype, expected_cm):
    """Test that the coding matrix is built properly for each type of ordinal
    decomposition."""
    classifier = OrdinalDecomposition()
    classifier.dtype = dtype
    cm = classifier._coding_matrix(classifier.dtype, 5)

    npt.assert_array_equal(cm, expected_cm)


def test_frank_hall_method():
    """Test that frank and hall method returns expected values for one toy problem
    (starting off predicted probabilities given by each binary classifier)."""
    classifier = OrdinalDecomposition(dtype="ordered_partitions")
    classifier.coding_matrix_ = classifier._coding_matrix(classifier.dtype, 5)

    # Predicted probabilities from a 5 class ordinal dataset (positive class)
    predictions = np.array(
        [
            [0.07495, 0.00003, 0.06861, 0.00005],
            [0.00017, 0.0, 0.03174, 0.00011],
            [0.99235, 0.04285, 0.0485, 0.00004],
            [0.95376, 0.16388, 0.03857, 0.00028],
            [0.99726, 0.20159, 0.61801, 0.00037],
            [1.0, 0.90501, 0.44459, 0.00011],
            [1.0, 0.97307, 0.99424, 0.14627],
            [1.0, 0.64663, 0.45326, 0.06143],
            [1.0, 0.83569, 0.9175, 0.94988],
            [1.0, 0.93172, 0.6774, 0.43379],
        ]
    )

    y_proba = classifier._frank_hall_method(predictions)
    expected_y_proba = np.array(
        [
            [0.925050, 0.040630, 0.0, 0.034270, 0.000050],
            [0.989363, 0.0, 0.0, 0.010527, 0.000110],
            [0.007650, 0.946675, 0.0, 0.045635, 0.000040],
            [0.046240, 0.789880, 0.125310, 0.038290, 0.000280],
            [0.002740, 0.587460, 0.0, 0.409430, 0.000370],
            [0.0, 0.094990, 0.460420, 0.444480, 0.000110],
            [0.0, 0.016345, 0.0, 0.837385, 0.146270],
            [0.0, 0.353370, 0.193370, 0.391830, 0.061430],
            [0.0, 0.098977, 0.0, 0.0, 0.901023],
            [0.0, 0.068280, 0.254320, 0.243610, 0.433790],
        ]
    )

    # Asserting similarity
    npt.assert_allclose(
        y_proba,
        expected_y_proba,
        rtol=1e-04,
        atol=0,
    )

    # Asserting each row is a valid probability distribution
    assert (y_proba >= 0).all()
    npt.assert_allclose(y_proba.sum(axis=1), 1.0)


def test_exponential_loss_method():
    """Test that exponential loss method returns expected values for one toy problem
    (starting off predicted probabilities given by each binary classifier)."""
    classifier = OrdinalDecomposition(dtype="ordered_partitions")
    classifier.coding_matrix_ = classifier._coding_matrix(classifier.dtype, 5)

    # Predicted probabilities from a 5 class ordinal dataset (positive class)
    predictions = np.array(
        [
            [0.07495, 0.00003, 0.06861, 0.00005],
            [0.00017, 0.0, 0.03174, 0.00011],
            [0.99235, 0.04285, 0.0485, 0.00004],
            [0.95376, 0.16388, 0.03857, 0.00028],
            [0.99726, 0.20159, 0.61801, 0.00037],
            [1.0, 0.90501, 0.44459, 0.00011],
            [1.0, 0.97307, 0.99424, 0.14627],
            [1.0, 0.64663, 0.45326, 0.06143],
            [1.0, 0.83569, 0.9175, 0.94988],
            [1.0, 0.93172, 0.6774, 0.43379],
        ]
    )

    # Interpoling values from [0, 1] range to [-1, 1]
    predictions = (2 * predictions) - 1

    e_losses = classifier._exponential_loss(predictions)
    expected_e_losses = np.array(
        [
            [1.5852, 3.49769, 5.8479, 7.79566, 10.14575],
            [1.49583, 3.84519, 6.19559, 8.35469, 10.70441],
            [3.85107, 1.54761, 3.64184, 5.70348, 8.05364],
            [3.7542, 1.67955, 3.12761, 5.24671, 7.59538],
            [4.88834, 2.55481, 3.82059, 3.34415, 5.69227],
            [6.2293, 3.87889, 2.07579, 2.29788, 4.64761],
            [8.47407, 6.12367, 3.93616, 1.62115, 3.15709],
            [5.3858, 3.0354, 2.44043, 2.62767, 4.61571],
            [9.43904, 7.08864, 5.64271, 3.77177, 1.71942],
            [7.39145, 5.04105, 3.09146, 2.36688, 2.63249],
        ]
    )

    # Asserting similarity
    npt.assert_allclose(e_losses, expected_e_losses, rtol=1e-04, atol=0)


def test_logarithmic_loss_method():
    """Test that logarithmic loss method returns expected values for one toy problem
    (starting off predicted probabilities given by each binary classifier)."""
    classifier = OrdinalDecomposition(dtype="ordered_partitions")
    classifier.coding_matrix_ = classifier._coding_matrix(classifier.dtype, 5)

    # Predicted probabilities from a 5 class ordinal dataset (positive class)
    predictions = np.array(
        [
            [0.07495, 0.00003, 0.06861, 0.00005],
            [0.00017, 0.0, 0.03174, 0.00011],
            [0.99235, 0.04285, 0.0485, 0.00004],
            [0.95376, 0.16388, 0.03857, 0.00028],
            [0.99726, 0.20159, 0.61801, 0.00037],
            [1.0, 0.90501, 0.44459, 0.00011],
            [1.0, 0.97307, 0.99424, 0.14627],
            [1.0, 0.64663, 0.45326, 0.06143],
            [1.0, 0.83569, 0.9175, 0.94988],
            [1.0, 0.93172, 0.6774, 0.43379],
        ]
    )

    # Interpoling values from [0, 1] range to [-1, 1]
    predictions = (2 * predictions) - 1

    l_losses = classifier._logarithmic_loss(predictions)
    expected_l_losses = np.array(
        [
            [0.58553, 2.28573, 4.28561, 6.01117, 8.01097],
            [0.52385, 2.52317, 4.52317, 6.39621, 8.39577],
            [2.52807, 0.55867, 2.38727, 4.19327, 6.19311],
            [2.47122, 0.65618, 2.00066, 3.84638, 5.84526],
            [3.46591, 1.47687, 2.67051, 2.19847, 4.19699],
            [4.64297, 2.64297, 1.02293, 1.24457, 3.24413],
            [6.48375, 4.48375, 2.59147, 0.61451, 2.02943],
            [3.91936, 1.91936, 1.33284, 1.5198, 3.27408],
            [7.49674, 5.49674, 4.15398, 2.48398, 0.68446],
            [5.69657, 3.69657, 1.96969, 1.26009, 1.52493],
        ]
    )

    # Asserting similarity
    npt.assert_allclose(l_losses, expected_l_losses, rtol=1e-04, atol=0)


def test_hinge_loss_method():
    """Test that hinge loss method returns expected values for one toy problem
    (starting off predicted probabilities given by each binary classifier)."""
    classifier = OrdinalDecomposition(dtype="ordered_partitions")
    classifier.coding_matrix_ = classifier._coding_matrix(classifier.dtype, 5)

    # Predicted probabilities from a 5 class ordinal dataset (positive class)
    predictions = np.array(
        [
            [0.07495, 0.00003, 0.06861, 0.00005],
            [0.00017, 0.0, 0.03174, 0.00011],
            [0.99235, 0.04285, 0.0485, 0.00004],
            [0.95376, 0.16388, 0.03857, 0.00028],
            [0.99726, 0.20159, 0.61801, 0.00037],
            [1.0, 0.90501, 0.44459, 0.00011],
            [1.0, 0.97307, 0.99424, 0.14627],
            [1.0, 0.64663, 0.45326, 0.06143],
            [1.0, 0.83569, 0.9175, 0.94988],
            [1.0, 0.93172, 0.6774, 0.43379],
        ]
    )

    # Interpoling values from [0, 1] range to [-1, 1]
    predictions = (2 * predictions) - 1

    h_losses = classifier._hinge_loss(predictions)
    expected_h_losses = np.array(
        [
            [0.28728, 1.98748, 3.98736, 5.71292, 7.71272],
            [0.06404, 2.06336, 4.06336, 5.9364, 7.93596],
            [2.16748, 0.19808, 2.02668, 3.83268, 5.83252],
            [2.31298, 0.49794, 1.84242, 3.68814, 5.68702],
            [3.63446, 1.64542, 2.83906, 2.36702, 4.36554],
            [4.69942, 2.69942, 1.07938, 1.30102, 3.30058],
            [6.22716, 4.22716, 2.33488, 0.35792, 1.77284],
            [4.32264, 2.32264, 1.73612, 1.92308, 3.67736],
            [7.40614, 5.40614, 4.06338, 2.39338, 0.59386],
            [6.08582, 4.08582, 2.35894, 1.64934, 1.91418],
        ]
    )

    # Asserting similarity
    npt.assert_allclose(h_losses, expected_h_losses, rtol=1e-04, atol=0)


def test_ordinal_decomposition_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = OrdinalDecomposition()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])


def test_frank_hall_method_raises_error(X, y):
    """Test that using frank_hall with invalid dtype raises a ValueError."""
    classifier = OrdinalDecomposition(dtype="one_vs_next", decision_method="frank_hall")
    with pytest.raises(ValueError):
        classifier.fit(X, y)


def test_ordinal_decomposition_sets_classes_and_n_features_in_after_fit(X, y):
    """Test that classes_ and n_features_in_ are set after fit."""
    classifier = OrdinalDecomposition().fit(X, y)

    assert isinstance(classifier.classes_, np.ndarray)
    np.testing.assert_array_equal(classifier.classes_, np.unique(y))
    assert isinstance(classifier.n_features_in_, int)
    assert classifier.n_features_in_ == X.shape[1]


def test_ordinal_decomposition_predict_raises_if_not_fitted(X):
    """Test that predict raises NotFittedError if called before fit."""
    classifier = OrdinalDecomposition()
    with pytest.raises(NotFittedError):
        classifier.predict(X)


def test_ordinal_decomposition_feature_names_in_when_dataframe(X, y):
    """Test that feature_names_in_ is set when X is a DataFrame."""
    df = pd.DataFrame(X, columns=["f0", "f1"])
    classifier = OrdinalDecomposition().fit(df, y)

    assert hasattr(classifier, "feature_names_in_")
    np.testing.assert_array_equal(
        classifier.feature_names_in_, np.array(["f0", "f1"], dtype=object)
    )


def test_ordinal_decomposition_parameter_constraints_match_init_params():
    """Test that _parameter_constraints keys match __init__ parameters."""
    init_params = set(inspect.signature(OrdinalDecomposition.__init__).parameters) - {
        "self"
    }
    assert set(OrdinalDecomposition._parameter_constraints) == init_params


def test_ordinal_decomposition_predict_rejects_wrong_n_features(X, y):
    """Test that predict rejects input with mismatched n_features."""
    classifier = OrdinalDecomposition().fit(X, y)
    with pytest.raises(ValueError):
        classifier.predict(X[:, :-1])


@pytest.mark.parametrize(
    "labels",
    [
        [1, 2, 3],
        [0, 1, 2],
        [-1, 0, 1],
        [3, 5, 7],
    ],
)
def test_ordinal_decomposition_label_roundtrip(labels):
    """Test that arbitrary ordinal label sets round-trip through fit/predict."""
    labels_array = np.array(labels)
    X = np.array(
        [[i, i] for i, _ in enumerate(np.repeat(labels_array, 3))], dtype=float
    )
    y = np.repeat(labels_array, 3)

    classifier = OrdinalDecomposition(
        base_classifier="SVC",
        parameters={"C": 1.0, "gamma": "scale", "probability": True},
    ).fit(X, y)

    np.testing.assert_array_equal(classifier.classes_, np.unique(labels_array))
    assert set(classifier.predict(X)).issubset(set(np.unique(labels_array)))


def test_ordinal_decomposition_decision_method_frozen_after_fit(X, y):
    """Test that decision_method is frozen at fit time, not read live by predict."""
    classifier = OrdinalDecomposition(decision_method="frank_hall").fit(X, y)
    y_proba_before = classifier.predict_proba(X)

    classifier.set_params(decision_method="hinge_loss")
    npt.assert_allclose(classifier.predict_proba(X), y_proba_before)

    # Refitting picks up the new decision_method
    classifier.fit(X, y)
    reference = OrdinalDecomposition(decision_method="hinge_loss").fit(X, y)
    npt.assert_allclose(classifier.predict_proba(X), reference.predict_proba(X))


def test_ordinal_decomposition_frank_hall_unreachable_by_set_params(X, y):
    """Test that set_params cannot reach frank_hall over a one_vs_next coding matrix."""
    classifier = OrdinalDecomposition(
        dtype="one_vs_next", decision_method="hinge_loss"
    ).fit(X, y)
    y_proba_before = classifier.predict_proba(X)

    classifier.set_params(decision_method="frank_hall")
    npt.assert_allclose(classifier.predict_proba(X), y_proba_before)

    with pytest.raises(ValueError, match="ordered_partitions must be used"):
        classifier.fit(X, y)
