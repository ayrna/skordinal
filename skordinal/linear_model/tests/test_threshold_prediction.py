"""Boundary decisions of the logistic threshold models."""

import numpy as np
import pytest

from skordinal.linear_model import LogisticAT, LogisticIT


@pytest.mark.parametrize("estimator_class", [LogisticAT, LogisticIT])
def test_predict_repeated_thresholds_and_adjacent_floats(estimator_class):
    """An exact tie precedes all repeated thresholds at that value."""
    classifier = estimator_class()
    classifier.classes_ = np.array([-10, 20, 50, 80])
    classifier.coef_ = np.array([1.0])
    classifier.thresholds_ = np.array([-0.5, 0.5, 0.5])
    classifier.n_features_in_ = 1
    X = np.array(
        [
            [-1e100],
            [-0.5],
            [np.nextafter(-0.5, np.inf)],
            [np.nextafter(0.5, -np.inf)],
            [0.5],
            [np.nextafter(0.5, np.inf)],
            [1e100],
        ]
    )

    np.testing.assert_array_equal(classifier.predict(X), [-10, -10, 20, 20, 20, 80, 80])
