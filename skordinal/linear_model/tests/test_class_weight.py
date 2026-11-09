"""Class weight validation shared by the linear ordinal classifiers."""

import numpy as np
import pytest

from skordinal.linear_model import POM, LogisticAT, LogisticIT


@pytest.mark.parametrize("estimator_class", [POM, LogisticAT, LogisticIT])
@pytest.mark.parametrize(
    "class_weight",
    [
        {10: 0.0, 20: 0.0, 30: 0.0},
        {10: -1.0, 20: 1.0, 30: 1.0},
        {10: np.nan},
        {10: np.inf},
        {10: -np.inf},
        {10: 1e308, 20: 1e308, 30: 1e308},
    ],
    ids=["zero", "negative", "nan", "infinity", "negative-infinity", "overflow"],
)
def test_fit_rejects_invalid_class_weights(estimator_class, class_weight):
    """Invalid weights raise before an undefined training objective is evaluated."""
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.repeat([10, 20, 30], 2)

    with pytest.raises(ValueError, match="weight"):
        estimator_class(class_weight=class_weight).fit(X, y)


@pytest.mark.parametrize("estimator_class", [POM, LogisticAT, LogisticIT])
def test_fit_accepts_individual_zero_class_weights(estimator_class):
    """A zero class weight remains valid when some samples have positive weight."""
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.repeat([10, 20, 30], 2)

    classifier = estimator_class(alpha=1.0, class_weight={10: 0.0}).fit(X, y)

    assert np.isfinite(classifier.loss_)
    assert np.isfinite(classifier.predict_proba(X)).all()
