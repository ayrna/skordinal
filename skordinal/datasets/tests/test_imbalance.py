"""Tests for target-based dataset imbalance metrics."""

import numpy as np
import pytest

from skordinal.datasets import (
    compute_dataset_air,
    compute_dataset_mir,
    load_dataset,
)


def test_imbalance_metrics_from_target_counts():
    """AIR and MIR are computed from class frequencies in a target vector."""
    y = np.array([0, 0, 0, 0, 1, 1, 2])

    # Counts [4, 2, 1] yield class IRs [3/8, 5/4, 3]: AIR is their mean,
    # (3/8 + 5/4 + 3) / 3 = 37/24. MIR is 4 / 1 = 4.
    assert compute_dataset_air(y) == pytest.approx(37 / 24)
    assert compute_dataset_mir(y) == pytest.approx(4)


def test_imbalance_metrics_support_string_labels():
    """Class labels need not be numeric or ordinally ordered."""
    y = ["small", "small", "large", "large"]

    # With two balanced classes, both class IRs are 1.
    assert compute_dataset_air(y) == pytest.approx(1)
    assert compute_dataset_mir(y) == pytest.approx(1)


def test_imbalance_metrics_accept_target_from_load_dataset():
    """The loader's target can be passed directly to each metric."""
    target = load_dataset("balance_scale").target
    # Balance Scale has class counts [288, 49, 288] (left, balanced, right).
    # Derive expected AIR/MIR independently from those documented frequencies.
    counts = np.array([288, 49, 288])
    class_irs = (counts.sum() - counts) / ((len(counts) - 1) * counts)

    assert compute_dataset_air(target) == pytest.approx(class_irs.mean())
    assert compute_dataset_mir(target) == pytest.approx(counts.max() / counts.min())


def test_single_class_target_raises_value_error():
    """At least two classes are required to compute imbalance ratios."""
    with pytest.raises(ValueError, match="at least two distinct classes"):
        compute_dataset_air([1, 1, 1])


def test_multidimensional_target_raises_value_error():
    """Targets must be one-dimensional class labels, not one-hot matrices."""
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_dataset_mir([[1, 0], [0, 1]])


def test_target_parameter_is_validated():
    """Public functions reject values that are not array-like targets."""
    with pytest.raises(ValueError, match="array-like"):
        compute_dataset_air(None)
