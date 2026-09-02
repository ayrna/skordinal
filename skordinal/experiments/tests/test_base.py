"""Tests for the experiments shared validation helpers."""

import os

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skordinal.experiments._base import (
    _check_input_preprocessing,
    _check_metric_names,
    _check_path_component,
    _check_resample_id,
    _compute_metric,
)


@pytest.mark.parametrize(
    "bad, exc_type, match",
    [
        pytest.param("", ValueError, "empty or a dot segment", id="empty"),
        pytest.param(".", ValueError, "empty or a dot segment", id="dot"),
        pytest.param("..", ValueError, "empty or a dot segment", id="dot-dot"),
        pytest.param("a/b", ValueError, "path separator", id="slash"),
        pytest.param(f"a{os.sep}b", ValueError, "path separator", id="os-sep"),
        # os.altsep is None on POSIX, so a backslash must be rejected outright
        pytest.param("x\\y", ValueError, "path separator", id="backslash"),
        pytest.param("..\\esc", ValueError, "path separator", id="backslash-up"),
        # "D:" would resolve outside the results root once read on Windows
        pytest.param("D:", ValueError, "colon", id="drive"),
        pytest.param(3, TypeError, "must be a str", id="int"),
        pytest.param(None, TypeError, "must be a str", id="none"),
        pytest.param(("x",), TypeError, "must be a str", id="tuple"),
    ],
)
def test_path_component_rejects_unsafe_names(bad, exc_type, match):
    """Every name that cannot safely name one directory raises."""
    with pytest.raises(exc_type, match=match):
        _check_path_component(bad, "model label")


@pytest.mark.parametrize("good", [0, -1, np.int64(3), "0"])
def test_resample_id_accepts_int_like(good):
    """Ints, numpy ints and int-like strings pass through."""
    _check_resample_id(good)


@pytest.mark.parametrize("bad", ["../../../../tmp/evil", "..", "", "a/b"])
def test_resample_id_rejects_traversal(bad):
    """A non-int id that fails path validation raises."""
    with pytest.raises(ValueError, match="resample_id"):
        _check_resample_id(bad)


@pytest.mark.parametrize(
    "bad, exc_type, match",
    [
        pytest.param(iter([]), ValueError, "non-empty", id="empty-one-shot-iterable"),
        pytest.param("", TypeError, "not a bare string", id="bare-string"),
        pytest.param(None, TypeError, "got NoneType", id="not-iterable"),
        pytest.param([3], TypeError, "only metric name strings", id="non-str-element"),
    ],
)
def test_metric_names_rejects_degenerate_input(bad, exc_type, match):
    """Emptiness is judged after materialising, so lazy iterables cannot hide."""
    with pytest.raises(exc_type, match=match):
        _check_metric_names(bad, param="eval_metrics")


@pytest.mark.parametrize(
    "metrics",
    [
        ["mean_absolute_error", " mean_absolute_error ", "accuracy_score"],
        np.array(["mean_absolute_error", " accuracy_score "]),
    ],
    ids=["repeated-name", "array-like"],
)
def test_metric_names_stripped_and_deduplicated(metrics):
    """Names come back stripped and unique, whatever iterable carried them."""
    expected = ["mean_absolute_error", "accuracy_score"]
    assert _check_metric_names(metrics, param="m") == expected


def test_input_preprocessing_accepts_a_transformer():
    """A transformer instance and None both pass the check."""
    _check_input_preprocessing(StandardScaler())
    _check_input_preprocessing(None)


@pytest.mark.parametrize(
    "bad",
    [SVC(), "std", StandardScaler],
    ids=["no-transform", "old-token", "class-not-instance"],
)
def test_input_preprocessing_rejects_a_non_transformer(bad):
    """A non-transformer, a removed token or a class (forgotten parens) raises."""
    with pytest.raises(TypeError, match="'input_preprocessing' must be None"):
        _check_input_preprocessing(bad)


def test_compute_metric_forwards_labels_where_accepted():
    """labels= reaches the metrics that take it and is dropped for the rest."""
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 2, 0, 2])
    scale = np.array([0, 1, 2])
    # With the middle class named, the extremes stay two apart, not adjacent
    assert _compute_metric("accuracy_off1_score", y_true, y_pred) == 1.0
    assert _compute_metric(
        "accuracy_off1_score", y_true, y_pred, labels=scale
    ) == pytest.approx(0.5)
    # accuracy_score takes no labels, so passing them must not raise
    assert _compute_metric(
        "accuracy_score", y_true, y_pred, labels=scale
    ) == pytest.approx(0.5)
