"""Tests for the experiments shared validation helpers."""

import os
from pathlib import PureWindowsPath

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


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", f"a{os.sep}b"])
def test_check_path_component_rejects_bad_strings(bad):
    """_check_path_component rejects empty, dotted or separator names."""
    with pytest.raises(ValueError):
        _check_path_component(bad, "classifier_name")


@pytest.mark.parametrize("bad", [3, None, ("x",)])
def test_check_path_component_rejects_non_str(bad):
    """_check_path_component rejects a non-string component."""
    with pytest.raises(TypeError):
        _check_path_component(bad, "classifier_name")


def test_check_path_component_rejects_a_windows_drive():
    """A drive-qualified name would resolve outside the results root."""
    # No Windows runner needed: PureWindowsPath shows the escape anywhere
    assert PureWindowsPath("C:/runs/exp") / "D:" / "ds" == PureWindowsPath("D:ds")
    with pytest.raises(ValueError, match="must not contain a colon"):
        _check_path_component("D:", "model label")


@pytest.mark.parametrize("bad", ["x\\y", "..\\esc"], ids=["nested", "traversal"])
def test_check_path_component_rejects_a_backslash(bad):
    """os.altsep is None on POSIX, so a backslash must be rejected outright."""
    with pytest.raises(ValueError, match="must not contain a path separator"):
        _check_path_component(bad, "model label")


@pytest.mark.parametrize("good", [0, -1, np.int64(3), "0"])
def test_check_resample_id_accepts_int_like(good):
    """_check_resample_id passes through ints, numpy ints and int-like strings."""
    _check_resample_id(good)


@pytest.mark.parametrize("bad", ["../../../../tmp/evil", "..", "", "a/b"])
def test_check_resample_id_rejects_traversal(bad):
    """_check_resample_id rejects a non-int id that fails path validation."""
    with pytest.raises(ValueError, match="resample_id"):
        _check_resample_id(bad)


@pytest.mark.parametrize(
    "bad, exc_type, match",
    [
        pytest.param(
            iter([]),
            ValueError,
            "non-empty",
            id="empty-one-shot-iterable",
        ),
        pytest.param("", TypeError, "not a bare string", id="empty-string"),
        pytest.param(None, TypeError, "got NoneType", id="none"),
    ],
)
def test_check_metric_names_rejects_degenerate_input(bad, exc_type, match):
    """Emptiness is judged after materialising, so lazy iterables cannot hide."""
    with pytest.raises(exc_type, match=match):
        _check_metric_names(bad, param="eval_metrics")


def test_check_metric_names_drops_an_exact_duplicate():
    """A repeat would compute the metric twice and duplicate evaluate's column."""
    names = _check_metric_names(
        ["mean_absolute_error", " mean_absolute_error ", "accuracy_score"], param="m"
    )
    assert names == ["mean_absolute_error", "accuracy_score"]


def test_check_metric_names_accepts_array_likes():
    """A numpy array of names materialises instead of raising ambiguous truth."""
    names = _check_metric_names(
        np.array(["mean_absolute_error", " accuracy_score "]), param="metrics"
    )
    assert names == ["mean_absolute_error", "accuracy_score"]


def test_check_input_preprocessing_accepts_a_transformer():
    """A transformer instance and None both pass the check."""
    _check_input_preprocessing(StandardScaler())
    _check_input_preprocessing(None)


@pytest.mark.parametrize(
    "bad",
    [SVC(), "std", StandardScaler],
    ids=["no-transform", "old-token", "class-not-instance"],
)
def test_check_input_preprocessing_rejects_a_non_transformer(bad):
    """A non-transformer, a removed token or a class (forgotten parens) raises."""
    with pytest.raises(TypeError, match="'input_preprocessing' must be None"):
        _check_input_preprocessing(bad)


def test_compute_metric_names_the_scale_only_where_accepted():
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
