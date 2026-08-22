"""Tests for parameter grid preparation and validation utilities."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skordinal.classifiers import (
    NNOP,
    NNPOM,
    REDSVM,
    SVOREX,
    SVORIM,
    OrdinalDecomposition,
)
from skordinal.model_selection._validation import (
    _add_random_state,
    _normalize_param_grid,
    _prepare_parameters_for_ensemble,
    check_for_random_state,
    is_ensemble,
    is_searchcv,
    prepare_param_grid,
)
from skordinal.utils._testing import TEST_RANDOM_STATE


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (None, False),
        (NNOP, True),
        (NNPOM, True),
        (OrdinalDecomposition, False),
        (REDSVM, False),
        (SVOREX, False),
        (SVORIM, False),
        (SVC, True),
        (LogisticRegression, True),
    ],
)
def test_check_for_random_state(estimator, expected):
    """Test that check_for_random_state correctly identifies classifiers that
    use random state."""
    assert check_for_random_state(estimator) is expected


@pytest.mark.parametrize(
    "param_grid, expected",
    [
        ({"C": 0.1}, False),
        ({"C": [0.1, 1]}, False),
        ({"base_classifier": "SVC"}, True),
        (
            {
                "base_classifier": "SVC",
                "parameters": {"C": [1], "gamma": [0.1]},
            },
            True,
        ),
    ],
)
def test_is_ensemble(param_grid, expected):
    """Test that is_ensemble correctly identifies ensemble methods with
    various parameter configurations."""
    assert is_ensemble(param_grid) is expected


@pytest.mark.parametrize(
    "param_grid, expected",
    [
        ({"C": 0.1}, False),
        ({"C": [0.1, 1]}, True),
        ({"C": [0.1], "kernel": "linear"}, False),
        ({"kernel_type": 0, "c": [0.1, 1], "k": [0.1, 1], "t": 0.001}, True),
        (
            {
                "base_classifier": "SVC",
                "parameters": {"C": 0.01, "gamma": 0.01},
            },
            False,
        ),
        (
            {
                "base_classifier": "SVC",
                "parameters": {"C": [0.01, 1], "gamma": [0.01, 1]},
            },
            True,
        ),
        (
            {
                "base_classifier": "SVC",
                "dtype": ["ordered", "unordered"],
                "parameters": None,
            },
            True,
        ),
    ],
)
def test_is_searchcv(param_grid, expected):
    """Test that is_searchcv correctly identifies cross-validation cases with
    various parameter configurations."""
    assert is_searchcv(param_grid) is expected


def test_is_searchcv_invalid_input():
    """Test that is_searchcv returns False when param_grid is not a dictionary."""
    assert not is_searchcv(None)
    assert not is_searchcv("not a dict")
    assert not is_searchcv(["param", "values"])


@pytest.mark.parametrize(
    "param_grid, expected",
    [
        (
            {
                "base_classifier": "SVC",
                "parameters": [{"C": 1, "gamma": 0.1}, {"C": 10, "gamma": 1.0}],
            },
            True,
        ),
        (
            {
                "base_classifier": "SVC",
                "parameters": [{"C": 1, "gamma": 0.1}],
            },
            False,
        ),
        (
            {
                "base_classifier": "SVC",
                "parameters": None,
                "dtype": ["ordered", "unordered"],
            },
            True,
        ),
        (
            {
                "base_classifier": "SVC",
                "parameters": [{"C": 1}],
                "dtype": ["ordered", "unordered"],
            },
            True,
        ),
    ],
)
def test_is_searchcv_with_ensemble_variants(param_grid, expected):
    """Test that is_searchcv detects cross-validation when ensemble variants are used."""
    assert is_searchcv(param_grid) is expected


def test_prepare_param_grid_no_cv():
    """Test that prepare_param_grid correctly handles non-cross-validation case."""
    param_grid = {"C": 0.1}
    result = prepare_param_grid(SVC, param_grid, TEST_RANDOM_STATE)
    assert result == {"C": 0.1, "random_state": TEST_RANDOM_STATE}


def test_prepare_param_grid_with_cv():
    """Test that prepare_param_grid correctly handles cross-validation case."""
    param_grid = {"C": [0.1, 1.0]}
    result = prepare_param_grid(SVC, param_grid, TEST_RANDOM_STATE)
    assert result == {"C": [0.1, 1.0], "random_state": [TEST_RANDOM_STATE]}


def test_prepare_parame_grid_mixed():
    """Test that prepare_param_grid correctly handles mixed single and multiple
    parameters."""
    param_grid = {"C": [0.1, 1.0], "gamma": 0.1}
    result = prepare_param_grid(SVC, param_grid, TEST_RANDOM_STATE)
    assert result == {
        "C": [0.1, 1.0],
        "gamma": [0.1],
        "random_state": [TEST_RANDOM_STATE],
    }


def test_prepare_param_grid_without_random_state():
    """Test that prepare_param_grid correctly handles cases without specifying a
    random seed."""
    param_grid = {"C": [0.1, 1.0]}
    result = prepare_param_grid(SVC, param_grid)
    assert "random_state" in result


def test_prepare_param_grid_simple_ensemble():
    """Test that prepare_param_grid correctly handles simple ensemble methods."""
    param_grid = {
        "dtype": "OrderedPartitions",
        "base_classifier": "SVC",
        "parameters": {"C": [1], "gamma": [1]},
    }
    result = prepare_param_grid(OrdinalDecomposition, param_grid, TEST_RANDOM_STATE)
    assert result["parameters"] == {
        "C": 1,
        "gamma": 1,
        "random_state": TEST_RANDOM_STATE,
    }


def test_prepare_param_grid_cv_ensemble():
    """Test that prepare_param_grid correctly handles ensemble methods with
    cross-validation for base_classifier."""
    param_grid = {
        "dtype": "OrderedPartitions",
        "base_classifier": "SVC",
        "parameters": {"C": [1, 10], "gamma": [1, 10], "probability": ["True"]},
    }
    prepared_params = prepare_param_grid(
        OrdinalDecomposition, param_grid, TEST_RANDOM_STATE
    )
    expected_params = {
        "dtype": ["OrderedPartitions"],
        "base_classifier": ["SVC"],
        "parameters": [
            {
                "C": 1,
                "gamma": 1,
                "probability": True,
                "random_state": TEST_RANDOM_STATE,
            },
            {
                "C": 1,
                "gamma": 10,
                "probability": True,
                "random_state": TEST_RANDOM_STATE,
            },
            {
                "C": 10,
                "gamma": 1,
                "probability": True,
                "random_state": TEST_RANDOM_STATE,
            },
            {
                "C": 10,
                "gamma": 10,
                "probability": True,
                "random_state": TEST_RANDOM_STATE,
            },
        ],
    }
    assert prepared_params == expected_params


def test_prepare_param_grid_invalid_input():
    """Test that prepare_param_grid raises error with invalid input."""
    with pytest.raises(ValueError, match="param_grid must be a dictionary"):
        prepare_param_grid(None, "not a dict")


def test_add_random_state():
    """Test that _add_random_state adds random_state if missing."""
    param_grid = {"C": 1.0}
    updated = _add_random_state(SVC, param_grid.copy(), TEST_RANDOM_STATE)
    assert updated["random_state"] == TEST_RANDOM_STATE

    param_grid = {"C": 1.0, "random_state": 999}
    updated = _add_random_state(SVC, param_grid.copy(), TEST_RANDOM_STATE)
    assert updated["random_state"] == 999

    param_grid = {"C": 1.0}
    updated = _add_random_state(NNOP, param_grid.copy(), TEST_RANDOM_STATE)
    assert updated["random_state"] == TEST_RANDOM_STATE


def test_normalize_param_grid():
    """Test that _normalize_param_grid wraps all scalar values into lists."""
    param_grid = {"C": 1.0, "kernel": "linear"}
    normalized = _normalize_param_grid(param_grid)
    assert normalized == {"C": [1.0], "kernel": ["linear"]}

    param_grid = {"C": [0.1, 1.0], "gamma": [0.01]}
    normalized = _normalize_param_grid(param_grid)
    assert normalized == param_grid


def test_prepare_parameters_for_ensemble():
    """Test that _prepare_parameters_for_ensemble correctly prepares parameters
    for ensemble methods."""
    param_grid = {
        "base_classifier": "SVC",
        "parameters": {"C": [1, 10], "gamma": [0.1, 1]},
    }
    result = _prepare_parameters_for_ensemble(param_grid, TEST_RANDOM_STATE)
    assert isinstance(result["parameters"], list)
    assert len(result["parameters"]) == 4
    assert all("random_state" in d for d in result["parameters"])

    param_grid = {"C": [1, 10], "gamma": [0.1, 1]}
    result = _prepare_parameters_for_ensemble(param_grid, TEST_RANDOM_STATE)
    assert result == param_grid


def test_prepare_parameters_for_ensemble_adds_random_state():
    """Test that _prepare_parameters_for_ensemble adds random_state if supported."""
    param_grid = {
        "base_classifier": "SVC",
        "parameters": {"C": [1], "gamma": [0.1]},
    }
    result = _prepare_parameters_for_ensemble(param_grid, TEST_RANDOM_STATE)

    assert all("random_state" in d for d in result["parameters"])
    assert all(d["random_state"] == TEST_RANDOM_STATE for d in result["parameters"])

    param_grid = {
        "base_classifier": "LogisticRegression",
        "parameters": {"C": [1.0], "penalty": ["l2"]},
    }
    result = _prepare_parameters_for_ensemble(param_grid, TEST_RANDOM_STATE)

    assert isinstance(result["parameters"], list)
    for params in result["parameters"]:
        assert "random_state" in params
        assert params["random_state"] == TEST_RANDOM_STATE

    param_grid = {
        "base_classifier": "SVOREX",
        "parameters": {"C": ["0.1"], "kappa": [0.001]},
    }
    result = _prepare_parameters_for_ensemble(param_grid, TEST_RANDOM_STATE)

    assert all("random_state" not in d for d in result["parameters"])


def test_prepare_parameters_for_ensemble_literal_eval_fallback():
    """Test that _prepare_parameters_for_ensemble ignores ValueError in literal_eval
    and leaves strings."""
    param_grid = {
        "base_classifier": "SVC",
        "parameters": {
            "C": [1],
            "gamma": [0.1],
            "kernel": ["linear"],
        },
    }

    result = _prepare_parameters_for_ensemble(param_grid, TEST_RANDOM_STATE)
    assert isinstance(result["parameters"][0]["kernel"], str)
    assert result["parameters"][0]["kernel"] == "linear"
