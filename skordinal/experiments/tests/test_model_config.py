"""Tests for the ModelConfig frozen dataclass."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from skordinal.experiments import ModelConfig


def test_modelconfig_stores_estimator_and_grid():
    """Store estimator by identity and param_grid (None default, verbatim)."""
    lr = LogisticRegression(max_iter=200)
    cfg_default = ModelConfig(lr)
    assert cfg_default.estimator is lr
    assert cfg_default.param_grid is None

    grid = {"C": [0.1, 1.0, 10.0]}
    cfg_with_grid = ModelConfig(LogisticRegression(), param_grid=grid)
    assert cfg_with_grid.param_grid is grid


def test_modelconfig_fields_are_frozen():
    """Assigning to a field of a frozen ``ModelConfig`` raises an error."""
    cfg = ModelConfig(LogisticRegression(), param_grid={"C": [1.0]})
    with pytest.raises((AttributeError, TypeError)):
        cfg.estimator = LogisticRegression(max_iter=500)  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_estimator",
    ["not an estimator", 42, None],
    ids=["string", "int", "none"],
)
def test_modelconfig_rejects_non_estimator(bad_estimator):
    """Reject a non-``BaseEstimator`` with a ``TypeError``."""
    with pytest.raises(TypeError, match="BaseEstimator"):
        ModelConfig(bad_estimator)


@pytest.mark.parametrize(
    "bad_grid",
    [[1, 2], "C=0.1", 42],
    ids=["list", "string", "int"],
)
def test_modelconfig_rejects_bad_param_grid(bad_grid):
    """Reject a non-dict, non-None ``param_grid`` with a ``TypeError``."""
    with pytest.raises(TypeError, match="param_grid"):
        ModelConfig(LogisticRegression(), param_grid=bad_grid)


def test_modelconfig_param_grid_is_keyword_only():
    """``param_grid`` as a second positional argument raises ``TypeError``."""
    with pytest.raises(TypeError):
        ModelConfig(LogisticRegression(), {"C": [1]})  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "param_grid, expected",
    [
        (None, False),
        ({}, False),
        ({"C": [0.5]}, False),
        ({"C": [0.1, 1.0]}, True),
        ({"C": [0.1, 1.0], "max_iter": 200}, True),
    ],
    ids=["none", "empty", "singleton", "multi_value", "mixed_multi"],
)
def test_needs_search(param_grid, expected):
    """Return True iff a grid value is a multi-element list or tuple."""
    cfg = ModelConfig(LogisticRegression(), param_grid=param_grid)
    assert cfg.needs_search is expected


@pytest.mark.parametrize(
    "param_grid, expected",
    [
        (None, {}),
        ({}, {}),
        ({"C": [0.5]}, {"C": 0.5}),
        ({"C": 0.5}, {"C": 0.5}),
        ({"C": []}, {}),
    ],
    ids=["none", "empty", "singleton_list", "scalar", "empty_list"],
)
def test_fixed_params(param_grid, expected):
    """Unwrap singleton lists, pass scalars through, skip empty lists."""
    cfg = ModelConfig(LogisticRegression(), param_grid=param_grid)
    assert cfg.fixed_params() == expected


def test_build_returns_distinct_clone_without_mutating_source():
    """``build`` returns a fresh clone and does not mutate the source."""
    cfg = ModelConfig(DecisionTreeClassifier(random_state=0))
    built = cfg.build(7)
    assert built is not cfg.estimator
    assert cfg.estimator.random_state == 0


@pytest.mark.parametrize(
    "estimator, check",
    [
        (
            DecisionTreeClassifier(),
            lambda built: built.random_state == 7,
        ),
        (
            Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier())]),
            lambda built: built.named_steps["clf"].random_state == 7,
        ),
    ],
    ids=["direct_estimator", "pipeline_clf_step"],
)
def test_build_forwards_random_state(estimator, check):
    """``build(7)`` sets the seed on the clone (direct or Pipeline)."""
    cfg = ModelConfig(estimator)
    built = cfg.build(7)
    assert check(built)


@pytest.mark.parametrize(
    "estimator, check",
    [
        (
            StandardScaler(),
            lambda built: "random_state" not in built.get_params(),
        ),
        (
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]),
            lambda built: built.named_steps["model"].random_state is None,
        ),
    ],
    ids=["no_random_state", "pipeline_step_not_clf"],
)
def test_build_ignores_random_state_when_absent(estimator, check):
    """``build(7)`` returns a clone and injects no seed when none applies."""
    cfg = ModelConfig(estimator)
    built = cfg.build(7)
    assert built is not cfg.estimator
    assert check(built)


def test_build_none_seed_preserves_defaults():
    """``build(None)`` returns a distinct clone, keeping defaults."""
    cfg = ModelConfig(DecisionTreeClassifier(random_state=5))
    built = cfg.build(None)
    assert built is not cfg.estimator
    assert built.random_state == 5
