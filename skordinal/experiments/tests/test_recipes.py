"""Tests for the recipe loading and validation feature."""

from pathlib import Path

import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skordinal.experiments import Benchmark, ModelConfig, load_recipe, validate_recipe

_MINIMAL_MODELS: dict[str, ModelConfig] = {"svc": ModelConfig(SVC())}
_MINIMAL_RECIPE: dict = {
    "models": _MINIMAL_MODELS,
    "datasets": ["balance_scale"],
}

_VALID_RECIPE_SRC = """\
from sklearn.svm import SVC
from skordinal.experiments import ModelConfig

RECIPE = {
    "models": {"svc": ModelConfig(SVC())},
    "datasets": ["balance_scale"],
}
"""

_NO_RECIPE_ATTR_SRC = """\
# Intentionally missing a top-level RECIPE attribute
MODELS = {"svc": None}
"""

_RECIPE_WITH_EXTRAS_SRC = """\
from sklearn.svm import SVC
from skordinal.experiments import ModelConfig

RECIPE = {
    "models": {"svc": ModelConfig(SVC(), param_grid={"C": [0.1, 1.0]})},
    "datasets": ["balance_scale"],
    "resamples": 2,
    "results_path": "/tmp/recipes_test_out",
}
"""


def _write(tmp_path: Path, src: str, name: str = "recipe.py") -> Path:
    """Write *src* into *tmp_path/name* and return the path."""
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


_FULL_RECIPE: dict = {
    "models": _MINIMAL_MODELS,
    "datasets": ["balance_scale"],
    "data_home": "/data",
    "eval_metrics": ["mean_absolute_error"],
    "results_path": "/tmp/out",
    "resamples": 10,
    "test_size": 0.25,
    "tuning_metric": "neg_mean_absolute_error",
    "cv": 3,
    "n_jobs": 1,
    "input_preprocessing": StandardScaler(),
    "random_state": 0,
    "overwrite": True,
    "verbose": False,
}


@pytest.mark.parametrize(
    "recipe", [_MINIMAL_RECIPE, _FULL_RECIPE], ids=["minimal", "every-optional-key"]
)
def test_validate_recipe_accepts_valid_recipes(recipe):
    """The required keys alone and every optional key both pass."""
    validate_recipe(recipe)


@pytest.mark.parametrize(
    "recipe, exc_type, match",
    [
        pytest.param(
            ["models", "datasets"], TypeError, "recipe must be a dict", id="list"
        ),
        pytest.param(None, TypeError, "recipe must be a dict", id="none"),
        pytest.param(
            {"datasets": ["balance_scale"]},
            ValueError,
            "missing required keys",
            id="missing-models",
        ),
        pytest.param(
            {"models": _MINIMAL_MODELS},
            ValueError,
            "missing required keys",
            id="missing-datasets",
        ),
        pytest.param(
            {"models": _MINIMAL_MODELS, "datasets": ["era"], "bogus": True},
            ValueError,
            "unknown keys",
            id="unknown-key",
        ),
        pytest.param(
            {"models": {}, "datasets": ["era"]},
            ValueError,
            "non-empty dict",
            id="empty-models",
        ),
        pytest.param(
            {"models": [_MINIMAL_MODELS], "datasets": ["era"]},
            TypeError,
            "'models' must be a dict",
            id="models-not-a-dict",
        ),
        pytest.param(
            {"models": {"svc": SVC()}, "datasets": ["era"]},
            TypeError,
            "ModelConfig",
            id="non-modelconfig-value",
        ),
        pytest.param(
            {"models": _MINIMAL_MODELS, "datasets": "era"},
            TypeError,
            r"not a bare string; pass \['era'\]",
            id="bare-string-datasets",
        ),
        pytest.param(
            {"models": _MINIMAL_MODELS, "datasets": []},
            ValueError,
            "non-empty",
            id="empty-datasets",
        ),
    ],
)
def test_validate_recipe_rejects_invalid_structures(recipe, exc_type, match):
    """Each structural defect raises its own error, before Benchmark is built."""
    with pytest.raises(exc_type, match=match):
        validate_recipe(recipe)


def test_load_recipe_returns_the_validated_dict(tmp_path):
    """The RECIPE dict comes back whole, its ModelConfig values constructed."""
    p = _write(tmp_path, _RECIPE_WITH_EXTRAS_SRC)
    recipe = load_recipe(p)

    assert recipe["datasets"] == ["balance_scale"]
    assert recipe["resamples"] == 2
    assert all(isinstance(cfg, ModelConfig) for cfg in recipe["models"].values())


def test_load_recipe_missing_attribute_raises_attribute_error(tmp_path):
    """A recipe file without a top-level ``RECIPE`` attribute raises ``AttributeError``."""
    p = _write(tmp_path, _NO_RECIPE_ATTR_SRC)
    with pytest.raises(AttributeError, match="RECIPE"):
        load_recipe(p)


@pytest.mark.parametrize("exists", [False, True], ids=["missing", "not-python"])
def test_load_recipe_rejects_an_unloadable_path(tmp_path, exists):
    """A missing path and a file no import loader accepts both fail loud."""
    path = tmp_path / "recipe.txt"
    if exists:
        path.write_text(_VALID_RECIPE_SRC, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_recipe(path)


def test_load_recipe_is_repeatable_and_leaves_sys_modules_clean(tmp_path):
    """Repeat loads see fresh state; the synthetic module never lingers."""
    import sys

    p = _write(tmp_path, _VALID_RECIPE_SRC)
    first = load_recipe(p)
    second = load_recipe(p)

    assert list(first["datasets"]) == list(second["datasets"])
    assert f"_skordinal_recipe_{p.stem}" not in sys.modules


_FROM_RECIPE_TMPL = """\
from sklearn.svm import SVC
from skordinal.experiments import ModelConfig

RECIPE = {{
    "models": {{"svc": ModelConfig(SVC())}},
    "datasets": ["balance_scale"],
    "eval_metrics": ["mean_absolute_error"],
    "results_path": "{results_path}",
    "resamples": 2,
    "test_size": 0.25,
    "verbose": False,
}}
"""


def test_from_recipe_attributes_match_recipe(tmp_path):
    """``Benchmark.from_recipe`` returns a ``Benchmark`` whose attributes reflect the recipe."""
    results_dir = tmp_path / "out"
    recipe_src = _FROM_RECIPE_TMPL.format(results_path=str(results_dir))
    p = _write(tmp_path, recipe_src)

    b = Benchmark.from_recipe(p)

    assert isinstance(b, Benchmark)
    assert b.datasets == ["balance_scale"]
    assert b.eval_metrics == ["mean_absolute_error"]
    assert b.resamples == 2
    assert b.test_size == 0.25
    assert b.verbose is False
    assert "svc" in b.models
    for key, cfg in b.models.items():
        assert isinstance(cfg, ModelConfig), (
            f"b.models[{key!r}] is {type(cfg)!r}, not ModelConfig"
        )


def test_from_recipe_override_wins_over_recipe(tmp_path):
    """Keyword overrides passed to ``from_recipe`` take precedence over recipe values."""
    recipe_src = _FROM_RECIPE_TMPL.format(results_path=str(tmp_path / "default_out"))
    p = _write(tmp_path, recipe_src)

    override_path = tmp_path / "override_out"
    b = Benchmark.from_recipe(p, results_path=override_path, resamples=7)

    assert Path(b.results_path) == override_path
    assert b.resamples == 7


def test_from_recipe_missing_file_raises(tmp_path):
    """``Benchmark.from_recipe`` propagates ``FileNotFoundError`` for bad paths."""
    with pytest.raises(FileNotFoundError):
        Benchmark.from_recipe(tmp_path / "ghost_recipe.py")
