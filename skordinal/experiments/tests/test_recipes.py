"""Tests for the recipe loading and validation feature."""

from pathlib import Path

import pytest
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


def test_validate_recipe_minimal_valid_does_not_raise():
    """A minimal recipe with ``models`` and ``datasets`` passes without error."""
    validate_recipe(_MINIMAL_RECIPE)


def test_validate_recipe_all_optional_keys_accepted():
    """A recipe with every optional key accepted alongside the required keys."""
    full: dict = {
        "models": _MINIMAL_MODELS,
        "datasets": ["balance_scale"],
        "data_home": "/data",
        "eval_metrics": ["mean_absolute_error"],
        "results_path": "/tmp/out",
        "resamples": 10,
        "tuning_metric": "neg_mean_absolute_error",
        "cv": 3,
        "n_jobs": 1,
        "input_preprocessing": "std",
        "random_state": 0,
        "verbose": False,
    }
    validate_recipe(full)


@pytest.mark.parametrize(
    "bad_recipe",
    [
        pytest.param(["models", "datasets"], id="list-not-dict"),
        pytest.param(None, id="none-not-dict"),
    ],
)
def test_validate_recipe_not_dict_raises_type_error(bad_recipe):
    """A non-dict recipe raises ``TypeError`` mentioning the type."""
    with pytest.raises(TypeError, match="recipe must be a dict"):
        validate_recipe(bad_recipe)


def test_validate_recipe_missing_models_raises_value_error():
    """A recipe without ``models`` raises ``ValueError`` mentioning the key."""
    with pytest.raises(ValueError, match="missing required keys"):
        validate_recipe({"datasets": ["balance_scale"]})


def test_validate_recipe_missing_datasets_raises_value_error():
    """A recipe without ``datasets`` raises ``ValueError`` mentioning the key."""
    with pytest.raises(ValueError, match="missing required keys"):
        validate_recipe({"models": _MINIMAL_MODELS})


def test_validate_recipe_empty_models_raises_value_error():
    """An empty ``models`` dict raises ``ValueError``."""
    with pytest.raises(ValueError, match="non-empty dict"):
        validate_recipe({"models": {}, "datasets": ["balance_scale"]})


def test_validate_recipe_models_value_not_modelconfig_raises_type_error():
    """A ``models`` value that is not a ``ModelConfig`` raises ``TypeError``."""
    with pytest.raises(TypeError, match="ModelConfig"):
        validate_recipe(
            {
                "models": {"svc": SVC()},  # raw estimator, not wrapped
                "datasets": ["balance_scale"],
            }
        )


def test_validate_recipe_empty_datasets_raises_value_error():
    """An empty ``datasets`` list raises ``ValueError``."""
    with pytest.raises(ValueError, match="non-empty"):
        validate_recipe({"models": _MINIMAL_MODELS, "datasets": []})


def test_validate_recipe_unknown_key_raises_value_error():
    """An unknown top-level key raises ``ValueError`` naming the unknown key."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_recipe(
            {
                "models": _MINIMAL_MODELS,
                "datasets": ["balance_scale"],
                "this_key_is_not_allowed": True,
            }
        )


def test_load_recipe_valid_file_returns_dict(tmp_path):
    """``load_recipe`` on a valid file returns a validated dict with ``ModelConfig`` values."""
    p = _write(tmp_path, _VALID_RECIPE_SRC)
    recipe = load_recipe(p)

    assert isinstance(recipe, dict)
    assert "models" in recipe
    assert "datasets" in recipe
    for name, cfg in recipe["models"].items():
        assert isinstance(cfg, ModelConfig), (
            f"models[{name!r}] is {type(cfg)!r}, expected ModelConfig"
        )


def test_load_recipe_missing_attribute_raises_attribute_error(tmp_path):
    """A recipe file without a top-level ``RECIPE`` attribute raises ``AttributeError``."""
    p = _write(tmp_path, _NO_RECIPE_ATTR_SRC)
    with pytest.raises(AttributeError, match="RECIPE"):
        load_recipe(p)


def test_load_recipe_file_not_found_raises(tmp_path):
    """A non-existent path raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_recipe(tmp_path / "nonexistent_recipe.py")


def test_load_recipe_is_repeatable(tmp_path):
    """Calling ``load_recipe`` twice on the same file returns equal results."""
    p = _write(tmp_path, _VALID_RECIPE_SRC)
    first = load_recipe(p)
    second = load_recipe(p)

    assert set(first.keys()) == set(second.keys())
    assert list(first["datasets"]) == list(second["datasets"])


def test_load_recipe_does_not_pollute_sys_modules(tmp_path):
    """The synthetic module name is removed from ``sys.modules`` after loading."""
    import sys

    p = _write(tmp_path, _VALID_RECIPE_SRC)
    module_name = f"_skordinal_recipe_{p.stem}"

    load_recipe(p)

    assert module_name not in sys.modules


def test_load_recipe_with_extra_fields_returns_all_keys(tmp_path):
    """Optional keys declared in the recipe are present in the returned dict."""
    p = _write(tmp_path, _RECIPE_WITH_EXTRAS_SRC)
    recipe = load_recipe(p)

    assert "resamples" in recipe
    assert recipe["resamples"] == 2


_FROM_RECIPE_TMPL = """\
from sklearn.svm import SVC
from skordinal.experiments import ModelConfig

RECIPE = {{
    "models": {{"svc": ModelConfig(SVC())}},
    "datasets": ["balance_scale"],
    "eval_metrics": ["mean_absolute_error"],
    "results_path": "{results_path}",
    "resamples": 2,
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
