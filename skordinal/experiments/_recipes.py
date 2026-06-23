"""Recipe loader and validator for benchmark configurations."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ._model_config import ModelConfig

_REQUIRED_KEYS: frozenset[str] = frozenset({"models", "datasets"})
_OPTIONAL_KEYS: frozenset[str] = frozenset(
    {
        "data_home",
        "eval_metrics",
        "results_path",
        "resamples",
        "tuning_metric",
        "cv",
        "n_jobs",
        "input_preprocessing",
        "random_state",
        "verbose",
    }
)
_ALLOWED_KEYS: frozenset[str] = _REQUIRED_KEYS | _OPTIONAL_KEYS


def validate_recipe(recipe: dict) -> None:
    """Structurally validate a recipe dict.

    Checks that ``recipe`` is a dict containing the required keys
    ``models`` and ``datasets``, that ``models`` is a non-empty dict
    mapping string labels to ``ModelConfig`` instances, and that
    ``datasets`` is a non-empty sequence.  All further value validation
    (scorer names, numeric ranges, estimator types) is deferred to
    ``Benchmark.__init__``.

    Parameters
    ----------
    recipe : dict
        Candidate recipe.  Keys must be a subset of the ``Benchmark``
        constructor parameter names.  ``models`` must be a non-empty
        dict of ``str`` -> ``ModelConfig``; ``datasets`` must be a
        non-empty list.

    Raises
    ------
    TypeError
        If ``recipe`` is not a dict, or if any value in ``models`` is
        not a ``ModelConfig`` instance.

    ValueError
        If required keys are missing, unknown keys are present,
        ``models`` is empty, or ``datasets`` is empty.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from skordinal.experiments import ModelConfig, validate_recipe
    >>> validate_recipe(
    ...     {"models": {"svc": ModelConfig(SVC())}, "datasets": ["era"]}
    ... )
    """
    if not isinstance(recipe, dict):
        raise TypeError(f"recipe must be a dict; got {type(recipe)!r}.")

    unknown = set(recipe) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"recipe contains unknown keys: {sorted(unknown)}. "
            f"Allowed keys: {sorted(_ALLOWED_KEYS)}."
        )

    missing = _REQUIRED_KEYS - set(recipe)
    if missing:
        raise ValueError(f"recipe missing required keys: {sorted(missing)}.")

    models = recipe["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError(f"'models' must be a non-empty dict; got {type(models)!r}.")
    bad = [k for k, v in models.items() if not isinstance(v, ModelConfig)]
    if bad:
        raise TypeError(
            f"All values in 'models' must be ModelConfig instances; "
            f"got non-ModelConfig value(s) for key(s): {bad}."
        )

    datasets = recipe["datasets"]
    if not datasets:
        raise ValueError("'datasets' must be a non-empty list; got an empty sequence.")


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Import a recipe ``.py`` file and return its validated ``RECIPE`` dict.

    The file is imported under a synthetic module name, which is removed from
    ``sys.modules`` after the call so successive loads of the same path always
    pick up fresh state.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the recipe file.

    Returns
    -------
    recipe : dict
        The top-level ``RECIPE`` dict, validated by ``validate_recipe``.
        Keys are a subset of the ``Benchmark`` constructor parameter names
        (e.g. ``models``, ``datasets``, ``eval_metrics``, ...).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or the module spec cannot be created.

    AttributeError
        If the module does not expose a top-level ``RECIPE`` attribute.

    TypeError
        If ``validate_recipe`` detects a structural type mismatch.

    ValueError
        If ``validate_recipe`` detects a structural constraint violation.

    Examples
    --------
    >>> from skordinal.experiments import load_recipe
    >>> recipe = load_recipe("/path/to/recipe.py")  # doctest: +SKIP
    >>> sorted(recipe.keys())  # doctest: +SKIP
    ['datasets', 'models']
    """
    recipe_path = Path(path).expanduser()
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}.")

    module_name = f"_skordinal_recipe_{recipe_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, recipe_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"Could not create module spec for recipe at {recipe_path}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if not hasattr(module, "RECIPE"):
            raise AttributeError(
                f"Recipe at {recipe_path} does not define a top-level RECIPE dict."
            )
        recipe: dict[str, Any] = module.RECIPE
        validate_recipe(recipe)
        return recipe
    finally:
        sys.modules.pop(module_name, None)
