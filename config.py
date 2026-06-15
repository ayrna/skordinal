"""Command-line runner for skordinal benchmark recipes."""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from skordinal.experiments import Benchmark


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Import a recipe ``.py`` file and return its ``RECIPE`` dict.

    The module is imported under a synthetic name and removed from
    ``sys.modules`` after the call, so successive loads of recipes placed at
    the same path always pick up fresh state.

    Parameters
    ----------
    path : str or Path
        Filesystem path to the recipe file.

    Returns
    -------
    recipe : dict
        The recipe's top-level ``RECIPE`` dict, with ``general_conf`` and
        ``configurations`` keys.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or is not a file, or if the module spec
        cannot be created.
    AttributeError
        If the module does not expose a ``RECIPE`` attribute.
    """
    recipe_path = Path(path).expanduser()
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}.")

    module_name = f"_skordinal_recipe_{recipe_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, recipe_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load recipe module at {recipe_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, "RECIPE"):
            raise AttributeError(
                f"Recipe at {recipe_path} does not define a top-level RECIPE dict."
            )
        return module.RECIPE
    finally:
        sys.modules.pop(module_name, None)


def main(general_conf: dict[str, Any], configurations: dict[str, Any]) -> None:
    if not general_conf["basedir"] or not general_conf["datasets"]:
        raise RuntimeError(
            "A dataset has to be defined to run this program.\n"
            + "For more information about using this framework, please refer to the README."
        )

    if not configurations:
        raise RuntimeError(
            "No configuration was defined.\n"
            + "For more information about using this framework, please refer to the README."
        )

    benchmark = Benchmark(
        configurations,
        data_path=general_conf["basedir"],
        datasets=general_conf["datasets"],
        eval_metrics=general_conf["metrics"],
        results_path=general_conf["output_folder"],
        tuning_metric=general_conf.get("cv_metric", "neg_mean_absolute_error"),
        cv=general_conf.get("hyperparam_cv_nfolds", 3),
        n_jobs=general_conf.get("jobs", 1),
        input_preprocessing=general_conf.get("input_preprocessing"),
        random_state=general_conf.get("random_state"),
    )
    benchmark.run()
    benchmark.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a skordinal experiment.")
    parser.add_argument("config", type=Path, help="Path to a Python recipe file (.py).")
    args = parser.parse_args()

    recipe = load_recipe(args.config)

    main(recipe["general_conf"], recipe["configurations"])
