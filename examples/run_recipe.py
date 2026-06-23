"""Command-line runner for skordinal benchmark recipes."""

import argparse

from skordinal.experiments import Benchmark


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the benchmark described by a recipe file.

    Parameters
    ----------
    argv : list of str or None, default=None
        Command-line arguments.  ``None`` falls back to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        prog="run_recipe",
        description="Run a skordinal benchmark recipe.",
        epilog=(
            "Examples:\n"
            "  python examples/run_recipe.py examples/recipes/nnop_demo.py\n"
            "  python examples/run_recipe.py examples/recipes/nnop_demo.py"
            " --results-dir /tmp/out"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recipe", help="Path to the recipe .py file.")
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="PATH",
        help="Override the results_path defined in the recipe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-run experiments even when result files already exist on disk.",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.results_dir is not None:
        overrides["results_path"] = args.results_dir

    benchmark = Benchmark.from_recipe(args.recipe, **overrides)
    benchmark.run()
    benchmark.summarize()


if __name__ == "__main__":
    main()
