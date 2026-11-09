# Configuration file for the Sphinx documentation builder.

import inspect
import sys
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder

# --- Project metadata -------------------------------------------------------
import skordinal  # noqa: E402

project = "skordinal"
copyright = "2026, AYRNA – Universidad de Córdoba"
author = "Ángel Sevilla Molina, Pedro A. Gutiérrez Peña, David Guijo Rubio"
release = skordinal.__version__

# --- Extensions -------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

# --- Theme ------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/ayrna/skordinal",
}

# --- Intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/stable", None),
}

# --- Autodoc / autosummary --------------------------------------------------

# Each api/*.rst lists its public names in an ``autosummary`` toctree, so the
# stub pages under api/generated/ are built from scratch on every run.
autosummary_generate = True

# Class pages document every public method, inherited ones included, as in scikit-learn
templates_path = ["_templates"]

# --- Source links -----------------------------------------------------------

# ``viewcode`` would republish the source of every documented object inside the
# site, scikit-learn's ``accuracy_score`` and ``mean_absolute_error`` included,
# since ``skordinal.metrics`` re-exports them. ``linkcode`` points at the real
# file on GitHub instead, and skips anything that is not skordinal source.

_SOURCE_URL = (
    "https://github.com/ayrna/skordinal/blob/main/skordinal/{path}#L{start}-L{end}"
)


def linkcode_resolve(domain, info):
    """Return the GitHub URL for a documented object, or ``None`` to omit it."""
    if domain != "py" or (info.get("module") or "").split(".")[0] != "skordinal":
        return None

    obj = sys.modules.get(info["module"])
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    try:
        file = inspect.getsourcefile(inspect.unwrap(obj))
        lines, start = inspect.getsourcelines(inspect.unwrap(obj))
    except (OSError, TypeError):
        return None
    if file is None:
        return None

    try:
        path = (
            Path(file).resolve().relative_to(Path(skordinal.__file__).resolve().parent)
        )
    except ValueError:
        return None

    return _SOURCE_URL.format(
        path=path.as_posix(), start=start, end=start + len(lines) - 1
    )


# --- numpydoc ---------------------------------------------------------------

# Single backticks are code, as in scikit-learn, so `n_features_in_` is not a link
default_role = "literal"

numpydoc_show_class_members = False

# --- Sphinx-Gallery ---------------------------------------------------------

# ``examples/`` holds two kinds of file. The gallery scripts live in one
# subdirectory per topic, each with the ``README.txt`` header that makes
# sphinx-gallery render it as a section. ``run_recipe.py`` is a CLI driver
# for ``examples/recipes/`` and not an example, so ``ignore_pattern`` drops
# it, and ``examples/recipes/`` needs no entry of its own because sphinx-
# gallery only descends into a subdirectory that carries a header

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "subsection_order": ExplicitOrder(
        [
            "../examples/getting_started",
            "../examples/classifiers",
            "../examples/metrics",
            "../examples/experiments",
        ]
    ),
    "within_subsection_order": "FileNameSortKey",
    "ignore_pattern": r"run_recipe\.py",
}

# --- Build targets ----------------------------------------------------------

# sphinx-gallery writes a per-gallery ``sg_execution_times.rst`` beside the
# generated pages and a project-wide one into this directory. Both are
# ``:orphan:`` build artefacts that do not belong on the site
exclude_patterns = ["_build", "**/sg_execution_times.rst"]
