# Configuration file for the Sphinx documentation builder.
#
# Source-docstring compatibility notes
# -------------------------------------
# All classifier docstrings include fitted-attribute names such as
# ``n_features_in_`` in type strings (e.g. "ndarray of shape
# (n_features_in_,)").  In plain RST, a word ending with ``_`` is a
# hyperlink reference, producing "Unknown target name" errors.  An
# ``autodoc-process-docstring`` hook escapes those trailing underscores
# to ``\_`` inside parenthesised shape expressions.

import inspect
import re
import sys
from pathlib import Path

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
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# --- Autodoc / autosummary --------------------------------------------------

# Each api/*.rst lists its public names in an ``autosummary`` toctree, so the
# stub pages under api/generated/ are built from scratch on every run.
autosummary_generate = True

# The C extension modules are compiled separately; mock them so autodoc can
# import the pure-Python wrappers without a compiled extension present.
autodoc_mock_imports = [
    "skordinal.classifiers._libsvmrank",
    "skordinal.classifiers._libsvorex",
    "skordinal.classifiers._libsvorim",
    "skordinal.classifiers._orensemble",
]

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

numpydoc_show_class_members = False

# --- Build targets ----------------------------------------------------------

exclude_patterns = ["_build"]

# --- Docstring post-processing ----------------------------------------------

# Trailing underscores in plain text inside RST are treated as hyperlink
# references (RST ``target_`` syntax).  Fitted sklearn attribute names such
# as ``n_features_in_`` appear bare in shape descriptions and cause
# "Unknown target name" ERROR nodes.  Escape them to ``\_``.
_TRAILING_UNDERSCORE_RE = re.compile(
    r"(?<![`\\])"  # not already escaped or inside a backtick role
    r"([A-Za-z][A-Za-z0-9_]*)"  # identifier
    r"_"  # trailing underscore (RST hyperlink ref syntax)
    r"(?![_`a-zA-Z])",  # not followed by another _ or backtick
)


def _fixup_docstring(app, what, name, obj, options, lines):
    """Normalise docstring lines for clean RST rendering.

    Escapes bare trailing-underscore identifiers (fitted attribute names)
    that RST would interpret as hyperlink references.
    """
    for i, line in enumerate(lines):
        line = _TRAILING_UNDERSCORE_RE.sub(r"\1\\_", line)
        lines[i] = line


def setup(app):
    app.connect("autodoc-process-docstring", _fixup_docstring)
    return {"version": "0.1", "parallel_read_safe": True}
