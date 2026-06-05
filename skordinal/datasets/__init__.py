"""Bundled ordinal classification datasets."""

from ._loaders import (
    load_balance_scale,
    load_era,
    load_esl,
    load_lev,
    load_swd,
)
from ._samples_generator import make_ordinal_classification

__all__ = [
    "load_balance_scale",
    "load_era",
    "load_esl",
    "load_lev",
    "load_swd",
    "make_ordinal_classification",
]
