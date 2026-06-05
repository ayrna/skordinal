"""Ordinal classification classifiers module."""

from ._nnop import NNOP
from ._nnpom import NNPOM
from ._ordinal_decomposition import OrdinalDecomposition
from ._redsvm import REDSVM
from ._regressor_wrapper import RegressorWrapper
from ._svorex import SVOREX

__all__ = [
    "NNOP",
    "NNPOM",
    "OrdinalDecomposition",
    "REDSVM",
    "RegressorWrapper",
    "SVOREX",
]
