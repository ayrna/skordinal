"""Ordinal classification classifiers module."""

from ._cost_sensitive_wrapper import CostSensitiveWrapper
from ._nnop import NNOP
from ._nnpom import NNPOM
from ._ordinal_decomposition import OrdinalDecomposition
from ._redsvm import REDSVM
from ._regressor_wrapper import RegressorWrapper
from ._svorex import SVOREX

__all__ = [
    "CostSensitiveWrapper",
    "NNOP",
    "NNPOM",
    "OrdinalDecomposition",
    "REDSVM",
    "RegressorWrapper",
    "SVOREX",
]
