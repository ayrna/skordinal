"""Ordinal classification classifiers module."""

from ._cost_sensitive_wrapper import CostSensitiveWrapper
from ._elmop import ELMOP
from ._nnop import NNOP
from ._nnpom import NNPOM
from ._orboost import ORBoost
from ._ordinal_decomposition import OrdinalDecomposition
from ._redsvm import REDSVM
from ._regressor_wrapper import RegressorWrapper
from ._svorex import SVOREX

__all__ = [
    "CostSensitiveWrapper",
    "ELMOP",
    "NNOP",
    "NNPOM",
    "ORBoost",
    "OrdinalDecomposition",
    "REDSVM",
    "RegressorWrapper",
    "SVOREX",
]
