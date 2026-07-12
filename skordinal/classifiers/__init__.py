"""Ordinal classification classifiers module."""

from ._cost_sensitive_wrapper import CostSensitiveWrapper
from ._elmop import ELMOP
from ._kdlor import KDLOR
from ._nnop import NNOP
from ._nnpom import NNPOM
from ._orboost import ORBoost
from ._ordinal_decomposition import OrdinalDecomposition
from ._pom import POM
from ._redsvm import REDSVM
from ._regressor_wrapper import RegressorWrapper
from ._svorex import SVOREX

__all__ = [
    "CostSensitiveWrapper",
    "ELMOP",
    "KDLOR",
    "NNOP",
    "NNPOM",
    "ORBoost",
    "OrdinalDecomposition",
    "POM",
    "REDSVM",
    "RegressorWrapper",
    "SVOREX",
]
