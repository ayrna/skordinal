"""Meta-estimators that combine estimators fitted on fixed ordinal subproblems."""

from ._cost_sensitive import CostSensitiveWrapper
from ._decomposition import OrdinalDecomposition
from ._opbe import OPBE
from ._regression import RegressorWrapper

__all__ = [
    "CostSensitiveWrapper",
    "OPBE",
    "OrdinalDecomposition",
    "RegressorWrapper",
]
