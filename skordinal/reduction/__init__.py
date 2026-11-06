"""Meta-estimators that combine estimators fitted on fixed ordinal subproblems."""

from ._cost_sensitive import CostSensitiveClassifier
from ._decomposition import OrdinalDecompositionClassifier
from ._opbe import OPBE
from ._regression import RegressionClassifier

__all__ = [
    "CostSensitiveClassifier",
    "OPBE",
    "OrdinalDecompositionClassifier",
    "RegressionClassifier",
]
