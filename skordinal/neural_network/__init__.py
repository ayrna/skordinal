"""Ordinal neural networks."""

from ._elmop import ELMOP
from ._mlp_clm import MLPCLMClassifier
from ._mlp_triangular import MLPTriangularClassifier
from ._nnop import NNOP
from ._nnpom import NNPOM

__all__ = [
    "ELMOP",
    "MLPCLMClassifier",
    "MLPTriangularClassifier",
    "NNOP",
    "NNPOM",
]
