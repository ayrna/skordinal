"""skordinal."""

import logging

__all__ = [
    "datasets",
    "discriminant_analysis",
    "ensemble",
    "experiments",
    "linear_model",
    "metrics",
    "neural_network",
    "preprocessing",
    "reduction",
    "svm",
    "utils",
]
__version__ = "0.2.0"

# Library convention: emit records, never configure handlers for the app
_logger = logging.getLogger("skordinal")
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
del _logger
