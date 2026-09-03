"""skordinal."""

import logging

__all__: list[str] = []
__version__ = "0.1.0"

# Library convention: emit records, never configure handlers for the app
_logger = logging.getLogger("skordinal")
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
del _logger
