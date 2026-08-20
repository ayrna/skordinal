"""Utilities for ordinal classification."""

from skordinal.utils.extmath import (
    cumproba_to_proba,
    proba_to_cumproba,
    repair_cumproba,
)
from skordinal.utils.validation import check_ordinal_targets, validate_thresholds

__all__ = [
    "check_ordinal_targets",
    "cumproba_to_proba",
    "proba_to_cumproba",
    "repair_cumproba",
    "validate_thresholds",
]
