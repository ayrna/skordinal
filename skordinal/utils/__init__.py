"""Utilities for ordinal classification."""

from skordinal.utils.extmath import (
    cumproba_to_proba,
    proba_to_cumproba,
    repair_cumproba,
)
from skordinal.utils.validation import check_ordinal_targets, check_thresholds

__all__ = [
    "check_ordinal_targets",
    "check_thresholds",
    "cumproba_to_proba",
    "proba_to_cumproba",
    "repair_cumproba",
]
