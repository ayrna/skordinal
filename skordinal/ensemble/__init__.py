"""Ordinal ensemble methods."""

from ._ebano import EBANO
from ._orboost import ORBoost
from ._ordinal_xgb import OrdinalXGBClassifier

__all__ = [
    "EBANO",
    "ORBoost",
    "OrdinalXGBClassifier",
]
