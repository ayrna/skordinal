"""Private losses of the threshold models, shared by the estimators."""

from skordinal._loss.loss import AllThresholdLoss, BaseLoss, ImmediateThresholdLoss
from skordinal._loss.margin import (
    BaseMargin,
    ExponentialMargin,
    LogisticMargin,
    get_margin,
)

__all__ = [
    "AllThresholdLoss",
    "BaseLoss",
    "BaseMargin",
    "ExponentialMargin",
    "ImmediateThresholdLoss",
    "LogisticMargin",
    "get_margin",
]
