"""Shared internal helpers for the experiments package."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from skordinal.metrics import get_ordinal_scorer
from skordinal.metrics._metrics import _resolve_label_metric


def _check_path_component(name: Any, what: str) -> None:
    """Reject a path component that cannot safely name one directory."""
    if not isinstance(name, str):
        raise TypeError(f"{what} must be a str; got {type(name).__name__}.")
    if name in ("", ".", ".."):
        raise ValueError(f"{what} must not be empty or a dot segment; got {name!r}.")
    # os.altsep is None on POSIX, so a backslash escapes only once read on Windows
    if any(sep in name for sep in (os.sep, "/", "\\", os.altsep) if sep):
        raise ValueError(f"{what} must not contain a path separator; got {name!r}.")
    # On Windows "D:" resolves outside the results root and "C:" drops a level
    if ":" in name:
        raise ValueError(f"{what} must not contain a colon; got {name!r}.")


def _check_resample_id(resample_id: Any) -> None:
    """Reject a resample id that is neither int-like nor a plain path component."""
    if isinstance(resample_id, (int, np.integer)):
        return
    _check_path_component(str(resample_id), "resample_id")


def _check_split(split: str, *, allow_both: bool) -> None:
    """Raise ValueError when split is not a recognised value."""
    valid = {"test", "train", "both"} if allow_both else {"test", "train"}
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)!r}, got {split!r}.")


def _check_metric_names(metrics: Any, *, param: str) -> list[str]:
    """Check an iterable of metric names, returning them stripped and unique."""
    if isinstance(metrics, str):
        raise TypeError(
            f"'{param}' must be an iterable of metric name strings, not a "
            f"bare string; pass [{metrics!r}] to use a single metric."
        )
    # An iterator is always truthy, and a name array has no single truth value
    try:
        metrics = list(metrics)
    except TypeError:
        raise TypeError(
            f"'{param}' must be an iterable of metric name strings; got "
            f"{type(metrics).__name__}."
        ) from None
    if not metrics:
        raise ValueError(f"'{param}' must be a non-empty list; got an empty sequence.")
    for name in metrics:
        if not isinstance(name, str):
            raise TypeError(
                f"'{param}' must contain only metric name strings; got "
                f"{type(name).__name__!r}."
            )
    # A repeat would compute the metric twice and duplicate evaluate's column
    metrics = list(dict.fromkeys(name.strip() for name in metrics))
    # Resolve now: a typo otherwise costs a full grid search
    for name in metrics:
        _resolve_label_metric(name)
    return metrics


def _check_tuning_metric(tuning_metric: Any) -> None:
    """Resolve the tuning metric, which a search-free run never would."""
    get_ordinal_scorer(tuning_metric)
