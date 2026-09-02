"""Shared internal helpers for the experiments package."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _check_path_component(name: Any, what: str) -> None:
    """Reject a path component that is empty, dotted or holds a separator."""
    if not isinstance(name, str):
        raise TypeError(f"{what} must be a str; got {type(name).__name__}.")
    if name in ("", ".", ".."):
        raise ValueError(f"{what} must not be empty or a dot segment; got {name!r}.")
    if any(sep in name for sep in (os.sep, "/", os.altsep) if sep):
        raise ValueError(f"{what} must not contain a path separator; got {name!r}.")


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
