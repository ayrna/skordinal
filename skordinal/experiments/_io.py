"""File-persistence primitives for the experiments results tree."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np

_TEMP_PREFIX = ".skordinal-tmp-"


def _check_path_component(name: Any, what: str) -> None:
    """Reject a path component that is empty, dotted or holds a separator."""
    if not isinstance(name, str):
        raise TypeError(f"{what} must be a str; got {type(name).__name__}.")
    if name in ("", ".", ".."):
        raise ValueError(f"{what} must not be empty or a dot segment; got {name!r}.")
    if any(sep in name for sep in (os.sep, "/", os.altsep) if sep):
        raise ValueError(f"{what} must not contain a path separator; got {name!r}.")


def _atomic_write(path: Path, content: str) -> None:
    """Write text to path atomically via a temp file, fsync and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=_TEMP_PREFIX, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_dump(path: Path, obj: Any) -> None:
    """Serialise obj to path atomically via a temp file, fsync and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=_TEMP_PREFIX, suffix=".tmp")
    try:
        # Close the mkstemp descriptor; joblib opens its own handle by path
        os.close(fd)
        joblib.dump(obj, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _sweep_orphaned_temp_files(base_dir: Path) -> None:
    """Unlink stray temp files a prior crash left under base_dir."""
    if not base_dir.is_dir():
        return
    for path in base_dir.rglob(f"{_TEMP_PREFIX}*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def _format_proba_column(proba: np.ndarray) -> list[str]:
    """Render probability rows as single-line ``"[p0, p1, ...]"`` cells."""
    return [
        "[" + ", ".join(repr(float(p)) for p in row) + "]" for row in np.asarray(proba)
    ]


def _parse_proba_column(series: Iterable[str]) -> np.ndarray:
    """Expand stringified ``"[p0, p1, ...]"`` cells into an (n, q) array."""
    return np.vstack([np.fromstring(x.strip("[]"), sep=",") for x in series])
