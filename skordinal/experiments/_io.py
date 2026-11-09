"""File-persistence primitives for the experiments results tree."""

from __future__ import annotations

import ast
import os
import sys
import tempfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

_TEMP_PREFIX = ".skordinal-tmp-"


def _read_results_csv(path: Path, *, identifier: int | str = 0) -> pd.DataFrame:
    """Preserve literal identifiers and pandas' normal parsing of other columns."""
    # Parse one snapshot twice because pandas versions differ in whether
    # converters bypass NA handling. Only the identifier pass disables it.
    content = path.read_bytes()
    frame = pd.read_csv(BytesIO(content), float_precision="round_trip")
    identifiers = pd.read_csv(
        BytesIO(content), usecols=[identifier], dtype=str, na_filter=False
    ).iloc[:, 0]
    column = frame.columns[identifier] if isinstance(identifier, int) else identifier
    frame[column] = identifiers
    return frame


def _read_report_csv(path: Path) -> pd.DataFrame:
    """Read metric columns normally while preserving every literal resample id."""
    report = _read_results_csv(path)
    index_name = report.columns[0]
    report = report.set_index(index_name)
    if index_name == "Unnamed: 0":
        report.index.name = None
    return report


def _canonical_resample_ids(identifiers: pd.Index) -> pd.Index:
    """Restore integer ids only when their spelling is unchanged by conversion."""
    if identifiers.empty:
        return identifiers
    try:
        numbers = [int(value) for value in identifiers]
    except (TypeError, ValueError):
        return identifiers
    if all(str(number) == value for number, value in zip(numbers, identifiers)):
        return pd.Index(numbers, name=identifiers.name)
    return identifiers


def _ensure_parent(path: Path) -> None:
    """Create path's parent directory, wrapping OSError with the failing path."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Could not create folder {path.parent} to store results."
        ) from exc


def _atomic_write(path: Path, content: str) -> None:
    """Write text to path atomically via a temp file, fsync and replace."""
    _ensure_parent(path)
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
    _ensure_parent(path)
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


def _format_split_files(
    split: str,
    *,
    index: np.ndarray | None,
    true_y: np.ndarray,
    predicted_y: np.ndarray,
    proba: np.ndarray | None,
    classes: np.ndarray,
    resample_id: int,
) -> dict[str, str]:
    """Encode one split's labels and map its per-seed file names to their text."""
    if not np.isin(true_y, classes).all():
        raise ValueError(
            f"'{split}' true labels contain classes unknown to the fitted model."
        )
    pattern_id = index if index is not None else np.arange(predicted_y.shape[0])
    target = np.searchsorted(classes, true_y)
    if not np.isin(predicted_y, classes).all():
        raise ValueError(
            f"'{split}' predicted labels contain classes unknown to the fitted model."
        )
    # Record the estimator's actual decision, never a proba argmax substitute
    prediction = np.searchsorted(classes, predicted_y)

    columns: dict[str, object] = {"Pattern ID": pattern_id, "Target": target}
    if proba is not None:
        if proba.shape != (true_y.shape[0], classes.size):
            raise ValueError(
                f"'{split}' probabilities have shape {proba.shape}; expected "
                f"({true_y.shape[0]}, {classes.size})."
            )
        columns["Prediction probabilities"] = _format_proba_column(proba)
    columns["Prediction"] = prediction

    cm = confusion_matrix(target, prediction, labels=np.arange(classes.size))
    body = np.array2string(
        cm, separator=", ", threshold=cm.size, max_line_width=sys.maxsize
    )
    return {
        f"{split}_predictions.csv": pd.DataFrame(columns).to_csv(index=False),
        f"{split}_confusion_matrix.txt": f"Seed {resample_id}\n{'=' * 21}\n{body}\n",
    }


def _read_confusion_matrix_size(path: Path) -> int | None:
    """Return K from a saved K x K confusion matrix, or None if unreadable."""
    try:
        body = path.read_text(encoding="utf-8").split("\n", 2)[2]
        matrix = ast.literal_eval(body.strip())
    except (OSError, IndexError, ValueError, SyntaxError):
        return None
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        return None
    return len(matrix)
