"""Tests for the experiments file-persistence primitives."""

import os

import joblib
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from skordinal.experiments._io import (
    _TEMP_PREFIX,
    _atomic_dump,
    _atomic_write,
    _ensure_parent,
    _format_proba_column,
    _parse_proba_column,
    _sweep_orphaned_temp_files,
    _write_split_files,
)


def test_format_proba_column_cells_round_trip():
    """Wide probability cells stay single-line and parse back exactly."""
    rng = np.random.default_rng(0)
    raw = rng.random((3, 20))
    proba = raw / raw.sum(axis=1, keepdims=True)

    for cell, row in zip(_format_proba_column(proba), proba):
        assert "\n" not in cell
        npt.assert_array_equal(np.fromstring(cell.strip("[]"), sep=","), row)


def test_parse_proba_column_round_trips_formatted_cells():
    """_parse_proba_column reconstructs the matrix _format_proba_column wrote."""
    rng = np.random.default_rng(0)
    raw = rng.random((3, 5))
    proba = raw / raw.sum(axis=1, keepdims=True)
    parsed = _parse_proba_column(pd.Series(_format_proba_column(proba)))
    npt.assert_array_equal(parsed, proba)


def test_atomic_write_success_leaves_no_temp(tmp_path):
    """_atomic_write writes full content and leaves no temp file."""
    target = tmp_path / "f.csv"
    _atomic_write(target, "a,b\n1,2\n")
    assert target.read_text() == "a,b\n1,2\n"
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_atomic_write_creates_missing_parent(tmp_path):
    """_atomic_write creates the target's parent directory if absent."""
    target = tmp_path / "nested" / "f.csv"
    _atomic_write(target, "a,b\n1,2\n")
    assert target.read_text() == "a,b\n1,2\n"


def test_ensure_parent_wraps_mkdir_failure(tmp_path, monkeypatch):
    """A failing parent mkdir raises a clear, wrapped OSError."""
    monkeypatch.setattr(
        "skordinal.experiments._io.Path.mkdir",
        lambda *a, **k: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(OSError, match="Could not create folder"):
        _ensure_parent(tmp_path / "nested" / "f.csv")


def test_atomic_write_cleans_up_on_failure(tmp_path, monkeypatch):
    """A failing os.replace unlinks the temp file and re-raises."""
    monkeypatch.setattr(
        "skordinal.experiments._io.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(OSError, match="boom"):
        _atomic_write(tmp_path / "f.csv", "data")
    assert not (tmp_path / "f.csv").exists()
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_atomic_write_fsyncs(tmp_path, monkeypatch):
    """_atomic_write calls os.fsync on the temp file descriptor."""
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(
        "skordinal.experiments._io.os.fsync",
        lambda fd: calls.append(fd) or real_fsync(fd),
    )
    _atomic_write(tmp_path / "f.txt", "x")
    assert len(calls) == 1


def test_atomic_dump_cleans_up_on_failure(tmp_path, monkeypatch):
    """A failing os.replace unlinks the dump's temp file and re-raises."""
    monkeypatch.setattr(
        "skordinal.experiments._io.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(OSError, match="boom"):
        _atomic_dump(tmp_path / "m.joblib", {"k": 1})
    assert not (tmp_path / "m.joblib").exists()
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_atomic_dump_round_trips_no_temp(tmp_path):
    """_atomic_dump serialises an object recoverable by joblib.load."""
    target = tmp_path / "m.joblib"
    _atomic_dump(target, {"k": [1, 2, 3]})
    assert joblib.load(target) == {"k": [1, 2, 3]}
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_sweep_orphaned_temp_files_ignores_missing_dir(tmp_path):
    """The sweep is a no-op when the base directory does not exist."""
    _sweep_orphaned_temp_files(tmp_path / "absent")


def test_confusion_matrix_not_elided_for_many_classes(tmp_path):
    """A large confusion matrix is written in full, without summarising."""
    labels = np.arange(40)
    _write_split_files(
        tmp_path,
        "train",
        index=None,
        true_y=labels,
        predicted_y=labels,
        proba=None,
        classes=labels,
        resample_id=0,
    )

    body = (
        (tmp_path / "train_confusion_matrix.txt")
        .read_text()
        .split("\n", 2)[2]
        .rstrip("\n")
    )
    assert "..." not in body
    # Check the matrix keeps one physical line per row (no wrapping)
    assert body.count("\n") == 39
