"""Tests for the experiments file-persistence primitives."""

import os

import joblib
import numpy as np
import numpy.testing as npt
import pytest

from skordinal.experiments._io import (
    _TEMP_PREFIX,
    _atomic_dump,
    _atomic_write,
    _ensure_parent,
    _format_proba_column,
    _parse_proba_column,
    _read_confusion_matrix_size,
    _sweep_orphaned_temp_files,
    _write_split_files,
)


def test_proba_column_round_trips():
    """Wide probability rows stay single-line cells and parse back exactly."""
    rng = np.random.default_rng(0)
    raw = rng.random((3, 20))
    proba = raw / raw.sum(axis=1, keepdims=True)

    cells = _format_proba_column(proba)
    assert all("\n" not in cell for cell in cells)
    npt.assert_array_equal(_parse_proba_column(cells), proba)


def test_atomic_write_round_trips(tmp_path):
    """Content lands whole under a created parent, with no temp file left."""
    target = tmp_path / "nested" / "f.csv"
    _atomic_write(target, "a,b\n1,2\n")
    assert target.read_text() == "a,b\n1,2\n"
    assert list(tmp_path.rglob(f"{_TEMP_PREFIX}*")) == []


def test_ensure_parent_wraps_mkdir_failure(tmp_path, monkeypatch):
    """A failing parent mkdir raises a clear, wrapped OSError."""
    monkeypatch.setattr(
        "skordinal.experiments._io.Path.mkdir",
        lambda *a, **k: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(OSError, match="Could not create folder"):
        _ensure_parent(tmp_path / "nested" / "f.csv")


@pytest.mark.parametrize(
    "write",
    [
        lambda path: _atomic_write(path, "data"),
        lambda path: _atomic_dump(path, {"k": 1}),
    ],
    ids=["write", "dump"],
)
def test_atomic_writers_clean_up_on_failure(tmp_path, monkeypatch, write):
    """A failing os.replace unlinks the temp file and re-raises."""
    monkeypatch.setattr(
        "skordinal.experiments._io.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(OSError, match="boom"):
        write(tmp_path / "target")
    assert not (tmp_path / "target").exists()
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


def test_atomic_dump_round_trips_no_temp(tmp_path):
    """_atomic_dump serialises an object recoverable by joblib.load."""
    target = tmp_path / "m.joblib"
    _atomic_dump(target, {"k": [1, 2, 3]})
    assert joblib.load(target) == {"k": [1, 2, 3]}
    assert list(tmp_path.glob(f"{_TEMP_PREFIX}*")) == []


def test_sweep_removes_only_stray_temp_files(tmp_path):
    """The sweep unlinks temp files, keeping directories; a missing root is a no-op."""
    _sweep_orphaned_temp_files(tmp_path / "absent")

    stray = tmp_path / "sub" / f"{_TEMP_PREFIX}leftover.tmp"
    stray.parent.mkdir()
    stray.write_text("junk")
    temp_dir = tmp_path / f"{_TEMP_PREFIX}dir"
    temp_dir.mkdir()
    _sweep_orphaned_temp_files(tmp_path)
    assert not stray.exists()
    assert temp_dir.is_dir()


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


def test_read_confusion_matrix_size(tmp_path):
    """K comes from the saved matrix, and an unusable file yields None."""
    good = tmp_path / "test_confusion_matrix.txt"
    good.write_text("Seed 0\n=====\n[[1, 0, 0],\n [0, 2, 0],\n [0, 0, 3]]\n")
    assert _read_confusion_matrix_size(good) == 3

    ragged = tmp_path / "ragged.txt"
    ragged.write_text("Seed 0\n=====\n[[1, 0], [0]]\n")
    assert _read_confusion_matrix_size(ragged) is None
    assert _read_confusion_matrix_size(tmp_path / "absent.txt") is None
