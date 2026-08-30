"""Tests for the dataset loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import Bunch
from sklearn.utils._param_validation import InvalidParameterError

from skordinal.datasets import (
    _base,
    clear_data_home,
    get_data_home,
    load_dataset,
    load_partitions,
)
from skordinal.datasets._base import _load_bundled, _resolve_target_names

_NAMED_CSV = """\
x_0,x_1,y
1.0,2.0,0
3.0,4.0,1
5.0,6.0,2
7.0,8.0,0
9.0,10.0,1
"""

_N_SAMPLES = 20
_N_FEATURES = 2


def _make_csv_content(n_samples=_N_SAMPLES):
    """Return a deterministic named-header CSV string with 3 classes."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, _N_FEATURES))
    y = (np.arange(n_samples) % 3).astype(int)
    lines = ["x_0,x_1,y"]
    for i in range(n_samples):
        lines.append(f"{X[i, 0]:.6f},{X[i, 1]:.6f},{y[i]}")
    return "\n".join(lines) + "\n"


def _write_csv(directory, name, n_samples=_N_SAMPLES):
    """Write a named-header CSV and return its path."""
    path = directory / f"{name}.csv"
    path.write_text(_make_csv_content(n_samples), encoding="utf-8")
    return path


def _write_masks(directory, name, masks):
    """Write a per-dataset ``<name>.masks.json`` and return its path."""
    path = directory / f"{name}.masks.json"
    path.write_text(json.dumps(masks), encoding="utf-8")
    return path


def _write_keyed_masks(directory, keyed):
    """Write a keyed ``train_masks.json`` and return its path."""
    path = directory / "train_masks.json"
    path.write_text(json.dumps(keyed), encoding="utf-8")
    return path


@pytest.fixture
def named_csv(tmp_path):
    """Minimal named-header CSV: two features, three classes, five rows."""
    path = tmp_path / "toy_dataset.csv"
    path.write_text(_NAMED_CSV, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "env_sub, arg_sub, expected_sub",
    [
        (None, "arg", "arg"),
        ("env", None, "env"),
        ("env", "arg", "arg"),
    ],
    ids=["explicit-only", "env-var", "explicit-overrides-env"],
)
def test_get_data_home_resolution(
    tmp_path, monkeypatch, env_sub, arg_sub, expected_sub
):
    """Resolve data_home from the argument, else ``SKORDINAL_DATA``."""
    if env_sub is not None:
        monkeypatch.setenv("SKORDINAL_DATA", str(tmp_path / env_sub))
    else:
        monkeypatch.delenv("SKORDINAL_DATA", raising=False)
    arg = (tmp_path / arg_sub) if arg_sub is not None else None
    result = get_data_home(arg)
    assert Path(result) == tmp_path / expected_sub
    assert Path(result).is_dir()
    assert isinstance(result, str)


def test_clear_data_home_removes_directory(tmp_path):
    """``clear_data_home`` deletes the cache directory."""
    target = tmp_path / "cache"
    target.mkdir()
    (target / "dummy.txt").write_text("x")
    clear_data_home(target)
    assert not target.exists()


def test_get_data_home_empty_env_var_treated_as_unset(tmp_path, monkeypatch):
    """``SKORDINAL_DATA=""`` falls back to the default home, not the cwd."""
    monkeypatch.setenv("SKORDINAL_DATA", "")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = get_data_home(data_home=None)
    assert Path(result) == tmp_path / "skordinal_data"
    assert Path(result) != Path.cwd()


def test_load_dataset_named_csv_bunch_contract(named_csv):
    """Named-header CSV yields a Bunch with correct shape, names, dtype."""
    bunch = load_dataset(named_csv)
    assert isinstance(bunch, Bunch)
    for key in (
        "data",
        "target",
        "feature_names",
        "target_names",
        "n_classes",
        "DESCR",
        "filename",
    ):
        assert key in bunch, f"Missing key: {key!r}"
    assert bunch.data.shape == (5, 2)
    assert bunch.target.shape == (5,)
    assert bunch.feature_names == ["x_0", "x_1"]
    assert bunch.n_classes == len(np.unique(bunch.target))
    assert bunch.target.dtype == np.int32
    assert bunch.filename == named_csv.name


@pytest.mark.parametrize(
    "csv_text, expected_feature_names, expected_target_names",
    [
        (
            "3,2,low,mid,high\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n",
            ["x0", "x1"],
            ["low", "mid", "high"],
        ),
        (
            "x_0,x_1,y\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n",
            ["x_0", "x_1"],
            ["0", "1", "2"],
        ),
        (
            "1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n",
            ["x0", "x1"],
            ["0", "1", "2"],
        ),
        (
            "3,2\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n",
            ["x0", "x1"],
            ["0", "1", "2"],
        ),
    ],
    ids=[
        "metadata-header",
        "named-header",
        "no-header",
        "metadata-header-no-class-names",
    ],
)
def test_load_dataset_header_styles(
    tmp_path, csv_text, expected_feature_names, expected_target_names
):
    """Each header style yields correct feature_names and target_names."""
    path = tmp_path / "style.csv"
    path.write_text(csv_text, encoding="utf-8")
    bunch = load_dataset(path)
    assert bunch.feature_names == expected_feature_names
    assert list(bunch.target_names) == expected_target_names
    assert bunch.data.shape == (3, 2)


@pytest.mark.parametrize(
    "csv_text, expected_feature_names",
    [
        ("3,2\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n", ["x0", "x1"]),
        ("x_0,x_1,y\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n", ["x_0", "x_1"]),
    ],
    ids=["metadata-header", "named-header"],
)
def test_load_dataset_strips_bom(tmp_path, csv_text, expected_feature_names):
    """A leading UTF-8 BOM is stripped, not folded into the first token."""
    path = tmp_path / "bom.csv"
    path.write_text(csv_text, encoding="utf-8-sig")
    bunch = load_dataset(path)
    assert bunch.feature_names == expected_feature_names
    assert bunch.data.shape == (3, 2)


def test_load_dataset_bundled_era():
    """Bundled ``era`` dataset loads with correct shape and rst DESCR."""
    bunch = load_dataset("era")
    assert bunch.data.shape == (1000, 4)
    # The bundled sidecar era.rst exists; DESCR must come from it, not the
    # generated "Dataset '...':" fallback
    assert not bunch.DESCR.startswith("Dataset '")
    assert len(bunch.DESCR) > 10


def test_load_dataset_data_home_does_not_inherit_bundled_descr(tmp_path):
    """A user's own ``era.csv`` must not inherit the bundled ERA DESCR."""
    _write_csv(tmp_path, "era")
    bunch = load_dataset("era", data_home=tmp_path)
    assert bunch.DESCR.startswith("Dataset 'era'")


@pytest.mark.parametrize(
    "csv_text, expected_feature_names, expected_shape",
    [
        ("1.0,2.0,0\n", ["x0", "x1"], (1, 2)),
        ("5\n3\n6\n", [], (3, 0)),
    ],
    ids=["single-row", "single-column"],
)
def test_load_dataset_degenerate_shapes(
    tmp_path, csv_text, expected_feature_names, expected_shape
):
    """Single-row and single-column CSVs parse as headerless data."""
    path = tmp_path / "edge.csv"
    path.write_text(csv_text, encoding="utf-8")
    bunch = load_dataset(path)
    assert bunch.feature_names == expected_feature_names
    assert bunch.data.shape == expected_shape


def test_load_dataset_uses_colocated_rst_sidecar(tmp_path):
    """Co-located ``.rst`` sidecar is used as ``DESCR``."""
    path = tmp_path / "ds.csv"
    path.write_text("x_0,x_1,y\n1.0,2.0,0\n3.0,4.0,1\n", encoding="utf-8")
    sidecar = tmp_path / "ds.rst"
    sidecar.write_text("Distinctive sidecar description.", encoding="utf-8")
    bunch = load_dataset(path)
    assert "Distinctive sidecar description." in bunch.DESCR


def test_load_dataset_return_X_y(named_csv):
    """``return_X_y=True`` returns an (ndarray, ndarray) tuple."""
    result = load_dataset(named_csv, return_X_y=True)
    assert isinstance(result, tuple) and len(result) == 2
    X, y = result
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 1
    assert X.shape[0] == y.shape[0]


def test_load_dataset_as_frame(named_csv):
    """``as_frame=True`` returns a DataFrame and a Series."""
    pd = pytest.importorskip("pandas")
    bunch = load_dataset(named_csv, as_frame=True)
    assert isinstance(bunch.data, pd.DataFrame)
    assert isinstance(bunch.target, pd.Series)
    assert bunch.data.shape[1] == len(bunch.feature_names)
    assert bunch.frame is not None


@pytest.mark.parametrize(
    "csv_text, match",
    [
        ("x_0,target,y\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n", "collides"),
        ("a,a,y\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n", "more than once"),
    ],
    ids=["collides-with-target", "duplicate-names"],
)
def test_load_dataset_as_frame_bad_feature_names_raise(tmp_path, csv_text, match):
    """A feature name that collides with ``target`` or repeats raises."""
    pytest.importorskip("pandas")
    path = tmp_path / "names.csv"
    path.write_text(csv_text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_dataset(path, as_frame=True)


@pytest.mark.parametrize("name", ["ds_home", "ds_home.csv"], ids=["stem", "filename"])
def test_load_dataset_data_home(tmp_path, name):
    """Stem or filename resolves against an explicit ``data_home`` directory."""
    _write_csv(tmp_path, "ds_home")
    bunch = load_dataset(name, data_home=tmp_path)
    assert bunch.data.shape == (_N_SAMPLES, _N_FEATURES)


def test_load_dataset_missing_file_raises(tmp_path):
    """``load_dataset`` raises ``FileNotFoundError`` if absent."""
    missing = tmp_path / "no_such_file.csv"
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_dataset(missing)


def test_load_dataset_empty_csv_raises(tmp_path):
    """An empty CSV file raises ``ValueError`` matching ``'empty'``."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(path)


@pytest.mark.parametrize(
    "bad_target",
    ["nan", "inf", "1e300", "2147483648", "2.7"],
    ids=["nan", "inf", "overflow", "int32-max-plus-one", "fractional"],
)
def test_load_dataset_invalid_target_raises(tmp_path, bad_target):
    """A non-integer target raises instead of casting to a corrupt class label."""
    path = tmp_path / "bad_target.csv"
    path.write_text(f"x_0,x_1,y\n1.0,2.0,0\n3.0,4.0,{bad_target}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Target column of .* is invalid"):
        load_dataset(path)


@pytest.mark.parametrize(
    "csv_text, match",
    [
        ("3,2\n1.0,2.0,0\n3.0,4.0,1\n", "declares 3 sample.*file has 2 data row"),
        ("1.0,2.0,0\n3.0,4.0\n5.0,6.0,2\n", "inconsistent lengths"),
        (
            "x_0,x_1,x_2,y\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n",
            "Expected 3 feature column",
        ),
        ("x_0,x_1,y\n", "header but no data rows"),
    ],
    ids=[
        "declared-count-mismatch",
        "ragged-rows",
        "wrong-column-count",
        "header-only",
    ],
)
def test_load_dataset_malformed_csv_raises(tmp_path, csv_text, match):
    """A structurally malformed CSV raises ``ValueError`` naming the file."""
    path = tmp_path / "malformed.csv"
    path.write_text(csv_text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_dataset(path)


def test_holdout_fallback_era():
    """Holdout fallback on ``era`` yields 3 independent 700/300 resamples."""
    bunches = list(load_partitions("era", resamples=3))
    assert len(bunches) == 3
    for bunch in bunches:
        # 700/300 matches era's published reference partitions
        assert bunch.data_train.shape[0] == 700
        assert bunch.data_test.shape[0] == 300


def test_holdout_fallback_id_stable_across_request_shapes():
    """Ids keep their split across a pair, a sparse pair and a plain count."""
    pair = {b.resample_id: b for b in load_partitions("era", resamples=[0, 1])}
    sparse = {b.resample_id: b for b in load_partitions("era", resamples=[0, 5])}
    by_count = {b.resample_id: b for b in load_partitions("era", resamples=6)}
    for other in (sparse, by_count):
        np.testing.assert_array_equal(pair[0].train_index, other[0].train_index)
    # Id 5 is second in the sparse pair but sixth in the count, so a seed
    # derived from list position rather than id would move its split
    np.testing.assert_array_equal(sparse[5].train_index, by_count[5].train_index)


def test_holdout_fallback_reproducible_from_documented_inputs(tmp_path):
    """An integer ``random_state`` and the id fully determine a split."""
    name = "ds_repro"
    _write_csv(tmp_path, name, n_samples=300)
    (bunch,) = list(
        load_partitions(name, resamples=[4], random_state=7, data_home=tmp_path)
    )
    # Rebuild the split from the documented inputs alone
    seed = int(np.random.SeedSequence([7, 4]).generate_state(1)[0])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    raw = load_dataset(name, data_home=tmp_path)
    train_indices, _ = next(splitter.split(raw.data, raw.target))
    np.testing.assert_array_equal(bunch.train_index, np.sort(train_indices))


def test_invalid_resamples_fail_before_csv_read(tmp_path, monkeypatch):
    """A bad ``resamples`` raises before the CSV is parsed."""
    name = "ds_prevalidation"
    _write_csv(tmp_path, name)
    # Unbinding the reader makes any premature read raise TypeError
    monkeypatch.setattr(_base, "_read_csv_any", None)
    with pytest.raises(ValueError, match="at least one id"):
        load_partitions(name, resamples=[], data_home=tmp_path)


def test_holdout_fallback_random_state_instance_varies_across_calls():
    """A ``RandomState`` instance is drawn from once, so calls differ."""
    rng = np.random.RandomState(0)
    first = next(iter(load_partitions("era", resamples=1, random_state=rng)))
    second = next(iter(load_partitions("era", resamples=1, random_state=rng)))
    assert not np.array_equal(first.train_index, second.train_index)


@pytest.mark.parametrize(
    "kwargs, per_class_train, per_class_test",
    [
        ({}, 70, 30),
        ({"test_size": 0.25}, 75, 25),
        ({"test_size": 0.5}, 50, 50),
    ],
    ids=["default", "orca-ratio", "half"],
)
def test_holdout_fallback_test_size(tmp_path, kwargs, per_class_train, per_class_test):
    """``test_size`` sets the ratio, and each class splits at that ratio."""
    name = "ds_test_size"
    _write_csv(tmp_path, name, n_samples=300)
    (bunch,) = list(load_partitions(name, resamples=1, data_home=tmp_path, **kwargs))
    _, train_counts = np.unique(bunch.target_train, return_counts=True)
    _, test_counts = np.unique(bunch.target_test, return_counts=True)
    assert train_counts.tolist() == [per_class_train] * 3
    assert test_counts.tolist() == [per_class_test] * 3


@pytest.mark.parametrize("test_size", [0, 1], ids=["0", "1"])
def test_holdout_fallback_rejects_out_of_range_test_size(test_size):
    """``test_size`` at the excluded (0, 1) boundaries is rejected up front."""
    # The constraint fires before any file is read, so no CSV is needed
    with pytest.raises(
        InvalidParameterError, match=r"'test_size' parameter.*range \(0\.0, 1\.0\)"
    ):
        load_partitions("era", resamples=1, test_size=test_size)


def test_per_dataset_masks(tmp_path):
    """Per-dataset masks.json sets exact train/test counts, complement."""
    name = "ds_perdataset"
    _write_csv(tmp_path, name)
    mask0 = [True] * 14 + [False] * 6
    mask1 = [False] * 8 + [True] * 12
    _write_masks(tmp_path, name, [mask0, mask1])
    bunches = list(load_partitions(name, resamples=2, data_home=tmp_path))
    assert bunches[0].data_train.shape[0] == 14
    assert bunches[0].data_test.shape[0] == 6
    assert bunches[1].data_train.shape[0] == 12
    assert bunches[1].data_test.shape[0] == 8
    # Train and test sizes sum to the full sample count
    total = bunches[0].data_train.shape[0] + bunches[0].data_test.shape[0]
    assert total == _N_SAMPLES


def test_integer_mask_accepted(tmp_path):
    """A masks file of 0/1 integers splits as if it held booleans."""
    name = "ds_int_mask"
    _write_csv(tmp_path, name)
    _write_masks(tmp_path, name, [[1] * 14 + [0] * 6])
    (bunch,) = list(load_partitions(name, resamples=1, data_home=tmp_path))
    assert bunch.data_train.shape[0] == 14
    assert bunch.data_test.shape[0] == 6


@pytest.mark.parametrize(
    "place_in_parent",
    [False, True],
    ids=["same-dir", "parent-dir"],
)
def test_keyed_masks_location(tmp_path, place_in_parent):
    """``train_masks.json`` is found in the CSV dir or its parent."""
    if place_in_parent:
        csv_dir = tmp_path / "data"
        csv_dir.mkdir()
        mask_dir = tmp_path
    else:
        csv_dir = tmp_path
        mask_dir = tmp_path
    name = "ds_keyed"
    _write_csv(csv_dir, name)
    keyed = {
        f"{name}_seed_0": [True] * 15 + [False] * 5,
        f"{name}_seed_1": [True] * 10 + [False] * 10,
    }
    _write_keyed_masks(mask_dir, keyed)
    bunches = list(load_partitions(name, resamples=2, data_home=csv_dir))
    assert bunches[0].data_train.shape[0] == 15
    assert bunches[1].data_train.shape[0] == 10


def test_partition_bunch_contract(tmp_path):
    """Yielded Bunch has all required fields with correct shapes and dtypes."""
    required = {
        "data_train",
        "target_train",
        "data_test",
        "target_test",
        "feature_names",
        "target_names",
        "dataset_name",
        "resample_id",
        "train_index",
        "test_index",
        "n_classes",
        "DESCR",
    }
    name = "ds_contract"
    _write_csv(tmp_path, name)
    mask = [True] * 14 + [False] * 6
    _write_masks(tmp_path, name, [mask])
    (bunch,) = list(load_partitions(name, resamples=1, data_home=tmp_path))
    assert isinstance(bunch, Bunch)
    assert required <= set(bunch.keys())
    # Feature count is consistent across train and test splits
    assert bunch.data_train.shape[1] == bunch.data_test.shape[1]
    # Targets are integer arrays
    assert np.issubdtype(bunch.target_train.dtype, np.integer)
    assert np.issubdtype(bunch.target_test.dtype, np.integer)
    # dataset_name and resample_id echo the requested values
    assert bunch.dataset_name == name
    assert bunch.resample_id == 0
    # DESCR is a non-empty string
    assert isinstance(bunch.DESCR, str) and len(bunch.DESCR) > 0


def test_partition_index_contract():
    """train_index/test_index are integer, ascending, disjoint and complete."""
    for bunch in load_partitions("era", resamples=3):
        for index, data in (
            (bunch.train_index, bunch.data_train),
            (bunch.test_index, bunch.data_test),
        ):
            assert isinstance(index, np.ndarray)
            assert np.issubdtype(index.dtype, np.integer)
            assert len(index) == data.shape[0]
            assert np.all(np.diff(index) > 0)
        n_total = bunch.data_train.shape[0] + bunch.data_test.shape[0]
        combined = np.concatenate([bunch.train_index, bunch.test_index])
        assert np.array_equal(np.sort(combined), np.arange(n_total))


def test_partition_index_matches_source_rows():
    """train_index/test_index select the same rows as data_train/test."""
    raw = load_dataset("era")
    bunches = list(load_partitions("era", resamples=3))
    for bunch in bunches:
        np.testing.assert_array_equal(raw.data[bunch.train_index], bunch.data_train)
        np.testing.assert_array_equal(raw.data[bunch.test_index], bunch.data_test)


def _setup_missing_csv(tmp_path):
    """Return args for missing-CSV error case; no files written."""
    return "nonexistent_calltime", {}


def _setup_bad_mask(tmp_path):
    """Return args for mask-length-mismatch error case."""
    name = "ds_calltime_badmask"
    _write_csv(tmp_path, name, n_samples=_N_SAMPLES)
    bad_mask = [True] * 13 + [False] * 6
    _write_masks(tmp_path, name, [bad_mask])
    return name, {}


def _setup_missing_keyed_entry(tmp_path):
    """Return args for missing keyed-entry error case."""
    name = "ds_calltime_missingkey"
    _write_csv(tmp_path, name)
    keyed = {f"{name}_seed_0": [True] * 14 + [False] * 6}
    _write_keyed_masks(tmp_path, keyed)
    return name, {"resamples": 2}


def _setup_single_member_class(tmp_path):
    """Return args for a dataset whose rarest class has one member."""
    name = "ds_single_member"
    rows = ["x_0,x_1,y"] + [f"{i}.0,{i}.5,0" for i in range(10)] + ["99.0,99.5,1"]
    (tmp_path / f"{name}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return name, {"resamples": 1}


def _setup_short_per_dataset_masks(tmp_path):
    """Return args for short per-dataset-masks IndexError case."""
    name = "ds_short_masks"
    _write_csv(tmp_path, name)
    mask0 = [True] * 14 + [False] * 6
    _write_masks(tmp_path, name, [mask0])
    return name, {"resamples": 2}


@pytest.mark.parametrize(
    "setup_fn, exc_type, match",
    [
        (_setup_missing_csv, FileNotFoundError, "Dataset file not found"),
        (_setup_bad_mask, ValueError, "Mask for resample 0"),
        (_setup_missing_keyed_entry, KeyError, r"_seed_1"),
        (_setup_short_per_dataset_masks, IndexError, r"\.masks\.json"),
        (_setup_single_member_class, ValueError, "least populated"),
    ],
    ids=[
        "missing-csv",
        "mask-length-mismatch",
        "missing-keyed-entry",
        "short-per-dataset-masks",
        "single-member-class",
    ],
)
def test_load_partitions_eager_errors(tmp_path, setup_fn, exc_type, match):
    """Errors raise eagerly at ``load_partitions`` call time."""
    name, extra_kwargs = setup_fn(tmp_path)
    with pytest.raises(exc_type, match=match):
        load_partitions(name, data_home=tmp_path, **extra_kwargs)


@pytest.mark.parametrize(
    "resamples, exc_type, match",
    [
        ([0, 0], ValueError, "duplicate ids"),
        ([-1], ValueError, "non-negative"),
        ([], ValueError, "at least one id"),
        ([0.0], TypeError, "must be integers"),
        ([True], TypeError, "must be integers"),
        (True, TypeError, "integer count or a list"),
    ],
    ids=["duplicate", "negative", "empty", "float", "bool-id", "bool-count"],
)
def test_invalid_resample_ids(tmp_path, resamples, exc_type, match):
    """Duplicate, negative and non-integer resample ids raise."""
    name = "ds_bad_ids"
    _write_csv(tmp_path, name)
    with pytest.raises(exc_type, match=match):
        load_partitions(name, resamples=resamples, data_home=tmp_path)


@pytest.mark.parametrize(
    "mask, match",
    [
        (["false"] * _N_SAMPLES, "boolean or integer"),
        ([0.4] * _N_SAMPLES, "boolean or integer"),
        ([None] * _N_SAMPLES, "boolean or integer"),
        ([True] * _N_SAMPLES, "degenerate"),
        ([False] * _N_SAMPLES, "degenerate"),
        ([[True, False]] * _N_SAMPLES, "shape"),
    ],
    ids=["string", "float", "null", "all-true", "all-false", "nested"],
)
def test_invalid_masks(tmp_path, mask, match):
    """Malformed masks raise instead of splitting the data silently."""
    name = "ds_bad_mask"
    _write_csv(tmp_path, name)
    _write_masks(tmp_path, name, [mask])
    with pytest.raises(ValueError, match=match):
        load_partitions(name, resamples=1, data_home=tmp_path)


def test_load_partitions_reads_csv_once(tmp_path, monkeypatch):
    """The CSV is parsed exactly once regardless of the number of resamples."""
    calls = 0
    original = _base._read_csv_any

    def counting_read_csv(path):
        """Count CSV reads and delegate to the original ``_read_csv_any``."""
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(_base, "_read_csv_any", counting_read_csv)
    name = "ds_read_once"
    _write_csv(tmp_path, name)
    mask = [True] * 14 + [False] * 6
    _write_masks(tmp_path, name, [mask, mask])
    list(load_partitions(name, resamples=2, data_home=tmp_path))
    assert calls == 1


@pytest.mark.parametrize(
    "resamples, expected_ids",
    [
        (3, [0, 1, 2]),
        (np.int64(3), [0, 1, 2]),
        ([0, 2], [0, 2]),
    ],
    ids=["int", "np.int64", "list"],
)
def test_resamples_input_types(resamples, expected_ids):
    """``int``, ``np.int64``, and list ``resamples`` all yield correct ids."""
    bunches = list(load_partitions("era", resamples=resamples))
    assert [b.resample_id for b in bunches] == expected_ids


def test_resamples_list_with_masks(tmp_path):
    """``resamples=[1]`` selects the second per-dataset mask."""
    name = "ds_list_mask"
    _write_csv(tmp_path, name)
    mask0 = [True] * 14 + [False] * 6
    mask1 = [True] * 10 + [False] * 10
    _write_masks(tmp_path, name, [mask0, mask1])
    (bunch,) = list(load_partitions(name, resamples=[1], data_home=tmp_path))
    assert bunch.resample_id == 1
    assert bunch.data_train.shape[0] == 10
    assert bunch.data_test.shape[0] == 10


@pytest.mark.parametrize(
    "header_class_names, target, expected",
    [
        (
            np.array(["low", "mid", "high"]),
            np.array([0, 1, 2]),
            ["low", "mid", "high"],
        ),
        (
            None,
            np.array([2, 0, 1, 0, 2]),
            ["0", "1", "2"],
        ),
    ],
    ids=["header-present", "header-none"],
)
def test_resolve_target_names(header_class_names, target, expected):
    """Return header names, else sorted unique targets as strings."""
    result = _resolve_target_names(header_class_names, target)
    assert list(result) == expected
    assert result.dtype.kind == "U"


def test_load_bundled_rejects_feature_name_count_mismatch():
    """Declaring the wrong number of column names for a bundled CSV raises."""
    with pytest.raises(ValueError, match="feature name"):
        _load_bundled("era", ["in1", "in2"], return_X_y=False, as_frame=False)
