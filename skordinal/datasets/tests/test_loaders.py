"""Tests for bundled dataset loaders."""

import numpy as np
import pytest
from sklearn.utils import Bunch

from skordinal.datasets import (
    load_balance_scale,
    load_dataset,
    load_era,
    load_esl,
    load_lev,
    load_swd,
)

ALL_LOADERS = [
    load_balance_scale,
    load_era,
    load_esl,
    load_lev,
    load_swd,
]


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_returns_bunch(loader):
    """Loader returns a sklearn Bunch with the standard skordinal fields."""
    bunch = loader()
    assert isinstance(bunch, Bunch)
    for key in (
        "data",
        "target",
        "feature_names",
        "target_names",
        "n_classes",
        "frame",
        "DESCR",
        "filename",
        "data_module",
    ):
        assert key in bunch
    assert bunch.n_classes == len(bunch.target_names)


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_shape_contract(loader):
    """data and target have consistent shapes; names line up with dimensions."""
    bunch = loader()
    assert bunch.data.ndim == 2
    assert bunch.target.shape == (bunch.data.shape[0],)
    assert len(bunch.feature_names) == bunch.data.shape[1]
    assert len(bunch.target_names) == len(np.unique(bunch.target))


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_target_zero_indexed(loader):
    """Target labels are integers in 0..n_classes-1 covering every class."""
    bunch = loader()
    assert np.issubdtype(bunch.target.dtype, np.integer)
    n_classes = len(bunch.target_names)
    assert set(np.unique(bunch.target)) == set(range(n_classes))


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_descr_nonempty(loader):
    """DESCR is a non-empty string referencing the dataset."""
    bunch = loader()
    assert isinstance(bunch.DESCR, str)
    assert len(bunch.DESCR) > 100


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_dtypes(loader):
    """Feature matrix is float64 and the target is int32."""
    bunch = loader()
    assert bunch.data.dtype == np.float64
    assert bunch.target.dtype == np.int32


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_idempotent(loader):
    """Calling the loader twice returns equal arrays."""
    a = loader()
    b = loader()
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_array_equal(a.target, b.target)


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_return_X_y(loader):
    """``return_X_y=True`` yields a (data, target) tuple instead of a Bunch."""
    out = loader(return_X_y=True)
    assert isinstance(out, tuple) and len(out) == 2
    X, y = out
    assert isinstance(X, np.ndarray) and X.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 1
    assert X.shape[0] == y.shape[0]


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_as_frame(loader):
    """``as_frame=True`` populates ``frame`` and yields pandas containers."""
    pd = pytest.importorskip("pandas")
    bunch = loader(as_frame=True)
    assert isinstance(bunch.frame, pd.DataFrame)
    assert isinstance(bunch.data, pd.DataFrame)
    assert isinstance(bunch.target, pd.Series)
    assert bunch.data.shape[1] == len(bunch.feature_names)
    assert bunch.frame.shape == (bunch.data.shape[0], bunch.data.shape[1] + 1)


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_return_X_y_as_frame(loader):
    """Combining both parameters yields pandas DataFrame and Series."""
    pd = pytest.importorskip("pandas")
    X, y = loader(return_X_y=True, as_frame=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_invalid_param(loader):
    """Non-boolean keyword arguments raise ``InvalidParameterError``."""
    with pytest.raises(Exception, match="return_X_y"):
        loader(return_X_y="yes")


@pytest.mark.parametrize(
    "loader,n_samples,n_features,target_names_expected",
    [
        (load_era, 1000, 4, ["1", "2", "3", "4", "5", "6", "7", "8", "9"]),
        (load_esl, 488, 4, ["1", "2", "3", "4", "5", "6", "7", "8", "9"]),
        (load_lev, 1000, 4, ["1", "2", "3", "4", "5"]),
        (load_swd, 1000, 10, ["1", "2", "3", "4"]),
        (load_balance_scale, 625, 4, ["L", "B", "R"]),
    ],
)
def test_loader_known_metadata(loader, n_samples, n_features, target_names_expected):
    """Each dataset has the documented number of samples, features, and class labels."""
    bunch = loader()
    assert bunch.data.shape == (n_samples, n_features)
    assert list(bunch.target_names) == target_names_expected


def test_load_balance_scale_feature_names():
    """Balance Scale exposes the original semantic feature names."""
    bunch = load_balance_scale()
    assert bunch.feature_names == [
        "left_weight",
        "left_distance",
        "right_weight",
        "right_distance",
    ]


@pytest.mark.parametrize("loader", ALL_LOADERS)
def test_loader_matches_load_dataset(loader):
    """A bundled loader agrees with ``load_dataset`` on every shared field."""
    stem = loader.__name__.removeprefix("load_")
    wrapper = loader()
    generic = load_dataset(stem)
    np.testing.assert_array_equal(wrapper.data, generic.data)
    np.testing.assert_array_equal(wrapper.target, generic.target)
    np.testing.assert_array_equal(wrapper.target_names, generic.target_names)
    assert wrapper.n_classes == generic.n_classes
    assert wrapper.DESCR == generic.DESCR
    assert wrapper.filename == generic.filename
    assert wrapper.data_module == generic.data_module
    # feature_names is the one field the wrapper overrides, since the metadata
    # header carries no column names and load_dataset generates x0..xd-1
    assert wrapper.feature_names != generic.feature_names


def test_loader_ignores_cwd_file(tmp_path, monkeypatch):
    """A same-named file in the working directory cannot shadow bundled data."""
    (tmp_path / "era").write_text("junk\n", encoding="utf-8")
    (tmp_path / "era.csv").write_text("1,1,a\n0.0,0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_era().data.shape == (1000, 4)
