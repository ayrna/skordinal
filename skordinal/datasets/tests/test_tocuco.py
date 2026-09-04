"""Tests for the remote Tocuco dataset loader."""

import json
import urllib
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
from sklearn.utils import Bunch

from skordinal.datasets import (
    load_partitions,
    load_tocuco_dataset,
    load_tocuco_partitions,
)

_CSV = """\
x_0,x_1,y
1.0,10.0,0
2.0,20.0,0
3.0,30.0,1
4.0,40.0,1
5.0,50.0,2
6.0,60.0,2
"""
_MASKS = {
    "0": [True, True, True, False, False, False],
    "1": [True, False, True, False, True, False],
}
_EXPECTED_BASE_URL = "https://www.uco.es/ayrna/tocuco/files/toy"


def _fake_downloader(csv_content=_CSV, masks=_MASKS):
    """Return a ``urlretrieve`` replacement backed by in-memory fixtures."""

    def download(url, filename):
        destination = Path(filename)
        if url.endswith(".csv"):
            destination.write_text(csv_content, encoding="utf-8")
        else:
            destination.write_text(json.dumps(masks), encoding="utf-8")
        return str(destination), None

    return download


def test_load_tocuco_partitions_downloads_and_returns_valid_bunch(monkeypatch):
    """A downloaded dataset produces scaled, disjoint train/test partitions."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    result = list(load_tocuco_partitions("toy", resamples=[0]))

    assert len(result) == 1
    bunch = result[0]
    assert isinstance(bunch, Bunch)
    assert bunch.dataset_name == "toy"
    assert bunch.resample_id == 0
    assert bunch.data_train.shape == (3, 2)
    assert bunch.data_test.shape == (3, 2)
    assert bunch.target_train.shape == (3,)
    assert bunch.target_test.shape == (3,)
    assert bunch.feature_names == ["x_0", "x_1"]
    np.testing.assert_array_equal(bunch.target_names, ["0", "1", "2"])
    assert bunch.n_classes == 3
    assert np.isfinite(bunch.data_train).all()
    assert np.isfinite(bunch.data_test).all()
    np.testing.assert_array_equal(bunch.train_index, [0, 1, 2])
    np.testing.assert_array_equal(bunch.test_index, [3, 4, 5])
    assert set(bunch.train_index).isdisjoint(bunch.test_index)
    np.testing.assert_allclose(bunch.data_train.mean(axis=0), 0)
    assert "toy resample 0" in bunch.DESCR


def test_load_tocuco_partitions_uses_expected_remote_files(monkeypatch):
    """The loader downloads the dataset CSV and its keyed mask file."""
    calls = []

    def download(url, filename):
        calls.append(url)
        _fake_downloader()(url, filename)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", download
    )

    next(load_tocuco_partitions("toy", resamples=1))

    assert calls == [
        f"{_EXPECTED_BASE_URL}/toy.csv",
        f"{_EXPECTED_BASE_URL}/train_masks.json",
    ]


def test_load_tocuco_partitions_accepts_count_and_selected_resamples(monkeypatch):
    """An integer requests IDs from zero, while a sequence preserves its IDs."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    results = list(load_tocuco_partitions("toy", resamples=2))
    selected = list(load_tocuco_partitions("toy", resamples=[1]))

    assert [item.resample_id for item in results] == [0, 1]
    assert [item.resample_id for item in selected] == [1]
    np.testing.assert_array_equal(selected[0].train_index, [0, 2, 4])


@pytest.mark.parametrize(
    "name",
    ["toy", Path("toy")],
    ids=["string_name", "path_like_name"],
)
def test_load_tocuco_partitions_normalizes_dataset_name(monkeypatch, name):
    """String and path-like names produce the same metadata and URLs."""
    calls = []

    def download(url, filename):
        calls.append(url)
        _fake_downloader()(url, filename)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", download
    )

    bunch = next(load_tocuco_partitions(name, resamples=1))

    assert bunch.dataset_name == "toy"
    assert calls[0].endswith("/toy/toy.csv")


def test_load_partitions_routes_tocuco_names(monkeypatch):
    """The generic partition API removes the ``tocuco_`` prefix."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    result = next(load_partitions("tocuco_toy", resamples=1))

    assert result.dataset_name == "toy"


def test_load_tocuco_partitions_rejects_missing_resample(monkeypatch):
    """A resample absent from the downloaded mask file raises ``KeyError``."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    with pytest.raises(KeyError, match="Mask key '2'"):
        next(load_tocuco_partitions("toy", resamples=[2]))


@pytest.mark.parametrize(
    "masks",
    [
        {"0": [True] * 4 + [False]},
        {"0": [True] * 7},
        {"0": [True] * 6},
    ],
    ids=["short_mask", "long_mask", "all_train"],
)
def test_load_tocuco_partitions_does_not_accept_invalid_mask_shape(monkeypatch, masks):
    """A mask must describe exactly one non-degenerate split per sample."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(masks=masks),
    )

    with pytest.raises((IndexError, ValueError)):
        next(load_tocuco_partitions("toy", resamples=1))


def test_load_tocuco_partitions_rejects_csv_without_target(monkeypatch):
    """A downloaded CSV without the required ``y`` column is invalid."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(csv_content="x_0,x_1,label\n1,2,0\n3,4,1\n"),
    )

    with pytest.raises(KeyError, match="y"):
        next(load_tocuco_partitions("toy", resamples=1))


def test_load_tocuco_partitions_rejects_corrupt_csv(monkeypatch):
    """Malformed CSV content is reported as a value error."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(csv_content="x_0,x_1,y\n1,2\nnot,csv,content,extra\n"),
    )

    with pytest.raises(ValueError):
        next(load_tocuco_partitions("toy", resamples=1))


def test_load_tocuco_partitions_rejects_corrupt_masks(monkeypatch):
    """Invalid JSON in the mask download is not silently ignored."""

    def corrupt_download(url, filename):
        if url.endswith("train_masks.json"):
            Path(filename).write_text("{not valid json", encoding="utf-8")
        else:
            _fake_downloader()(url, filename)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", corrupt_download
    )

    with pytest.raises(json.JSONDecodeError):
        next(load_tocuco_partitions("toy", resamples=1))


def test_load_tocuco_partitions_translates_non_404_http_error(monkeypatch):
    """HTTP errors other than 404 are surfaced with their status code."""

    def server_error(url, filename):
        raise HTTPError(url, 503, "Unavailable", hdrs=None, fp=None)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", server_error
    )

    with pytest.raises(RuntimeError, match="HTTP error 503"):
        next(load_tocuco_partitions("toy", resamples=1))


def test_load_tocuco_partitions_downloads_masks_before_iteration(monkeypatch):
    """Both remote files are loaded before the first partition is yielded."""
    calls = []

    def download(url, filename):
        calls.append(url)
        _fake_downloader()(url, filename)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", download
    )

    partitions = load_tocuco_partitions("toy", resamples=[0, 1])
    assert len(calls) == 2
    next(partitions)
    assert len(calls) == 2


def test_load_tocuco_partitions_scales_test_using_train_statistics(monkeypatch):
    """Test features are transformed with the training scaler, not refit."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    bunch = next(load_tocuco_partitions("toy", resamples=1))

    np.testing.assert_allclose(bunch.data_train.mean(axis=0), 0)
    np.testing.assert_allclose(bunch.data_train.std(axis=0), 1)
    assert not np.allclose(bunch.data_test.mean(axis=0), 0)


def test_load_tocuco_partitions_preserves_original_targets(monkeypatch):
    """Scaling changes features only; target values and row membership remain intact."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    bunch = next(load_tocuco_partitions("toy", resamples=1))

    np.testing.assert_array_equal(bunch.target_train, [0, 0, 1])
    np.testing.assert_array_equal(bunch.target_test, [1, 2, 2])
    np.testing.assert_array_equal(
        np.sort(np.concatenate((bunch.train_index, bunch.test_index))),
        np.arange(6),
    )


def test_load_tocuco_partitions_rejects_html_dataset(monkeypatch):
    """A successful HTTP response containing an error page is rejected."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(csv_content="<html>not found</html>\n"),
    )

    with pytest.raises(ValueError, match="HTML web page"):
        next(load_tocuco_partitions("missing", resamples=1))


def test_load_tocuco_partitions_translates_http_404(monkeypatch):
    """A missing remote dataset has an actionable ``ValueError``."""

    def not_found(url, filename):
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", not_found
    )

    with pytest.raises(ValueError, match="dataset 'missing' not found"):
        next(load_tocuco_partitions("missing", resamples=1))


def test_load_tocuco_partitions_translates_connection_error(monkeypatch):
    """Network failures are surfaced as a runtime error with the reason."""

    def unavailable(url, filename):
        raise URLError("offline")

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", unavailable
    )

    with pytest.raises(RuntimeError, match="Connection error.*offline"):
        next(load_tocuco_partitions("toy", resamples=1))


@pytest.mark.network
@pytest.mark.parametrize("name", ["dr04_forestfires", "oc03_newthyroid"])
def test_real_tocuco_dataset_is_available_and_valid(name):
    """Known Tocuco datasets can be downloaded and yield a usable partition."""
    bunch = next(load_tocuco_partitions(name, resamples=1))

    assert bunch.data_train.ndim == bunch.data_test.ndim == 2
    assert bunch.data_train.shape[1] == bunch.data_test.shape[1]
    assert bunch.target_train.shape == (bunch.data_train.shape[0],)
    assert bunch.target_test.shape == (bunch.data_test.shape[0],)
    assert bunch.n_classes >= 2
    assert len(bunch.feature_names) == bunch.data_train.shape[1]
    assert np.isfinite(bunch.data_train).all()
    assert np.isfinite(bunch.data_test).all()
    assert set(bunch.train_index).isdisjoint(bunch.test_index)


@pytest.mark.network
def test_real_tocuco_dataset_has_all_thirty_valid_resamples():
    """The canonical Tocuco masks expose thirty complete holdout partitions."""
    partitions = list(load_tocuco_partitions("dr04_forestfires", resamples=30))

    assert [partition.resample_id for partition in partitions] == list(range(30))
    for partition in partitions:
        assert partition.data_train.shape[1] == partition.data_test.shape[1]
        assert partition.data_train.shape[0] == partition.train_index.size
        assert partition.data_test.shape[0] == partition.test_index.size
        assert set(partition.train_index).isdisjoint(partition.test_index)
        indices = np.sort(np.concatenate((partition.train_index, partition.test_index)))
        np.testing.assert_array_equal(indices, np.arange(indices.size))
        assert np.isfinite(partition.data_train).all()
        assert np.isfinite(partition.data_test).all()


@pytest.mark.network
def test_real_tocuco_dataset_not_found():
    """A request for a non-existent dataset on the real server is handled correctly."""
    with pytest.raises((ValueError, urllib.error.HTTPError)):
        next(load_tocuco_partitions("dataset_inventado_12345", resamples=1))


@pytest.mark.network
def test_real_tocuco_dataset_dtypes():
    """Real datasets have float64 features and integer targets."""
    bunch = next(load_tocuco_partitions("dr04_forestfires", resamples=1))

    assert bunch.data_train.dtype == np.float64
    assert bunch.data_test.dtype == np.float64
    assert np.issubdtype(bunch.target_train.dtype, np.integer)
    assert np.issubdtype(bunch.target_test.dtype, np.integer)


def test_load_tocuco_dataset_downloads_and_returns_valid_bunch(monkeypatch):
    """A downloaded dataset produces a standard Bunch."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    bunch = load_tocuco_dataset("toy")

    assert isinstance(bunch, Bunch)
    assert bunch.data.shape == (6, 2)
    assert bunch.target.shape == (6,)
    assert bunch.feature_names == ["x_0", "x_1"]
    np.testing.assert_array_equal(bunch.target_names, ["0", "1", "2"])
    assert bunch.n_classes == 3
    assert bunch.filename == "toy.csv"
    assert bunch.data_module is None
    assert bunch.frame is None
    assert "TOC-UCO Dataset 'toy': 6 samples, 2 features, 3 classes" in bunch.DESCR


def test_load_tocuco_dataset_return_X_y(monkeypatch):
    """The return_X_y parameter yields exactly a (data, target) tuple."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    result = load_tocuco_dataset("toy", return_X_y=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    X, y = result
    assert X.shape == (6, 2)
    assert y.shape == (6,)


def test_load_tocuco_dataset_as_frame(monkeypatch):
    """The as_frame parameter wraps the dataset into pandas objects."""
    import pandas as pd

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(),
    )

    bunch = load_tocuco_dataset("toy", as_frame=True)

    assert isinstance(bunch.frame, pd.DataFrame)
    assert isinstance(bunch.data, pd.DataFrame)
    assert isinstance(bunch.target, pd.Series)
    assert bunch.frame.shape == (6, 3)
    assert list(bunch.data.columns) == ["x_0", "x_1"]
    assert bunch.target.name == "target"


def test_load_tocuco_dataset_uses_expected_remote_files(monkeypatch):
    """The loader downloads only the dataset CSV, not the mask file."""
    calls = []

    def download(url, filename):
        calls.append(url)
        _fake_downloader()(url, filename)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", download
    )

    load_tocuco_dataset("toy")

    assert calls == [f"{_EXPECTED_BASE_URL}/toy.csv"]


@pytest.mark.parametrize(
    "name",
    ["toy", Path("toy")],
    ids=["string_name", "path_like_name"],
)
def test_load_tocuco_dataset_normalizes_dataset_name(monkeypatch, name):
    """String and path-like names produce the same dataset."""
    calls = []

    def download(url, filename):
        calls.append(url)
        _fake_downloader()(url, filename)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", download
    )

    bunch = load_tocuco_dataset(name)

    assert bunch.filename == "toy.csv"
    assert calls[0].endswith("/toy/toy.csv")


def test_load_tocuco_dataset_rejects_html_dataset(monkeypatch):
    """A successful HTTP response containing an error page is rejected."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(csv_content="<html>not found</html>\n"),
    )

    with pytest.raises(ValueError, match="HTML web page"):
        load_tocuco_dataset("missing")


def test_load_tocuco_dataset_translates_http_404(monkeypatch):
    """A missing remote dataset raises a clear ValueError for the user."""

    def not_found(url, filename):
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve", not_found
    )

    with pytest.raises(ValueError, match="not found"):
        load_tocuco_dataset("missing")


def test_load_tocuco_dataset_rejects_corrupt_csv(monkeypatch):
    """Malformed CSV content raises a parsing error."""
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.urllib.request.urlretrieve",
        _fake_downloader(csv_content="x_0,x_1,y\n1,2\nnot,csv,content,extra\n"),
    )

    with pytest.raises(ValueError):
        load_tocuco_dataset("toy")


@pytest.mark.network
@pytest.mark.parametrize("name", ["dr04_forestfires", "oc03_newthyroid"])
def test_real_tocuco_dataset_load_dataset_is_available(name):
    """A real Tocuco dataset can be downloaded as a full bunch without partitions."""
    bunch = load_tocuco_dataset(name)

    assert bunch.data.ndim == 2
    assert bunch.target.shape == (bunch.data.shape[0],)
    assert bunch.n_classes >= 2
    assert len(bunch.feature_names) == bunch.data.shape[1]
    assert np.isfinite(bunch.data).all()
    assert np.issubdtype(bunch.target.dtype, np.integer)
