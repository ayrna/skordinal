"""Tests for the TOC-UCO dataset API (fetch, partition access)."""

from __future__ import annotations

import contextlib
import json
import urllib
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from skordinal.datasets import (
    _tocuco,
    fetch_tocuco,
    fetch_tocuco_partition,
    load_partitions,
    load_tocuco_partitions,
)

_CSV_CONTENT = """\
x_0,x_1,y
1.0,2.0,0
3.0,4.0,1
5.0,6.0,2
7.0,8.0,0
9.0,10.0,1
11.0,12.0,2
13.0,14.0,0
15.0,16.0,1
17.0,18.0,2
19.0,20.0,0
"""


_DATASET_MASKS_LIST = [[True] * 7 + [False] * 3, [True, False] * 5]


_SITE_TRAIN_MASKS = {"0": _DATASET_MASKS_LIST[0], "1": _DATASET_MASKS_LIST[1]}


_METADATA_HEADER = (
    "dataset,is_oc,n_patterns_train,n_patterns_test,n_features,n_classes,"
    "class_distr,imbalance_ratio"
)


_METADATA_CSV = (
    _METADATA_HEADER + "\n" + 'oc03_fake,True,7,3,2,3,"[0.4 0.3 0.3]",1.33\n'
)


_REQUIRED_KEYS = {
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


_TOCUCO_DATASET_KEYS = {
    "data",
    "target",
    "frame",
    "feature_names",
    "target_names",
    "n_classes",
    "filename",
    "data_module",
    "DESCR",
    "dataset_name",
    "is_oc",
    "n_patterns_train",
    "n_patterns_test",
    "class_distr",
    "imbalance_ratio",
    "url",
}


_SOFT_404_HTML = "<html><body>Dataset not found, redirecting...</body></html>"


def _build_dataset_tree(tocuco_root, name, is_oc=True):
    """Write a valid per-dataset TOC-UCO tree named *name* under *tocuco_root*."""
    dataset_dir = tocuco_root / name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / f"{name}.csv").write_text(_CSV_CONTENT, encoding="utf-8")
    metadata = (
        "dataset,is_oc,n_patterns_train,n_patterns_test,n_features,n_classes,"
        "class_distr,imbalance_ratio\n"
        f'{name},{is_oc},7,3,2,3,"[0.4 0.3 0.3]",1.33\n'
    )
    (dataset_dir / "metadata.csv").write_text(metadata, encoding="utf-8")
    (dataset_dir / f"{name}.masks.json").write_text(
        json.dumps(_DATASET_MASKS_LIST), encoding="utf-8"
    )
    return dataset_dir


def _build_partial_tocuco_dataset_tree(tocuco_root, name, omit):
    """Write a per-dataset TOC-UCO tree under *tocuco_root* missing *omit*."""
    dataset_dir = tocuco_root / name
    dataset_dir.mkdir(parents=True)
    if omit != "csv":
        (dataset_dir / f"{name}.csv").write_text(_CSV_CONTENT, encoding="utf-8")
    if omit != "metadata":
        (dataset_dir / "metadata.csv").write_text(_METADATA_CSV, encoding="utf-8")
    if omit != "masks":
        (dataset_dir / f"{name}.masks.json").write_text(
            json.dumps(_DATASET_MASKS_LIST), encoding="utf-8"
        )


def _assert_nothing_published(tmp_path):
    """Assert the cache root under *tmp_path* holds no dataset, marker or staging."""
    root = tmp_path / "tocuco"
    assert not root.exists() or list(root.iterdir()) == []


class _FakeHeaders:
    """Minimal stand-in for urlretrieve's returned HTTP headers."""

    def __init__(self, content_type):
        self._content_type = content_type

    def get_content_type(self):
        """Return the fake response content type."""
        return self._content_type


class _FakeUrlretrieve:
    """Record calls and write fake per-dataset site payloads by URL suffix."""

    def __init__(self):
        self.calls = []
        self.site_masks = _SITE_TRAIN_MASKS
        self.fail_suffix = None
        self.fail_exc = None
        self.html_names = set()
        self.html_suffix = None
        # URL suffix -> (body, content type), served instead of the defaults
        self.raw_payloads = {}
        # Called with each URL before it is served, for mid-download side effects
        self.on_request = None

    def __call__(self, url, filename):
        """Mimic urlretrieve(url, filename), returning the (filename, headers) pair."""
        self.calls.append(url)
        if self.on_request is not None:
            self.on_request(url)
        if self.fail_suffix is not None and url.endswith(self.fail_suffix):
            raise self.fail_exc
        for suffix, (body, content_type) in self.raw_payloads.items():
            if url.endswith(suffix):
                Path(filename).write_text(body, encoding="utf-8")
                return filename, _FakeHeaders(content_type)
        # The name is the URL's second-to-last segment in .../{name}/{file}
        name = url.split("/")[-2]
        is_soft_404 = name in self.html_names or (
            self.html_suffix is not None and url.endswith(self.html_suffix)
        )
        if is_soft_404:
            Path(filename).write_text(_SOFT_404_HTML, encoding="utf-8")
            return filename, _FakeHeaders("text/html")
        if url.endswith("/train_masks.json"):
            content, content_type = json.dumps(self.site_masks), "application/json"
        elif url.endswith("/metadata.csv"):
            content, content_type = _METADATA_CSV, "application/octet-stream"
        elif url.endswith(".csv"):
            content, content_type = _CSV_CONTENT, "application/octet-stream"
        else:
            raise AssertionError(f"unexpected URL in fake_urlretrieve: {url}")
        Path(filename).write_text(content, encoding="utf-8")
        return filename, _FakeHeaders(content_type)


def _fetch_partition_by_name(name, **kwargs):
    """Call fetch_tocuco_partition for resample 0, matching fetch_tocuco's shape."""
    return fetch_tocuco_partition(name, 0, **kwargs)


# The two per-dataset entry points share their whole cache preamble
_FETCHERS = [fetch_tocuco, _fetch_partition_by_name]


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Autouse: raise on any unmocked urlretrieve call unless marked network."""
    if request.node.get_closest_marker("network") is not None:
        return

    def _unexpected(*args, **kwargs):
        """Raise to catch accidental real network access in offline tests."""
        raise AssertionError("unexpected network access")

    monkeypatch.setattr("skordinal.datasets._tocuco.urlretrieve", _unexpected)


@pytest.fixture
def tocuco_root(tmp_path):
    """Minimal fake TOC-UCO cache root used by all non-network tests."""
    _build_dataset_tree(tmp_path / "tocuco", "oc03_fake")
    return tmp_path / "tocuco"


@pytest.fixture
def fake_urlretrieve(monkeypatch):
    """Monkeypatch urlretrieve with a call-recording per-dataset site fake."""
    fake = _FakeUrlretrieve()
    monkeypatch.setattr("skordinal.datasets._tocuco.urlretrieve", fake)
    return fake


@pytest.mark.parametrize(
    "status,reason,expected_attempts,retried",
    [
        (403, "Forbidden", 1, False),
        (408, "Request Timeout", 3, True),
        (429, "Too Many Requests", 3, True),
        (503, "Service Unavailable", 3, True),
    ],
    ids=["403-not-retried", "408-retried", "429-retried", "503-retried"],
)
def test_fetch_remote_retry_policy_by_status(
    tmp_path, monkeypatch, status, reason, expected_attempts, retried
):
    """A permanent 4xx fails fast, while 408/429/5xx retry then raise."""
    n_retries = 2
    call_count = {"n": 0}

    def fake_urlretrieve(url, filename):
        """Raise the parametrized HTTP status on every call."""
        call_count["n"] += 1
        raise HTTPError(url, status, reason, hdrs=None, fp=None)

    monkeypatch.setattr("skordinal.datasets._tocuco.urlretrieve", fake_urlretrieve)
    remote = _tocuco.RemoteFileMetadata(
        "oc03_fake.csv", "https://example.test/oc03_fake.csv"
    )
    warns_ctx = (
        pytest.warns(UserWarning, match="Retry")
        if retried
        else contextlib.nullcontext()
    )

    with warns_ctx, pytest.raises(HTTPError):
        _tocuco._fetch_remote(remote, tmp_path, n_retries=n_retries, delay=0.0)

    assert call_count["n"] == expected_attempts


def test_fetch_remote_retries_then_succeeds(tmp_path, monkeypatch):
    """A single transient failure is retried once, then the download succeeds."""
    call_count = {"n": 0}

    def flaky_urlretrieve(url, filename):
        """Fail once with URLError, then write the file and succeed."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise URLError("simulated transient failure")
        Path(filename).write_text(_CSV_CONTENT, encoding="utf-8")
        return filename, _FakeHeaders("application/octet-stream")

    monkeypatch.setattr("skordinal.datasets._tocuco.urlretrieve", flaky_urlretrieve)
    remote = _tocuco.RemoteFileMetadata(
        "oc03_fake.csv", "https://example.test/oc03_fake.csv"
    )

    with pytest.warns(UserWarning, match="Retry"):
        result = _tocuco._fetch_remote(remote, tmp_path, n_retries=2, delay=0.0)

    assert call_count["n"] == 2
    assert result.is_file()


def test_fetch_tocuco_bunch_contract(tocuco_root, tmp_path):
    """The dataset Bunch carries the loader core keys plus TOC-UCO metadata."""
    bunch = fetch_tocuco("oc03_fake", data_home=tmp_path)

    assert isinstance(bunch, Bunch)
    assert _TOCUCO_DATASET_KEYS <= set(bunch.keys())
    assert bunch.dataset_name == "oc03_fake"
    assert "oc03_fake" in bunch.DESCR
    assert bunch.url == f"{_tocuco._DATASET_BASE_URL}/oc03_fake"
    # data holds every sample, train and test together
    assert bunch.data.shape == (10, 2)

    assert bunch.data.shape[0] == bunch.n_patterns_train + bunch.n_patterns_test
    assert isinstance(bunch.is_oc, bool)
    assert isinstance(bunch.n_patterns_train, int)
    assert isinstance(bunch.n_patterns_test, int)
    assert isinstance(bunch.imbalance_ratio, float)
    np.testing.assert_array_equal(bunch.class_distr, [0.4, 0.3, 0.3])


def test_fetch_tocuco_return_x_y_returns_bare_tuple(tocuco_root, tmp_path):
    """return_X_y=True yields a bare (X, y) tuple with no metadata fields."""
    result = fetch_tocuco("oc03_fake", data_home=tmp_path, return_X_y=True)
    assert isinstance(result, tuple)
    X, y = result
    assert X.shape == (10, 2)
    assert y.shape == (10,)


def test_fetch_tocuco_as_frame_returns_dataframe_and_series(tocuco_root, tmp_path):
    """as_frame=True returns a DataFrame data and a Series target."""
    bunch = fetch_tocuco("oc03_fake", data_home=tmp_path, as_frame=True)
    assert isinstance(bunch.data, pd.DataFrame)
    assert isinstance(bunch.target, pd.Series)


def test_fetch_tocuco_metadata_with_bom(tocuco_root, tmp_path):
    """A UTF-8 BOM on metadata.csv does not hide the dataset's row."""
    meta_path = tocuco_root / "oc03_fake" / "metadata.csv"
    meta_path.write_text(
        "\ufeff" + meta_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert fetch_tocuco("oc03_fake", data_home=tmp_path).is_oc is True


def test_fetch_tocuco_download_publishes_normalised_layout(tmp_path, fake_urlretrieve):
    """A fresh download publishes the 3-file layout with masks as an ordered list."""
    fetch_tocuco("oc03_fake", data_home=tmp_path)
    dest = tmp_path / "tocuco" / "oc03_fake"
    assert {p.name for p in dest.iterdir()} == {
        "oc03_fake.csv",
        "metadata.csv",
        "oc03_fake.masks.json",
    }
    # Element k of the list is the site dict's key str(k), order preserved
    published = json.loads((dest / "oc03_fake.masks.json").read_text(encoding="utf-8"))
    assert published == [_SITE_TRAIN_MASKS["0"], _SITE_TRAIN_MASKS["1"]]
    bunch = fetch_tocuco_partition("oc03_fake", 0, data_home=tmp_path)
    assert bunch.data_train.shape[0] == sum(_SITE_TRAIN_MASKS["0"])


def test_fetch_tocuco_idempotent_skips_download(tmp_path, fake_urlretrieve):
    """A second call with a populated cache issues zero urlretrieve calls."""
    fetch_tocuco("oc03_fake", data_home=tmp_path)
    n_calls_after_first = len(fake_urlretrieve.calls)
    fetch_tocuco("oc03_fake", data_home=tmp_path)
    assert len(fake_urlretrieve.calls) == n_calls_after_first


def test_fetch_tocuco_differently_cased_name_is_not_a_cache_hit(tocuco_root, tmp_path):
    """A miscased name must not adopt another dataset's cache directory."""
    # Only discriminates on a case-insensitive filesystem (macOS, Windows)
    # Must read as "not cached", never as a stale cache to delete
    with pytest.raises(OSError, match="not found at") as excinfo:
        fetch_tocuco("OC03_FAKE", data_home=tmp_path, download_if_missing=False)
    assert "Delete this directory" not in str(excinfo.value)
    assert (tocuco_root / "oc03_fake" / "oc03_fake.csv").is_file()


@pytest.mark.parametrize(
    "omit,expected_fragment",
    [
        ("csv", "'oc03_fake.csv'"),
        ("metadata", "'metadata.csv'"),
        ("masks", "'oc03_fake.masks.json'"),
    ],
    ids=["missing-csv", "missing-metadata", "missing-masks"],
)
def test_fetch_tocuco_partial_cache_raises(omit, expected_fragment, tmp_path):
    """A tocuco/{name}/ dir missing one file raises OSError naming it."""
    _build_partial_tocuco_dataset_tree(tmp_path / "tocuco", "oc03_fake", omit)
    with pytest.raises(OSError, match=expected_fragment):
        fetch_tocuco("oc03_fake", data_home=tmp_path, download_if_missing=False)


@pytest.mark.parametrize(
    "metadata_csv,expected_fragment",
    [
        (
            '{header}\nother_dataset,True,7,3,2,3,"[0.4 0.3 0.3]",1.33\n',
            "no entry for",
        ),
        (
            "is_oc,n_patterns_train,n_patterns_test,n_features,n_classes,"
            "class_distr,imbalance_ratio\n"
            'True,7,3,2,3,"[0.4 0.3 0.3]",1.33\n',
            "no entry for",
        ),
        (
            '{header}\noc03_fake,True,N/A,3,2,3,"[0.4 0.3 0.3]",1.33\n',
            "malformed",
        ),
        (
            "dataset,is_oc,n_patterns_test,n_features,n_classes,"
            "class_distr,imbalance_ratio\n"
            'oc03_fake,True,3,2,3,"[0.4 0.3 0.3]",1.33\n',
            "malformed",
        ),
        (
            '{header}\noc03_fake,True,7,3,2,3,"[0.4 x 0.3]",1.33\n',
            "malformed",
        ),
    ],
    ids=[
        "row-for-another-dataset",
        "no-dataset-column",
        "non-numeric-cell",
        "missing-numeric-column",
        "non-numeric-class-distr",
    ],
)
def test_fetch_tocuco_unusable_metadata_raises_oserror(
    metadata_csv, expected_fragment, tocuco_root, tmp_path
):
    """A structurally complete cache with unusable metadata raises OSError."""
    (tocuco_root / "oc03_fake" / "metadata.csv").write_text(
        metadata_csv.format(header=_METADATA_HEADER), encoding="utf-8"
    )
    with pytest.raises(OSError, match=expected_fragment):
        fetch_tocuco("oc03_fake", data_home=tmp_path)


def test_fetch_tocuco_atomic_on_download_failure(tmp_path, fake_urlretrieve):
    """A download failure on train_masks.json leaves no staging or dest dir."""
    fake_urlretrieve.fail_suffix = "/train_masks.json"
    fake_urlretrieve.fail_exc = URLError("simulated network failure")
    with pytest.raises(URLError):
        fetch_tocuco("oc03_fake", data_home=tmp_path, n_retries=0, delay=0.0)
    _assert_nothing_published(tmp_path)


def test_fetch_tocuco_unknown_name_raises_value_error(tmp_path, fake_urlretrieve):
    """A soft-404 (HTML body, HTTP 200) for an unknown name raises ValueError."""
    fake_urlretrieve.html_names = {"no_such_dataset"}
    with pytest.raises(ValueError, match="not found in the TOC-UCO repository"):
        fetch_tocuco("no_such_dataset", data_home=tmp_path)
    assert not (tmp_path / "tocuco" / "no_such_dataset").exists()
    # The CSV is fetched first and fails at once, so nothing else follows
    assert fake_urlretrieve.calls == [
        f"{_tocuco._DATASET_BASE_URL}/no_such_dataset/no_such_dataset.csv"
    ]


@pytest.mark.network
def test_fetch_tocuco_real_unknown_name_is_a_soft_404(tmp_path):
    """The live site answers an unknown name with a web page, not an HTTP error."""
    with pytest.raises(ValueError, match="not found in the TOC-UCO repository"):
        fetch_tocuco("no_such_dataset_xyz", data_home=tmp_path)
    _assert_nothing_published(tmp_path)


def test_fetch_tocuco_soft_404_on_metadata_raises_value_error(
    tmp_path, fake_urlretrieve
):
    """A soft-404 on the second (metadata.csv) request raises ValueError, no publish."""
    fake_urlretrieve.html_suffix = "/metadata.csv"
    with pytest.raises(ValueError, match="not found in the TOC-UCO repository"):
        fetch_tocuco("oc03_fake", data_home=tmp_path)
    assert not (tmp_path / "tocuco" / "oc03_fake").exists()
    # The CSV succeeds, metadata fails as a soft-404, and masks never follows
    assert fake_urlretrieve.calls == [
        f"{_tocuco._DATASET_BASE_URL}/oc03_fake/oc03_fake.csv",
        f"{_tocuco._DATASET_BASE_URL}/oc03_fake/metadata.csv",
    ]


@pytest.mark.parametrize(
    "site_masks",
    [
        [True, False],
        {"a": [True] * 7 + [False] * 3, "b": [True, False] * 5},
        {"00": [True] * 7 + [False] * 3, "01": [True, False] * 5},
        {},
        {"0": True, "1": False},
        {"0": [True] * 7 + [False] * 3, "2": [True, False] * 5},
        {"0": ["true"] * 7 + ["false"] * 3, "1": ["true", "false"] * 5},
        {"0": [1] * 7 + [0] * 3, "1": [1, 0] * 5},
        {
            "0": [True, False, True, False, True, False, True, None, True, False],
            "1": [True, False] * 5,
        },
        {"0": [[True]] * 7 + [[False]] * 3, "1": [[True], [False]] * 5},
        {"0": [True] * 7 + [False] * 3, "1": [True] * 5},
        {"0": [True] * 10, "1": [True, False] * 5},
    ],
    ids=[
        "non-dict-json",
        "non-int-keys",
        "zero-padded-keys",
        "empty-dict",
        "scalar-values",
        "non-contiguous-keys",
        "list-of-strings",
        "list-of-ints",
        "null-values",
        "nested-lists",
        "mismatched-lengths",
        "all-true-mask",
    ],
)
def test_fetch_tocuco_corrupt_masks_variants_raise_and_do_not_publish(
    site_masks, tmp_path, fake_urlretrieve
):
    """Every malformed train_masks.json shape raises OSError, publishing nothing."""
    fake_urlretrieve.site_masks = site_masks
    with pytest.raises(OSError, match="is malformed"):
        fetch_tocuco("oc03_fake", data_home=tmp_path)
    _assert_nothing_published(tmp_path)


def test_fetch_tocuco_soft_404_backstop_on_masks(tmp_path, fake_urlretrieve):
    """An HTML masks body mislabelled as JSON is caught by the mask JSON backstop."""
    fake_urlretrieve.raw_payloads["/train_masks.json"] = (
        _SOFT_404_HTML,
        "application/json",
    )
    with pytest.raises(OSError, match="is malformed"):
        fetch_tocuco("oc03_fake", data_home=tmp_path)
    _assert_nothing_published(tmp_path)


def test_fetch_tocuco_concurrent_publish_race_survives(tmp_path, fake_urlretrieve):
    """A dest fully published by a concurrent fetch wins, with no nested staging."""
    dest = tmp_path / "tocuco" / "oc03_fake"

    def rival_publishes(url):
        """Publish a full tree at dest while this fetch still downloads."""
        if url.endswith("/train_masks.json"):
            _build_dataset_tree(dest.parent, dest.name)
            (dest / "sentinel.txt").write_text("winner", encoding="utf-8")

    fake_urlretrieve.on_request = rival_publishes
    bunch = fetch_tocuco("oc03_fake", data_home=tmp_path)

    assert bunch.dataset_name == "oc03_fake"
    assert {p.name for p in dest.iterdir()} == {
        "oc03_fake.csv",
        "metadata.csv",
        "oc03_fake.masks.json",
        "sentinel.txt",
    }


def test_fetch_tocuco_rename_failure_without_valid_tree_raises_oserror(
    tmp_path, fake_urlretrieve, monkeypatch
):
    """A genuine rename failure (no tree adopted) raises OSError, not FileNotFoundError."""

    def failing_rename(src, dst):
        """Simulate a genuine (non-benign) publish failure."""
        raise OSError("simulated rename failure")

    monkeypatch.setattr("skordinal.datasets._tocuco.os.rename", failing_rename)

    with pytest.raises(OSError) as exc_info:
        fetch_tocuco("oc03_fake", data_home=tmp_path)

    # A swallowed publish failure would surface later as a FileNotFoundError
    # from load_dataset instead of the real cause
    assert not isinstance(exc_info.value, FileNotFoundError)
    assert not (tmp_path / "tocuco" / "oc03_fake").exists()


def test_fetch_tocuco_partition_bunch_contract(tocuco_root, tmp_path):
    """The partition Bunch carries every documented field with its own type."""
    bunch = fetch_tocuco_partition("oc03_fake", 1, data_home=tmp_path)

    assert isinstance(bunch, Bunch)
    assert _REQUIRED_KEYS <= set(bunch.keys())
    assert bunch.dataset_name == "oc03_fake"
    assert bunch.resample_id == 1
    assert bunch.feature_names == ["x_0", "x_1"]
    assert list(bunch.target_names) == ["0", "1", "2"]
    assert bunch.n_classes == 3
    assert "oc03_fake" in bunch.DESCR
    # Resample 1 is the alternating mask, so the split is 5/5
    assert bunch.data_train.shape == (5, 2)
    assert bunch.data_test.shape == (5, 2)
    assert bunch.target_train.shape == (5,)
    assert bunch.target_test.shape == (5,)
    assert np.issubdtype(bunch.data_train.dtype, np.floating)
    assert np.issubdtype(bunch.target_train.dtype, np.integer)


def test_fetch_tocuco_partition_applies_the_published_mask(tocuco_root, tmp_path):
    """Rows the mask marks True go to train, the rest to test, in file order."""
    bunch = fetch_tocuco_partition("oc03_fake", 0, data_home=tmp_path)
    expected = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ]
    )
    np.testing.assert_array_equal(bunch.data_train, expected[:7])
    np.testing.assert_array_equal(bunch.data_test, expected[7:])
    assert bunch.train_index.tolist() == list(range(7))


def test_fetch_tocuco_partition_descr_falls_back_when_metadata_has_no_entry(tmp_path):
    """DESCR falls back to the bare form when metadata.csv lacks the name's row."""
    tocuco_root_dir = tmp_path / "tocuco"
    _build_dataset_tree(tocuco_root_dir, "oc03_fake")
    divergent_metadata = (
        "dataset,is_oc,n_patterns_train,n_patterns_test,n_features,n_classes,"
        "class_distr,imbalance_ratio\n"
        'other_dataset,True,7,3,2,3,"[0.4 0.3 0.3]",1.33\n'
    )
    (tocuco_root_dir / "oc03_fake" / "metadata.csv").write_text(
        divergent_metadata, encoding="utf-8"
    )
    bunch = fetch_tocuco_partition("oc03_fake", 0, data_home=tmp_path)
    assert bunch.DESCR == "TOC-UCO dataset 'oc03_fake'."


def test_fetch_tocuco_partition_missing_seed_raises(tocuco_root, tmp_path):
    """IndexError for a resample_id past the end of the dataset's masks."""
    with pytest.raises(IndexError, match=r"No mask for resample 99"):
        fetch_tocuco_partition("oc03_fake", 99, data_home=tmp_path)


def test_fetch_tocuco_partition_downloads_only_the_requested_dataset(
    tmp_path, fake_urlretrieve
):
    """fetch_tocuco_partition with nothing cached downloads only that one dataset."""
    fetch_tocuco_partition("oc03_fake", 0, data_home=tmp_path)
    assert {p.name for p in (tmp_path / "tocuco").iterdir()} == {"oc03_fake"}
    assert len(fake_urlretrieve.calls) == 3


@pytest.mark.parametrize("fetcher", _FETCHERS, ids=["fetch", "partition"])
def test_fetch_missing_cache_no_download_raises(fetcher, tmp_path):
    """Both fetchers refuse an absent cache when download_if_missing=False."""
    with pytest.raises(OSError, match="TOC-UCO dataset 'oc03_fake' not found"):
        fetcher("oc03_fake", data_home=tmp_path, download_if_missing=False)


@pytest.mark.parametrize("fetcher", _FETCHERS, ids=["fetch", "partition"])
def test_fetch_downloaded_tree_missing_masks_raises(fetcher, tmp_path, monkeypatch):
    """A maskless tree adopted mid-race raises OSError, never a generated holdout."""

    def fake_download(name, data_home, n_retries, delay):
        """Simulate adopting a rival tree that lacks the masks file."""
        _build_partial_tocuco_dataset_tree(tmp_path / "tocuco", name, omit="masks")
        return tmp_path / "tocuco" / name

    monkeypatch.setattr(
        "skordinal.datasets._tocuco._download_tocuco_dataset", fake_download
    )
    with pytest.raises(OSError, match=r"oc03_fake\.masks\.json"):
        fetcher("oc03_fake", data_home=tmp_path)


@pytest.mark.parametrize(
    "call,fail_suffix,n_calls_before",
    [
        (lambda **kw: fetch_tocuco("oc03_fake", **kw), "/train_masks.json", 2),
        (
            lambda **kw: fetch_tocuco_partition("oc03_fake", 0, **kw),
            "/train_masks.json",
            2,
        ),
    ],
    ids=["fetch", "partition"],
)
def test_n_retries_and_delay_reach_the_retry_loop(
    call, fail_suffix, n_calls_before, tmp_path, fake_urlretrieve, monkeypatch
):
    """n_retries and delay drive the retry loop, which leaves no temp file behind."""
    n_retries = 2
    delay = 1.5
    sleep_calls = []
    fake_urlretrieve.fail_suffix = fail_suffix
    fake_urlretrieve.fail_exc = URLError("simulated network failure")
    monkeypatch.setattr(
        "skordinal.datasets._tocuco.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    with pytest.warns(UserWarning, match="Retry"), pytest.raises(URLError):
        call(data_home=tmp_path, n_retries=n_retries, delay=delay)

    # The files before the failing one are fetched once each
    assert len(fake_urlretrieve.calls) == n_calls_before + n_retries + 1
    assert sleep_calls == [delay] * n_retries
    assert list(tmp_path.rglob("*.part_*")) == []
    _assert_nothing_published(tmp_path)


@pytest.mark.parametrize(
    "call",
    [
        lambda tmp_path: fetch_tocuco("a/b", data_home=tmp_path),
        lambda tmp_path: fetch_tocuco("", data_home=tmp_path),
        lambda tmp_path: fetch_tocuco_partition("../x", 0, data_home=tmp_path),
        lambda tmp_path: fetch_tocuco(".", data_home=tmp_path),
        lambda tmp_path: fetch_tocuco("..", data_home=tmp_path),
        lambda tmp_path: fetch_tocuco(".hidden", data_home=tmp_path),
        lambda tmp_path: fetch_tocuco("oc03_fake.csv", data_home=tmp_path),
    ],
    ids=[
        "fetch_tocuco-embedded-slash",
        "fetch_tocuco-empty-name",
        "fetch_tocuco_partition-dotdot",
        "fetch_tocuco-dot",
        "fetch_tocuco-dotdot",
        "fetch_tocuco-hidden",
        "fetch_tocuco-csv-suffix",
    ],
)
def test_unsafe_or_empty_dataset_name_raises_value_error(call, tmp_path):
    """An unsafe or empty dataset name is rejected with ValueError before any I/O."""
    with pytest.raises(ValueError, match="(?i)name"):
        call(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "call",
    [
        lambda path: fetch_tocuco("oc03_fake", path),
        lambda path: fetch_tocuco_partition("oc03_fake", 0, path),
    ],
    ids=["fetch", "partition"],
)
def test_data_home_is_keyword_only(call, tmp_path):
    """Every public entry point rejects a positional data_home."""
    with pytest.raises(TypeError):
        call(tmp_path)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "name",
    [
        "fetch_tocuco",
        "fetch_tocuco_partition",
    ],
)
def test_public_name_is_listed_in_all(name):
    """Each public TOC-UCO function is listed in the datasets package's __all__."""
    import skordinal.datasets as datasets_pkg

    assert name in datasets_pkg.__all__


def test_load_partitions_on_tocuco_layout(tocuco_root):
    """load_partitions resolves a cached dataset directory directly."""
    bunches = list(
        load_partitions("oc03_fake", resamples=2, data_home=tocuco_root / "oc03_fake")
    )
    assert [b.resample_id for b in bunches] == [0, 1]
    assert bunches[0].data_train.shape == (7, 2)
    assert bunches[1].data_train.shape == (5, 2)
    for bunch in bunches:
        assert _REQUIRED_KEYS <= set(bunch.keys())


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

    with pytest.raises(ValueError, match="Dataset 'missing' not found"):
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
