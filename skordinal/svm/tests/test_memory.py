"""Regression checks for allocations retained by the native prediction bridges."""

import os
import platform
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(
    platform.libc_ver()[0] != "glibc", reason="requires glibc mallinfo2"
)
@pytest.mark.parametrize("estimator", ["SVOR", "REDSVM"])
@pytest.mark.parametrize("n_features, n_samples", [(257, 1), (2, 257)])
def test_repeated_predictions_release_native_allocations(
    estimator, n_features, n_samples
):
    """Repeated predictions release list headers and uncached dimension integers."""
    script = textwrap.dedent(
        """
        import ctypes
        import gc
        import sys

        import numpy as np

        from skordinal.svm import REDSVM, SVOR, _libsvmrank, _libsvor

        class Mallinfo2(ctypes.Structure):
            _fields_ = [(name, ctypes.c_size_t) for name in (
                "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
                "fsmblks", "uordblks", "fordblks", "keepcost"
            )]

        libc = ctypes.CDLL(None)
        if not hasattr(libc, "mallinfo2"):
            sys.exit(77)
        libc.mallinfo2.restype = Mallinfo2
        estimator, n_features, n_samples = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        X = np.tile(np.arange(6, dtype=float)[:, None], (1, n_features))
        y = np.repeat(np.arange(3), 2)
        classifier = {"SVOR": SVOR, "REDSVM": REDSVM}[estimator](kernel="linear").fit(X, y)
        predict = {"SVOR": _libsvor, "REDSVM": _libsvmrank}[estimator].predict
        features = np.tile(X[:1], (n_samples, 1)).tolist()
        expected = predict(features, classifier.model_)
        for _ in range(200):
            predict(features, classifier.model_)
        gc.collect()
        before = libc.mallinfo2().uordblks
        for _ in range(4000):
            predict(features, classifier.model_)
        gc.collect()
        print(libc.mallinfo2().uordblks - before)
        assert predict(features, classifier.model_) == expected
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, estimator, str(n_features), str(n_samples)],
        env={**os.environ, "PYTHONMALLOC": "malloc"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("requires glibc mallinfo2")
    assert result.returncode == 0, result.stderr
    retained_bytes = int(result.stdout.strip().splitlines()[-1])
    assert retained_bytes < 65536, f"Native prediction retained {retained_bytes} bytes"
