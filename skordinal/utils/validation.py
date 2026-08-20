"""Validation utilities for ordinal classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets


def check_ordinal_targets(
    y: ArrayLike,
) -> tuple[NDArray, NDArray[np.intp]]:
    """Validate an ordinal target vector and return its integer encoding.

    Accepts integer or integer-valued float labels. Rejects string/object
    arrays, continuous floats, arrays with fewer than 2 unique classes,
    and empty or multi-dimensional inputs.

    The return order ``(classes, y_encoded)`` mirrors
    ``np.unique(y, return_inverse=True)`` so callers can write
    ``self.classes_, y_enc = check_ordinal_targets(y)`` in ``fit()``.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels. Must be 1-D and have a numeric dtype. Labels need
        not form a contiguous range; gaps are allowed (e.g. ``[3, 5, 7]``
        is mapped to ``[0, 1, 2]``). Integer-valued floats (e.g.
        ``[1.0, 2.0]``) are accepted; continuous floats (e.g.
        ``[0.5, 1.5]``) are rejected by the upstream sklearn check.

    Returns
    -------
    classes : ndarray of shape (n_classes,)
        Unique labels sorted in ascending order. Dtype matches the
        original dtype of ``y``.

    y_encoded : ndarray of shape (n_samples,), dtype np.intp
        Zero-based contiguous encoding such that
        ``classes[y_encoded[i]] == y[i]`` for every sample ``i``.

    Raises
    ------
    ValueError
        If ``y`` is ``None``, empty, multi-dimensional, has a
        non-numeric dtype, or contains fewer than 2 unique classes.
        Upstream ``ValueError`` from ``check_array`` (e.g. NaN inputs,
        object arrays) and from ``check_classification_targets`` (e.g.
        continuous targets) are propagated unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.validation import check_ordinal_targets
    >>> classes, y_enc = check_ordinal_targets(np.array([3, 1, 2, 1, 3]))
    >>> classes
    array([1, 2, 3])
    >>> y_enc
    array([2, 0, 1, 0, 2])
    """
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None.")

    y = check_array(
        y, ensure_2d=False, dtype="numeric", ensure_min_samples=1, input_name="y"
    )

    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y.shape}.")

    check_classification_targets(y)

    classes, y_encoded = np.unique(y, return_inverse=True)

    if classes.size < 2:
        raise ValueError(
            f"y must contain at least 2 unique classes, got {classes.size}."
        )

    return classes, y_encoded.astype(np.intp)


def _rank_encode_labels(
    y: ArrayLike,
    classes: ArrayLike,
    *,
    input_name: str = "y",
) -> NDArray[np.intp]:
    """Rank-encode ``y`` against ``classes``, rejecting non-finite or
    unknown ``y`` values.

    A label above the last class encodes to ``len(classes)`` without
    raising, so a caller with a documented out-of-range penalty can
    still apply it. Every other unknown label raises. ``classes`` must
    already be non-empty and sorted ascending, which is not validated
    here.
    """
    classes = np.asarray(classes)
    y = np.asarray(y)
    if y.dtype.kind == "f" and not np.isfinite(y).all():
        raise ValueError(f"{input_name} contains non-finite values.")
    idx = np.searchsorted(classes, y)
    # in_bounds excludes only labels above the last class. A mismatch
    # after clipping flags every other unmatched label as unknown.
    in_bounds = idx < len(classes)
    clipped = np.clip(idx, 0, len(classes) - 1)
    mismatched = in_bounds & (classes[clipped] != y)
    if np.any(mismatched):
        bad = np.unique(y[mismatched])
        raise ValueError(
            f"{input_name} contains labels not present in the given "
            f"label set: {bad.tolist()}; labels={classes.tolist()}."
        )
    return idx


def check_thresholds(thresholds: ArrayLike) -> None:
    """Check that thresholds are strictly increasing and finite.

    A valid threshold vector must be 1-D, contain only finite values,
    have at least one entry, and have strictly positive consecutive
    differences.

    Parameters
    ----------
    thresholds : array-like of shape (n_classes - 1,)
        Threshold values defining the boundaries between ordinal classes.
        Must be strictly increasing. Length must be at least 1, i.e.
        ``n_classes >= 2``. Length 1 (binary case) trivially satisfies
        the monotonicity check.

    Raises
    ------
    ValueError
        If ``thresholds`` is not 1-D, is empty, contains non-finite
        values, or is not strictly increasing.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.validation import check_thresholds
    >>> check_thresholds(np.array([-1.0, 0.0, 1.0]))  # returns None
    """
    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim != 1:
        raise ValueError(
            f"thresholds must be a 1D array, got shape {thresholds.shape}."
        )

    if thresholds.size < 1:
        raise ValueError("thresholds must have length >= 1, got 0.")

    if not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be finite, got non-finite values.")

    diffs = np.diff(thresholds)
    if (diffs <= 0).any():
        raise ValueError(
            f"thresholds must be strictly increasing, got differences {diffs!r}."
        )
