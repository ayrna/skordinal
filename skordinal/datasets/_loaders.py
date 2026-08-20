"""Loaders for bundled ordinal classification datasets."""

import csv
from importlib import resources

import numpy as np
from sklearn.utils import Bunch
from sklearn.utils._param_validation import validate_params

DATA_MODULE = "skordinal.datasets.data"
DESCR_MODULE = "skordinal.datasets.descr"


def _load_csv_data(data_file_name, *, descr_file_name):
    """Load a sklearn-style CSV with metadata header and a description file.

    The CSV's first row stores ``n_samples,n_features,target_name_0,...``;
    subsequent rows store ``feature_0,...,feature_d-1,target_int``.  Targets
    are read as zero-indexed integers.
    """
    data_path = resources.files(DATA_MODULE) / data_file_name
    with data_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        n_samples = int(header[0])
        n_features = int(header[1])
        target_names = np.array(header[2:])
        data = np.empty((n_samples, n_features), dtype=np.float64)
        target = np.empty((n_samples,), dtype=np.int32)
        for i, row in enumerate(reader):
            data[i] = np.asarray(row[:-1], dtype=np.float64)
            target[i] = int(row[-1])
    descr = (resources.files(DESCR_MODULE) / descr_file_name).read_text(
        encoding="utf-8"
    )
    return data, target, target_names, descr


def _convert_data_dataframe(caller_name, data, target, feature_names, target_columns):
    """Combine ``data`` and ``target`` into a pandas frame for ``as_frame=True``."""
    try:
        import pandas as pd
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised in environments without pandas
        raise ImportError(f"{caller_name} with as_frame=True requires pandas.") from exc
    data_df = pd.DataFrame(data, columns=feature_names, copy=False)
    target_df = pd.DataFrame(target, columns=target_columns)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_columns]
    if y.shape[1] == 1:  # pragma: no branch
        y = y.iloc[:, 0]
    return combined_df, X, y


def _bundle(
    data,
    target,
    target_names,
    descr,
    feature_names,
    filename,
    *,
    return_X_y,
    as_frame,
    caller_name,
):
    """Common return path for the public loaders."""
    frame = None
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            caller_name, data, target, feature_names, ["target"]
        )
    if return_X_y:
        return data, target
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=descr,
        feature_names=feature_names,
        filename=filename,
        data_module=DATA_MODULE,
    )


@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_era(*, return_X_y=False, as_frame=False):
    """Load and return the ERA dataset (ordinal classification).

    Four ordinal input attributes describing job candidates; the target is
    an overall acceptance level on a 1-9 scale (9 ordered classes).

    =================   ==============
    Classes                          9
    Samples total                 1000
    Dimensionality                   4
    Features                  ordinal
    =================   ==============

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas Series. If
        ``return_X_y`` is True, then ``(data, target)`` will be a pandas
        DataFrame and Series.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object with the following attributes.

        data : {ndarray, dataframe} of shape (1000, 4)
            The data matrix. If ``as_frame=True``, ``data`` is a DataFrame.
        target : {ndarray, Series} of shape (1000,)
            Ordinal target labels, zero-indexed (``0`` to ``8``).
        feature_names : list of str
            ``["in1", "in2", "in3", "in4"]``.
        target_names : ndarray of str
            ``["1", "2", "3", "4", "5", "6", "7", "8", "9"]``.
        frame : DataFrame of shape (1000, 5) or None
            ``None`` if ``as_frame=False``; otherwise a DataFrame
            combining ``data`` and ``target``.
        DESCR : str
            Full description of the dataset.
        filename : str
            Name of the CSV file inside the data module.
        data_module : str
            Python module path used by :func:`importlib.resources` to
            locate the data file.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays. The first contains a 2D array of shape
        ``(1000, 4)`` with each row representing one sample and each
        column representing a feature. The second array of shape ``(1000,)``
        contains the ordinal target labels.

    Examples
    --------
    >>> from skordinal.datasets import load_era
    >>> bunch = load_era()
    >>> bunch.data.shape
    (1000, 4)
    >>> int(bunch.target.min()), int(bunch.target.max())
    (0, 8)

    References
    ----------
    .. [1] A. Ben-David, "Automatic generation of symbolic multiattribute
           ordinal knowledge-based DSSs: methodology and applications",
           Decision Sciences, vol. 23, no. 6, pp. 1357-1372, 1992.

    """
    filename = "era.csv"
    data, target, target_names, descr = _load_csv_data(
        filename, descr_file_name="era.rst"
    )
    feature_names = [f"in{i + 1}" for i in range(4)]
    return _bundle(
        data,
        target,
        target_names,
        descr,
        feature_names,
        filename,
        return_X_y=return_X_y,
        as_frame=as_frame,
        caller_name="load_era",
    )


@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_esl(*, return_X_y=False, as_frame=False):
    """Load and return the ESL dataset (ordinal classification).

    Four ordinal psychometric scores assigned to industrial-job candidates
    by expert psychologists; the target is an overall fitness rating on a
    1-9 scale (9 ordered classes).

    =================   ==============
    Classes                          9
    Samples total                  488
    Dimensionality                   4
    Features                  ordinal
    =================   ==============

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas Series. If
        ``return_X_y`` is True, then ``(data, target)`` will be a pandas
        DataFrame and Series.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        See :func:`load_era` for the field layout. ``data`` has shape
        ``(488, 4)``; ``target_names`` are ``["1", ..., "9"]``.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays; see :func:`load_era` for details.

    Examples
    --------
    >>> from skordinal.datasets import load_esl
    >>> load_esl().data.shape
    (488, 4)

    References
    ----------
    .. [1] A. Ben-David, "Automatic generation of symbolic multiattribute
           ordinal knowledge-based DSSs: methodology and applications",
           Decision Sciences, vol. 23, no. 6, pp. 1357-1372, 1992.

    """
    filename = "esl.csv"
    data, target, target_names, descr = _load_csv_data(
        filename, descr_file_name="esl.rst"
    )
    feature_names = [f"in{i + 1}" for i in range(4)]
    return _bundle(
        data,
        target,
        target_names,
        descr,
        feature_names,
        filename,
        return_X_y=return_X_y,
        as_frame=as_frame,
        caller_name="load_esl",
    )


@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_lev(*, return_X_y=False, as_frame=False):
    """Load and return the LEV dataset (ordinal classification).

    Four ordinal student-rating attributes collected in anonymous university
    course evaluations; the target is an overall lecturer rating on a
    1-5 scale (5 ordered classes).

    =================   ==============
    Classes                          5
    Samples total                 1000
    Dimensionality                   4
    Features                  ordinal
    =================   ==============

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas Series. If
        ``return_X_y`` is True, then ``(data, target)`` will be a pandas
        DataFrame and Series.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        See :func:`load_era` for the field layout. ``data`` has shape
        ``(1000, 4)``; ``target_names`` are ``["1", ..., "5"]``.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays; see :func:`load_era` for details.

    Examples
    --------
    >>> from skordinal.datasets import load_lev
    >>> load_lev().data.shape
    (1000, 4)

    References
    ----------
    .. [1] A. Ben-David, "Automatic generation of symbolic multiattribute
           ordinal knowledge-based DSSs: methodology and applications",
           Decision Sciences, vol. 23, no. 6, pp. 1357-1372, 1992.

    """
    filename = "lev.csv"
    data, target, target_names, descr = _load_csv_data(
        filename, descr_file_name="lev.rst"
    )
    feature_names = [f"in{i + 1}" for i in range(4)]
    return _bundle(
        data,
        target,
        target_names,
        descr,
        feature_names,
        filename,
        return_X_y=return_X_y,
        as_frame=as_frame,
        caller_name="load_lev",
    )


@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_swd(*, return_X_y=False, as_frame=False):
    """Load and return the SWD dataset (ordinal classification).

    Ten ordinal risk-assessment attributes filled in by qualified social
    workers for child-safety cases; the target is the ordinal risk level
    used in family-court decisions, on a 1-4 scale (4 ordered classes).

    =================   ==============
    Classes                          4
    Samples total                 1000
    Dimensionality                  10
    Features                  ordinal
    =================   ==============

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas Series. If
        ``return_X_y`` is True, then ``(data, target)`` will be a pandas
        DataFrame and Series.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        See :func:`load_era` for the field layout. ``data`` has shape
        ``(1000, 10)``; ``target_names`` are ``["1", "2", "3", "4"]``.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays; see :func:`load_era` for details.

    Examples
    --------
    >>> from skordinal.datasets import load_swd
    >>> load_swd().data.shape
    (1000, 10)

    References
    ----------
    .. [1] A. Ben-David, "Automatic generation of symbolic multiattribute
           ordinal knowledge-based DSSs: methodology and applications",
           Decision Sciences, vol. 23, no. 6, pp. 1357-1372, 1992.

    """
    filename = "swd.csv"
    data, target, target_names, descr = _load_csv_data(
        filename, descr_file_name="swd.rst"
    )
    feature_names = [f"in{i + 1}" for i in range(10)]
    return _bundle(
        data,
        target,
        target_names,
        descr,
        feature_names,
        filename,
        return_X_y=return_X_y,
        as_frame=as_frame,
        caller_name="load_swd",
    )


@validate_params(
    {"return_X_y": ["boolean"], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_balance_scale(*, return_X_y=False, as_frame=False):
    """Load and return the Balance Scale dataset (ordinal classification).

    A synthetic dataset modelling Piaget-style balance-scale experiments.
    Each sample describes the weight and distance of objects placed on the
    left and right pans; the ordinal target indicates which way the scale
    tips: ``L`` (left), ``B`` (balanced), ``R`` (right).

    =================   ==============
    Classes                          3
    Samples per class     [288, 49, 288]
    Samples total                  625
    Dimensionality                   4
    Features            integer 1..5
    =================   ==============

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas Series. If
        ``return_X_y`` is True, then ``(data, target)`` will be a pandas
        DataFrame and Series.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        See :func:`load_era` for the field layout. ``feature_names`` are
        ``["left_weight", "left_distance", "right_weight",
        "right_distance"]``; ``target_names`` are ``["L", "B", "R"]``.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays; see :func:`load_era` for details.

    Examples
    --------
    >>> from skordinal.datasets import load_balance_scale
    >>> bunch = load_balance_scale()
    >>> bunch.data.shape
    (625, 4)
    >>> bunch.target_names.tolist()
    ['L', 'B', 'R']

    References
    ----------
    .. [1] R. S. Siegler, "Three aspects of cognitive development",
           Cognitive Psychology, vol. 8, pp. 481-520, 1976.

    """
    filename = "balance_scale.csv"
    data, target, target_names, descr = _load_csv_data(
        filename, descr_file_name="balance_scale.rst"
    )
    feature_names = [
        "left_weight",
        "left_distance",
        "right_weight",
        "right_distance",
    ]
    return _bundle(
        data,
        target,
        target_names,
        descr,
        feature_names,
        filename,
        return_X_y=return_X_y,
        as_frame=as_frame,
        caller_name="load_balance_scale",
    )
